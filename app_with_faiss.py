# save as app_with_faiss.py (replace your current file)
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from datetime import datetime
import plotly.express as px
import pandas as pd
import json
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import glob

# ---------------- FAISS import with fallback ----------------
try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    faiss = None
    _FAISS_AVAILABLE = False

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Enterprise Missing Person Detection (FAISS)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles (omitted in the interest of brevity - copy your previous CSS if needed)
st.markdown("""
<style>
/* keep your previous styles if desired */
</style>
""", unsafe_allow_html=True)

st.title("🔍 Enterprise Missing Person Detection — FAISS edition")
st.caption("FAISS (or fallback) for fast vector search. Upload reference image, point to footage folder, hit Process.")

# =====================================================================
# SESSION STATE
# =====================================================================
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# =====================================================================
# MODEL LOADING
# =====================================================================
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        mtcnn = MTCNN(
            keep_all=True,
            device=device,
            post_process=False,
            select_largest=False,
            min_face_size=30,
            thresholds=[0.6, 0.7, 0.7],
        )
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        if device.type == 'cuda':
            # half precision for speed
            try:
                resnet = resnet.half()
            except Exception:
                pass
        return resnet, mtcnn, device, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False

# =====================================================================
# EMBEDDING & FAISS UTILITIES
# =====================================================================
def normalize_vectors(x: np.ndarray, eps=1e-10):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)

def get_face_embedding_batch(images_list, resnet, device):
    """Batch embed faces. returns numpy array shape (N, D) dtype float32"""
    if not images_list:
        return np.zeros((0, 512), dtype=np.float32)
    try:
        tensors = []
        for img in images_list:
            img_resized = img.resize((160, 160))
            img_array = np.array(img_resized).astype(np.float32)
            img_tensor = torch.tensor(img_array).permute(2, 0, 1).float()
            img_tensor = (img_tensor - 127.5) / 128.0
            tensors.append(img_tensor)
        batch = torch.stack(tensors).to(device)
        if device.type == 'cuda':
            # keep dtype consistent: resnet may be half, but we'll cast back to float32 after forward
            try:
                batch = batch.half()
            except Exception:
                pass
        with torch.no_grad():
            emb = resnet(batch)
        emb = emb.cpu().numpy().astype(np.float32)
        return emb
    except Exception as e:
        st.error(f"Embedding batch error: {e}")
        return np.zeros((0, 512), dtype=np.float32)

def build_faiss_index(ref_embeddings: np.ndarray):
    """Build FAISS IndexFlatIP on normalized vectors. Returns (index, ref_norms)"""
    if ref_embeddings.size == 0:
        return None, None
    ref_norm = normalize_vectors(ref_embeddings.copy())
    dim = ref_norm.shape[1]
    if _FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(dim)  # cosine via normalized vectors
        index.add(ref_norm.astype(np.float32))
        return index, ref_norm
    else:
        # fallback: store ref_norm so we can dot product manually
        return None, ref_norm

# =====================================================================
# HELPER: single cosine function (fallback)
# =====================================================================
def batch_cosine_similarity(embs: np.ndarray, refs_norm: np.ndarray):
    """embs: (n, d) not normalized; refs_norm: (m, d) normalized -> returns (n, m) similarities"""
    if embs.size == 0 or refs_norm.size == 0:
        return np.zeros((embs.shape[0], refs_norm.shape[0]), dtype=np.float32)
    embs_norm = normalize_vectors(embs.copy())
    # dot product
    return np.dot(embs_norm, refs_norm.T)

# =====================================================================
# OTHER UTILITIES (motion detection, drawing)
# =====================================================================
def detect_motion(frame1, frame2, threshold=25):
    if frame1 is None or frame2 is None:
        return True
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    motion_pixels = np.sum(thresh > 0)
    total_pixels = thresh.shape[0] * thresh.shape[1]
    return (motion_pixels / total_pixels) > 0.005

def draw_box(frame, bbox, confidence, label="Match"):
    x1, y1, x2, y2 = map(int, bbox)
    if confidence >= 0.8:
        color = (0, 255, 0)
    elif confidence >= 0.6:
        color = (255, 165, 0)
    else:
        color = (255, 0, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    text = f"{label} {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
    cv2.putText(frame, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), thickness)
    return frame

# =====================================================================
# VIDEO PROCESSING (modified to use FAISS)
# =====================================================================
def process_single_video(video_path, faiss_index, refs_norm, ref_ids, resnet, mtcnn, device, 
                        confidence_threshold, sample_rate, min_face_size,
                        use_motion_detection, progress_callback=None, batch_size=16):
    """Process a single video; uses faiss_index/refs_norm for fast search."""
    detections = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"Failed to open {video_path}", "detections": []}
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = 0
        processed_count = 0
        skipped_motion = 0
        prev_frame = None
        face_batch = []
        face_metadata = []
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
            if use_motion_detection and prev_frame is not None:
                if not detect_motion(prev_frame, frame_bgr):
                    skipped_motion += 1
                    frame_count += 1
                    prev_frame = frame_bgr
                    continue
            prev_frame = frame_bgr.copy()
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            boxes, probs = mtcnn.detect(frame_pil)
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob < 0.9:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    face_width = x2 - x1
                    face_height = y2 - y1
                    if face_width < min_face_size or face_height < min_face_size:
                        continue
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(width, x2); y2 = min(height, y2)
                    face_crop = frame_rgb[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    face_pil = Image.fromarray(face_crop)
                    face_batch.append(face_pil)
                    face_metadata.append({
                        'frame_number': frame_count,
                        'bbox': [x1, y1, x2, y2],
                        'frame_bgr': frame_bgr.copy(),
                        'face_pil': face_pil
                    })
                    if len(face_batch) >= batch_size:
                        embeddings = get_face_embedding_batch(face_batch, resnet, device)  # (N, D)
                        if embeddings.shape[0] > 0:
                            embs_norm = normalize_vectors(embeddings.copy())
                            if _FAISS_AVAILABLE and faiss_index is not None:
                                # search top1
                                D, I = faiss_index.search(embs_norm.astype(np.float32), 1)
                                sims = D[:, 0]  # cosine similarities
                                ids = I[:, 0]
                            else:
                                # fallback: dot with refs_norm
                                sims_all = batch_cosine_similarity(embeddings, refs_norm)  # (N, M)
                                ids = np.argmax(sims_all, axis=1)
                                sims = sims_all[np.arange(sims_all.shape[0]), ids]
                            for sim, id_, meta in zip(sims, ids, face_metadata[:len(sims)]):
                                if sim >= confidence_threshold:
                                    timestamp = meta['frame_number'] / fps
                                    annotated_frame = meta['frame_bgr'].copy()
                                    annotated_frame = draw_box(annotated_frame, meta['bbox'], float(sim))
                                    detections.append({
                                        'video_path': video_path,
                                        'frame_number': meta['frame_number'],
                                        'timestamp': timestamp,
                                        'confidence': float(sim),
                                        'bbox': meta['bbox'],
                                        'face_image': meta['face_pil'],
                                        'annotated_frame': cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                        'matched_ref_id': ref_ids[id_] if ref_ids is not None else None
                                    })
                        # clear batch
                        face_batch = []
                        face_metadata = []
            processed_count += 1
            frame_count += 1
            if progress_callback and frame_count % 50 == 0:
                progress_callback(frame_count, total_frames, len(detections))
        # leftover batch
        if face_batch:
            embeddings = get_face_embedding_batch(face_batch, resnet, device)
            if embeddings.shape[0] > 0:
                embs_norm = normalize_vectors(embeddings.copy())
                if _FAISS_AVAILABLE and faiss_index is not None:
                    D, I = faiss_index.search(embs_norm.astype(np.float32), 1)
                    sims = D[:, 0]
                    ids = I[:, 0]
                else:
                    sims_all = batch_cosine_similarity(embeddings, refs_norm)
                    ids = np.argmax(sims_all, axis=1)
                    sims = sims_all[np.arange(sims_all.shape[0]), ids]
                for sim, id_, meta in zip(sims, ids, face_metadata):
                    if sim >= confidence_threshold:
                        timestamp = meta['frame_number'] / fps
                        annotated_frame = meta['frame_bgr'].copy()
                        annotated_frame = draw_box(annotated_frame, meta['bbox'], float(sim))
                        detections.append({
                            'video_path': video_path,
                            'frame_number': meta['frame_number'],
                            'timestamp': timestamp,
                            'confidence': float(sim),
                            'bbox': meta['bbox'],
                            'face_image': meta['face_pil'],
                            'annotated_frame': cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                            'matched_ref_id': ref_ids[id_] if ref_ids is not None else None
                        })
        cap.release()
        return {
            'video_path': video_path,
            'detections': detections,
            'total_frames': total_frames,
            'processed_frames': processed_count,
            'skipped_motion': skipped_motion,
            'fps': fps
        }
    except Exception as e:
        return {"error": str(e), "video_path": video_path, "detections": []}

# =====================================================================
# SIDEBAR: Inputs & settings
# =====================================================================
with st.sidebar:
    st.header("📤 Reference Image")
    ref_image_file = st.file_uploader("Upload Reference Photo (frontal)", type=["jpg","jpeg","png"])
    st.markdown("---")
    st.header("📁 Video Source")
    source_type = st.radio("Select Video Source:", ["📂 Local Folder Path", "📤 Upload Files"], index=0)
    video_files = []
    if source_type == "📂 Local Folder Path":
        folder_path = st.text_input("Enter Folder Path:", placeholder="C:/CCTV or /mnt/videos")
        if folder_path:
            if os.path.exists(folder_path):
                extensions = ['*.mp4','*.avi','*.mov','*.mkv','*.MP4']
                for ext in extensions:
                    video_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
                if video_files:
                    st.success(f"Found {len(video_files)} videos")
                    with st.expander("First 20 files"):
                        for vf in video_files[:20]:
                            st.text(str(Path(vf).name))
                else:
                    st.warning("No videos found.")
            else:
                st.error("Folder path does not exist.")
    else:
        uploaded_videos = st.file_uploader("Upload Video Files (200MB limit per file)", type=["mp4","avi","mov"], accept_multiple_files=True)
        if uploaded_videos:
            import tempfile
            temp_dir = tempfile.mkdtemp()
            for up in uploaded_videos:
                temp_path = os.path.join(temp_dir, up.name)
                with open(temp_path, 'wb') as f:
                    f.write(up.getbuffer())
                video_files.append(temp_path)
            st.success(f"Uploaded {len(video_files)} files (temporary)")

    st.markdown("---")
    st.header("⚙️ Processing Settings")
    confidence_threshold = st.slider("Confidence Threshold (cosine)", 0.3, 0.95, 0.55, 0.01)
    min_face_size = st.slider("Minimum Face Size (px)", 20, 200, 40)
    sample_rate = st.slider("Frame Sampling Rate (every Nth frame)", 1, 120, 15)
    use_motion_detection = st.checkbox("Enable Motion Detection", value=True)
    parallel_videos = st.slider("Parallel Video Processing (threads)", 1, min(8, max(1, os.cpu_count() or 4)), value=min(4, os.cpu_count() or 4))
    st.markdown("---")
    st.header("System Info")
    device_info = "GPU (FP16)" if torch.cuda.is_available() else "CPU"
    st.info(device_info)
    st.metric("CPU Cores", os.cpu_count() or "Unknown")
    st.markdown(f"FAISS available: {'✅' if _FAISS_AVAILABLE else '❌ (fallback enabled)'}")

# =====================================================================
# MAIN: build ref embeddings + index
# =====================================================================
if not ref_image_file:
    st.info("Upload a reference image to begin.")
    st.stop()

# load models
with st.spinner("Loading models..."):
    resnet, mtcnn, device, ok = load_models()
    if not ok:
        st.error("Model loading failed.")
        st.stop()

# process reference image(s)
ref_img = Image.open(ref_image_file).convert("RGB")
st.image(ref_img, width=200, caption="Reference Image")
with st.spinner("Extracting reference face embedding..."):
    ref_boxes, ref_probs = mtcnn.detect(ref_img)
    if ref_boxes is None or len(ref_boxes) == 0:
        st.error("No face found in reference image.")
        st.stop()
    best_idx = int(np.argmax(ref_probs)) if len(ref_probs) > 1 else 0
    ref_box = ref_boxes[best_idx]
    x1, y1, x2, y2 = map(int, ref_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(ref_img.width, x2), min(ref_img.height, y2)
    ref_face = ref_img.crop((x1, y1, x2, y2))
    ref_embedding = get_face_embedding_batch([ref_face], resnet, device)  # (1, D)
    if ref_embedding.size == 0:
        st.error("Failed to compute reference embedding.")
        st.stop()
    # Build FAISS index (or fallback)
    # For now we support a single reference; extend by accepting multiple reference images if needed.
    ref_ids = ["ref_0"]
    faiss_index, refs_norm = build_faiss_index(ref_embedding)
    st.success("Reference embedding prepared.")

# =====================================================================
# PROCESSING: start batch
# =====================================================================
if not video_files:
    st.info("No videos selected. Choose a folder or upload files.")
    st.stop()

if st.button("🚀 Start Batch Processing"):
    st.session_state.is_processing = True
    start_t = time.time()
    all_detections = []
    total_videos = len(video_files)
    processed_videos = 0

    overall_progress = st.progress(0)
    status_container = st.empty()
    live_container = st.empty()

    def progress_cb(fnum, total, detcount):
        # optional: can be enhanced to push per-video progress into UI
        return

    with ThreadPoolExecutor(max_workers=parallel_videos) as executor:
        future_map = {}
        for v in video_files:
            fut = executor.submit(
                process_single_video,
                v,
                faiss_index,
                refs_norm,
                ref_ids,
                resnet,
                mtcnn,
                device,
                confidence_threshold,
                sample_rate,
                min_face_size,
                use_motion_detection,
                progress_cb,
                16  # batch size
            )
            future_map[fut] = v
        for fut in as_completed(future_map):
            vpath = future_map[fut]
            try:
                res = fut.result()
                if 'error' in res:
                    st.warning(f"Error on {Path(vpath).name}: {res.get('error')}")
                else:
                    all_detections.extend(res['detections'])
            except Exception as e:
                st.error(f"Exception processing {Path(vpath).name}: {e}")
            processed_videos += 1
            overall_progress.progress(int((processed_videos/total_videos)*100))
            elapsed = time.time() - start_t
            speed = processed_videos / elapsed if elapsed > 0 else 0
            eta = (total_videos - processed_videos) / speed if speed > 0 else 0
            status_container.markdown(f"**Processed**: {processed_videos}/{total_videos} • ETA: {int(eta//60)}:{int(eta%60):02d} • Matches: {len(all_detections)}")
            if len(all_detections) > 0:
                live_container.success(f"🎯 {len(all_detections)} matches found so far")

    total_time = time.time() - start_t
    st.success("✅ All videos processed.")
    st.session_state.is_processing = False

    # ================== RESULTS UI ==================
    st.header("📊 Detection Results (FAISS)")
    if not all_detections:
        st.warning("No matches found. Try lowering the threshold or sampling more frames.")
    else:
        # metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Detections", len(all_detections))
        with col2:
            avg_conf = float(np.mean([d['confidence'] for d in all_detections]))
            st.metric("Avg Confidence", f"{avg_conf*100:.1f}%")
        with col3:
            max_conf = float(max([d['confidence'] for d in all_detections]))
            st.metric("Best Match", f"{max_conf*100:.1f}%")
        with col4:
            st.metric("Total Time", f"{int(total_time//60)}:{int(total_time%60):02d}")

        # timeline scatter
        df = pd.DataFrame([{
            'Video': Path(d['video_path']).name,
            'Time (s)': d['timestamp'],
            'Confidence (%)': d['confidence'] * 100,
            'Frame': d['frame_number']
        } for d in all_detections])

        fig = px.scatter(df, x='Time (s)', y='Confidence (%)', color='Video', hover_data=['Frame'], title='All Detections Across Videos', height=500)
        st.plotly_chart(fig, use_container_width=True)

        # detections per video
        st.markdown("---")
        st.subheader("Detections by Video")
        counts = df.groupby('Video').size().reset_index(name='Count').sort_values('Count', ascending=False)
        bar = px.bar(counts, x='Video', y='Count', title='Detections per Video', height=350)
        st.plotly_chart(bar, use_container_width=True)

        # show top N detections with images
        st.markdown("---")
        st.subheader("Top Detections (by confidence)")
        sorted_dets = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        for i, d in enumerate(sorted_dets[:10]):
            with st.expander(f"#{i+1} • {Path(d['video_path']).name} • Time {int(d['timestamp']//60)}:{int(d['timestamp']%60):02d} • {d['confidence']*100:.1f}%"):
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(d['face_image'], caption="Detected Face", use_column_width=True)
                    st.metric("Frame", f"{d['frame_number']:,}")
                    st.metric("Confidence", f"{d['confidence']*100:.1f}%")
                with c2:
                    st.image(d['annotated_frame'], caption="Annotated Frame (RGB)", use_column_width=True)

        # export report
        st.markdown("---")
        st.subheader("Export Report")
        report = {
            "processing_date": datetime.now().isoformat(),
            "total_videos": total_videos,
            "total_detections": len(all_detections),
            "avg_confidence": float(avg_conf),
            "detections": [
                {"video": Path(d['video_path']).name, "frame": int(d['frame_number']), "time_s": float(d['timestamp']), "confidence": float(d['confidence'])}
                for d in sorted_dets
            ]
        }
        st.download_button("Download JSON Report", json.dumps(report, indent=2), file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
        # also text
        text_report = f"DETECTION REPORT\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\nTotal Videos: {total_videos}\nTotal Detections: {len(all_detections)}\n\nTOP MATCHES:\n"
        for i, d in enumerate(sorted_dets[:20]):
            text_report += f"{i+1}. {Path(d['video_path']).name} | Frame {d['frame_number']} | Time {int(d['timestamp']//60)}:{int(d['timestamp']%60):02d} | {d['confidence']*100:.1f}%\n"
        st.download_button("Download Text Report", text_report, file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain")

    st.balloons()

