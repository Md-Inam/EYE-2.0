# app_with_faiss.py
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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Enterprise Missing Person Detection (FAISS)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Enterprise Missing Person Detection — FAISS edition")
st.caption("FAISS (or fallback) for fast vector search. Upload reference image, point to footage folder, hit Process.")

# ---------------- SESSION STATE ----------------
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# ---------------- MODEL LOADING ----------------
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
            try:
                resnet = resnet.half()
            except Exception:
                pass
        return resnet, mtcnn, device, True
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, False

# ---------------- EMBEDDING & FAISS UTILITIES ----------------
def normalize_vectors(x: np.ndarray, eps=1e-10):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)

def get_face_embedding_batch(images_list, resnet, device):
    if not images_list:
        return np.zeros((0, 512), dtype=np.float32)
    try:
        tensors = []
        for img in images_list:
            img_resized = img.resize((160, 160))
            img_array = np.array(img_resized).astype(np.float32)
            img_tensor = torch.tensor(img_array).permute(2,0,1).float()
            img_tensor = (img_tensor - 127.5)/128.0
            tensors.append(img_tensor)
        batch = torch.stack(tensors).to(device)
        if device.type == 'cuda':
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
    if ref_embeddings.size == 0:
        return None, None
    ref_norm = normalize_vectors(ref_embeddings.copy())
    dim = ref_norm.shape[1]
    if _FAISS_AVAILABLE:
        index = faiss.IndexFlatIP(dim)
        index.add(ref_norm.astype(np.float32))
        return index, ref_norm
    else:
        return None, ref_norm

def batch_cosine_similarity(embs: np.ndarray, refs_norm: np.ndarray):
    if embs.size == 0 or refs_norm.size == 0:
        return np.zeros((embs.shape[0], refs_norm.shape[0]), dtype=np.float32)
    embs_norm = normalize_vectors(embs.copy())
    return np.dot(embs_norm, refs_norm.T)

# ---------------- MOTION DETECTION & DRAW ----------------
def detect_motion(frame1, frame2, threshold=25):
    if frame1 is None or frame2 is None:
        return True
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
    motion_pixels = np.sum(thresh > 0)
    total_pixels = thresh.shape[0]*thresh.shape[1]
    return (motion_pixels/total_pixels) > 0.005

def draw_box(frame, bbox, confidence, label="Match"):
    x1, y1, x2, y2 = map(int, bbox)
    if confidence >= 0.8:
        color = (0,255,0)
    elif confidence >= 0.6:
        color = (255,165,0)
    else:
        color = (255,0,0)
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
    text = f"{label} {confidence*100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1+text_w, y1), color, -1)
    cv2.putText(frame, text, (x1, y1-5), font, font_scale, (255,255,255), thickness)
    return frame

# ---------------- VIDEO PROCESSING ----------------
def process_single_video(video_path, faiss_index, refs_norm, ref_ids, resnet, mtcnn, device,
                         confidence_threshold, sample_rate, min_face_size, use_motion_detection,
                         progress_callback=None, batch_size=16):
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
                        embeddings = get_face_embedding_batch(face_batch, resnet, device)
                        if embeddings.shape[0] > 0:
                            embs_norm = normalize_vectors(embeddings.copy())
                            if _FAISS_AVAILABLE and faiss_index is not None:
                                D, I = faiss_index.search(embs_norm.astype(np.float32), 1)
                                sims = D[:,0]
                                ids = I[:,0]
                            else:
                                sims_all = batch_cosine_similarity(embeddings, refs_norm)
                                ids = np.argmax(sims_all, axis=1)
                                sims = sims_all[np.arange(sims_all.shape[0]), ids]
                            for sim, id_, meta in zip(sims, ids, face_metadata[:len(sims)]):
                                if sim >= confidence_threshold:
                                    timestamp = meta['frame_number']/fps
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
                    sims = D[:,0]
                    ids = I[:,0]
                else:
                    sims_all = batch_cosine_similarity(embeddings, refs_norm)
                    ids = np.argmax(sims_all, axis=1)
                    sims = sims_all[np.arange(sims_all.shape[0]), ids]
                for sim, id_, meta in zip(sims, ids, face_metadata):
                    if sim >= confidence_threshold:
                        timestamp = meta['frame_number']/fps
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
  st.markdown("---")
        st.subheader("Top Detections (by confidence)")
        sorted_dets = sorted(all_detections, key=lambda x: x['confidence'], reverse=True)
        for i, d in enumerate(sorted_dets[:10]):
            with st.expander(f"#{i+1} • {Path(d['video_path']).name} • Time {int(d['timestamp']//60)}:{int(d['timestamp']%60):02d} • {d['confidence']*100:.1f}%"):
                c1, c2 = st.columns([1,2])
                with c1:
                    st.image(d['face_image'], caption="Detected Face", use_column_width=True)
                    st.metric("Frame", f"{d['frame_number']:,}")
                    st.metric("Confidence", f"{d['confidence']*100:.1f}%")
                with c2:
                    st.image(d['annotated_frame'], caption="Annotated Frame (RGB)", use_column_width=True)

        st.markdown("---")
        st.subheader("Export Report")
                report = {
            "processing_date": datetime.now().isoformat(),
            "total_videos": total_videos,
            "total_detections": len(all_detections),
            "avg_confidence": float(avg_conf),
            "detections": [
                {
                    "video": Path(d['video_path']).name,
                    "frame": int(d['frame_number']),
                    "time_s": float(d['timestamp']),
                    "confidence": float(d['confidence'])
                } 
                for d in sorted_dets
            ]
        }

        st.download_button(
            "Download JSON Report",
            json.dumps(report, indent=2),
            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

        # Text report
        text_report = f"DETECTION REPORT\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text_report += f"Total Videos: {total_videos}\nTotal Detections: {len(all_detections)}\n\nTOP MATCHES:\n"
        for i, d in enumerate(sorted_dets[:20]):
            text_report += (
                f"{i+1}. {Path(d['video_path']).name} | Frame {d['frame_number']} | "
                f"Time {int(d['timestamp']//60)}:{int(d['timestamp']%60):02d} | "
                f"{d['confidence']*100:.1f}%\n"
            )

        st.download_button(
            "Download Text Report",
            text_report,
            file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

    st.balloons()
