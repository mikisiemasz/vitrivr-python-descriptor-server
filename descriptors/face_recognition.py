import base64
import json
import os

import cv2
import numpy as np
import torch
from apiflask import APIBlueprint
from flask import request, jsonify
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from facenet_pytorch import InceptionResnetV1

face_recognition = APIBlueprint('face_recognition', __name__)

_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
_app.prepare(ctx_id=0, det_size=(640, 640))

# Second embedder for the model comparison. Detection stays with buffalo_l SCRFD;
_facenet_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_facenet = InceptionResnetV1(pretrained='vggface2').eval().to(_facenet_device)


def _facenet_embed(aligned_bgr: np.ndarray) -> np.ndarray:
    """Embed a 160x160 aligned BGR face crop with FaceNet (InceptionResnetV1).

    Applies FaceNet's fixed image standardization ((x - 127.5) / 128) on RGB input and
    L2-normalizes the output so cosine distance matches the ArcFace path.
    """
    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    tensor = (tensor - 127.5) / 128.0
    with torch.no_grad():
        emb = _facenet(tensor.to(_facenet_device)).cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def _decode_image(data: str) -> np.ndarray:
    """Decode base64 data URL to OpenCV BGR image."""
    _, encoded = data.split("base64,", 1)
    raw = base64.b64decode(encoded)
    img_array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@face_recognition.post("/extract/face_embedding")
@face_recognition.doc(
    summary="Extract per-face ArcFace embeddings with bounding box and quality score. "
            "Returns a JSON array of objects, one per detected face. "
            "Empty array if no faces detected."
)
def extract_face_embedding():
    """
    Returns a JSON array where each element contains:
      - embedding: 512-d normalized ArcFace vector
      - bbox:      [x1, y1, x2, y2] bounding box in pixels
      - score:     detection confidence score
    """
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify([]), 200

    try:
        img = _decode_image(data)
        faces = _app.get(img)

        if not faces:
            return jsonify([]), 200

        results = []
        for f in faces:
            emb = f.embedding / np.linalg.norm(f.embedding)
            results.append({
                "embedding": emb.tolist(),
                "bbox": f.bbox.tolist(),
                "score": float(f.det_score),
            })

        return jsonify(results), 200

    except Exception as e:
        print(f"[ERROR] face_embedding failed: {e}")
        return jsonify([]), 200


@face_recognition.post("/extract/face_embedding_facenet")
@face_recognition.doc(
    summary="Extract per-face FaceNet embeddings with bounding box and quality score. "
            "Detection uses the same buffalo_l SCRFD as /extract/face_embedding; only the "
            "embedding network differs. Returns a JSON array of objects, one per detected face. "
            "Empty array if no faces detected."
)
def extract_face_embedding_facenet():
    """
    Returns a JSON array where each element contains:
      - embedding: 512-d L2-normalized FaceNet (InceptionResnetV1, vggface2) vector
      - bbox:      [x1, y1, x2, y2] bounding box in pixels
      - score:     detection confidence score

    Same response schema as /extract/face_embedding so the engine reuses FaceDetectionResult.
    """
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify([]), 200

    try:
        img = _decode_image(data)
        faces = _app.get(img)

        if not faces:
            return jsonify([]), 200

        results = []
        for f in faces:
            try:
                # Align with the SCRFD landmarks (NOT MTCNN) so the detector stays constant.
                # norm_crop only supports sizes that are multiples of 112 or 128, so align at
                # 224 (=112*2) and downscale to FaceNet's 160 input — preserves more detail than
                # aligning at 112 and upscaling.
                aligned = face_align.norm_crop(img, landmark=f.kps, image_size=224)
                aligned = cv2.resize(aligned, (160, 160), interpolation=cv2.INTER_AREA)
                emb = _facenet_embed(aligned)
                results.append({
                    "embedding": emb.tolist(),
                    "bbox": f.bbox.tolist(),
                    "score": float(f.det_score),
                })
            except Exception as e:
                # Skip a single unembeddable face rather than dropping the whole frame's results.
                print(f"[ERROR] face_embedding_facenet face skipped: {e!r}")

        return jsonify(results), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] face_embedding_facenet failed: {e!r}")
        return jsonify([]), 200


@face_recognition.post("/cluster/face_embeddings")
@face_recognition.doc(
    summary="Cluster a batch of face embeddings using HDBSCAN. "
            "Returns integer cluster labels (-1 = noise) aligned with the input list."
)
def cluster_face_embeddings():
    """
    Request JSON body:
      {
        "embeddings":          [[...512 floats...], ...],
        "detection_ids":       ["uuid1", "uuid2", ...],   // optional, echoed back
        "min_cluster_size":    5,
        "min_samples":         3,
        "min_gallery_cluster_size": 10
      }

    Response:
      {
        "labels":    [0, -1, 1, 0, ...],
        "n_clusters": <int>
      }
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        embeddings_raw = body.get("embeddings", [])
        min_cluster_size = int(body.get("min_cluster_size", 5))
        min_samples = int(body.get("min_samples", 3))
        # 'eom' picks the most *stable* condensed-tree nodes, which at corpus scale merges
        # every identity reachable through low-density bridge faces into one mega-cluster.
        # 'leaf' selects the finest granularity instead. Per-request override wins; the
        # HDBSCAN_SELECTION env var lets engine builds that don't send the field opt in.
        selection = str(
            body.get("cluster_selection_method")
            or os.environ.get("HDBSCAN_SELECTION", "eom")
        ).lower()
        if selection not in ("eom", "leaf"):
            return jsonify({"error": f"invalid cluster_selection_method '{selection}'"}), 400

        if not embeddings_raw:
            return jsonify({"labels": [], "n_clusters": 0}), 200

        X = np.array(embeddings_raw, dtype=np.float32)

        # Normalise rows (caller should already normalise, but be defensive)
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            cluster_selection_method=selection,
        )
        labels = clusterer.fit_predict(X).tolist()
        n_clusters = len(set(l for l in labels if l >= 0))

        return jsonify({"labels": labels, "n_clusters": n_clusters}), 200

    except Exception as e:
        print(f"[ERROR] cluster_face_embeddings failed: {e}")
        return jsonify({"error": str(e)}), 500
