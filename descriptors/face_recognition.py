import base64
import json

import cv2
import numpy as np
from apiflask import APIBlueprint
from flask import request, jsonify
from insightface.app import FaceAnalysis

face_recognition = APIBlueprint('face_recognition', __name__)

_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
_app.prepare(ctx_id=0, det_size=(640, 640))


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


@face_recognition.post("/cluster/face_embeddings")
@face_recognition.doc(
    summary="Cluster a batch of face embeddings using HDBSCAN. "
            "Returns integer cluster labels (−1 = noise) aligned with the input list."
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
        )
        labels = clusterer.fit_predict(X).tolist()
        n_clusters = len(set(l for l in labels if l >= 0))

        return jsonify({"labels": labels, "n_clusters": n_clusters}), 200

    except Exception as e:
        print(f"[ERROR] cluster_face_embeddings failed: {e}")
        return jsonify({"error": str(e)}), 500
