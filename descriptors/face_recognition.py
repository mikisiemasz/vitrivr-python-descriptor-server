import base64
import json
import tempfile
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from apiflask import APIBlueprint
from flask import request, jsonify
from insightface.app import FaceAnalysis

face_recognition = APIBlueprint('face_recognition', __name__)

_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
_app.prepare(ctx_id=0, det_size=(640, 640))

_known_gallery = {}  


def _decode_image(data: str) -> np.ndarray:
    """Decode base64 data URL to OpenCV BGR image."""
    _, encoded = data.split("base64,", 1)
    raw = base64.b64decode(encoded)
    img_array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@face_recognition.post("/extract/face_embedding")
@face_recognition.doc(
    summary="Extract ArcFace embedding(s) for all detected faces. "
            "Returns a single 512-d vector (mean of all faces) for vitrivr indexing."
)
def extract_face_embedding():
    # Returns a single FloatVector (512-d) that is the L2-normalized mean
    # of all ArcFace embeddings found in the image.
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify([0.0] * 512), 200

    try:
        img = _decode_image(data)
        faces = _app.get(img)

        if not faces:
            return jsonify([0.0] * 512), 200

        # Return mean embedding (single-person frames)
        embeddings = [f.embedding / np.linalg.norm(f.embedding) for f in faces]
        mean_emb = np.mean(embeddings, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)

        return jsonify(mean_emb.tolist()), 200

    except Exception as e:
        print(f"[ERROR] face_embedding failed: {e}")
        return jsonify([0.0] * 512), 200


@face_recognition.post("/extract/face_embeddings_multi")
@face_recognition.doc(
    summary="Extract per-face ArcFace embeddings with bounding boxes. "
            "Returns a list of {bbox, embedding, identity} for each detected face."
)
def extract_face_embeddings_multi():
    """
    Returns detailed per-face information.
    """
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify({"faces": []}), 200

    try:
        img = _decode_image(data)
        faces = _app.get(img)

        results = []
        for face in faces:
            emb = face.embedding
            emb_norm = (emb / np.linalg.norm(emb)).tolist()
            bbox = face.bbox.astype(int).tolist()

            # match against known gallery
            identity = "unknown"
            confidence = 0.0
            for name, ref_emb in _known_gallery.items():
                sim = float(np.dot(emb_norm, ref_emb))
                if sim > confidence:
                    confidence = sim
                    if sim > 0.4:
                        identity = name

            results.append({
                "bbox": bbox,
                "embedding": emb_norm,
                "identity": identity,
                "confidence": round(confidence, 4)
            })

        return jsonify({"faces": results}), 200

    except Exception as e:
        print(f"[ERROR] face_embeddings_multi failed: {e}")
        return jsonify({"faces": []}), 200