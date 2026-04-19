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


def _decode_image(data: str) -> np.ndarray:
    """Decode base64 data URL to OpenCV BGR image."""
    _, encoded = data.split("base64,", 1)
    raw = base64.b64decode(encoded)
    img_array = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


@face_recognition.post("/extract/face_embedding")
@face_recognition.doc(
    summary="Extract per-face ArcFace embeddings. "
            "Returns a JSON array of 512-d vectors, one per detected face. "
            "Empty array if no faces detected."
)
def extract_face_embedding():
    """
    Returns: JSON array of arrays, where each inner array is a 512-d face embedding.
    """
    data = request.form.get('data', '')
    if not data or "base64," not in data:
        return jsonify([]), 200

    try:
        img = _decode_image(data)
        faces = _app.get(img)

        if not faces:
            return jsonify([]), 200

        # One embedding per face
        results = []
        for f in faces:
            emb = f.embedding / np.linalg.norm(f.embedding)
            results.append(emb.tolist())

        return jsonify(results), 200

    except Exception as e:
        print(f"[ERROR] face_embedding failed: {e}")
        return jsonify([]), 200