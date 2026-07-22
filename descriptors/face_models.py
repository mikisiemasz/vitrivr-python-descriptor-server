"""Additional face-embedding models for the recognition-model comparison.

Every endpoint here shares the buffalo_l SCRFD detector loaded in face_recognition
and returns the same JSON shape as /extract/face_embedding,
so the engine reuses FaceDetectionResult and FaceDetectionTransformer unchanged.

Models:
  /extract/face_embedding_sface         SFace via DeepFace          128-d
  /extract/face_embedding_ghostfacenet  GhostFaceNet via DeepFace   512-d
  /extract/face_embedding_adaface       AdaFace IR-101 (vendored)   512-d

AdaFace needs two deployment steps (endpoint returns 500 until they're done):
  git clone https://github.com/mk-minchul/AdaFace ~/AdaFace
  # download adaface_ir101_webface12m.ckpt (Google Drive, see their README) into ~/AdaFace/
Override locations with env vars ADAFACE_REPO / ADAFACE_CKPT.

DeepFace models download their weights to ~/.deepface/weights on first request.
"""
import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import sys

import cv2
import numpy as np
import torch
from apiflask import APIBlueprint
from flask import request, jsonify
from insightface.utils import face_align

from descriptors.face_recognition import _app, _decode_image

face_models = APIBlueprint('face_models', __name__)


def _detect_and_embed(embed_fn, align_size: int):
    """Shared handler: SCRFD detect (same _app as ArcFace), align with the SCRFD
    landmarks, then embed each face with the given model"""
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
                aligned = face_align.norm_crop(img, landmark=f.kps, image_size=align_size)
                emb = embed_fn(aligned)
                results.append({
                    "embedding": emb.tolist(),
                    "bbox": f.bbox.tolist(),
                    "score": float(f.det_score),
                })
            except Exception as e:
                # Skip a single unembeddable face rather than dropping the whole frame.
                print(f"[ERROR] face skipped: {e!r}")

        return jsonify(results), 200

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] face_models embedding failed: {e!r}")
        return jsonify([]), 200


def _l2(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb)
    return emb / n if n > 0 else emb


def _deepface_embed(aligned_bgr: np.ndarray, model_name: str) -> np.ndarray:
    """Embed an already-aligned BGR crop with a DeepFace-wrapped model. Detection and
    alignment are skipped — they already happened via SCRFD upstream."""
    from deepface import DeepFace
    rep = DeepFace.represent(
        img_path=aligned_bgr,
        model_name=model_name,
        detector_backend="skip",
        enforce_detection=False,
        align=False,
    )
    return _l2(np.asarray(rep[0]["embedding"], dtype=np.float32))


@face_models.post("/extract/face_embedding_sface")
@face_models.doc(summary="Per-face SFace embeddings (128-d). Same SCRFD detection as "
                         "/extract/face_embedding; same response shape.")
def extract_face_embedding_sface():
    return _detect_and_embed(lambda a: _deepface_embed(a, "SFace"), align_size=112)


@face_models.post("/extract/face_embedding_ghostfacenet")
@face_models.doc(summary="Per-face GhostFaceNet embeddings (512-d). Same SCRFD detection as "
                         "/extract/face_embedding; same response shape.")
def extract_face_embedding_ghostfacenet():
    return _detect_and_embed(lambda a: _deepface_embed(a, "GhostFaceNet"), align_size=112)


_ADAFACE_REPO = os.environ.get("ADAFACE_REPO", os.path.expanduser("~/AdaFace"))
_ADAFACE_CKPT = os.environ.get(
    "ADAFACE_CKPT", os.path.join(_ADAFACE_REPO, "adaface_ir101_webface12m.ckpt")
)
_adaface = None


def _get_adaface():
    """Lazy-load the AdaFace IR-101 backbone from the cloned upstream repo + checkpoint."""
    global _adaface
    if _adaface is None:
        sys.path.insert(0, _ADAFACE_REPO)
        import net
        model = net.build_model('ir_101')
        ckpt = torch.load(_ADAFACE_CKPT, map_location='cpu')
        # Lightning checkpoint: strip the 'model.' prefix off backbone weights.
        sd = {k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        model.load_state_dict(sd)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _adaface = (model.eval().to(device), device)
    return _adaface


def _adaface_embed(aligned_bgr: np.ndarray) -> np.ndarray:
    """AdaFace preprocessing: BGR input (their training convention), ((x/255)-0.5)/0.5.
    Input is the SCRFD-aligned 112x112 crop — same size ArcFace uses, no resize needed."""
    model, device = _get_adaface()
    t = torch.from_numpy(aligned_bgr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    t = ((t / 255.0) - 0.5) / 0.5
    with torch.no_grad():
        emb, _ = model(t.to(device))  # AdaFace forward returns (embedding, norm)
    return _l2(emb[0].cpu().numpy())


@face_models.post("/extract/face_embedding_adaface")
@face_models.doc(summary="Per-face AdaFace IR-101 embeddings (512-d). Same SCRFD detection as "
                         "/extract/face_embedding; same response shape. Returns 500 until the "
                         "AdaFace repo + checkpoint are deployed (see module docstring).")
def extract_face_embedding_adaface():
    if not os.path.isfile(_ADAFACE_CKPT):
        return jsonify({"error": f"AdaFace checkpoint not found at {_ADAFACE_CKPT}; "
                                 "clone the repo and download the ckpt (see face_models.py)"}), 500
    return _detect_and_embed(_adaface_embed, align_size=112)
