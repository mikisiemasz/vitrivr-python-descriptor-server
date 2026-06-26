"""
Experimental Face tracking endpoint. Inspired loosely by ByteTrack

Inputs pre-computed FACE_DETECTION rows (bbox + embedding + time, grouped per source video)
and emits per-video tracks: groupings of detections that the appearance-driven greedy
associator considers the same identity across a contiguous time window.

per frame:
(a unique time_ns), score every (open_track, detection) pair as
    alpha * cosine + (1 - alpha) * IoU
sort descending, greedily assign one-to-one. Hard cosine floor prevents IoU from
extending a track on a face that doesn't look like the track's running mean. Tracks
that go max_gap_sec without a match are closed.
"""

from collections import defaultdict

import numpy as np
from apiflask import APIBlueprint
from flask import jsonify, request

face_tracking = APIBlueprint("face_tracking", __name__)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _TrackState:
    __slots__ = (
        "track_id", "members", "mean_emb", "n",
        "started_ns", "last_time_ns", "last_bbox",
    )

    def __init__(self, track_id, det, emb):
        self.track_id = track_id
        self.members = [det]
        self.mean_emb = emb.copy()
        self.n = 1
        self.started_ns = det["time_ns"]
        self.last_time_ns = det["time_ns"]
        self.last_bbox = det["bbox"]

    def add(self, det, emb):
        m = (self.mean_emb * self.n + emb) / (self.n + 1)
        nrm = float(np.linalg.norm(m))
        if nrm > 0:
            m = m / nrm
        self.mean_emb = m
        self.n += 1
        self.last_time_ns = det["time_ns"]
        self.last_bbox = det["bbox"]


def _track_one_video(detections, alpha, sim_threshold, max_gap_ns):
    embs = [np.asarray(d["embedding"], dtype=np.float32) for d in detections]

    by_time = defaultdict(list)
    for i, d in enumerate(detections):
        by_time[d["time_ns"]].append(i)

    next_id = 0
    open_tr = []
    closed = []

    for t in sorted(by_time):
        still = []
        for tr in open_tr:
            if t - tr.last_time_ns > max_gap_ns:
                closed.append(tr)
            else:
                still.append(tr)
        open_tr = still

        det_idxs = by_time[t]

        if not open_tr:
            for di in det_idxs:
                open_tr.append(_TrackState(next_id, detections[di], embs[di]))
                next_id += 1
            continue

        scored = []
        for ti, tr in enumerate(open_tr):
            for di in det_idxs:
                cos = float(np.dot(tr.mean_emb, embs[di]))
                box_iou = _iou(tr.last_bbox, detections[di]["bbox"])
                scored.append((alpha * cos + (1 - alpha) * box_iou, cos, ti, di))
        scored.sort(reverse=True)

        used_t, used_d = set(), set()
        for _, cos, ti, di in scored:
            if ti in used_t or di in used_d:
                continue
            if cos < sim_threshold:
                continue
            open_tr[ti].add(detections[di], embs[di])
            used_t.add(ti)
            used_d.add(di)

        for di in det_idxs:
            if di not in used_d:
                open_tr.append(_TrackState(next_id, detections[di], embs[di]))
                next_id += 1

    closed.extend(open_tr)
    return closed


@face_tracking.post("/track/face_tracks")
@face_tracking.doc(
    summary="Greedy appearance-driven face tracking. "
            "Input: pre-computed FACE_DETECTIONs grouped per source video (bbox + "
            "embedding + time_ns). Output: per-video tracks with mean embedding."
)
def track_face_tracks():
    """
    Request JSON:
      {
        "videos": {
          "<source_id>": {
            "detections": [
              {
                "face_id":   "<uuid>",
                "time_ns":   1234567890,
                "bbox":      [x1, y1, x2, y2],     # any consistent space; IoU is the only consumer
                "embedding": [0.01, ...]           # 512-d L2-normalized
              },
              ...
            ]
          },
          ...
        },
        "alpha":         0.85,    # appearance weight (1.0 = cosine only)
        "threshold":     0.55,    # hard cosine floor; below this never extends a track
        "max_gap_sec":   60.0     # close a track after this much idle time
      }

    Response:
      {
        "videos": {
          "<source_id>": {
            "tracks": [
              {
                "track_id":         0,
                "start_ns":         <ns>,
                "end_ns":           <ns>,
                "detection_count":  <int>,
                "member_face_ids":  ["<uuid>", ...],
                "mean_embedding":   [..512 floats..]
              },
              ...
            ]
          },
          ...
        }
      }
    """
    try:
        body = request.get_json(force=True, silent=True) or {}
        videos_in = body.get("videos") or {}
        alpha = float(body.get("alpha", 0.85))
        threshold = float(body.get("threshold", 0.55))
        max_gap_sec = float(body.get("max_gap_sec", 60.0))
        max_gap_ns = int(max_gap_sec * 1_000_000_000)

        videos_out = {}
        for source_id, payload in videos_in.items():
            detections = payload.get("detections") or []
            if not detections:
                videos_out[source_id] = {"tracks": []}
                continue
            detections = sorted(detections, key=lambda d: d.get("time_ns", 0))
            tracks = _track_one_video(detections, alpha, threshold, max_gap_ns)
            videos_out[source_id] = {
                "tracks": [
                    {
                        "track_id": tr.track_id,
                        "start_ns": int(tr.started_ns),
                        "end_ns": int(tr.last_time_ns),
                        "detection_count": tr.n,
                        "member_face_ids": [m["face_id"] for m in tr.members],
                        "mean_embedding": tr.mean_emb.tolist(),
                    }
                    for tr in tracks
                ]
            }

        return jsonify({"videos": videos_out}), 200

    except Exception as e:
        print(f"[ERROR] track_face_tracks failed: {e}")
        return jsonify({"error": str(e)}), 500
