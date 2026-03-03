"""
Face recognition — compare input image against enrolled database.

Usage:
    python3 recognize.py --image test.jpg
    python3 recognize.py --image test.jpg --threshold 0.5

Output: prints who is in the image (or "Unknown" if below threshold).
"""
import argparse

import cv2
import numpy as np
import onnxruntime as ort

# ── Constants ──────────────────────────────────────────────────────────────────
DETECT_INPUT_HW   = (256, 256)
SCORE_THRESHOLD   = 0.75
CAVAFACE_INPUT_HW = (112, 112)
DEFAULT_THRESHOLD = 0.45   # cosine similarity — ปรับได้, ยิ่งสูงยิ่ง strict

# ArcFace reference landmark positions for 112×112 aligned output
ARCFACE_DST = np.array([
    [38.2946, 51.6963],  # right eye
    [73.5318, 51.5014],  # left eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # right mouth corner
    [70.7299, 92.2041],  # left mouth corner
], dtype=np.float32)


# ── (reuse same helpers as build_database.py) ──────────────────────────────────
def generate_anchors(input_size: int = 256) -> np.ndarray:
    strides, anchors_per_cell = [16, 32], [2, 6]
    rows = []
    for stride, n in zip(strides, anchors_per_cell):
        grid = input_size // stride
        for y in range(grid):
            for x in range(grid):
                cx, cy = (x + 0.5) / grid, (y + 0.5) / grid
                for _ in range(n):
                    rows.append([cx, cy, 1.0, 1.0])
    return np.array(rows, dtype=np.float32).reshape(-1, 2, 2)


def resize_pad(img_rgb: np.ndarray, target_hw: tuple):
    h, w = img_rgb.shape[:2]
    th, tw = target_hw
    scale = min(th / h, tw / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_rgb, (nw, nh))
    pt, pl = (th - nh) // 2, (tw - nw) // 2
    padded = np.zeros((th, tw, 3), dtype=np.uint8)
    padded[pt:pt+nh, pl:pl+nw] = resized
    tensor = (padded.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return tensor, scale, pt, pl


def decode_boxes(raw, anchors, img_hw):
    H, W = img_hw
    center = anchors[:, 0:1, :] * np.array([[W, H]], dtype=np.float32)
    scale  = anchors[:, 1:2, :]
    K = raw.shape[1]
    mask = np.ones((K, 1), dtype=np.float32); mask[1] = 0.0
    return raw * scale + center * mask


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2-x1) * max(0.0, y2-y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thr):
    order = scores.argsort()[::-1].tolist()
    keep = []
    while order:
        i = order.pop(0); keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def detect_face(img_rgb, det_sess, anchors):
    """Return ArcFace-aligned face [112,112,3] or None if no face found."""
    inp, scale, pt, pl = resize_pad(img_rgb, DETECT_INPUT_HW)
    name = det_sess.get_inputs()[0].name
    c1, c2, s1, s2 = det_sess.run(None, {name: inp})

    coords = np.concatenate([c1[0], c2[0]], axis=0).reshape(-1, 8, 2)
    scores = np.concatenate([s1[0], s2[0]], axis=0).reshape(-1)
    scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100.0, 100.0)))

    decoded = decode_boxes(coords, anchors, DETECT_INPUT_HW)

    mask = scores >= SCORE_THRESHOLD
    if not mask.any():
        return None

    best_idx = int(np.where(mask)[0][scores[mask].argmax()])

    # Extract 5 facial landmarks and map back to original image space
    kps = decoded[best_idx, 2:7, :].copy()
    kps[:, 0] = (kps[:, 0] - pl) / scale
    kps[:, 1] = (kps[:, 1] - pt) / scale

    # Affine-align to ArcFace reference positions → 112×112
    M, _ = cv2.estimateAffinePartial2D(kps, ARCFACE_DST, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img_rgb, M, (CAVAFACE_INPUT_HW[1], CAVAFACE_INPUT_HW[0]))


def get_embedding(face_rgb, cava_sess):
    inp  = (face_rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    name = cava_sess.get_inputs()[0].name
    emb  = cava_sess.run(None, {name: inp})[0].reshape(-1)
    return emb / (np.linalg.norm(emb) + 1e-8)


# ── Recognition ────────────────────────────────────────────────────────────────
def recognize(image_path: str, db_path: str, threshold: float,
              detector_path: str, cavaface_path: str) -> None:

    # Load database
    db_file = np.load(db_path)
    database = {name: db_file[name] for name in db_file.files}
    if not database:
        print("Database is empty. Run build_database.py first.")
        return
    print(f"Database loaded: {list(database.keys())}")

    # Load models
    anchors   = generate_anchors()
    det_sess  = ort.InferenceSession(detector_path)
    cava_sess = ort.InferenceSession(cavaface_path)

    # Load input image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print(f"Cannot read image: {image_path}")
        return
    img_rgb = img_bgr[:, :, ::-1]

    # Detect face
    face = detect_face(img_rgb, det_sess, anchors)
    if face is None:
        print("No face detected in the image.")
        return

    # Get embedding
    query_emb = get_embedding(face, cava_sess)

    # Compare with database (cosine similarity)
    best_name  = "Unknown"
    best_score = -1.0
    for name, emb in database.items():
        score = float(np.dot(query_emb, emb))   # both L2-normalized → cosine sim
        print(f"  {name}: {score:.3f}")
        if score > best_score:
            best_score = score
            best_name  = name

    print()
    if best_score >= threshold:
        print(f"Result : {best_name}  (similarity={best_score:.3f})")
    else:
        print(f"Result : Unknown  (best match={best_name}, similarity={best_score:.3f} < threshold={threshold})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition")
    parser.add_argument("--image",     required=True, help="Input image path")
    parser.add_argument("--db",        default="face_db.npz",
                        help="Database file from build_database.py (default: face_db.npz)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Cosine similarity threshold (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--detector",  default="FaceDetector.onnx")
    parser.add_argument("--cavaface",  default="CavaFace.onnx")
    args = parser.parse_args()

    recognize(args.image, args.db, args.threshold, args.detector, args.cavaface)
