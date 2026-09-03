"""
face_guard.py
──────────────────────────────────────────────────────────────
ตรวจสอบภาพก่อนส่งเข้า ViT:
  1. มีใบหน้าหรือไม่           → บล็อก
  2. มีหลายหน้าหรือไม่          → บล็อก (ไม่รู้ว่าจะวัดใคร)
  3. ใบหน้าเล็กเกินไปหรือไม่     → บล็อก
  4. ภาพเบลอ / มืด / สว่างเกิน  → เตือน (ไม่บล็อก)

Backend: mediapipe → opencv haar → none (fail-open พร้อมคำเตือน)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw

# ══════════════════════════════════════════════════════════
# THRESHOLDS — ปรับได้ตามความเข้มงวดที่ต้องการ
# ══════════════════════════════════════════════════════════
MIN_FACE_AREA_RATIO = 0.030   # พื้นที่ใบหน้า / พื้นที่ภาพ ขั้นต่ำ (3%)
MIN_FACE_PIXELS     = 72      # ความกว้างใบหน้าขั้นต่ำ (px)
MIN_IMAGE_SIDE      = 160     # ด้านสั้นสุดของภาพขั้นต่ำ
BLUR_WARN           = 55.0    # variance of Laplacian ต่ำกว่านี้ = เบลอ
BLUR_BLOCK          = 18.0    # เบลอหนักมาก → บล็อก
DARK_WARN           = 55      # ค่าเฉลี่ยความสว่างต่ำกว่านี้ = มืด
BRIGHT_WARN         = 212     # สูงกว่านี้ = สว่างจ้า/โอเวอร์
MP_CONFIDENCE       = 0.5


# ══════════════════════════════════════════════════════════
# RESULT SCHEMA
# ══════════════════════════════════════════════════════════
@dataclass
class FaceCheckResult:
    ok: bool
    code: str                       # OK / NO_FACE / MULTIPLE_FACES / ...
    message: str
    hint: str = ""
    faces: list = field(default_factory=list)     # [(x, y, w, h), ...]
    warnings: list = field(default_factory=list)  # [(code, msg), ...]
    backend: str = "none"
    metrics: dict = field(default_factory=dict)

    @property
    def face_count(self) -> int:
        return len(self.faces)


# ══════════════════════════════════════════════════════════
# BACKEND INIT (lazy + singleton)
# ══════════════════════════════════════════════════════════
_BACKEND: Optional[str] = None
_DETECTOR = None
_DETECTOR_PROFILE = None


def _init_backend() -> str:
    global _BACKEND, _DETECTOR, _DETECTOR_PROFILE
    if _BACKEND is not None:
        return _BACKEND

    # ---------- 1) MediaPipe (แม่นสุด รองรับหน้าเอียง/หลายมุม) ----------
    try:
        import mediapipe as mp
        _DETECTOR = mp.solutions.face_detection.FaceDetection(
            model_selection=1,                      # 1 = ระยะไกลถึง 5 เมตร
            min_detection_confidence=MP_CONFIDENCE,
        )
        _BACKEND = "mediapipe"
        return _BACKEND
    except Exception:
        pass

    # ---------- 2) OpenCV Haar Cascade (ติดมากับ opencv-python) ----------
    try:
        import cv2
        front = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        prof = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml")
        if front.empty():
            raise RuntimeError("โหลด cascade ไม่สำเร็จ")
        _DETECTOR, _DETECTOR_PROFILE = front, prof
        _BACKEND = "opencv"
        return _BACKEND
    except Exception:
        pass

    # ---------- 3) ไม่มีอะไรเลย → fail-open ----------
    _BACKEND = "none"
    return _BACKEND


def get_backend_name() -> str:
    return _init_backend()


# ══════════════════════════════════════════════════════════
# DETECTION
# ══════════════════════════════════════════════════════════
def detect_faces(img: Image.Image) -> list[tuple[int, int, int, int]]:
    backend = _init_backend()
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    if backend == "mediapipe":
        res = _DETECTOR.process(arr)
        boxes = []
        if res.detections:
            for det in res.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                bw = min(w - x, int(bb.width * w))
                bh = min(h - y, int(bb.height * h))
                if bw > 0 and bh > 0:
                    boxes.append((x, y, bw, bh))
        return boxes

    if backend == "opencv":
        import cv2
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)          # ช่วยกรณีแสงไม่ดี
        found = _DETECTOR.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
        boxes = [tuple(map(int, b)) for b in found]

        # ถ้าไม่เจอหน้าตรง ลองหน้าด้านข้าง (ทั้งภาพปกติและภาพกลับซ้าย-ขวา)
        if not boxes and _DETECTOR_PROFILE is not None \
                and not _DETECTOR_PROFILE.empty():
            for flip in (False, True):
                g = cv2.flip(gray, 1) if flip else gray
                p = _DETECTOR_PROFILE.detectMultiScale(
                    g, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
                for (x, y, bw, bh) in p:
                    x = w - x - bw if flip else x
                    boxes.append((int(x), int(y), int(bw), int(bh)))
                if boxes:
                    break
        return _dedup(boxes)

    return []   # backend == "none"


def _dedup(boxes, iou_thr: float = 0.4):
    """รวมกล่องที่ทับซ้อนกัน (Haar มักเจอซ้ำหลายกล่องบนหน้าเดียว)"""
    out = []
    for b in sorted(boxes, key=lambda x: -x[2] * x[3]):
        if all(_iou(b, k) < iou_thr for k in out):
            out.append(b)
    return out


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


# ══════════════════════════════════════════════════════════
# IMAGE QUALITY METRICS (numpy ล้วน ไม่ต้องพึ่ง cv2)
# ══════════════════════════════════════════════════════════
def _laplacian_variance(gray: np.ndarray) -> float:
    """ยิ่งค่าต่ำ = ยิ่งเบลอ (ขอบภาพน้อย)"""
    g = gray.astype(np.float32)
    lap = (4 * g[1:-1, 1:-1]
           - g[:-2, 1:-1] - g[2:, 1:-1]
           - g[1:-1, :-2] - g[1:-1, 2:])
    return float(lap.var())


def _quality(img: Image.Image, box=None) -> dict:
    region = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3])) if box else img
    gray = np.array(region.convert("L"))
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return {"blur": 999.0, "brightness": 128.0, "contrast": 50.0}
    return {
        "blur": _laplacian_variance(gray),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
    }


# ══════════════════════════════════════════════════════════
# MAIN GUARD
# ══════════════════════════════════════════════════════════
def check_face(img: Image.Image, allow_multiple: bool = False) -> FaceCheckResult:
    backend = _init_backend()
    W, H = img.size

    # ---------- 0) ความละเอียดภาพ ----------
    if min(W, H) < MIN_IMAGE_SIDE:
        return FaceCheckResult(
            ok=False, code="LOW_RESOLUTION", backend=backend,
            message=f"ภาพมีความละเอียดต่ำเกินไป ({W}×{H} px)",
            hint=f"กรุณาใช้ภาพที่ด้านสั้นสุดอย่างน้อย {MIN_IMAGE_SIDE} px "
                 f"เพื่อให้ ViT จับรายละเอียดใบหน้าได้",
        )

    # ---------- backend ไม่พร้อม → fail-open ----------
    if backend == "none":
        return FaceCheckResult(
            ok=True, code="GUARD_UNAVAILABLE", backend=backend,
            message="ข้ามการตรวจใบหน้า (ไม่พบไลบรารีตรวจจับ)",
            warnings=[("GUARD_UNAVAILABLE",
                       "ไม่พบ mediapipe หรือ opencv-python จึงข้ามการตรวจใบหน้า "
                       "ติดตั้งด้วย: pip install mediapipe หรือ pip install opencv-python")],
        )

    # ---------- 1) ตรวจจับใบหน้า ----------
    faces = detect_faces(img)

    if not faces:
        return FaceCheckResult(
            ok=False, code="NO_FACE", backend=backend, faces=[],
            message="ไม่พบใบหน้าคนในภาพนี้",
            hint="กรุณาอัปโหลดภาพถ่ายใบหน้าคนจริง หันหน้าตรง มองกล้อง "
                 "ในที่มีแสงสว่างเพียงพอ และไม่สวมหน้ากาก/แว่นกันแดด",
        )

    if len(faces) > 1 and not allow_multiple:
        return FaceCheckResult(
            ok=False, code="MULTIPLE_FACES", backend=backend, faces=faces,
            message=f"พบใบหน้า {len(faces)} คนในภาพ",
            hint="ระบบประเมินได้ครั้งละ 1 คน กรุณาใช้ภาพที่มีใบหน้าเพียงคนเดียว "
                 "หรือครอปเฉพาะบุคคลที่ต้องการประเมิน",
        )

    # ---------- 2) ขนาดใบหน้า ----------
    faces.sort(key=lambda b: -b[2] * b[3])
    x, y, fw, fh = faces[0]
    area_ratio = (fw * fh) / float(W * H)

    if area_ratio < MIN_FACE_AREA_RATIO or fw < MIN_FACE_PIXELS:
        return FaceCheckResult(
            ok=False, code="FACE_TOO_SMALL", backend=backend, faces=faces,
            message=f"ใบหน้าในภาพเล็กเกินไป (กว้าง {fw}px, "
                    f"คิดเป็น {area_ratio*100:.1f}% ของภาพ)",
            hint="กรุณาถ่ายใกล้ขึ้น หรือครอปให้ใบหน้าเต็มเฟรมมากกว่านี้ "
                 f"(ต้องการอย่างน้อย {MIN_FACE_AREA_RATIO*100:.0f}% ของภาพ)",
            metrics={"face_area_ratio": area_ratio, "face_width": fw},
        )

    # ---------- 3) คุณภาพภาพ (เตือน ไม่บล็อก ยกเว้นเบลอหนัก) ----------
    q = _quality(img, (x, y, fw, fh))
    warns = []

    if q["blur"] < BLUR_BLOCK:
        return FaceCheckResult(
            ok=False, code="TOO_BLURRY", backend=backend, faces=faces,
            message=f"ภาพเบลอมากเกินไป (ค่าความคมชัด {q['blur']:.1f})",
            hint="กรุณาถ่ายใหม่ให้ภาพคมชัด ถือกล้องนิ่ง ๆ และโฟกัสที่ใบหน้า",
            metrics=q,
        )
    if q["blur"] < BLUR_WARN:
        warns.append(("BLURRY",
                      f"ภาพค่อนข้างเบลอ (ค่าความคมชัด {q['blur']:.1f}) "
                      f"ผลลัพธ์อาจคลาดเคลื่อน"))
    if q["brightness"] < DARK_WARN:
        warns.append(("TOO_DARK",
                      f"ใบหน้าค่อนข้างมืด (ความสว่าง {q['brightness']:.0f}/255) "
                      f"แนะนำถ่ายในที่แสงสว่างกว่านี้"))
    elif q["brightness"] > BRIGHT_WARN:
        warns.append(("TOO_BRIGHT",
                      f"ภาพสว่างจ้า/โอเวอร์ (ความสว่าง {q['brightness']:.0f}/255) "
                      f"รายละเอียดใบหน้าอาจหายไป"))
    if q["contrast"] < 22:
        warns.append(("LOW_CONTRAST",
                      "ภาพมีคอนทราสต์ต่ำ อาจกระทบความแม่นยำ"))

    q["face_area_ratio"] = area_ratio
    q["face_width"] = fw
    return FaceCheckResult(
        ok=True, code="OK", backend=backend, faces=faces,
        message=f"ตรวจพบใบหน้า 1 คน (ขนาด {fw}×{fh} px)",
        warnings=warns, metrics=q,
    )


# ══════════════════════════════════════════════════════════
# PREVIEW — วาดกรอบใบหน้าให้ผู้ใช้เห็นว่า AI มองอะไร
# ══════════════════════════════════════════════════════════
def draw_boxes(img: Image.Image, faces, color=(14, 165, 233), width=None) -> Image.Image:
    out = img.copy().convert("RGB")
    if not faces:
        return out
    d = ImageDraw.Draw(out)
    w = width or max(2, int(min(out.size) * 0.006))
    for i, (x, y, fw, fh) in enumerate(faces):
        c = color if i == 0 else (148, 163, 184)   # หน้าหลัก = ฟ้า, อื่น ๆ = เทา
        d.rectangle([x, y, x + fw, y + fh], outline=c, width=w)
        # มุมเน้น (corner accent) ให้ดูเป็นมืออาชีพ
        L = int(min(fw, fh) * 0.22)
        for (px, py, dx, dy) in [(x, y, 1, 1), (x + fw, y, -1, 1),
                                 (x, y + fh, 1, -1), (x + fw, y + fh, -1, -1)]:
            d.line([px, py, px + dx * L, py], fill=c, width=w + 1)
            d.line([px, py, px, py + dy * L], fill=c, width=w + 1)
    return out