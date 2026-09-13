import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_facial_morphometry(image_bgr):
    """
    สกัดตัวชี้วัดกายวิภาคใบหน้า (Facial Morphometry) จาก MediaPipe Face Mesh
    ตามงานวิจัย: Coetzee (2009), Wen & Guo (2013), Lee & Kim (2014)
    """
    h, w, _ = image_bgr.shape
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        
        if not results.multi_face_landmarks:
            return None, "ไม่พบใบหน้าในภาพ"

        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks])

        # --- 1. ตรวจสอบ Aleatoric Proxy: Roll / Yaw / Pitch (Pose Check) ---
        left_eye_center = coords[33][:2]
        right_eye_center = coords[263][:2]
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        roll_angle = np.degrees(np.arctan2(dY, dX))

        # ตรวจสอบ Yaw โดยดูความต่างของแก้มซ้าย-ขวาเทียบกับแนวกึ่งกลางจมูก
        nose_tip = coords[1][:2]
        left_cheek_outer = coords[234][:2]
        right_cheek_outer = coords[454][:2]
        dist_left = np.linalg.norm(left_cheek_outer - nose_tip)
        dist_right = np.linalg.norm(right_cheek_outer - nose_tip)
        symmetry_diff = abs(dist_left - dist_right) / max(dist_left, dist_right)

        # เกณฑ์ปฏิเสธ Aleatoric (ภาพเอียงหรือหันข้างเกินเกณฑ์)
        if abs(roll_angle) > 15.0 or symmetry_diff > 0.50:
            return None, f"ภาพถ่ายเอียงหรือหันข้างเกินไป (Symmetry diff: {symmetry_diff:.1%}, Roll: {roll_angle:.1f}°)"

        # --- 2. คำนวณ Morphometric Metrics ---
        # จุดอ้างอิงสำคัญตาม Anthropometry:
        # Zygion (โหนกแก้ม): 234 (ซ้าย), 454 (ขวา)
        # Gonion (มุมกราม): 172 (ซ้าย), 397 (ขวา) หรือ 58, 288
        # Nasion (หว่างคิ้ว): 168
        # Menton (ปลายคาง): 152
        # Eyebrows (คิ้ว): 70, 300 | Upper Lip (ริมฝีปากบน): 0

        bizygomatic_width = calculate_distance(coords[234][:2], coords[454][:2])
        bigonial_width = calculate_distance(coords[172][:2], coords[397][:2])
        lower_face_height = calculate_distance(coords[168][:2], coords[152][:2]) # Nasion to Menton
        mid_face_height = calculate_distance(coords[168][:2], coords[0][:2])    # Nasion to Lip

        # Metric 1: Cheek-to-Jaw Width Ratio (CJWR)
        # สรีรวิทยา: สะท้อนการสะสมไขมันช่วงล่าง (Buccal/Jowl fat) ยิ่งอ้วน ค่ายิ่งต่ำลงเข้าใกล้ 1
        cjwr = bizygomatic_width / bigonial_width if bigonial_width > 0 else 0.0

        # Metric 2: Lower-Face Width to Face Height Ratio (LFWR)
        # สรีรวิทยา: Lee & Kim (2014) ชี้ว่าสัมพันธ์กับ Visceral Fat ของประชากรเอเชีย
        lfwr = bigonial_width / lower_face_height if lower_face_height > 0 else 0.0

        # Metric 3: Facial Width-to-Height Ratio (FWHR)
        fwhr = bizygomatic_width / mid_face_height if mid_face_height > 0 else 0.0

        # Metric 4: Perimeter-to-Area Ratio (PAR) ของกรอบหน้าส่วนล่าง
        # รวบรวม Landmark แนวขากรรไกรและคาง
        lower_jaw_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
        jaw_polygon = coords[lower_jaw_indices][:, :2].astype(np.int32)
        perimeter = cv2.arcLength(jaw_polygon, closed=False)
        area = cv2.contourArea(jaw_polygon)
        
        # ปรับ Normalize ขนาดด้วยระยะห่างระหว่างตา (Interpupillary distance)
        eye_dist = calculate_distance(coords[33][:2], coords[263][:2])
        norm_perimeter = perimeter / eye_dist if eye_dist > 0 else perimeter
        norm_area = area / (eye_dist ** 2) if eye_dist > 0 else area
        par = norm_perimeter / np.sqrt(norm_area) if norm_area > 0 else 0.0

        return {
            "CJWR": round(float(cjwr), 3),
            "LFWR": round(float(lfwr), 3),
            "FWHR": round(float(fwhr), 3),
            "PAR": round(float(par), 3),
            "symmetry_diff": round(float(symmetry_diff), 3),
            "roll_angle": round(float(roll_angle), 2),
            "coords": coords
        }, None