"""
pdf_export.py
──────────────────────────────────────────────────────────────
สร้างรายงาน PDF ภาษาไทยจากผลลัพธ์ predict_health_risk()
ต้องมีฟอนต์ไทยที่ assets/fonts/Sarabun-Regular.ttf (+ Bold)
ถ้าไม่มี → fallback เป็นรายงานภาษาอังกฤษอัตโนมัติ
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import Optional

from fpdf import FPDF
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
FONT_DIR = BASE_DIR.parent / "assets" / "fonts"
FONT_REG = FONT_DIR / "Sarabun-Regular.ttf"
FONT_BOLD = FONT_DIR / "Sarabun-Bold.ttf"

# ---------- Palette ----------
NAVY = (15, 23, 42)
SLATE = (100, 116, 139)
LIGHT = (241, 245, 249)
LINE = (226, 232, 240)
TEAL = (13, 148, 136)
CYAN = (8, 145, 178)
WHITE = (255, 255, 255)

PAGE_W, MARGIN = 210, 16
CONTENT_W = PAGE_W - MARGIN * 2

LIFESTYLE_PDF = {
    "sedentary": {
        "level": 0,
        "th": "กิจกรรมน้อย (Sedentary)",
        "en": "Sedentary",
    },
    "normal": {
        "level": 1,
        "th": "กิจกรรมปานกลาง (Normal)",
        "en": "Moderate activity",
    },
    "active": {
        "level": 2,
        "th": "ออกกำลังกายหนัก (Active)",
        "en": "Vigorous activity",
    },
}

def fonts_available() -> bool:
    return FONT_REG.exists()


# ══════════════════════════════════════════════════════════
def _txt(th: str, en: str, thai: bool) -> str:
    return th if thai else en


class HealthPDF(FPDF):
    def __init__(self, thai: bool):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.thai = thai
        self.set_auto_page_break(auto=True, margin=18)
        self.set_margins(MARGIN, MARGIN, MARGIN)
        if thai:
            self.add_font("TH", "", str(FONT_REG))
            self.add_font("TH", "B", str(FONT_BOLD) if FONT_BOLD.exists() else str(FONT_REG))
            self.base_font = "TH"
        else:
            self.base_font = "Helvetica"

    def f(self, size: float, bold: bool = False, color=NAVY):
        self.set_font(self.base_font, "B" if bold else "", size)
        self.set_text_color(*color)

    def footer(self):
        self.set_y(-14)
        self.f(7.5, color=SLATE)
        label = _txt("AI Health Screening · เอกสารคัดกรองเบื้องต้น ไม่ใช่ผลวินิจฉัยทางการแพทย์",
                     "AI Health Screening - Preliminary screening, not a medical diagnosis",
                     self.thai)
        self.cell(0, 5, f"{label}   |   {self.page_no()}", align="C")


# ---------- primitives ----------
def _bar(pdf, x, y, w, h, pct, color, track=(226, 232, 240)):
    pdf.set_fill_color(*track)
    pdf.rect(x, y, w, h, "F")
    fill_w = max(0.0, min(1.0, pct / 100.0)) * w
    if fill_w > 0:
        pdf.set_fill_color(*color)
        pdf.rect(x, y, fill_w, h, "F")


def _seg_bar(pdf, x, y, w, h, value, vmin, vmax, cuts, colors, marker=NAVY):
    """แถบไล่โซนพร้อมเข็มชี้ตำแหน่ง (ใช้กับ BMI / Body Fat)"""
    bounds = [vmin] + list(cuts) + [vmax]
    for i in range(len(bounds) - 1):
        sx = x + (bounds[i] - vmin) / (vmax - vmin) * w
        sw = (bounds[i + 1] - bounds[i]) / (vmax - vmin) * w
        pdf.set_fill_color(*colors[i])
        pdf.rect(sx, y, sw, h, "F")
    pos = x + max(0.0, min(1.0, (value - vmin) / (vmax - vmin))) * w
    pdf.set_fill_color(*marker)
    pdf.rect(pos - 0.7, y - 1.6, 1.4, h + 3.2, "F")


def _chip(pdf, x, y, text, bg=LIGHT, fg=SLATE, size=8.0, pad=3.0, h=6.2):
    pdf.f(size, color=fg)
    w = pdf.get_string_width(text) + pad * 2
    pdf.set_fill_color(*bg)
    pdf.rect(x, y, w, h, "F")
    pdf.set_xy(x, y)
    pdf.cell(w, h, text, align="C")
    return w


# ══════════════════════════════════════════════════════════
# MAIN BUILDER
# ══════════════════════════════════════════════════════════
def build_pdf(res: dict, include_photo: bool = True,
              photo: Optional[Image.Image] = None) -> bytes:
    thai = fonts_available()
    pdf = HealthPDF(thai)
    pdf.add_page()

    T = lambda th, en: _txt(th, en, thai)
    now = datetime.now().strftime("%d/%m/%Y  %H:%M")

    # ══════ HEADER ══════
    pdf.set_fill_color(*CYAN)
    pdf.rect(0, 0, PAGE_W, 34, "F")
    pdf.set_xy(MARGIN, 9)
    pdf.f(17, True, WHITE)
    pdf.cell(0, 8, T("รายงานผลคัดกรองสุขภาพด้วย AI", "AI Health Screening Report"))
    pdf.set_xy(MARGIN, 19)
    pdf.f(8.5, color=(224, 242, 254))
    pdf.cell(0, 6, T("Vision Transformer + XGBoost  |  ออกรายงานเมื่อ " + now,
                     "Vision Transformer + XGBoost  |  Generated " + now))

    y = 42

    # ══════ ข้อมูลผู้รับการประเมิน + รูป ══════
    photo_w = 0
    img = photo if photo is not None else res.get("image")
    if include_photo and img is not None:
        photo_w = 30
        buf = io.BytesIO()
        thumb = img.copy()
        thumb.thumbnail((420, 420))
        # ครอปเป็นสี่เหลี่ยมจัตุรัสให้ดูเรียบร้อย
        s = min(thumb.size)
        L = (thumb.width - s) // 2
        Tp = (thumb.height - s) // 2
        thumb.crop((L, Tp, L + s, Tp + s)).save(buf, format="PNG")
        buf.seek(0)
        pdf.image(buf, x=PAGE_W - MARGIN - photo_w, y=y, w=photo_w, h=photo_w)
        pdf.set_draw_color(*LINE)
        pdf.rect(PAGE_W - MARGIN - photo_w, y, photo_w, photo_w)

    info_w = CONTENT_W - photo_w - (5 if photo_w else 0)
    pdf.set_xy(MARGIN, y)
    pdf.f(9.5, True)
    pdf.cell(info_w, 6, T("ข้อมูลผู้รับการประเมิน", "Subject Information"), ln=1)

    g = str(res["gender"]).lower()
    gender_th = "ชาย" if g == "male" else "หญิง"
    waist_txt = (f"{res['waist_cm']:.1f} " + T("ซม.", "cm")) if res["has_waist"] \
        else T("ไม่ได้ระบุ", "Not provided")
    lifestyle_key = str(res.get("lifestyle", "normal")).lower()
    lifestyle_level = int(res.get("lifestyle_level", 1))

    ls = LIFESTYLE_PDF.get(
        lifestyle_key,
        LIFESTYLE_PDF["normal"],
    )

    lifestyle_text = T(
        ls["th"],
        ls["en"],
    )

    rows = [
        (T("เพศ / อายุ", "Gender / Age"),
         f"{T(gender_th, g.capitalize())}  ·  {res['age']} " + T("ปี", "yrs")),
        (T("รอบเอว", "Waist"), waist_txt),
        (T("กิจกรรมทางกาย", "Physical activity"),
         f"{lifestyle_text} (Level {lifestyle_level})",),
        (T("โหมดประเมิน", "Mode"),
         T(res["mode_text"], "Full (with waist)" if res["has_waist"] else "Basic (no waist)")),
    ]
    yy = y + 7
    for k, v in rows:
        pdf.set_xy(MARGIN, yy)
        pdf.f(8.5, color=SLATE)
        pdf.cell(34, 5.4, k)
        pdf.f(8.5, True)
        pdf.cell(info_w - 34, 5.4, str(v))
        yy += 5.6

    y = max(yy, y + photo_w) + 6

    # ══════ SECTION 1 : BMI ══════
    def section_header(y, num, title):
        pdf.set_fill_color(*TEAL)
        pdf.rect(MARGIN, y, 2.2, 6.5, "F")
        pdf.set_xy(MARGIN + 5, y)
        pdf.f(10.5, True)
        pdf.cell(0, 6.5, f"{num}  {title}")
        return y + 9

    y = section_header(y, "01", T("ดัชนีมวลกาย (BMI) — จากภาพใบหน้าด้วย ViT",
                                  "Body Mass Index - from face via ViT"))

    card_h = 26
    pdf.set_fill_color(*LIGHT)
    pdf.rect(MARGIN, y, CONTENT_W, card_h, "F")

    pdf.set_xy(MARGIN + 6, y + 5)
    pdf.f(26, True)
    pdf.cell(34, 11, f"{res['bmi']:.1f}")
    pdf.set_xy(MARGIN + 6 + pdf.get_string_width(f"{res['bmi']:.1f}") + 2, y + 10)
    pdf.f(9, color=SLATE)
    pdf.cell(20, 5, "kg/m²")

    pdf.set_xy(MARGIN + 6, y + 18)
    pdf.f(9.5, True, TEAL)
    pdf.cell(70, 5, res["bmi_cat"])

    bx = MARGIN + 88
    bw = CONTENT_W - 94
    _seg_bar(pdf, bx, y + 12, bw, 5.5, res["bmi"], 12, 40,
             [18.5, 23, 25],
             [(191, 219, 254), (167, 243, 208), (254, 240, 138), (254, 202, 202)])
    pdf.set_xy(bx, y + 19)
    pdf.f(6.8, color=SLATE)
    for lbl, frac in [("12", 0), ("18.5", .232), ("23", .393), ("25", .464), ("40", .96)]:
        pdf.set_xy(bx + frac * bw - 3, y + 19)
        pdf.cell(6, 4, lbl, align="C")

    y += card_h + 7

    # ══════ SECTION 2 : Body Fat ══════
    y = section_header(y, "02", T(
        "เปอร์เซ็นต์ไขมันในร่างกาย — XGBoost + กิจกรรมทางกาย",
        "Total Body Fat % - XGBoost + Physical Activity",
    ),)

    pdf.set_fill_color(*LIGHT)
    pdf.rect(MARGIN, y, CONTENT_W, card_h, "F")

    pdf.set_xy(MARGIN + 6, y + 5)
    pdf.f(26, True)
    pdf.cell(34, 11, f"{res['bodyfat']:.1f}")
    pdf.set_xy(MARGIN + 6 + pdf.get_string_width(f"{res['bodyfat']:.1f}") + 2, y + 10)
    pdf.f(9, color=SLATE)
    pdf.cell(10, 5, "%")

    pdf.set_xy(MARGIN + 6, y + 18)
    pdf.f(9.5, True, TEAL)
    pdf.cell(70, 5, res["bf_cat"])

    c = res["bf_cuts"]
    _seg_bar(pdf, bx, y + 12, bw, 5.5, res["bodyfat"], 0, 50, c,
             [(191, 219, 254), (167, 243, 208), (254, 240, 138), (254, 202, 202)])
    for lbl, val in [("0", 0), (str(c[0]), c[0]), (str(c[1]), c[1]),
                     (str(c[2]), c[2]), ("50", 50)]:
        pdf.set_xy(bx + (val / 50) * bw - 3, y + 19)
        pdf.f(6.8, color=SLATE)
        pdf.cell(6, 4, lbl, align="C")

    y += card_h + 6

    # ══════ Lifestyle Feature Note ══════
    feature_list = (
        "BMI, AGE, GENDER_NUM, WAIST_CM, LIFESTYLE_LEVEL"
        if res["has_waist"]
        else "BMI, AGE, GENDER_NUM, LIFESTYLE_LEVEL"
    )

    note = T(
        (
            f"กิจกรรมทางกายถูกใช้เป็นฟีเจอร์ของโมเดลโดยตรง: "
            f"LIFESTYLE_LEVEL={lifestyle_level} ({ls['th']}) "
            f"ร่วมกับข้อมูล BMI อายุ เพศ"
            f"{' และรอบเอว' if res['has_waist'] else ''} "
            f"เพื่อทำนายเปอร์เซ็นต์ไขมันจาก XGBoost "
            f"โดยไม่มีการบวกหรือลบค่าคงที่ภายหลังการทำนาย"
        ),
        (
            f"Physical activity is used directly as a trained model feature: "
            f"LIFESTYLE_LEVEL={lifestyle_level} ({ls['en']}). "
            f"Body fat is predicted directly by XGBoost without "
            f"post-prediction fixed-value calibration."
        ),
    )

    pdf.set_fill_color(240, 249, 255)
    nh = 16
    pdf.rect(MARGIN, y, CONTENT_W, nh, "F")

    pdf.set_fill_color(14, 165, 233)
    pdf.rect(MARGIN, y, 1.6, nh, "F")

    pdf.set_xy(MARGIN + 5, y + 1.8)
    pdf.f(7.6, color=(12, 74, 110))
    pdf.multi_cell(
        CONTENT_W - 9,
        3.6,
        note,
    )

    y += nh + 5

    # ══════ SECTION 3 : NCDs ══════
    y = section_header(y, "03", T("ความเสี่ยงโรคไม่ติดต่อเรื้อรัง (NCDs)",
                                  "Non-Communicable Disease Risk"))
    for name_th, name_en, pct, lv in [
        (T("โรคเบาหวาน", "Diabetes"), "Diabetes",
         res["diabetes_pct"], res["diabetes_level"]),
        (T("โรคความดันโลหิตสูง", "Hypertension"), "Hypertension",
         res["hypertension_pct"], res["hypertension_level"]),
    ]:
        rh = 16.5
        pdf.set_fill_color(250, 250, 251)
        pdf.rect(MARGIN, y, CONTENT_W, rh, "F")
        pdf.set_fill_color(*lv["rgb"])
        pdf.rect(MARGIN, y, 1.6, rh, "F")

        pdf.set_xy(MARGIN + 6, y + 3)
        pdf.f(10, True)
        pdf.cell(80, 5.5, name_th)

        pdf.set_xy(PAGE_W - MARGIN - 40, y + 2.5)
        pdf.f(16, True, lv["rgb"])
        pdf.cell(34, 7, f"{pct:.1f}%", align="R")

        _bar(pdf, MARGIN + 6, y + 11.5, CONTENT_W - 52, 4.2, pct, lv["rgb"])

        pdf.set_xy(PAGE_W - MARGIN - 40, y + 10.5)
        pdf.f(8, True, lv["rgb"])
        pdf.cell(34, 5, lv["label"], align="R")
        y += rh + 2.5

    # ══════ คำแนะนำ ══════
    hi = max(res["diabetes_pct"], res["hypertension_pct"])
    if hi >= 50:
        rec_bg, rec_fg = (254, 226, 226), (185, 28, 28)
        rec = T("แนะนำให้พบแพทย์ — ผลประเมินพบความเสี่ยงในระดับสูง "
                "ควรเข้ารับการตรวจเลือดและวัดความดันโลหิตที่สถานพยาบาลเพื่อยืนยันผล",
                "Consult a physician - high risk detected. Please confirm with lab tests.")
    elif hi >= 25:
        rec_bg, rec_fg = (254, 249, 195), (161, 98, 7)
        rec = T("ควรเฝ้าระวัง — แนะนำปรับพฤติกรรมการรับประทานอาหาร เพิ่มการเคลื่อนไหว "
                "และตรวจสุขภาพประจำปีอย่างสม่ำเสมอ",
                "Monitor - adjust diet, increase activity, annual check-up recommended.")
    else:
        rec_bg, rec_fg = (220, 252, 231), (21, 128, 61)
        rec = T("อยู่ในเกณฑ์ดี — รักษาพฤติกรรมสุขภาพนี้ต่อไป "
                "และเข้ารับการตรวจสุขภาพประจำปีตามปกติ",
                "Good - maintain current lifestyle and routine annual check-up.")

    y += 1.5
    pdf.set_fill_color(*rec_bg)
    pdf.rect(MARGIN, y, CONTENT_W, 11, "F")
    pdf.set_xy(MARGIN + 5, y + 1.5)
    pdf.f(8.2, True, rec_fg)
    pdf.multi_cell(CONTENT_W - 10, 3.8, rec)
    y += 15

    # ══════ ภาคผนวก: ค่าเชิงเทคนิค ══════
    # ปิด auto page break เพื่อป้องกัน FPDF ตัดขึ้นหน้าใหม่ตอนวาดตารางท้ายกระดาษ
    pdf.set_auto_page_break(auto=False)

    pdf.set_xy(MARGIN, y)
    pdf.f(8, True, SLATE)
    pdf.cell(0, 4.5, T("ภาคผนวก — ค่าทางเทคนิค", "Appendix - Technical values"))
    tech = [
        ("BMI (ViT output)",
         f"{res['bmi']:.4f}",),
        ("BMI uncertainty",
         f"SD {res.get('bmi_std', 0.0):.4f}",),
        ("Lifestyle",
         lifestyle_key,),
        ("Lifestyle level",
         str(lifestyle_level),),
        ("Body Fat (XGBoost)",
         f"{res['bodyfat']:.4f}%",),
        ("P(Diabetes)",
         f"{res['diabetes_pct'] / 100:.4f}",),
        ("P(Hypertension)",
         f"{res['hypertension_pct'] / 100:.4f}",),
        ("Model route", ("with_waist" if res["has_waist"] else "no_waist"),),
    ]
    cx, cy = MARGIN, y + 5.0
    for i, (k, v) in enumerate(tech):
        col = i % 2
        pdf.set_xy(cx + col * (CONTENT_W / 2), cy + (i // 2) * 4.2)
        pdf.f(7.0, color=SLATE)
        pdf.cell(32, 4.0, k)
        pdf.f(7.0, True, NAVY)
        pdf.cell(30, 4.0, v)
    y = cy + ((len(tech) + 1) // 2) * 4.2 + 3

    # ══════ Disclaimer ══════
    pdf.set_fill_color(255, 251, 235)
    pdf.rect(MARGIN, y, CONTENT_W, 14, "F")
    pdf.set_xy(MARGIN + 4, y + 1.5)
    pdf.f(6.8, color=(146, 64, 14))
    pdf.multi_cell(CONTENT_W - 8, 3.4, T(
        "ข้อจำกัดความรับผิดชอบ: รายงานฉบับนี้เป็นผลจากแบบจำลอง AI เพื่อการคัดกรองเบื้องต้นเท่านั้น "
        "ไม่ใช่การวินิจฉัยทางการแพทย์ และไม่สามารถใช้แทนการตรวจโดยแพทย์หรือผลตรวจทางห้องปฏิบัติการได้ "
        "ค่าที่ได้เป็นการประมาณทางสถิติซึ่งมีความคลาดเคลื่อน หากมีอาการผิดปกติหรือข้อสงสัย "
        "กรุณาปรึกษาแพทย์หรือบุคลากรทางการแพทย์ที่มีใบอนุญาต",
        "Disclaimer: This report is generated by an AI screening model for preliminary "
        "purposes only. It is not a medical diagnosis and cannot replace examination by "
        "a licensed physician or laboratory testing."), align="L")

    return bytes(pdf.output())


def safe_build_pdf(res: dict, **kw):
    """คืน (pdf_bytes, error_message) — ไม่ให้ UI พังถ้าสร้าง PDF ไม่สำเร็จ"""
    try:
        return build_pdf(res, **kw), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"