"""
app.py — AI Health Screening Dashboard
รัน:  cd scripts && streamlit run app.py
"""
import os
import sys
from datetime import datetime

import torch
import streamlit as st
import plotly.graph_objects as go

from predict_pipeline import (
    load_all_models, predict_health_risk, check_weights,
    get_device, LIFESTYLE_META, FaceGuardError,
)
import face_guard as fg
from pdf_export import safe_build_pdf, fonts_available

st.set_page_config(page_title="AI Health Screening", page_icon="🩺",
                   layout="wide", initial_sidebar_state="collapsed")

# ══════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════
st.markdown("""
<style>
 @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@400;500;600;700&display=swap');
 html,body,[class*="css"],.stApp{font-family:'IBM Plex Sans Thai',sans-serif;}
 #MainMenu,footer,header{visibility:hidden;}
 .block-container{padding-top:2rem;padding-bottom:3rem;max-width:1180px;}
 .hero{background:linear-gradient(120deg,#0f766e,#0891b2 55%,#0ea5e9);
   border-radius:20px;padding:32px 36px;color:#fff;margin-bottom:26px;
   box-shadow:0 12px 30px rgba(8,145,178,.22);}
 .hero h1{font-size:30px;font-weight:700;margin:0 0 6px;line-height:1.3;}
 .hero p{font-size:15px;opacity:.93;margin:0;}
 .hero-chips{margin-top:16px;display:flex;gap:8px;flex-wrap:wrap;}
 .chip{background:rgba(255,255,255,.18);border:1px solid rgba(255,255,255,.28);
   padding:5px 13px;border-radius:999px;font-size:12.5px;font-weight:500;}
 .card-title{font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;
   letter-spacing:.6px;margin-bottom:14px;}
 .badge{display:inline-block;padding:5px 14px;border-radius:999px;
   font-size:13px;font-weight:600;margin-top:10px;}
 .b-normal{background:#dcfce7;color:#15803d}.b-under{background:#dbeafe;color:#1d4ed8}
 .b-over{background:#fef9c3;color:#a16207}.b-obese{background:#fee2e2;color:#b91c1c}
 .b-lean{background:#dbeafe;color:#1d4ed8}.b-fit{background:#dcfce7;color:#15803d}
 .risk-card{border-radius:16px;padding:22px 24px;margin-bottom:14px;border:1px solid;}
 .risk-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;}
 .risk-name{font-size:16px;font-weight:600;color:#ffffff;}
 .risk-pct{font-size:30px;font-weight:700;}
 .risk-bar{height:11px;background:#e2e8f0;border-radius:99px;overflow:hidden;}
 .risk-fill{height:100%;border-radius:99px;transition:width .6s ease;}
 .risk-label{font-size:13px;font-weight:600;margin-top:9px;}
 .note{background:#f0f9ff;border-left:4px solid #0ea5e9;border-radius:10px;
   padding:15px 18px;font-size:14px;color:#0c4a6e;line-height:1.65;}
 .guard-ok{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
   padding:11px 15px;font-size:13px;color:#15803d;}
 .guard-bad{background:#fef2f2;border:1px solid #fecaca;border-radius:12px;
   padding:14px 17px;font-size:13.5px;color:#b91c1c;line-height:1.65;}
 .disclaimer{background:#fffbeb;border:1px solid #fde68a;border-radius:12px;
   padding:14px 18px;font-size:12.8px;color:#92400e;line-height:1.6;}
 .stButton>button{border-radius:12px;height:50px;font-size:16px;font-weight:600;
   background:linear-gradient(120deg,#0f766e,#0891b2);border:none;color:#fff;}
 .stButton>button:hover{opacity:.92;color:#fff;}
 .stDownloadButton>button{border-radius:12px;height:46px;font-weight:600;
   background:#fff;border:1.5px solid #0891b2;color:#0891b2;}
</style>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# CACHED RESOURCES
# ══════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def get_models():
    return load_all_models()

@st.cache_resource(show_spinner=False)
def warm_face_backend():
    return fg.get_backend_name()

@st.cache_data(show_spinner=False, max_entries=8)
def cached_face_check(img_bytes: bytes):
    from predict_pipeline import to_pil
    img = to_pil(img_bytes)
    rep = fg.check_face(img)
    preview = fg.draw_boxes(img, rep.faces) if rep.faces else img
    return rep, preview

# ══════════════════════════════════════════════════════════
# CHART HELPERS
# ══════════════════════════════════════════════════════════
def gauge(value, rng, steps, title, suffix=""):
    val_text = f"{value:.1f}{suffix}"
    fig = go.Figure()
    
    # วาดเฉพาะเกจวัด (ไม่ใช้ mode="+number" เพื่อกันตัวเลขโดนผลักไปขวา)
    fig.add_trace(go.Indicator(
        mode="gauge",
        value=value,
        title={"text": title, "font": {"size": 14, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": rng, "tickwidth": 1, "tickcolor": "#cbd5e1",
                     "tickfont": {"size": 10, "color": "#94a3b8"}},
            "bar": {"color": "#38bdf8", "thickness": 0.22},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps
        }
    ))
    
    # ตรึงตัวเลขไว้ตรงกลางช่องครึ่งวงกลมพอดี 100%
    fig.add_annotation(
        x=0.5, y=0.22,
        text=val_text,
        showarrow=False,
        font=dict(size=34, color="#ffffff", family="Arial Black, sans-serif")
    )
    
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def mini_gauge(value, rng, steps, title):
    val_text = f"{value:.2f}"
    fig = go.Figure()
    
    fig.add_trace(go.Indicator(
        mode="gauge",
        value=value,
        title={"text": title, "font": {"size": 12, "color": "#cbd5e1"}},
        gauge={
            "axis": {"range": rng, "tickwidth": 1, "tickcolor": "#64748b",
                     "tickfont": {"size": 9, "color": "#94a3b8"}},
            "bar": {"color": "#38bdf8", "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": steps
        }
    ))
    
    fig.add_annotation(
        x=0.5, y=0.18,
        text=val_text,
        showarrow=False,
        font=dict(size=20, color="#ffffff", family="Arial Black, sans-serif")
    )
    
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=30, b=5),
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def render_risk(name, pct, level, desc):
    st.markdown(f"""
    <div class="risk-card" style="background:{level['bg']};border-color:{level['color']}33;">
      <div class="risk-head">
        <div><div class="risk-name">{level['icon']} {name}</div>
          <div style="font-size:12.5px;color:#64748b;margin-top:3px;">{desc}</div></div>
        <div class="risk-pct" style="color:{level['color']};">{pct:.1f}%</div>
      </div>
      <div class="risk-bar"><div class="risk-fill"
        style="width:{min(pct,100):.1f}%;background:{level['color']};"></div></div>
      <div class="risk-label" style="color:{level['color']};">{level['label']}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>🩺 AI Health Screening</h1>
  <p>คัดกรองสุขภาพเบื้องต้นจากภาพใบหน้า ด้วย Vision Transformer + XGBoost</p>
  <div class="hero-chips">
    <span class="chip">🛡️ Two-Tier Safety Guard</span>
    <span class="chip">🧬 Explainable Morphometry</span>
    <span class="chip">📷 Face → BMI</span>
    <span class="chip">📊 Body Fat %</span>
    <span class="chip">🩸 Diabetes</span>
    <span class="chip">💓 Hypertension</span>
  </div>
</div>""", unsafe_allow_html=True)

missing = check_weights()
if missing:
    st.error(f"❌ ไม่พบไฟล์โมเดล: {', '.join(missing)} — ตรวจสอบโฟลเดอร์ weights/")
    st.stop()

BACKEND = warm_face_backend()
BACKEND_LABEL = {"mediapipe": "MediaPipe BlazeFace",
                 "opencv": "OpenCV Haar Cascade",
                 "none": "ปิดใช้งาน (ไม่พบไลบรารี)"}[BACKEND]

# ══════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════
col_in, col_prev = st.columns([1.35, 1], gap="large")

with col_in:
    st.markdown('<div class="card-title">ขั้นตอนที่ 1 — ภาพใบหน้า</div>',
                unsafe_allow_html=True)
    t_up, t_cam = st.tabs(["📁 อัปโหลดรูป", "📸 ถ่ายภาพ"])
    with t_up:
        uploaded = st.file_uploader("เลือกรูปใบหน้าตรง แสงสว่างพอ",
                                    type=["jpg", "jpeg", "png", "webp"])
    with t_cam:
        captured = st.camera_input("ถ่ายภาพใบหน้า")
    image_input = uploaded or captured

    # ---------- FACE GUARD ----------
    guard_rep, guard_preview, img_bytes = None, None, None
    if image_input is not None:
        img_bytes = image_input.getvalue()
        with st.spinner("🛡️ กำลังตรวจสอบใบหน้าในภาพ..."):
            try:
                guard_rep, guard_preview = cached_face_check(img_bytes)
            except Exception as e:
                st.warning(f"ตรวจใบหน้าไม่สำเร็จ ({e}) — จะข้ามการตรวจ")

    st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">ขั้นตอนที่ 2 — ข้อมูลพื้นฐาน</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age = st.number_input("อายุ (ปี)", 12, 99, 30, 1)
    with c2:
        gender_th = st.radio("เพศ", ["ชาย", "หญิง"], horizontal=True)
        gender = "male" if gender_th == "ชาย" else "female"

    st.markdown('<div class="card-title">ขั้นตอนที่ 3 — รูปแบบการใช้ชีวิต</div>',
                unsafe_allow_html=True)
    ls_order = ["active", "normal", "sedentary"]
    lifestyle = st.radio(
        "เลือกที่ใกล้เคียงตัวคุณที่สุด", ls_order, index=1,
        format_func=lambda k: f"{LIFESTYLE_META[k]['icon']}  {LIFESTYLE_META[k]['title']}",
        captions=[LIFESTYLE_META[k]["subtitle"] for k in ls_order])

    with st.expander("⚙️ ตัวเลือกขั้นสูง (ไม่บังคับ)"):
        st.caption("หากคุณมีสายวัด การใส่รอบเอวจะทำให้ระบบสลับไปใช้โมเดลชุดเต็ม "
                   "ซึ่งประเมินไขมันช่องท้องและความเสี่ยงได้แม่นยำขึ้น")
        use_waist = st.toggle("📏 ฉันมีสายวัด และต้องการใส่รอบเอว", value=False)
        waist_cm = None
        if use_waist:
            waist_cm = st.slider("รอบเอว (เซนติเมตร)", 50.0, 160.0, 80.0, 0.5)
            st.caption("💡 วัดที่ระดับสะดือ หายใจออกปกติ ไม่ต้องแขม่วท้อง")

        st.divider()
        strict = st.toggle("🛡️ เปิดระบบตรวจสอบใบหน้า (Face Guard)", value=True,
                           help=f"Backend: {BACKEND_LABEL}")
        st.caption(f"เครื่องมือตรวจจับที่ใช้งานอยู่: **{BACKEND_LABEL}**")

    # แสดงปุ่ม Checkbox ให้ติ๊ก Bypass ทันทีเมื่อมีการอัปโหลดภาพ
    bypass_guard = False
    if image_input is not None:
        has_issue = (guard_rep is not None and (not guard_rep.ok or len(guard_rep.warnings) > 0))
        if has_issue:
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            bypass_guard = st.checkbox(
                "⚠️ **ยินยอมข้ามการตรวจสอบคุณภาพภาพถ่าย (Bypass Guard)** เพื่อทำการวิเคราะห์ต่อ",
                value=False,
                help="เปิดใช้งานเพื่อบังคับให้ระบบประมวลผลต่อ แม้ภาพจะมีความเบลอ เอียง หรือหันข้างเกินเกณฑ์"
            )

    can_run = image_input is not None and (
        not strict or bypass_guard or guard_rep is None or guard_rep.ok)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    run = st.button("🔍 เริ่มวิเคราะห์สุขภาพ", use_container_width=True,
                    disabled=not can_run)
    if image_input is None:
        st.caption("⬆️ กรุณาอัปโหลดหรือถ่ายภาพใบหน้าก่อนเริ่มวิเคราะห์")
    elif strict and guard_rep is not None and not guard_rep.ok:
        if bypass_guard:
            st.caption("⚠️ คุณเปิดใช้งาน Bypass — สามารถกดวิเคราะห์ต่อได้ (ผลลัพธ์อาจคลาดเคลื่อน)")
        else:
            st.caption("🚫 ภาพไม่ผ่านการตรวจสอบ — กรุณาเปลี่ยนภาพ หรือเปิด 'ตัวเลือกขั้นสูง' เพื่อติ๊กข้ามการตรวจ")
with col_prev:
    st.markdown('<div class="card-title">ภาพที่เลือก</div>', unsafe_allow_html=True)
    if guard_preview is not None:
        st.image(guard_preview, use_column_width=True,
                 caption="กรอบฟ้า = ใบหน้าที่ระบบตรวจพบ")
    elif image_input is not None:
        st.image(image_input, use_column_width=True)
    else:
        st.markdown("""
        <div style="border:2px dashed #cbd5e1;border-radius:16px;height:300px;
          display:flex;flex-direction:column;align-items:center;justify-content:center;
          color:#94a3b8;background:#f8fafc;">
          <div style="font-size:46px;">👤</div>
          <div style="font-size:14px;margin-top:8px;">ยังไม่มีภาพ</div>
        </div>""", unsafe_allow_html=True)

    if guard_rep is not None:
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        if guard_rep.ok:
            m = guard_rep.metrics
            detail = ""
            if m:
                detail = (f" · ขนาด {m.get('face_area_ratio',0)*100:.0f}% ของภาพ"
                          f" · ความคมชัด {m.get('blur',0):.0f}")
            st.markdown(f'<div class="guard-ok">✅ <b>ผ่านการตรวจสอบ</b> — '
                        f'{guard_rep.message}{detail}</div>', unsafe_allow_html=True)
            for _, w in guard_rep.warnings:
                st.warning(f"⚠️ {w}")
        else:
            st.markdown(f'<div class="guard-bad">🚫 <b>{guard_rep.message}</b><br>'
                        f'{guard_rep.hint}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# RUN PREDICTION
# ══════════════════════════════════════════════════════════
if run and image_input is not None:
    with st.spinner("🧠 AI กำลังวิเคราะห์สรีรวิทยาและคัดกรองโรค..."):
        try:
            use_guard = strict and not bypass_guard
            st.session_state["result"] = predict_health_risk(
                image=img_bytes, age=int(age), gender=gender,
                waist_cm=waist_cm, lifestyle=lifestyle,
                models=get_models(), face_guard=use_guard)
            st.session_state.pop("pdf", None)
        except FaceGuardError as e:
            st.session_state.pop("result", None)
            st.error(f"🚫 {e.report.message}\n\n{e.report.hint}")
        except ValueError as e:
            st.session_state.pop("result", None)
            st.error(f"🚫 {e}")
        except Exception as e:
            st.session_state.pop("result", None)
            st.error(f"เกิดข้อผิดพลาดระหว่างวิเคราะห์: {type(e).__name__}: {e}")

# ══════════════════════════════════════════════════════════
# DISPLAY RESULT (วางโค้ดทั้งหมดไว้ใต้บล็อก if res:)
# ══════════════════════════════════════════════════════════
res = st.session_state.get("result")
if res:
    st.markdown("<hr style='margin:34px 0 26px;border:none;border-top:1px solid #e2e8f0;'>",
                unsafe_allow_html=True)
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown("### 📋 ผลการวิเคราะห์")
    with h2:
        if not fonts_available():
            st.caption("⚠️ ไม่พบฟอนต์ไทย — PDF จะออกเป็นภาษาอังกฤษ")
        if st.button("📄 สร้างรายงาน PDF", use_container_width=True):
            with st.spinner("กำลังสร้างเอกสาร..."):
                pdf_bytes, err = safe_build_pdf(res, include_photo=True)
            if err:
                st.error(f"สร้าง PDF ไม่สำเร็จ: {err}")
            else:
                st.session_state["pdf"] = pdf_bytes
        if st.session_state.get("pdf"):
            st.download_button(
                "⬇️ ดาวน์โหลด PDF", st.session_state["pdf"],
                file_name=f"health_report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                mime="application/pdf", use_container_width=True)

    waist_txt = f"{res['waist_cm']:.1f} ซม." if res["has_waist"] else "ไม่ได้ระบุ"
    mode_icon = "🎯" if res["has_waist"] else "⚡"
    st.markdown(f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:20px;">
      <span style="background:#f1f5f9;color:#475569;padding:6px 14px;border-radius:999px;
        font-size:12.5px;">{res['gender'].capitalize()} · {res['age']} ปี</span>
      <span style="background:#f1f5f9;color:#475569;padding:6px 14px;border-radius:999px;
        font-size:12.5px;">📏 รอบเอว: {waist_txt}</span>
      <span style="background:#f1f5f9;color:#475569;padding:6px 14px;border-radius:999px;
        font-size:12.5px;">{res['lifestyle_meta']['icon']} {res['lifestyle_meta']['title']}</span>
      <span style="background:#e0f2fe;color:#0369a1;padding:6px 14px;border-radius:999px;
        font-size:12.5px;font-weight:600;">{mode_icon} {res['mode_text']}</span>
    </div>""", unsafe_allow_html=True)

    g1, g2 = st.columns(2, gap="large")
    with g1:
        st.markdown('<div class="card-title">📷 Stage 1 — BMI จากใบหน้า (ViT + MC Dropout)</div>',
                    unsafe_allow_html=True)
        st.plotly_chart(gauge(res["bmi"], [12, 40], [
            {"range": [12, 18.5], "color": "#dbeafe"},
            {"range": [18.5, 23], "color": "#bbf7d0"},
            {"range": [23, 25], "color": "#fef08a"},
            {"range": [25, 40], "color": "#fecaca"}], "Body Mass Index"),
            use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div style="text-align:center;"><span class="badge '
                    f'b-{res["bmi_key"]}">{res["bmi_cat"]}</span>'
                    f'<div style="font-size:12px;color:#64748b;margin-top:6px;">'
                    f'Epistemic Uncertainty: ±{res["bmi_std"]:.2f} (95% CI: [{res["ci_range"][0]:.1f} - {res["ci_range"][1]:.1f}])</div>'
                    f'</div>',
                    unsafe_allow_html=True)
    with g2:
        st.markdown('<div class="card-title">📊 Stage 1.5 — เปอร์เซ็นต์ไขมัน (XGBoost)</div>',
                    unsafe_allow_html=True)
        c = res["bf_cuts"]
        st.plotly_chart(gauge(res["bodyfat"], [0, 50], [
            {"range": [0, c[0]], "color": "#dbeafe"},
            {"range": [c[0], c[1]], "color": "#bbf7d0"},
            {"range": [c[1], c[2]], "color": "#fef08a"},
            {"range": [c[2], 50], "color": "#fecaca"}], "Total Body Fat", "%"),
            use_container_width=True, config={"displayModeBar": False})
        bcls = {"lean": "b-lean", "fit": "b-fit",
                "normal": "b-normal", "obese": "b-obese"}[res["bf_key"]]
        st.markdown(f'<div style="text-align:center;"><span class="badge '
                    f'{bcls}">{res["bf_cat"]}</span></div>', unsafe_allow_html=True)

    # ---------- โครงสร้างกายวิภาคใบหน้า (EXPLAINABLE MORPHOMETRY GAUGE) ----------
    morph = res.get("morphometry")
    if morph:
        st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
        st.subheader("🧬 การตีความสรีรวิทยาใบหน้า (Explainable Facial Morphometry)")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            # LFWR: ปกติ 0.8 - 1.1 (ยิ่งต่ำ = ไขมันสะสมน้อย)
            st.plotly_chart(mini_gauge(
                morph['LFWR'], [0.7, 1.4], [
                    {"range": [0.7, 1.0], "color": "#bbf7d0"},
                    {"range": [1.0, 1.15], "color": "#fef08a"},
                    {"range": [1.15, 1.4], "color": "#fecaca"}
                ], "LFWR (กรามล่าง/ความสูง)"
            ), use_container_width=True, config={"displayModeBar": False})
            st.caption("<div style='text-align:center; font-size:11px; color:#94a3b8;'>สัมพันธ์กับ Visceral Fat (Lee 2014)</div>", unsafe_allow_html=True)

        with m_col2:
            # CJWR: ปกติ 1.15 - 1.45 (ยิ่งสูง = โหนกแก้มเด่น ไขมันแก้มล่างน้อย)
            st.plotly_chart(mini_gauge(
                morph['CJWR'], [0.9, 1.6], [
                    {"range": [0.9, 1.1], "color": "#fecaca"},
                    {"range": [1.1, 1.25], "color": "#fef08a"},
                    {"range": [1.25, 1.6], "color": "#bbf7d0"}
                ], "CJWR (โหนกแก้ม/กราม)"
            ), use_container_width=True, config={"displayModeBar": False})
            st.caption("<div style='text-align:center; font-size:11px; color:#94a3b8;'>ตรวจจับไขมันแก้มล่าง (Coetzee 2009)</div>", unsafe_allow_html=True)

        with m_col3:
            # PAR: ความมนกลมกรอบหน้า
            st.plotly_chart(mini_gauge(
                morph['PAR'], [1.5, 4.0], [
                    {"range": [1.5, 2.3], "color": "#bbf7d0"},
                    {"range": [2.3, 3.0], "color": "#fef08a"},
                    {"range": [3.0, 4.0], "color": "#fecaca"}
                ], "PAR (ความกลมของกราม)"
            ), use_container_width=True, config={"displayModeBar": False})
            st.caption("<div style='text-align:center; font-size:11px; color:#94a3b8;'>ความมนกลมของหน้าล่าง (Wen 2013)</div>", unsafe_allow_html=True)

        with m_col4:
            # FWHR: มิติใบหน้าส่วนกลาง
            st.plotly_chart(mini_gauge(
                morph['FWHR'], [1.4, 2.6], [
                    {"range": [1.4, 1.8], "color": "#bbf7d0"},
                    {"range": [1.8, 2.1], "color": "#fef08a"},
                    {"range": [2.1, 2.6], "color": "#fecaca"}
                ], "FWHR (กว้าง/สูงใบหน้า)"
            ), use_container_width=True, config={"displayModeBar": False})
            st.caption("<div style='text-align:center; font-size:11px; color:#94a3b8;'>มิติกระดูกและไขมันแก้มส่วนบน</div>", unsafe_allow_html=True)

        # ---------- DYNAMIC CLINICAL INTERPRETATION ----------
        lfwr_val = morph['LFWR']
        cjwr_val = morph['CJWR']
        par_val = morph['PAR']

        findings = []
        risk_notes = []

        # 1. วิเคราะห์แนวกรามและไขมันช่วงล่าง (LFWR & CJWR)
        if cjwr_val >= 1.22 and lfwr_val <= 0.98:
            findings.append("🔹 **โครงสร้างใบหน้าลีน/คมชัด (Chiseled Phenotype):** โหนกแก้มเด่นชัดเมื่อเทียบกับขากรรไกรล่าง ไม่พบการสะสมของถุงไขมันกระพุ้งแก้ม (Buccal Fat) หรือรอยพับใต้คาง")
        elif cjwr_val < 1.12 or lfwr_val > 1.10:
            findings.append("⚠️ **การสะสมไขมันช่วงล่างใบหน้า (Lower Facial Adiposity):** ขากรรไกรล่างกว้างขึ้นเมื่อเทียบกับโหนกแก้ม บ่งชี้การขยายตัวของชั้นไขมันใต้ผิวหนัง (Buccal & Jowl Fat Pads)")
            risk_notes.append("สัดส่วนใบหน้าส่วนล่างที่หนาขึ้น ทางการแพทย์ (Lee & Kim 2014) พบว่ามีความสัมพันธ์เชิงบวกกับภาวะไขมันสะสมในช่องท้อง (Visceral Adiposity)")
        else:
            findings.append("🔹 **สัดส่วนโครงหน้าสมดุล (Balanced Structure):** การกระจายตัวของเนื้อเยื่อและกล้ามเนื้อใบหน้าอยู่ในเกณฑ์มาตรฐานประชากรทั่วไป")

        # 2. วิเคราะห์ความกลมมนของกรอบหน้า (PAR)
        if par_val >= 2.8:
            findings.append("🔹 **รูปทรงกรอบหน้ามีความมนกลม (Rounded Contour):** เส้นรอบรูปกรอบหน้าส่วนล่างกลืนเป็นแนวโค้ง สะท้อนการสะสมของไขมันเนื้อเยื่อรอบแนวกราม")
        elif par_val <= 2.2:
            findings.append("🔹 **กรอบกระดูกกรามและคางเด่นชัด (Angular Jawline):** ปรากฏรอยต่อกระดูกขากรรไกรชัดเจน สัมพันธ์กับผู้ที่มีเปอร์เซ็นต์ไขมันต่ำหรือมวลกล้ามเนื้อสูง")

        # รวมข้อความสรุป
        summary_text = "\n\n".join(findings)
        if risk_notes:
            summary_text += "\n\n💡 **ข้อสังเกตทางสรีรวิทยา:** " + " ".join(risk_notes)

        st.info(f"🧬 **การประเมินลักษณะทางกายวิภาค (Anatomical Insight)**\n\n{summary_text}")
    

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    d, ls = res["calibration_delta"], res["lifestyle_meta"]
    if abs(d) < 0.01:
        note = (f"<b>{ls['icon']} ไม่มีการปรับค่า</b> — {ls['reason']}<br>"
                f"ค่าไขมันที่แสดง <b>{res['bodyfat']:.2f}%</b> คือผลจากโมเดลโดยตรง")
    else:
        sign, arrow = ("ลด", "↓") if d < 0 else ("เพิ่ม", "↑")
        clamp = ("<br><span style='color:#b45309;'>⚠️ ค่าถูกจำกัดที่ 5% "
                 "ซึ่งเป็นระดับไขมันจำเป็นขั้นต่ำของร่างกาย</span>"
                 if res["was_clamped"] else "")
        note = (f"<b>{ls['icon']} การปรับค่าตามไลฟ์สไตล์ (Lifestyle Calibration)</b><br>"
                f"{ls['reason']}<br><br><span style='font-family:monospace;font-size:13.5px;'>"
                f"ค่าดิบจากโมเดล <b>{res['raw_bodyfat']:.2f}%</b> {arrow} {sign} "
                f"<b>{abs(d):.2f}%</b> → ค่าที่ใช้จริง <b>{res['bodyfat']:.2f}%</b>"
                f"</span>{clamp}")
    st.markdown(f'<div class="note">{note}</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🩺 Stage 2 — ความเสี่ยงโรค NCDs</div>',
                unsafe_allow_html=True)
    r1, r2 = st.columns(2, gap="medium")
    with r1:
        render_risk("โรคเบาหวาน (Diabetes)", res["diabetes_pct"],
                    res["diabetes_level"], "ความน่าจะเป็นที่จะมีภาวะน้ำตาลในเลือดสูง")
    with r2:
        render_risk("ความดันโลหิตสูง (Hypertension)", res["hypertension_pct"],
                    res["hypertension_level"], "ความน่าจะเป็นที่จะมีภาวะความดันสูง")

    hi = max(res["diabetes_pct"], res["hypertension_pct"])
    if hi >= 50:
        st.error("🔴 **แนะนำให้พบแพทย์** — ควรตรวจเลือดและวัดความดันจริงเพื่อยืนยัน")
    elif hi >= 25:
        st.warning("🟡 **ควรเฝ้าระวัง** — ปรับพฤติกรรมและตรวจสุขภาพประจำปีสม่ำเสมอ")
    else:
        st.success("🟢 **อยู่ในเกณฑ์ดี** — รักษาพฤติกรรมสุขภาพนี้ต่อไป")

    with st.expander("🔬 ดูรายละเอียดเชิงเทคนิค"):
        fr = res.get("face_report")
        face_line = (f"| Face Guard | `{fr.backend}` · {fr.message} |\n"
                     if fr else "| Face Guard | ปิดใช้งาน |\n")
        st.markdown(f"""
| รายการ | ค่า |
|---|---|
| โหมดการประเมิน | {res['mode_text']} |
| โมเดล Body Fat | `{'with_waist' if res['has_waist'] else 'no_waist'}` |
{face_line}| BMI (ViT output) | `{res['bmi']:.4f}` |
| Epistemic Uncertainty (±SD) | `±{res['bmi_std']:.4f}` |
| Body Fat ดิบ | `{res['raw_bodyfat']:.4f}%` |
| Calibration Delta | `{res['calibration_delta']:+.2f}%` |
| Body Fat หลังปรับ | `{res['bodyfat']:.4f}%` |
| P(Diabetes) | `{res['diabetes_pct']/100:.4f}` |
| P(Hypertension) | `{res['hypertension_pct']/100:.4f}` |
| Device | `{res.get('device','-')}` |""")

    st.markdown("""<div style="height:20px"></div>
    <div class="disclaimer">⚠️ <b>ข้อจำกัดความรับผิดชอบ</b> — ผลลัพธ์นี้เป็นการ
    <b>คัดกรองเบื้องต้น</b>จากโมเดล AI ไม่ใช่การวินิจฉัยทางการแพทย์
    และไม่สามารถใช้แทนการตรวจโดยแพทย์หรือผลตรวจทางห้องปฏิบัติการได้
    หากมีอาการผิดปกติกรุณาปรึกษาแพทย์<br><br>
    🔒 <b>ความเป็นส่วนตัว</b> — ภาพใบหน้าถูกประมวลผลในหน่วยความจำเท่านั้น
    ไม่มีการจัดเก็บลงเซิร์ฟเวอร์</div>""", unsafe_allow_html=True)