import base64
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT_DIR, "scripts", "results")
OUT_HTML = os.path.join(ROOT_DIR, "reports", "slide_results_blocks.html")


def get_base64_img(filename):
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def build_html():
    fig1_b64 = get_base64_img("fig1_stage15_scatter.png")
    fig2_b64 = get_base64_img("fig2_roc_curves.png")
    fig3_b64 = get_base64_img("fig3_calibration.png")
    fig4_b64 = get_base64_img("fig4_confusion_matrix.png")

    html = f"""<!DOCTYPE html>
<html lang="th">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face2Health — Experimental Results & Evaluation Slides (Monochrome Luxury)</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Thai:wght@400;500;600;700;800&family=Prompt:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg-canvas: #0b0b0c;
    --title-main: #ffffff;
    --subtitle-silver: #d4d4d8;
    --card-ivory: #f7f7f5;
    --card-text: #18181b;
    --card-border-left: #18181b;
    --table-bg: #131316;
    --table-border: #2e2e36;
    --th-bg: #1e1e24;
    --th-text: #ffffff;
    --td-text: #e4e4e7;
    --row-highlight: #23232b;
    --hl-green: #34d399;
    --hl-red: #fb7185;
    --hl-yellow: #fbbf24;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  html, body {{
    width: 100vw;
    height: 100vh;
    background-color: var(--bg-canvas);
    color: var(--td-text);
    font-family: 'Prompt', 'Noto Sans Thai', 'Segoe UI', sans-serif;
    overflow: hidden;
    user-select: none;
    -webkit-font-smoothing: antialiased;
  }}

  /* Viewport wrapper that centers a locked 16:9 1920x1080 stage */
  .viewport {{
    width: 100vw;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: var(--bg-canvas);
  }}

  .slide-deck {{
    width: 1920px;
    height: 1080px;
    background-color: var(--bg-canvas);
    position: relative;
    transform-origin: center center;
  }}

  .slide {{
    display: none;
    width: 1920px;
    height: 1080px;
    padding: 38px 64px 36px 64px;
    background-color: var(--bg-canvas);
    flex-direction: column;
    justify-content: space-between;
  }}

  .slide.active {{
    display: flex;
  }}

  /* Header Section with Editorial Platinum Accent */
  .slide-header {{
    margin-bottom: 16px;
    flex-shrink: 0;
    transition: opacity 0.2s ease;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    padding-bottom: 12px;
  }}

  .main-title {{
    font-size: 42px;
    font-weight: 800;
    color: #ffffff !important;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin-bottom: 4px;
  }}

  .sub-title {{
    font-size: 23px;
    font-weight: 500;
    color: var(--subtitle-silver);
    line-height: 1.28;
  }}

  .sub-title span.tag-pill {{
    background-color: #ffffff;
    color: #0b0b0c;
    border-radius: 6px;
    padding: 2px 10px;
    font-weight: 700;
    display: inline-block;
    margin-left: 6px;
    font-size: 21px;
    letter-spacing: 0.2px;
  }}

  /* Content Layouts */
  .content-split {{
    display: flex;
    gap: 32px;
    flex: 1;
    align-items: stretch;
    min-height: 0;
  }}

  .col-left-52 {{ width: 52%; display: flex; flex-direction: column; justify-content: flex-start; gap: 20px; }}
  .col-right-48 {{ width: 48%; display: flex; flex-direction: column; justify-content: center; gap: 16px; }}
  .col-left-55 {{ width: 55%; display: flex; flex-direction: column; justify-content: flex-start; gap: 20px; }}
  .col-right-45 {{ width: 45%; display: flex; flex-direction: column; justify-content: center; gap: 15px; }}

  .content-full {{
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    flex: 1;
    gap: 20px;
    min-height: 0;
  }}

  /* Section Label above tables */
  .block-label {{
    font-size: 19px;
    font-weight: 700;
    color: var(--subtitle-silver);
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
    letter-spacing: 0.2px;
  }}

  .block-label::before {{
    content: "";
    display: inline-block;
    width: 6px;
    height: 18px;
    background-color: #ffffff;
    border-radius: 2px;
  }}

  /* Data Tables - Deep Graphite & Platinum Borders */
  .theme-table {{
    width: 100%;
    border-collapse: collapse;
    background-color: var(--table-bg);
    border: 1.5px solid var(--table-border);
    border-radius: 8px;
    overflow: hidden;
  }}

  .theme-table th {{
    background-color: var(--th-bg);
    color: var(--th-text);
    font-weight: 700;
    font-size: 17px;
    padding: 12px 14px;
    border: 1px solid var(--table-border);
    text-align: center;
    line-height: 1.25;
  }}

  .theme-table td {{
    color: var(--td-text);
    font-size: 16.5px;
    padding: 11px 14px;
    border: 1px solid var(--table-border);
    text-align: center;
    vertical-align: middle;
    line-height: 1.32;
  }}

  .theme-table td.left, .theme-table th.left {{
    text-align: left;
  }}

  .theme-table tr.ours-row {{
    background-color: var(--row-highlight);
  }}

  .theme-table tr.ours-row td:first-child {{
    border-left: 4px solid #ffffff !important;
  }}

  .theme-table tr.ours-row td {{
    font-weight: 600;
  }}

  .theme-table tr.sub-group-header td {{
    background-color: #1a1a20;
    color: #ffffff;
    font-weight: 700;
    font-size: 16px;
    text-align: left;
    padding: 7px 14px;
    letter-spacing: 0.2px;
    border-top: 1.5px solid var(--table-border);
    border-bottom: 1.5px solid var(--table-border);
  }}

  /* Semantic Metric Highlights */
  .hl-white-badge {{
    display: inline-block;
    color: #ffffff;
    background-color: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.2px;
  }}

  .hl-green {{
    color: var(--hl-green);
    font-weight: 800;
  }}

  .hl-red {{
    color: var(--hl-red);
    font-weight: 700;
  }}

  .hl-yellow {{
    color: var(--hl-yellow);
    font-weight: 700;
  }}

  .sub-ci {{
    display: block;
    font-size: 13px;
    color: #a1a1aa;
    font-weight: 400;
    margin-top: 2px;
  }}

  /* Inverted Ivory Cards - Editorial & Luxurious */
  .cream-card {{
    background-color: var(--card-ivory);
    color: var(--card-text);
    border-radius: 12px;
    border: 1px solid #e4e4e7;
    border-left: 6px solid var(--card-border-left);
    padding: 16px 22px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
  }}

  .cream-card h4 {{
    font-size: 19.5px;
    font-weight: 800;
    color: #09090b;
    margin-bottom: 8px;
    border-bottom: 1.5px solid #e4e4e5;
    padding-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}

  .cream-card ul {{
    list-style: none;
    padding: 0;
    margin: 0;
  }}

  .cream-card ul li {{
    font-size: 16.5px;
    line-height: 1.45;
    margin-bottom: 6px;
    position: relative;
    padding-left: 18px;
    color: var(--card-text);
    font-weight: 500;
  }}

  .cream-card ul li:last-child {{
    margin-bottom: 0;
  }}

  .cream-card ul li::before {{
    content: "▪";
    position: absolute;
    left: 0;
    color: #18181b;
    font-size: 15px;
  }}

  .cream-card p {{
    font-size: 16.5px;
    line-height: 1.48;
    color: var(--card-text);
    font-weight: 500;
  }}

  .cream-card strong {{
    color: #09090b;
    font-weight: 800;
  }}

  .cream-card .stat-pill {{
    display: inline-block;
    background-color: #18181b;
    color: #ffffff;
    font-weight: 700;
    padding: 2px 9px;
    border-radius: 6px;
    font-size: 15.5px;
    margin: 0 2px;
    letter-spacing: 0.2px;
  }}

  /* Figure Containers in Crisp White Frame */
  .figure-card {{
    background-color: #ffffff;
    border-radius: 14px;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
    border: 1px solid rgba(255, 255, 255, 0.15);
    height: 100%;
    min-height: 0;
    overflow: hidden;
    gap: 10px;
  }}

  .figure-card img {{
    flex: 1;
    min-height: 0;
    width: 100%;
    max-width: 100%;
    object-fit: contain;
    border-radius: 8px;
  }}

  .figure-caption {{
    flex-shrink: 0;
    width: 100%;
    background-color: #f1f1f4;
    border-left: 5px solid #18181b;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 15.5px;
    font-weight: 700;
    color: #18181b;
    text-align: left;
    line-height: 1.38;
  }}

  /* Dual figure stack on Slide 3 */
  .figure-grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    height: 100%;
    align-items: stretch;
  }}

  /* 3 Cards Grid on Slide 4 */
  .cards-grid-3 {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }}

  /* Traffic Light Badges for Slide 4 */
  .tier-badge {{
    display: inline-flex;
    align-items: center;
    gap: 10px;
    font-weight: 800;
    font-size: 17.5px;
    padding: 6px 14px;
    border-radius: 999px;
    white-space: nowrap;
  }}

  .tier-low {{ background-color: rgba(52, 211, 153, 0.15); color: #34d399; border: 1.5px solid #34d399; }}
  .tier-watch {{ background-color: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1.5px solid #fbbf24; }}
  .tier-pos {{ background-color: rgba(251, 146, 60, 0.18); color: #fb923c; border: 1.5px solid #fb923c; }}
  .tier-urgent {{ background-color: rgba(251, 113, 133, 0.22); color: #fb7185; border: 1.5px solid #fb7185; }}

  /* Floating Navigation HUD */
  .nav-hud {{
    position: fixed;
    bottom: 16px;
    right: 24px;
    background: rgba(19, 19, 22, 0.95);
    border: 1px solid var(--table-border);
    padding: 8px 16px;
    border-radius: 999px;
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 14px;
    color: var(--subtitle-silver);
    z-index: 999;
    box-shadow: 0 6px 16px rgba(0,0,0,0.6);
  }}

  body.hide-headers .nav-hud {{
    display: none;
  }}

  .nav-btn {{
    background: var(--th-bg);
    color: var(--th-text);
    border: 1px solid var(--table-border);
    border-radius: 6px;
    padding: 4px 10px;
    cursor: pointer;
    font-family: inherit;
    font-weight: 700;
    font-size: 13px;
    transition: all 0.15s ease;
  }}

  .nav-btn:hover {{
    background: #2e2e36;
    color: #ffffff;
  }}
</style>
</head>
<body>

<div class="viewport">
  <div class="slide-deck" id="slideDeck">

    <!-- ================================================================= -->
    <!-- SLIDE 1: STAGE 1 (VISION BMI) & STAGE 1.5 (DEXA REGRESSION)       -->
    <!-- ================================================================= -->
    <section class="slide active" data-slide="1">
      <header class="slide-header">
        <div class="main-title">Experimental Results</div>
        <div class="sub-title">Stage 1: การประมาณค่า BMI จากใบหน้า (Vision BMI) &amp; Stage 1.5: การประมาณเปอร์เซ็นต์ไขมันร่างกาย DEXA</div>
      </header>

      <div class="content-split">
        <div class="col-left-52">
          <div>
            <div class="block-label">ตารางสรุปความแม่นยำ Stage 1 (VisualBMI) และ Stage 1.5 (NHANES DEXA)</div>
            <table class="theme-table">
              <thead>
                <tr>
                  <th class="left" style="width: 32%;">โมเดล / ขั้นตอน (Stage &amp; Model)</th>
                  <th class="left" style="width: 24%;">ชุดข้อมูลทดสอบ (Dataset)</th>
                  <th style="width: 14%;">MAE</th>
                  <th style="width: 14%;">RMSE</th>
                  <th style="width: 16%;">R² / การคัดกรอง</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="left"><strong>Stage 1: ViT-H/14</strong><br><span class="sub-ci">ก่อนกรองความไม่แน่นอน</span></td>
                  <td class="left">VisualBMI Test<br><span class="sub-ci">Pooled (N = 793 ภาพ)</span></td>
                  <td>3.26 kg/m²</td>
                  <td>4.40 kg/m²</td>
                  <td>R² = 0.735</td>
                </tr>
                <tr class="ours-row">
                  <td class="left"><strong>Stage 1 + MC Dropout (Ours)</strong><br><span class="sub-ci">สุ่ม 25 รอบ, คัดทิ้งเมื่อ σ &gt; 1.80</span></td>
                  <td class="left">VisualBMI ที่ผ่านเกณฑ์<br><span class="sub-ci">Retained (n = 760 ภาพ)</span></td>
                  <td><span class="hl-green">3.08 kg/m²</span></td>
                  <td><span class="hl-white-badge">4.12 kg/m²</span></td>
                  <td><span class="hl-white-badge">คัดทิ้ง 4.2%</span><br><span class="sub-ci">(33 ภาพ MAE = 7.43)</span></td>
                </tr>
                <tr>
                  <td class="left"><strong>Stage 1: Controlled Subset</strong><br><span class="sub-ci">ภาพหน้าตรงมาตรฐาน + MC Gate</span></td>
                  <td class="left">VisualBMI Frontal<br><span class="sub-ci">Standardized (n = 412)</span></td>
                  <td><span class="hl-green">2.14 kg/m²</span></td>
                  <td><span class="hl-green">2.82 kg/m²</span></td>
                  <td>R² = 0.68<br><span class="sub-ci">(หลังกรอง σ &gt; 1.80)</span></td>
                </tr>
                <tr class="ours-row">
                  <td class="left"><strong>Stage 1.5: DEXA XGBoost (Primary)</strong><br><span class="sub-ci">ใช้ [BMI, อายุ, เพศ, กิจกรรมทางกาย]</span></td>
                  <td class="left">ไม่ใช้สายวัดรอบเอว<br><span class="sub-ci">NHANES Test (N = 531)</span></td>
                  <td><span class="hl-green">3.42 pp</span></td>
                  <td>4.25 pp</td>
                  <td><span class="hl-white-badge" style="font-size:18px;">R² = 0.74</span></td>
                </tr>
                <tr>
                  <td class="left"><strong>Stage 1.5: DEXA XGBoost (Auxiliary)</strong><br><span class="sub-ci">เพิ่มความยาวรอบเอว (Waist Circum.)</span></td>
                  <td class="left">ใช้สายวัดรอบเอวเสริม<br><span class="sub-ci">NHANES Test (N = 531)</span></td>
                  <td><span class="hl-green">2.85 pp</span></td>
                  <td>3.62 pp</td>
                  <td><span class="hl-green" style="font-size:18px;">R² = 0.82</span></td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="cream-card">
            <h4>💡 สรุปสาระสำคัญ (Key Findings — Stage 1 &amp; Stage 1.5)</h4>
            <ul>
              <li><strong>Epistemic Gate Work:</strong> กลไก MC Dropout (25 รอบ, &sigma; &gt; 1.80) ดักจับและคัดทิ้งภาพที่มีความคลาดเคลื่อนสูงกว่าปกติถึง <span class="stat-pill">2.4 เท่า</span> (กลุ่มถูกคัดทิ้ง 33 ภาพมี MAE = 7.43 kg/m² เทียบกับภาพที่ผ่านเกณฑ์ 3.08 kg/m²)</li>
              <li><strong>DEXA Feature Mapping:</strong> โมเดล Stage 1.5 แปลงข้อมูลพื้นฐานเข้าสู่สเกลเปอร์เซ็นต์ไขมันร่างกายจริง (DEXA Body Fat %) ได้แม่นยำถึง <span class="stat-pill">R² = 0.74</span> (MAE = 3.42 pp) โดย<strong>ไม่ต้องใช้สายวัดหรืออุปกรณ์ฮาร์ดแวร์เสริม</strong></li>
            </ul>
          </div>
        </div>

        <div class="col-right-48">
          <div class="figure-card">
            <img src="{fig1_b64}" alt="Stage 1.5 DEXA Scatter Plot">
            <div class="figure-caption">
              📊 <strong>กราฟเปรียบเทียบค่าเปอร์เซ็นต์ไขมันร่างกายจริงจาก DEXA กับค่าที่ทำนายได้ (N = 531):</strong><br>
              • (ซ้าย) โมเดลหลักไม่ใช้รอบเอว R² = 0.74, MAE = 3.42 pp | (ขวา) โมเดลเสริมใช้รอบเอว R² = 0.82, MAE = 2.85 pp
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- SLIDE 2: STAGE 2 CLASSIFICATION & F2-SCORE THRESHOLD OPTIMIZATION -->
    <!-- ================================================================= -->
    <section class="slide" data-slide="2">
      <header class="slide-header">
        <div class="main-title">Experimental Results</div>
        <div class="sub-title">Stage 2: ประสิทธิภาพการคัดกรองโรคเรื้อรัง &amp; การปรับจุดตัดด้วย <span class="tag-pill">F₂-Score Optimization</span></div>
      </header>

      <div class="content-split">
        <div class="col-left-55">
          <div>
            <div class="block-label">ตารางเปรียบเทียบประสิทธิภาพการคัดกรองโรคเบาหวาน (DIQ010) และความดันโลหิตสูง (BPQ020)</div>
            <table class="theme-table" style="font-size: 14.5px;">
              <thead>
                <tr>
                  <th class="left" style="width: 27%; padding: 7px 8px;">โมเดล (Model)</th>
                  <th style="width: 19%; padding: 7px 8px;">เกณฑ์จุดตัด (Rule)</th>
                  <th style="width: 10%; padding: 7px 8px;">Cutoff (τ)</th>
                  <th style="width: 14%; padding: 7px 8px;">ROC-AUC</th>
                  <th style="width: 16%; padding: 7px 8px;">Sensitivity (Recall)</th>
                  <th style="width: 14%; padding: 7px 8px;">Spec. / Prec.</th>
                </tr>
              </thead>
              <tbody>
                <tr class="sub-group-header">
                  <td colspan="6" style="padding: 4px 8px;">1. โรคเบาหวาน (Diabetes DIQ010 — N = 531 ราย | ผู้ป่วยจริง 55 ราย, ความชุก 10.36%)</td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Logistic Regression (Baseline)</td>
                  <td style="padding: 5px 8px;">Default / Opt. F2</td>
                  <td style="padding: 5px 8px;">0.500 / 0.068</td>
                  <td style="padding: 5px 8px;">0.742<span class="sub-ci">(0.678–0.806)</span></td>
                  <td style="padding: 5px 8px;">0.0% &rarr; 89.1%</td>
                  <td style="padding: 5px 8px;">39.3% / 14.5%</td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Primary Monotonic XGB (Ours)</td>
                  <td style="padding: 5px 8px;">Default (ปกติ)</td>
                  <td style="padding: 5px 8px;">0.500</td>
                  <td style="padding: 5px 8px;">0.762<span class="sub-ci">(0.701–0.823)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-red">0.00% (0/55 ราย)</span><span class="sub-ci">ผลจาก Class Imbalance</span></td>
                  <td style="padding: 5px 8px;">100.0% / 0.0%</td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Primary Monotonic XGB (Ours)</td>
                  <td style="padding: 5px 8px;">Youden's J</td>
                  <td style="padding: 5px 8px;">0.082</td>
                  <td style="padding: 5px 8px;">0.762<span class="sub-ci">(0.701–0.823)</span></td>
                  <td style="padding: 5px 8px;">78.18% (43/55 ราย)</td>
                  <td style="padding: 5px 8px;">59.0% / 18.1%</td>
                </tr>
                <tr class="ours-row">
                  <td class="left" style="padding: 5px 8px;"><strong>Primary Monotonic XGB (Ours)</strong></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">Opt. F2 (คัดกรอง)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.060</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.762</span><span class="sub-ci">(0.701–0.823)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-green" style="font-size:17px;">90.91% (50/55 ราย)</span></td>
                  <td style="padding: 5px 8px;"><strong>48.7% / 17.0%</strong></td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Auxiliary XGB (ใช้สายวัดเอว)</td>
                  <td style="padding: 5px 8px;">Opt. F2 (คัดกรอง)</td>
                  <td style="padding: 5px 8px;">0.061</td>
                  <td style="padding: 5px 8px;">0.763<span class="sub-ci">(0.702–0.824)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-green">89.09% (49/55 ราย)</span></td>
                  <td style="padding: 5px 8px;">48.9% / 16.8%</td>
                </tr>

                <tr class="sub-group-header">
                  <td colspan="6" style="padding: 4px 8px;">2. โรคความดันโลหิตสูง (Hypertension BPQ020 — N = 533 ราย | ผู้ป่วยจริง 181 ราย, ความชุก 33.96%)</td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Logistic Regression (Baseline)</td>
                  <td style="padding: 5px 8px;">Default / Opt. F2</td>
                  <td style="padding: 5px 8px;">0.500 / 0.165</td>
                  <td style="padding: 5px 8px;">0.740<span class="sub-ci">(0.697–0.783)</span></td>
                  <td style="padding: 5px 8px;">48.1% &rarr; 89.5%</td>
                  <td style="padding: 5px 8px;">34.7% / 41.3%</td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Primary Monotonic XGB (Ours)</td>
                  <td style="padding: 5px 8px;">Default / Youden's J</td>
                  <td style="padding: 5px 8px;">0.500 / 0.339</td>
                  <td style="padding: 5px 8px;">0.757<span class="sub-ci">(0.715–0.799)</span></td>
                  <td style="padding: 5px 8px;">50.3% / 71.3%</td>
                  <td style="padding: 5px 8px;">82.7% / 67.0%</td>
                </tr>
                <tr class="ours-row">
                  <td class="left" style="padding: 5px 8px;"><strong>Primary Monotonic XGB (Ours)</strong></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">Opt. F2 (คัดกรอง)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.155</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.757</span><span class="sub-ci">(0.715–0.799)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-green" style="font-size:17px;">90.61% (164/181 ราย)</span></td>
                  <td style="padding: 5px 8px;"><strong>40.9% / 44.1%</strong></td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Auxiliary XGB (ใช้สายวัดเอว)</td>
                  <td style="padding: 5px 8px;">Opt. F2 (คัดกรอง)</td>
                  <td style="padding: 5px 8px;">0.158</td>
                  <td style="padding: 5px 8px;">0.755<span class="sub-ci">(0.713–0.797)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-green">88.40% (160/181 ราย)</span></td>
                  <td style="padding: 5px 8px;">41.8% / 43.8%</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="cream-card" style="padding: 10px 16px;">
            <h4 style="font-size: 17.5px; margin-bottom: 4px; padding-bottom: 4px;">🎯 เหตุผลทางคลินิก: ทำไมต้องใช้ F₂-Score (β = 2) ปรับจุดตัดในงานคัดกรองโรค?</h4>
            <ul>
              <li style="font-size: 14px; line-height: 1.36; margin-bottom: 4px;"><strong>ต้นทุนไม่สมมาตร &amp; ถ่วงน้ำหนัก Recall 2 เท่า (β = 2):</strong> การปล่อยผู้ป่วยหลุดรอด (<strong>False Negative</strong>) อันตรายกว่าการเตือนเผื่อ (<strong>False Positive</strong>) สมการ F<sub>2</sub>-Score จึงให้น้ำหนัก Recall สูงกว่า Precision 2 เท่าเพื่อแก้ปัญหา Class Imbalance (ผู้ป่วยเบาหวานมีเพียง 10.36% ทำให้จุดตัดปกติ &tau; = 0.50 พลาดผู้ป่วยทั้งหมด)</li>
              <li style="font-size: 14px; line-height: 1.36;"><strong>กู้คืนความไวสู่ระดับคลินิก &amp; ชนะ Baseline:</strong> การปรับจุดตัด F<sub>2</sub>-Optimal (&tau; = 0.060 / 0.155) ดึง Sensitivity พุ่งขึ้นเป็น <span class="stat-pill">90.91% (50/55 ราย)</span> ในเบาหวาน และ <span class="stat-pill">90.61% (164/181 ราย)</span> ในความดัน พร้อมชนะ Logistic Regression อย่างมีนัยสำคัญทางสถิติ (<span class="stat-pill">&Delta;AUC = +0.020, p = 0.015</span> และ <span class="stat-pill">&Delta;AUC = +0.017, p = 0.006</span>)</li>
            </ul>
          </div>
        </div>

        <div class="col-right-45">
          <div class="figure-card">
            <img src="{fig4_b64}" alt="Confusion Matrix Comparison across Thresholds">
            <div class="figure-caption">
              📊 <strong>Confusion Matrices โรคเบาหวาน (N = 531):</strong><br>
              • <strong>(a) Default (&tau; = 0.500):</strong> ดักจับผู้ป่วยได้ 0/55 ราย (0.0% Sensitivity)<br>
              • <strong>(b) Youden's J (&tau; = 0.082):</strong> ดักจับผู้ป่วยได้ 43/55 ราย (78.2% Sensitivity)<br>
              • <strong>(c) F₂-Optimal (&tau; = 0.060):</strong> ดักจับผู้ป่วยได้สูงถึง <strong>50/55 ราย (90.91% Sensitivity)</strong>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- SLIDE 3: ABLATION, CALIBRATION & NOISE ROBUSTNESS                 -->
    <!-- ================================================================= -->
    <section class="slide" data-slide="3">
      <header class="slide-header">
        <div class="main-title">Model Evaluation &amp; Robustness</div>
        <div class="sub-title">การวิเคราะห์องค์ประกอบ (Ablation), Platt Calibration และความทนทานต่อสัญญาณรบกวน (Noise Sweep)</div>
      </header>

      <div class="content-split">
        <div class="col-left-52">
          <div>
            <div class="block-label">1. Component Ablation Study (ผลกระทบเมื่อตัดองค์ประกอบออก — เบาหวาน N = 531)</div>
            <table class="theme-table" style="margin-bottom: 8px; font-size: 14.5px;">
              <thead>
                <tr>
                  <th class="left" style="width: 34%; padding: 6px 8px;">โครงสร้างโมเดล (Configuration)</th>
                  <th style="width: 12%; padding: 6px 8px;">ROC-AUC</th>
                  <th style="width: 13%; padding: 6px 8px;">Sens (F2)</th>
                  <th style="width: 13%; padding: 6px 8px;">Spec (F2)</th>
                  <th style="width: 13%; padding: 6px 8px;">Brier Score</th>
                  <th style="width: 15%; padding: 6px 8px;">Monotonicity Violations</th>
                </tr>
              </thead>
              <tbody>
                <tr class="ours-row">
                  <td class="left" style="padding: 5px 8px;"><strong>Full Face2Health Pipeline (Ours)</strong></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.762</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-green">90.9%</span></td>
                  <td style="padding: 5px 8px;">48.7%</td>
                  <td style="padding: 5px 8px;"><span class="hl-green">0.081</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-green">0 ครั้ง (สมบูรณ์)</span></td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">No Body Fat (ตัด Stage 1.5 ออก)</td>
                  <td style="padding: 5px 8px;">0.748</td>
                  <td style="padding: 5px 8px;">83.6%</td>
                  <td style="padding: 5px 8px;">51.3%</td>
                  <td style="padding: 5px 8px;">0.085</td>
                  <td style="padding: 5px 8px;">0 ครั้ง</td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">No Monotonicity (ไม่บังคับทิศทาง)</td>
                  <td style="padding: 5px 8px;">0.747</td>
                  <td style="padding: 5px 8px;">85.5%</td>
                  <td style="padding: 5px 8px;">50.0%</td>
                  <td style="padding: 5px 8px;">0.084</td>
                  <td style="padding: 5px 8px;"><span class="hl-red">14 ครั้ง (เกิดความย้อนแย้ง)</span></td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">No Monot. &amp; No Calibration</td>
                  <td style="padding: 5px 8px;">0.747</td>
                  <td style="padding: 5px 8px;">80.0%</td>
                  <td style="padding: 5px 8px;">52.5%</td>
                  <td style="padding: 5px 8px;"><span class="hl-red">0.142 (เพี้ยนสูง)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-red">14 ครั้ง (เกิดความย้อนแย้ง)</span></td>
                </tr>
              </tbody>
            </table>

            <div class="block-label">2. ความทนทานต่อความคลาดเคลื่อนของภาพ (100 Monte Carlo Noise Draws &amp; Residual Resampling)</div>
            <table class="theme-table" style="font-size: 14.5px;">
              <thead>
                <tr>
                  <th class="left" style="width: 34%; padding: 6px 8px;">ระดับสัญญาณรบกวน (Noise Regime)</th>
                  <th style="width: 16%; padding: 6px 8px;">เบาหวาน Sens</th>
                  <th style="width: 17%; padding: 6px 8px;">เบาหวาน AUC (ΔDemo)</th>
                  <th style="width: 16%; padding: 6px 8px;">ความดัน Sens</th>
                  <th style="width: 17%; padding: 6px 8px;">ความดัน AUC (ΔDemo)</th>
                </tr>
              </thead>
              <tbody>
                <tr class="ours-row">
                  <td class="left" style="padding: 5px 8px;"><strong>ระดับควบคุม (Lab Frontal: σ = 2.82)</strong></td>
                  <td style="padding: 5px 8px;"><span class="hl-green">88.7% ± 2.4%</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.757</span> <span class="sub-ci">(+0.045, p &lt; 0.001)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-green">89.2% ± 1.6%</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.752</span> <span class="sub-ci">(+0.044, p &lt; 0.001)</span></td>
                </tr>
                <tr class="ours-row">
                  <td class="left" style="padding: 5px 8px;"><strong>ระดับใช้งานจริง (Pooled Test: σ = 4.40)</strong></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">85.1% ± 3.2%</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.745</span> <span class="sub-ci">(+0.033, p = 0.007)</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">86.4% ± 2.1%</span></td>
                  <td style="padding: 5px 8px;"><span class="hl-white-badge">0.741</span> <span class="sub-ci">(+0.033, p = 0.001)</span></td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">ระดับโซเชียลมีเดีย (Unconstrained: σ = 5.60)</td>
                  <td style="padding: 5px 8px;">81.3% ± 3.9%</td>
                  <td style="padding: 5px 8px;">0.735 <span class="sub-ci">(+0.024, p = 0.068)</span></td>
                  <td style="padding: 5px 8px;">83.7% ± 2.5%</td>
                  <td style="padding: 5px 8px;">0.732 <span class="sub-ci">(+0.025, p = 0.026)</span></td>
                </tr>
                <tr>
                  <td class="left" style="padding: 5px 8px;">Empirical Residual Resampling (Pooled)</td>
                  <td style="padding: 5px 8px;"><span class="hl-green">84.5% ± 2.9%</span></td>
                  <td style="padding: 5px 8px;">0.744 ± 0.008</td>
                  <td style="padding: 5px 8px;"><span class="hl-green">86.2% ± 1.9%</span></td>
                  <td style="padding: 5px 8px;">0.740 ± 0.006</td>
                </tr>
              </tbody>
            </table>
          </div>

          <div class="cream-card" style="padding: 10px 16px;">
            <h4 style="font-size: 17.5px; margin-bottom: 4px; padding-bottom: 4px;">🛡️ ความน่าเชื่อถือทางคลินิก &amp; จุดยืนการใช้งาน (Zero-Hardware Fallback)</h4>
            <ul>
              <li style="font-size: 14px; line-height: 1.36; margin-bottom: 3px;"><strong>Platt Calibration &amp; Monotonicity:</strong> ลดความคลาดเคลื่อนของความน่าจะเป็น (Brier Score) จาก <span class="stat-pill">0.142 &rarr; 0.081</span> และขจัดความย้อนแย้งทางคลินิกจาก <span class="stat-pill">14 ครั้ง &rarr; 0 ครั้ง</span> (ความเสี่ยงไม่ลดลงเมื่อ BMI หรือไขมันเพิ่มขึ้น)</li>
              <li style="font-size: 14px; line-height: 1.36;"><strong>Zero-Hardware Fallback:</strong> ออกแบบเป็นระบบสำรองเมื่อไม่มีตราชั่งหรือไม่ทราบน้ำหนัก-ส่วนสูงตัวเอง โดยยังคงให้ความแม่นยำชนะข้อมูลประชากรศาสตร์เดี่ยวๆ อย่างมีนัยสำคัญ (<i>p</i> &le; 0.007) จนถึงระดับ Noise ใช้งานจริง (&sigma;<sub>sim</sub> = 4.40 kg/m²)</li>
            </ul>
          </div>
        </div>

        <div class="col-right-48">
          <div class="figure-grid-2">
            <div class="figure-card">
              <img src="{fig3_b64}" alt="Platt Calibration Reliability Curves">
              <div class="figure-caption">
                📊 <strong>Calibration Reliability Curves:</strong><br>
                ลด Brier Score เบาหวานเหลือ 0.081 และความดันเหลือ 0.185 (เส้น Probability ทาบสนิทกับเส้นอุดมคติ)
              </div>
            </div>
            <div class="figure-card">
              <img src="{fig2_b64}" alt="ROC Curves Comparison">
              <div class="figure-caption">
                📈 <strong>ROC Curves Comparison:</strong><br>
                Primary Monotonic XGBoost ชนะ Logistic Regression Baseline อย่างมีนัยสำคัญทั้ง 2 โรค
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ================================================================= -->
    <!-- SLIDE 4: 4-TIER CLINICAL TRIAGE PROTOCOL                          -->
    <!-- ================================================================= -->
    <section class="slide" data-slide="4">
      <header class="slide-header">
        <div class="main-title">Clinical Triage Protocol</div>
        <div class="sub-title">การจัดลำดับความเสี่ยง 4 ระดับ (4-Tier Triage) และผลกระทบต่อการคัดกรองทางคลินิก</div>
      </header>

      <div class="content-full">
        <div>
          <div class="block-label">ตารางโปรโตคอลคัดกรอง 4 ระดับจากความน่าจะเป็นที่ผ่านการปรับเทียบแล้ว (Calibrated Posterior Probability)</div>
          <table class="theme-table">
            <thead>
              <tr>
                <th class="left" style="width: 17%;">ระดับความเสี่ยง (Triage Tier)</th>
                <th style="width: 18%;">เกณฑ์ความน่าจะเป็น (Probability)</th>
                <th style="width: 12%;">สัดส่วนเบาหวาน<br>(N = 531)</th>
                <th style="width: 13%;">ป่วยจริงเบาหวาน<br>(True Prevalence)</th>
                <th style="width: 12%;">สัดส่วนความดัน<br>(N = 533)</th>
                <th style="width: 13%;">ป่วยจริงความดัน<br>(True Prevalence)</th>
                <th class="left" style="width: 15%;">แนวทางปฏิบัติทางคลินิก<br>(Clinical Action)</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td class="left"><span class="tier-badge tier-low">🟢 Low Risk</span></td>
                <td>p &lt; 6.0% (เบาหวาน)<br><span class="sub-ci">p &lt; 15.5% (ความดัน)</span></td>
                <td><strong>44.6%</strong><span class="sub-ci">(237 ราย)</span></td>
                <td><span class="hl-green" style="font-size:20px;">2.11%</span><span class="sub-ci">(ป่วยจริงเพียง 5/237)</span></td>
                <td><strong>30.2%</strong><span class="sub-ci">(161 ราย)</span></td>
                <td><span class="hl-green">10.56%</span><span class="sub-ci">(17/161 ราย)</span></td>
                <td class="left">ตรวจสุขภาพประจำปีตามปกติ<br><span class="sub-ci">(ไม่ต้องเจาะเลือดซ้ำ)</span></td>
              </tr>
              <tr>
                <td class="left"><span class="tier-badge tier-watch">🟡 Watchful</span></td>
                <td>6.0% – 10.0% (เบาหวาน)<br><span class="sub-ci">15.5% – 25.0% (ความดัน)</span></td>
                <td><strong>23.9%</strong><span class="sub-ci">(127 ราย)</span></td>
                <td><span class="hl-yellow" style="font-size:20px;">7.87%</span><span class="sub-ci">(ป่วยจริง 10/127)</span></td>
                <td><strong>22.9%</strong><span class="sub-ci">(122 ราย)</span></td>
                <td><span class="hl-yellow">27.05%</span><span class="sub-ci">(33/122 ราย)</span></td>
                <td class="left">เจาะน้ำตาลปลายนิ้ว (Capillary)<br><span class="sub-ci">/ วัดความดันโลหิตซ้ำที่จุดคัดกรอง</span></td>
              </tr>
              <tr>
                <td class="left"><span class="tier-badge tier-pos">🟠 Screen Positive</span></td>
                <td>10.0% – 20.0% (เบาหวาน)<br><span class="sub-ci">25.0% – 50.0% (ความดัน)</span></td>
                <td><strong>22.8%</strong><span class="sub-ci">(121 ราย)</span></td>
                <td><span class="hl-white-badge" style="font-size:18px; color:#fb923c; border-color:#fb923c;">18.18%</span><span class="sub-ci">(ป่วยจริง 22/121)</span></td>
                <td><strong>18.4%</strong><span class="sub-ci">(98 ราย)</span></td>
                <td><span class="hl-white-badge" style="font-size:18px; color:#fb923c; border-color:#fb923c;">40.82%</span><span class="sub-ci">(40/98 ราย)</span></td>
                <td class="left">ส่งตรวจแล็บเลือดดำมาตรฐาน<br><span class="sub-ci">(Fasting Glucose / HbA1c)</span></td>
              </tr>
              <tr class="ours-row">
                <td class="left"><span class="tier-badge tier-urgent">🔴 Urgent</span></td>
                <td><strong>p &ge; 20.0% (เบาหวาน)</strong><br><span class="sub-ci"><strong>p &ge; 50.0% (ความดัน)</strong></span></td>
                <td><strong>8.7%</strong><span class="sub-ci">(46 ราย)</span></td>
                <td><span class="hl-red" style="font-size:22px; font-weight:800;">39.13%</span><span class="sub-ci">(ป่วยจริงพุ่งถึง 18/46)</span></td>
                <td><strong>28.5%</strong><span class="sub-ci">(152 ราย)</span></td>
                <td><span class="hl-red" style="font-size:22px; font-weight:800;">59.87%</span><span class="sub-ci">(ป่วยจริง 91/152)</span></td>
                <td class="left"><strong style="color:#ffffff;">ส่งต่อพบแพทย์ทันที</strong><br><span class="sub-ci">(Priority Medical Referral)</span></td>
              </tr>
            </tbody>
          </table>
        </div>

        <div class="cards-grid-3">
          <div class="cream-card">
            <h4>✅ 1. ความปลอดภัยสูง (NPV = 97.89%)</h4>
            <p>
              คัดแยกผู้ที่มีความเสี่ยงต่ำเกือบครึ่งหนึ่ง (<span class="stat-pill">44.6%</span> หรือ 237/531 คน) ออกจากการเจาะเลือดที่ไม่จำเป็นได้อย่างปลอดภัย โดยมีอัตราผู้ป่วยหลุดรอด (False Omission Rate) เพียง <span class="stat-pill">2.11%</span> (5/237 ราย)
            </p>
          </div>

          <div class="cream-card">
            <h4>📈 2. ความเข้มข้นผู้ป่วย (3.8x Enrichment)</h4>
            <p>
              กลุ่ม <strong>Urgent</strong> มีความชุกโรคเบาหวานจริงสูงถึง <span class="stat-pill">39.13%</span> ซึ่งเข้มข้นกว่าค่าเฉลี่ยประชากรทั่วไป (10.36%) ถึง <strong>3.8 เท่า</strong> และสูงกว่ากลุ่ม Low Risk ถึง <strong>18.5 เท่า</strong> (ส่วนความดันพุ่งสูงถึง <span class="stat-pill">59.87%</span>)
            </p>
          </div>

          <div class="cream-card">
            <h4>💰 3. โหมดประหยัดงบคลินิก (Resource-Saving)</h4>
            <p>
              ในคลินิกชุมชนที่มีงบประมาณจำกัด หากส่งตรวจแล็บเฉพาะ 2 กลุ่มบน (<strong>Screen Positive + Urgent</strong> ซึ่งคิดเป็นเพียง <span class="stat-pill">31.5%</span> ของผู้รับบริการ) ยังคงดักจับผู้ป่วยเบาหวานได้สูงถึง <span class="stat-pill">72.7% (40/55 ราย)</span>
            </p>
          </div>
        </div>
      </div>
    </section>

  </div>
</div>

<!-- Navigation HUD -->
<div class="nav-hud">
  <button class="nav-btn" onclick="changeSlide(-1)">◀ Prev</button>
  <span id="slideIndicator">Slide 1 / 4</span>
  <button class="nav-btn" onclick="changeSlide(1)">Next ▶</button>
  <span style="opacity:0.65; margin-left:6px;">[กดลูกศร ◀ ▶ เลื่อนหน้า | กด <b>H</b> ซ่อนหัวข้อเพื่อแคปจอ]</span>
</div>

<script>
  let currentSlide = 1;
  const totalSlides = 4;

  function showSlide(n) {{
    if (n < 1) n = totalSlides;
    if (n > totalSlides) n = 1;
    currentSlide = n;

    document.querySelectorAll('.slide').forEach(s => {{
      s.classList.remove('active');
    }});
    const target = document.querySelector(`.slide[data-slide="${{currentSlide}}"]`);
    if (target) target.classList.add('active');

    const indicator = document.getElementById('slideIndicator');
    if (indicator) indicator.textContent = `Slide ${{currentSlide}} / ${{totalSlides}}`;

    try {{
      window.location.hash = 'slide' + currentSlide;
    }} catch(e) {{}}
  }}

  function changeSlide(delta) {{
    showSlide(currentSlide + delta);
  }}

  // Responsive 16:9 scaling to fit any browser window cleanly
  function fitDeckToViewport() {{
    const deck = document.getElementById('slideDeck');
    const scaleX = window.innerWidth / 1920;
    const scaleY = window.innerHeight / 1080;
    const scale = Math.min(scaleX, scaleY);
    deck.style.transform = `scale(${{scale}})`;
  }}

  window.addEventListener('resize', fitDeckToViewport);
  window.addEventListener('DOMContentLoaded', () => {{
    fitDeckToViewport();
    const params = new URLSearchParams(window.location.search);
    if (params.get('capture') === '1') {{
      document.body.classList.add('hide-headers');
    }}
    const hash = window.location.hash.replace('#slide', '').replace('#', '');
    const initialSlide = parseInt(params.get('slide') || hash || '1', 10);
    if (!isNaN(initialSlide) && initialSlide >= 1 && initialSlide <= totalSlides) {{
      showSlide(initialSlide);
    }}
  }});

  window.addEventListener('keydown', (e) => {{
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') {{
      e.preventDefault();
      changeSlide(1);
    }} else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {{
      e.preventDefault();
      changeSlide(-1);
    }} else if (e.key === 'h' || e.key === 'H') {{
      document.body.classList.toggle('hide-headers');
    }} else if (e.key >= '1' && e.key <= '4') {{
      showSlide(parseInt(e.key, 10));
    }}
  }});
</script>

</body>
</html>
"""

    os.makedirs(os.path.dirname(OUT_HTML), exist_ok=True)
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[+] Successfully generated {OUT_HTML} ({len(html):,} bytes)")


if __name__ == "__main__":
    build_html()
