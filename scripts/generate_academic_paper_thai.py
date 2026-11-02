import os
import re
import sys
import shutil
import pymupdf
import pythainlp.tokenize

# Ensure we can import from the same directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
PDF_OUT = os.path.join(REPORTS_DIR, "paper_thai.pdf")

# Step 1: Ensure figures exist in results/
try:
    import regenerate_figures_fix
    regenerate_figures_fix.main()
except Exception as e:
    print(f"[!] Note: Automatic figure generation call: {e}")

required_figs = [
    "fig0_pipeline_architecture.png",
    "fig1_stage15_scatter.png",
    "fig2_roc_curves.png",
    "fig3_calibration.png",
    "fig4_confusion_matrix.png"
]
for fig in required_figs:
    fig_path = os.path.join(RESULTS_DIR, fig)
    if not os.path.exists(fig_path):
        print(f"[WARNING] Required figure missing: {fig_path}")

thai_char_pat = re.compile(r'[\u0e00-\u0e7f]')


def add_zws(text):
    """
    Inserts zero-width space ONLY between two adjacent Thai words so that
    parentheses, numbers, punctuation, and English tokens are never split awkwardly.
    """
    if not thai_char_pat.search(text):
        return text
    words = pythainlp.tokenize.word_tokenize(text, engine='newmm')
    res = []
    no_break_before = {')', ']', '}', '”', '"', "'", ',', '.', ':', ';', '%', '!', '?'}
    no_break_after = {'(', '[', '{', '“', '"', "'"}
    for i, w in enumerate(words):
        if i > 0:
            prev = words[i - 1]
            if (
                thai_char_pat.search(prev)
                and thai_char_pat.search(w)
                and prev.strip() not in no_break_after
                and w.strip() not in no_break_before
            ):
                res.append('\u200b')
        res.append(w)
    return ''.join(res)


def insert_zws_to_html(html):
    """
    Applies Thai zero-width space tokenization to text nodes outside of <table class="paper-table">
    and <div class="title">, preserving explicit line breaks and cell formatting.
    """
    parts = re.split(
        r'(<table[^>]*class="[^"]*paper-table[^"]*"[^>]*>.*?</table>|<div[^>]*class="[^"]*title[^"]*"[^>]*>.*?</div>)',
        html,
        flags=re.DOTALL
    )
    processed_parts = []
    for part in parts:
        if (part.startswith('<table') and 'paper-table' in part[:120]) or (part.startswith('<div') and 'title' in part[:80]):
            processed_parts.append(part)
        else:
            tokens = re.split(r'(<[^>]+>|&[a-zA-Z0-9#]+;)', part)
            new_tokens = []
            for t in tokens:
                if not t:
                    continue
                if t.startswith('<') or (t.startswith('&') and t.endswith(';')):
                    new_tokens.append(t)
                else:
                    new_tokens.append(add_zws(t))
            processed_parts.append(''.join(new_tokens))
    return ''.join(processed_parts)


def prepare_tables(html_content, target_table_w=240.0):
    """
    Parses <table> tags with data-w (widths) attributes,
    calculating exact point widths for MuPDF (accounting for cell padding and borders)
    and injecting inline styles onto cells to guarantee clean rendering without clipping.
    """
    def replace_table(match):
        table_open_tag = match.group(1)
        inner_content = match.group(2)

        w_match = re.search(r'data-w="([^"]+)"', table_open_tag)
        nw_match = re.search(r'data-nw="([^"]+)"', table_open_tag)

        widths = [float(x.strip()) for x in w_match.group(1).split(',')] if w_match else []
        nw_cols = [int(x.strip()) for x in nw_match.group(1).split(',')] if nw_match else []

        num_cols = max(len(widths), 1)
        # Reserve 1.8pt per column for minimal horizontal cell padding + borders
        avail_w = target_table_w - (num_cols * 1.8)
        pt_widths = [(w / 100.0) * avail_w for w in widths]

        rows = re.findall(r'(<tr.*?>.*?</tr>)', inner_content, re.DOTALL)
        new_rows = []

        for row in rows:
            cells = re.findall(r'(<t[hd].*?>)(.*?)(</t[hd]>)', row, re.DOTALL)
            new_cells = []
            for col_idx, (open_tag, content, close_tag) in enumerate(cells):
                styles = []
                if col_idx < len(pt_widths):
                    styles.append(f"width: {pt_widths[col_idx]:.1f}pt")
                if col_idx in nw_cols:
                    styles.append("white-space: nowrap")

                if styles:
                    style_str = "; ".join(styles) + ";"
                    if 'style="' in open_tag:
                        open_tag = open_tag.replace('style="', f'style="{style_str} ')
                    else:
                        open_tag = open_tag.replace('>', f' style="{style_str}">')
                new_cells.append(f"{open_tag}{content}{close_tag}")
            new_rows.append("<tr>" + "".join(new_cells) + "</tr>")

        return f"{table_open_tag}{''.join(new_rows)}</table>"

    pattern = r'(<table[^>]*class="[^"]*paper-table[^"]*"[^>]*>)(.*?)(</table>)'
    return re.sub(pattern, replace_table, html_content, flags=re.DOTALL)


FONT_FACE_CSS = """
@font-face {
    font-family: 'TH SarabunPSK';
    src: url('THSarabunNew.ttf');
    font-weight: normal;
    font-style: normal;
}
@font-face {
    font-family: 'TH SarabunPSK';
    src: url('THSarabunNew-Bold.ttf');
    font-weight: bold;
    font-style: normal;
}
@font-face {
    font-family: 'TH SarabunPSK';
    src: url('THSarabunNew-Italic.ttf');
    font-weight: normal;
    font-style: italic;
}
@font-face {
    font-family: 'TH SarabunPSK';
    src: url('THSarabunNew-BoldItalic.ttf');
    font-weight: bold;
    font-style: italic;
}
@font-face {
    font-family: 'TH Sarabun New';
    src: url('THSarabunNew.ttf');
    font-weight: normal;
    font-style: normal;
}
@font-face {
    font-family: 'TH Sarabun New';
    src: url('THSarabunNew-Bold.ttf');
    font-weight: bold;
    font-style: normal;
}
@font-face {
    font-family: 'TH Sarabun New';
    src: url('THSarabunNew-Italic.ttf');
    font-weight: normal;
    font-style: italic;
}
@font-face {
    font-family: 'TH Sarabun New';
    src: url('THSarabunNew-BoldItalic.ttf');
    font-weight: bold;
    font-style: italic;
}
"""


def build_thai_pdf():
    print("[*] กำลังคอมไพล์เอกสารวิจัยฉบับภาษาไทย (Face2Health Thai Academic Paper PDF)...")

    # =========================================================================
    # PAGE 1: SINGLE-COLUMN FULL-WIDTH HEADER + ABSTRACT + KEYWORDS + FOOTNOTE
    # (Matches reports/[THAI]-67076012-Short Paper.pdf Page 1)
    # =========================================================================
    PAGE1_TOP_HTML = f"""
<style>
{FONT_FACE_CSS}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'TH SarabunPSK', 'TH Sarabun New', sans-serif;
    font-size: 14pt;
    line-height: 1.35;
    color: #000000;
}}
sup {{ font-size: 9pt; vertical-align: super; line-height: 0; font-weight: bold; }}
.header-box {{
    text-align: center;
    margin-bottom: 24pt;
}}
.title {{
    font-size: 18pt;
    font-weight: bold;
    line-height: 24pt;
    margin-bottom: 14pt;
    text-align: center;
    color: #000000;
}}
.author-name {{
    font-size: 14pt;
    font-weight: bold;
    line-height: 20pt;
    margin-bottom: 8pt;
    text-align: center;
    color: #000000;
}}
.author-inst {{
    font-size: 14pt;
    font-weight: bold;
    line-height: 20pt;
    margin-bottom: 6pt;
    text-align: center;
    color: #000000;
}}
.author-loc {{
    font-size: 14pt;
    font-weight: bold;
    line-height: 20pt;
    margin-bottom: 26pt;
    text-align: center;
    color: #000000;
}}
.abstract-heading {{
    font-size: 16pt;
    font-weight: bold;
    text-align: left;
    margin-bottom: 6pt;
    color: #000000;
}}
.abstract-body {{
    font-size: 14pt;
    font-weight: normal;
    line-height: 1.38;
    text-align: left;
    text-indent: 28pt;
    margin-bottom: 20pt;
    color: #000000;
}}
.keywords-line {{
    font-size: 14pt;
    font-weight: normal;
    line-height: 1.35;
    text-align: left;
    color: #000000;
}}
</style>
<div class="header-box">
  <div class="title">Face2Health: การคัดกรองความเสี่ยงโรคเบาหวานและความดันโลหิตสูง<br>โดยใช้วิชันทรานส์ฟอร์เมอร์สและเกรเดียนต์บูสต์ทรีส์</div>
  <div class="author-name">พลาธิป เหมวุฒิ<sup>*1</sup> และ เอื้ออังกูร ชัยวิวัฒน์พร<sup>*2</sup></div>
  <div class="author-inst">คณะเทคโนโลยีสารสนเทศ สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง</div>
  <div class="author-loc">เลขที่ 1 ซอยฉลองกรุง 1 แขวงลาดกระบัง เขตลาดกระบัง กรุงเทพฯ 10520</div>
</div>

<div class="abstract-heading">บทคัดย่อ</div>
<div class="abstract-body">บทความนี้นำเสนอกรอบการทำงานแบบแยกส่วน Face2Health สำหรับการคัดกรองความเสี่ยงเบื้องต้นของโรคเบาหวานที่ได้รับการวินิจฉัยโดยแพทย์ (DIQ010) และโรคความดันโลหิตสูง (BPQ020) จากภาพถ่ายใบหน้าสองมิติ กระบวนการทำงานประกอบด้วยการเชื่อมต่อ 3 ขั้นตอนหลักอย่างเป็นลำดับ ได้แก่ (1) การประมาณค่าดัชนีมวลกาย (BMI) ด้วยแบบจำลองวิชันทรานส์ฟอร์เมอร์ส (ViT-H/14) แบบตรึงโครงข่ายหลักร่วมกับโครงข่าย MLP 5 ชั้น พร้อมการคัดกรองความไม่แน่นอนด้วยเทคนิค Monte Carlo Dropout (2) การประมาณค่าเปอร์เซ็นต์ไขมันในร่างกายทั้งหมด (DEXA Total Body Fat) จากข้อมูลสัดส่วนทางกายภาพด้วยแบบจำลอง XGBoost Regressor และ (3) การจำแนกความเสี่ยงของโรคแบบไบนารีด้วยแบบจำลอง XGBoost Classifier ที่กำหนดเงื่อนไขความชันแบบโมโนโทนิกและปรับเทียบความน่าจะเป็นด้วยวิธี Platt Scaling ผลการประเมินประสิทธิภาพบนชุดข้อมูลประชากรผู้ใหญ่ CDC NHANES รอบ 2003–2004 ภายใต้แบบจำลองสัญญาณรบกวนจากการประมาณค่าทางสายตา (Gaussian Noise Perturbation) พบว่าแบบจำลองหลักยังคงรักษาค่าความไว (Sensitivity) ได้ที่ร้อยละ 81–85 สำหรับโรคเบาหวาน และร้อยละ 84–86 สำหรับโรคความดันโลหิตสูง ณ ระดับสัญญาณรบกวน σ<sub>sim</sub> = 4.40–5.60 kg/m² (เทียบกับประมาณร้อยละ 91 ณ ระดับสัญญาณรบกวนต่ำสุด σ<sub>sim</sub> = 2.14 kg/m²) โดยให้ค่า AUC สูงกว่าแบบจำลองฐานที่ใช้เฉพาะข้อมูลประชากรศาสตร์อย่างมีนัยสำคัญทางสถิติ ยกเว้นกรณีโรคเบาหวานที่ระดับสัญญาณรบกวนสูงสุด σ<sub>sim</sub> = 5.60 kg/m² (p = 0.068) ผลการศึกษานี้สนับสนุนบทบาทของ Face2Health ในการคัดกรองจัดลำดับความเสี่ยงเบื้องต้นแบบฉวยโอกาส (Opportunistic Pre-Screening Triage) ในฐานะทางเลือกสำรองที่ไม่ต้องใช้อุปกรณ์วัดทางกายภาพ (Zero-Hardware Fallback) เมื่อไม่มีเครื่องชั่งน้ำหนักมาตรฐานหรือข้อมูลการรายงานตนเองที่น่าเชื่อถือ มิใช่เครื่องมือสำหรับการวินิจฉัยโรคทางการแพทย์</div>

<div class="keywords-line"><b>คำสำคัญ:</b> การคัดกรองความเสี่ยงโรค, ดัชนีมวลกาย, ภาพถ่ายใบหน้า, วิชันทรานส์ฟอร์เมอร์ส, เกรเดียนต์บูสต์ทรีส์</div>
"""

    PAGE1_FOOTNOTE_HTML = f"""
<style>
{FONT_FACE_CSS}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: 'TH SarabunPSK', 'TH Sarabun New', sans-serif;
    font-size: 13pt;
    line-height: 1.35;
    color: #000000;
}}
.fn-sep {{
    font-size: 12pt;
    letter-spacing: 0.5px;
    margin-bottom: 4pt;
    color: #000000;
}}
.fn-line {{
    font-size: 13pt;
    line-height: 1.4;
    text-align: left;
    margin-bottom: 2pt;
    color: #000000;
}}
</style>
<div class="fn-sep">-------------------------------------------------------------------------------------------------------------------------------------------</div>
<div class="fn-line">*Author's Email: 67070255@kmitl.ac.th, 67070302@kmitl.ac.th</div>
<div class="fn-line"><sup>1,2</sup> นักศึกษาปริญญาตรี คณะเทคโนโลยีสารสนเทศ สถาบันเทคโนโลยีพระจอมเกล้าเจ้าคุณทหารลาดกระบัง</div>
"""

    # =========================================================================
    # PAGES 2+: TWO-COLUMN BODY CONTENT (5 CHAPTERS IN ARABIC DECIMAL FORMAT)
    # =========================================================================
    BODY_HTML = f"""
<style>
{FONT_FACE_CSS}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    margin: 0; padding: 0;
    font-family: 'TH SarabunPSK', 'TH Sarabun New', sans-serif;
    font-size: 14pt;
    line-height: 1.25;
    text-align: left;
    color: #000000;
    background-color: transparent;
}}
h2 {{
    font-family: 'TH SarabunPSK', 'TH Sarabun New', sans-serif;
    font-size: 16pt;
    font-weight: bold;
    text-align: left;
    margin: 8pt 0 3pt 0;
    background-color: transparent;
}}
h3 {{
    font-family: 'TH SarabunPSK', 'TH Sarabun New', sans-serif;
    font-size: 14pt;
    font-weight: bold;
    font-style: normal;
    text-align: left;
    margin: 6pt 0 2pt 0;
    background-color: transparent;
}}
h4 {{
    font-family: 'TH SarabunPSK', 'TH Sarabun New', sans-serif;
    font-size: 14pt;
    font-weight: bold;
    font-style: normal;
    text-align: left;
    text-indent: 14pt;
    margin: 5pt 0 2pt 0;
    background-color: transparent;
}}
p {{
    margin: 0 0 4.0pt 0;
    text-indent: 18pt;
    line-height: 1.25;
    background-color: transparent;
    text-align: left;
}}
.sub-item {{
    margin: 0 0 3.5pt 0;
    text-indent: 24pt;
    line-height: 1.25;
    text-align: left;
}}
.table-container {{
    break-inside: avoid;
    page-break-inside: avoid;
    margin: 5pt 0 5pt 0;
    width: 100%;
}}
.table-caption {{
    break-inside: avoid;
    page-break-inside: avoid;
    break-after: avoid;
    page-break-after: avoid;
    font-size: 12pt;
    font-weight: bold;
    font-style: italic;
    text-align: left;
    margin: 0 0 2.5pt 0;
    line-height: 1.2;
}}
.table-note {{
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 10pt;
    text-align: left;
    margin-top: 2pt;
    line-height: 1.15;
    color: #000000;
}}
table.paper-table {{
    break-inside: avoid;
    page-break-inside: avoid;
    width: 240pt;
    margin: 2pt 0;
    border-collapse: collapse;
    font-size: 9.2pt;
    line-height: 1.15;
    text-align: center;
    table-layout: fixed;
}}
table.dense-table {{
    font-size: 8.2pt;
    line-height: 1.10;
}}
table.paper-table tr {{
    break-inside: avoid;
    page-break-inside: avoid;
}}
table.paper-table th {{
    border: 0.5pt solid #000000;
    font-weight: bold;
    padding: 1.5pt 0.4pt;
    background-color: transparent;
    color: #000000;
    vertical-align: middle;
}}
table.paper-table td {{
    border: 0.5pt solid #000000;
    padding: 1.5pt 0.4pt;
    color: #000000;
    vertical-align: middle;
}}
img {{
    object-fit: contain;
    width: 212pt;
    max-width: 212pt;
    height: auto;
    display: block;
    margin: 0 auto;
}}
.fig-container {{
    break-inside: avoid;
    page-break-inside: avoid;
    text-align: center;
    margin: 2pt 0 3pt 0;
    width: 100%;
}}
.fig-caption {{
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 11.5pt;
    font-style: italic;
    text-align: center;
    margin-top: 2pt;
    margin-bottom: 2pt;
    line-height: 1.18;
}}
.equation {{
    text-align: center;
    font-style: italic;
    margin: 4pt 0;
    font-size: 11.5pt;
    break-inside: avoid;
    page-break-inside: avoid;
}}
.ref-item {{
    font-size: 10pt;
    line-height: 1.15;
    text-align: left;
    text-indent: -16pt;
    margin: 0 0 1.0pt 16pt;
    break-inside: avoid;
    page-break-inside: avoid;
}}
</style>

<h2>1. บทนำ</h2>
<h3>1.1 ที่มาและความสำคัญ</h3>
<p>กลุ่มโรคไม่ติดต่อเรื้อรัง (Non-Communicable Diseases: NCDs) ซึ่งครอบคลุมถึงโรคระบบหัวใจและหลอดเลือด โรคมะเร็ง โรคทางเดินหายใจเรื้อรัง และโรคเบาหวาน ถือเป็นวิกฤตทางระบาดวิทยาที่สำคัญที่สุดในยุคปัจจุบัน ข้อมูลขององค์การอนามัยโลก (World Health Organization: WHO, 2022) ระบุว่ากลุ่มโรคดังกล่าวเป็นสาเหตุของการเสียชีวิตมากกว่า 41 ล้านคนต่อปี หรือคิดเป็นร้อยละ 74 ของการเสียชีวิตทั้งหมดทั่วโลก [1] ในบรรดาโรคเหล่านี้ โรคเบาหวาน (Diabetes Mellitus) และโรคความดันโลหิตสูง (Hypertension) จัดเป็นปัจจัยขับเคลื่อนทางเมแทบอลิซึมหลักที่สามารถปรับเปลี่ยนพฤติกรรมเพื่อป้องกันการเจ็บป่วยก่อนวัยอันควรได้ รายงานของสหพันธ์โรคเบาหวานนานาชาติ (International Diabetes Federation: IDF, 2021) ชี้ให้เห็นว่าในประเทศที่มีรายได้ปานกลางและประเทศอุตสาหกรรมใหม่ ร้อยละ 30 ถึง 45 ของผู้ป่วยโรคเบาหวานไม่ได้รับการวินิจฉัยจนกระทั่งเกิดภาวะแทรกซ้อนทางคลินิกระยะทุติยภูมิ เช่น โรคไตจากเบาหวาน โรคจอประสาทตาเสื่อม โรคหลอดเลือดสมอง และโรคหลอดเลือดหัวใจ [2]</p>

<p>ในปัจจุบันแนวทางการวินิจฉัยที่เป็นมาตรฐานอ้างอิงสูงสุด (Gold Standard) จำเป็นต้องอาศัยการเจาะเลือดจากหลอดเลือดดำเพื่อตรวจระดับน้ำตาลในพลาสมาหลังอดอาหาร (≥ 126 mg/dL) หรือค่าน้ำตาลสะสม HbA1c (≥ 6.5%) ร่วมกับการวัดความดันโลหิตด้วยเครื่องวัดทางการแพทย์ แม้ว่าวิธีเหล่านี้จะมีความแม่นยำทางชีวเคมีสูง แต่ยังมีข้อจำกัดเชิงโครงสร้างที่ต้องใช้วัสดุทางการแพทย์ บุคลากรที่มีใบอนุญาต และห้องปฏิบัติการส่วนกลาง ในขณะเดียวกัน ความก้าวหน้าของปัญญาประดิษฐ์ (AI) และการเรียนรู้ของเครื่อง (Machine Learning: ML) โดยเฉพาะแบบจำลอง Vision Transformer (ViT) ร่วมกับอัลกอริทึมเกรเดียนต์บูสต์ทรีส์ (Gradient-Boosted Trees) ทำให้สามารถวิเคราะห์ลักษณะสัณฐานวิทยาของเนื้อเยื่ออ่อนบนใบหน้าที่สะท้อนถึงการสะสมไขมันส่วนกลางและความเสี่ยงทางเมแทบอลิซึมได้โดยไม่ต้องเจาะเลือด</p>

<p>การศึกษานี้จึงประยุกต์ใช้เทคนิคคอมพิวเตอร์วิทัศน์ร่วมกับการเรียนรู้ของเครื่องบนตารางข้อมูล เพื่อพัฒนากรอบการทำงานแบบแยกส่วนภายใต้ชื่อ Face2Health สำหรับคัดกรองความเสี่ยงโรคเบาหวาน และโรคความดันโลหิตสูงจากภาพถ่ายใบหน้าสองมิติและข้อมูลพื้นฐานส่วนบุคคล โดยมุ่งแก้ไขความท้าทายเชิงอัลกอริทึม 3 ประการ ได้แก่ การแพร่กระจายของความคลาดเคลื่อนระหว่างขั้นตอน (Multi-Stage Error Propagation) ความไม่สอดคล้องทางชีววิทยาของต้นไม้ตัดสินใจ (Non-Monotonic Risk Surfaces) และปัญหาความไม่สมดุลของคลาส (Class Imbalance) บนฐานข้อมูลสุขภาพแห่งชาติ CDC NHANES</p>

<h3>1.2 วัตถุประสงค์ของการศึกษา</h3>
<p class="sub-item">1.2.1 เพื่อศึกษาและพัฒนาไปป์ไลน์การประมาณค่าดัชนีมวลกาย (BMI) และเปอร์เซ็นต์ไขมันในร่างกาย (DEXA Body Fat) จากภาพถ่ายใบหน้าและข้อมูลพื้นฐานส่วนบุคคล</p>
<p class="sub-item">1.2.2 เพื่อพัฒนาและประเมินประสิทธิภาพแบบจำลองการเรียนรู้ของเครื่องแบบกำหนดเงื่อนไขโมโนโทนิกสำหรับการคัดกรองความเสี่ยงโรคเบาหวาน และโรคความดันโลหิตสูงบนชุดข้อมูล CDC NHANES</p>

<h3>1.3 ประโยชน์ที่คาดว่าจะได้รับ</h3>
<p class="sub-item">1.3.1 ได้แนวทางการเชื่อมโยงแบบจำลองคอมพิวเตอร์วิทัศน์เข้ากับแบบจำลองตารางข้อมูลที่สามารถลดทอนความคลาดเคลื่อนและขจัดความย้อนแย้งทางคลินิก</p>
<p class="sub-item">1.3.2 ได้ระบบสนับสนุนการตัดสินใจทางคลินิกแบบไม่รุกล้ำ (Zero-Hardware Fallback) สำหรับช่วยคัดกรองและจัดลำดับความเร่งด่วนของผู้ป่วยเบื้องต้นในพื้นที่ห่างไกล</p>

<h2>2. ทฤษฎีและงานวิจัยที่เกี่ยวข้อง</h2>
<h3>2.1 สัณฐานวิทยาใบหน้าและภาวะไขมันสะสม (Facial Morphometry)</h3>
<p>สัณฐานวิทยาของใบหน้าและกะโหลกศีรษะทำหน้าที่เป็นตัวบ่งชี้ภายนอกของภาวะไขมันสะสมในร่างกาย เนื่องจากการกระจายตัวของชั้นไขมันใต้ผิวหนังบริเวณแก้มและแนวกราม Coetzee และคณะ (2009) ได้แสดงให้เห็นว่าความหนาแน่นของไขมันบนใบหน้าที่รับรู้ได้เป็นสัญญาณบ่งชี้ทางสายตาที่สัมพันธ์กับดัชนีมวลกายและสุขภาพของระบบหัวใจและหลอดเลือดอย่างมีนัยสำคัญ [3] ต่อมา Wen และ Guo (2013) ได้พิสูจน์ว่าคุณลักษณะทางเรขาคณิตของใบหน้าที่สกัดด้วยคอมพิวเตอร์สามารถใช้ทำนายค่าดัชนีมวลกายได้อย่างแม่นยำ [4] ในทางกายวิภาค การสะสมของไขมันสามารถอธิบายผ่านอัตราส่วนเรขาคณิตที่ไม่ขึ้นกับขนาดภาพ ได้แก่ อัตราส่วนความกว้างต่อความสูงของใบหน้าส่วนล่าง (LFWR) อัตราส่วนความกว้างของแก้มต่อขากรรไกร (CJWR) อัตราส่วนความกว้างต่อความสูงของใบหน้า (FWHR) และสัดส่วนเส้นรอบรูปต่อพื้นที่ของขากรรไกรล่างที่ปรับขนาดแล้ว (PAR) ซึ่งสอดคล้องกับการศึกษาของ Lee และ Kim (2014) ที่พบว่าระยะห่างและขนาดความกว้างของขากรรไกรล่างเป็นตัวทำนายสำคัญของภาวะอ้วนลงพุงและไขมันในช่องท้อง [5]</p>

<h3>2.2 การประมาณค่าดัชนีมวลกายจากภาพใบหน้าด้วย Deep Learning</h3>
<p>การประมาณค่าตัวชี้วัดสุขภาพจากใบหน้าในยุคแรกอาศัยโครงข่ายประสาทเทียมคอนโวลูชันแบบลึก (Deep Convolutional Neural Networks) เช่น สถาปัตยกรรม ResNet ที่พัฒนาโดย He และคณะ (2016) [6] ต่อมา Kocabey และคณะ (2017) ได้นำเสนอระบบ Face-to-BMI โดยใช้ภาพถ่ายจากสื่อสังคมออนไลน์และสร้างเกณฑ์มาตรฐานการทำนายค่า BMI ด้วยความคลาดเคลื่อนเฉลี่ย (MAE) ประมาณ 2.8 kg/m² [7] ขณะที่ Stephen และคณะ (2017) ได้ยืนยันความถูกต้องของลักษณะโครงสร้างใบหน้าในการบ่งชี้สุขภาพทางสรีรวิทยาข้ามกลุ่มประชากรคอเคเชียน แอฟริกัน และเอเชีย [8] นอกจากนี้ Jiang, Guo และ Mu (2020) ได้พัฒนาการประมาณค่า BMI จากภาพใบหน้าด้วยเทคนิคการเรียนรู้การแจกแจงป้ายกำกับ (Label Distribution Learning) เพื่อลดความผันผวนของป้ายกำกับ [9]</p>

<h3>2.3 การตรวจวัดองค์ประกอบร่างกาย DEXA และฐานข้อมูล CDC NHANES</h3>
<p>แม้ว่าค่าดัชนีมวลกาย (BMI = น้ำหนัก/ส่วนสูง²) จะถูกใช้อย่างแพร่หลาย แต่มีข้อจำกัดในการแยกแยะมวลกล้ามเนื้อออกจากเนื้อเยื่อไขมัน การตรวจวัดความหนาแน่นกระดูกและไขมันด้วยรังสีเอกซ์คู่ (Dual-Energy X-ray Absorptiometry: DEXA) จึงเป็นมาตรฐานอ้างอิงสูงสุดในการวัดองค์ประกอบร่างกาย การสำรวจสุขภาพและโภชนาการแห่งชาติ (NHANES) โดย Centers for Disease Control and Prevention (CDC, 2018) ได้รวบรวมผลการสแกน DEXA ทั้งตัว (เปอร์เซ็นต์ไขมันในร่างกายรวม DXDTOPF) การวัดสัดส่วนร่างกายมาตรฐาน (BMX) การตรวจชีวเคมีในห้องปฏิบัติการ (HbA1c: LBXGH) และแบบสัมภาษณ์การวินิจฉัยโรคโดยแพทย์ (DIQ010 สำหรับโรคเบาหวาน และ BPQ020 สำหรับโรคความดันโลหิตสูง) ในกลุ่มประชากรผู้ใหญ่อายุ 20 ปีขึ้นไป [10]</p>

<h3>2.4 Vision Transformer (ViT) และ Monte Carlo Dropout</h3>
<p>Dosovitskiy และคณะ (2021) ได้นำเสนอสถาปัตยกรรม Vision Transformer (ViT) ที่ใช้กลไก Self-Attention ในการเรียนรู้ความสัมพันธ์เชิงพื้นที่ระยะไกลข้ามแพตช์ภาพได้ดีกว่าโครงข่ายคอนโวลูชันแบบดั้งเดิม [11] อย่างไรก็ตาม โครงข่ายประสาทเทียมแบบกำหนดแน่นอนมักให้ผลการทำนายที่ผิดพลาดเมื่อพบภาพนอกการแจกแจง (Out-of-Distribution) Gal และ Ghahramani (2016) ได้พิสูจน์ว่าการใช้เทคนิค Monte Carlo Dropout (MC Dropout) โดยเปิดใช้งานชั้น Dropout ระหว่างการอนุมานและสุ่มส่งผ่านข้อมูลไปข้างหน้าจำนวน T รอบ สามารถประมาณค่าเฉลี่ยการทำนาย μ(x*) และความไม่แน่นอนของแบบจำลอง (Epistemic Uncertainty) σ²(x*) ได้ดังสมการ [12]:</p>
<div class="equation">μ(x*) = (1/T) Σ<sub>t=1</sub><sup>T</sup> f<sub>W<sub>t</sub></sub>(x*), &nbsp;&nbsp;&nbsp;&nbsp; σ²(x*) = [1/(T-1)] Σ<sub>t=1</sub><sup>T</sup> [f<sub>W<sub>t</sub></sub>(x*) - μ(x*)]²</div>

<h3>2.5 Extreme Gradient Boosting (XGBoost) แบบกำหนดเงื่อนไขโมโนโทนิก</h3>
<p>Chen และ Guestrin (2016) ได้พัฒนาอัลกอริทึม Extreme Gradient Boosting (XGBoost) ซึ่งเป็นเทคนิค Ensemble Learning แบบเร่งเกรเดียนต์ที่มีประสิทธิภาพสูงบนข้อมูลตาราง [13], [14] อย่างไรก็ตาม การสร้างต้นไม้ตัดสินใจแบบไร้เงื่อนไขบนตัวแปรที่มีความสัมพันธ์กันสูงอาจสร้างพื้นผิวการตัดสินใจที่ขัดแย้งกับหลักสรีรวิทยา เช่น ค่า BMI เพิ่มขึ้นแต่ความเสี่ยงโรคลดลง ดังนั้น XGBoost จึงอนุญาตให้กำหนดเงื่อนไขความชันแบบโมโนโทนิก (Monotonic Constraints: c<sub>j</sub> = +1) เพื่อบังคับให้ความน่าจะเป็นของโรคเพิ่มขึ้นตามอายุ ค่า BMI และเปอร์เซ็นต์ไขมันในร่างกายอย่างสอดคล้องกับหลักการแพทย์</p>

<h3>2.6 การปรับเทียบความน่าจะเป็น (Platt Scaling) และการหาจุดตัดด้วย F2-Score</h3>
<p>คะแนนผลลัพธ์จากต้นไม้ตัดสินใจมักมีการกระจายตัวบิดเบี้ยวไม่ตรงกับความน่าจะเป็นจริง Platt (2000) ได้นำเสนอวิธี Platt Scaling โดยปรับฟังก์ชันซิกมอยด์ P(Y=1|z) = 1 / [1 + exp(A·z + B)] บนคะแนนของแบบจำลองผ่านการตรวจสอบไขว้ [15] ซึ่งสอดคล้องกับการศึกษาของ Niculescu-Mizil และ Caruana (2005) ที่ยืนยันว่า Platt Scaling ช่วยลดคะแนนความคลาดเคลื่อน Brier Score ของต้นไม้แบบบูสต์ได้อย่างมีนัยสำคัญ [16] นอกจากนี้ ในปัญหาที่มีความไม่สมดุลของคลาสสูง (ความชุกโรคเบาหวานเพียงร้อยละ 10.32) การใช้จุดตัดมาตรฐานที่ 0.50 จะทำให้ไม่สามารถตรวจจับผู้ป่วยได้ จึงต้องปรับจุดตัดการตัดสินใจ (τ) โดยอาศัยค่าสถิติ Youden's J ของ Youden (1950) [17] และหลักการเรียนรู้แบบคำนึงถึงต้นทุน (Cost-Sensitive Learning) ของ Elkan (2001) [18] ผ่านมาตรวัด F2-Score ที่ให้น้ำหนักแก่ค่า Recall เป็นสองเท่าของค่า Precision</p>

<h3>2.7 จุดต่างของการศึกษานี้จากงานวิจัยที่ผ่านมา</h3>
<p>งานวิจัยด้านคอมพิวเตอร์วิทัศน์ที่ผ่านมาส่วนใหญ่ยุติกระบวนการทำงานเพียงแค่การประมาณค่า BMI จากภาพใบหน้า [4], [7], [9] โดยไม่ได้คัดกรองความไม่แน่นอนของแบบจำลองและไม่ได้เชื่อมโยงไปสู่การจำแนกโรคเรื้อรัง การศึกษานี้จึงต่อยอดด้วยการบูรณาการการคัดกรองความไม่แน่นอนของภาพด้วย MC Dropout เข้ากับการประมาณค่าเปอร์เซ็นต์ไขมันร่างกายมาตรฐาน DEXA และตัวจำแนก XGBoost แบบบังคับเงื่อนไขโมโนโทนิกที่ปรับเทียบความน่าจะเป็นแล้ว เพื่อจัดลำดับความเสี่ยงทางคลินิก 4 ระดับอย่างครบวงจร</p>

<h2>3. วิธีการดำเนินงานวิจัย</h2>
<h3>3.1 ภาพรวมกระบวนการดำเนินงาน (Process overview)</h3>
<p>กรอบการทำงานของระบบ Face2Health แบ่งออกเป็น 3 ขั้นตอนหลักที่เชื่อมต่อกันตามลำดับ (Stage 1, Stage 1.5 และ Stage 2) โดยพัฒนาด้วยภาษา Python ร่วมกับไลบรารี PyTorch, Scikit-learn [19] และ XGBoost ขั้นตอนการดำเนินงานทั้งหมดแสดงดังรูปที่ 1 และมีรายละเอียดในแต่ละขั้นตอนย่อยดังนี้</p>

<!-- FIGURE 1 -->
<div class="fig-container" id="fig_1">
  <img src="fig0_pipeline_architecture.png" style="object-fit: contain; width: 210pt; max-width: 210pt; height: 299pt;" alt="แผนภาพภาพรวมของกระบวนการดำเนินงาน Face2Health">
  <div class="fig-caption">รูปที่ 1 แผนภาพภาพรวมของกระบวนการดำเนินงาน Face2Health</div>
</div>

<h4>3.1.1 การเตรียมข้อมูลและการคัดกรองกลุ่มประชากร (Data Preparation)</h4>
<p>ข้อมูลที่ใช้ในการศึกษานี้ประกอบด้วย 2 ส่วนหลัก ส่วนแรกคือชุดข้อมูลภาพถ่ายใบหน้าจากฐานข้อมูลมาตรฐาน VisualBMI ของ Kocabey และคณะ (2017) [7] จำนวน 3,962 ภาพ (ภาพทั้งหมดมาจากชุดข้อมูลนี้ รวมถึงชุดย่อยภาพหน้าตรง 412 ภาพ) สำหรับฝึกฝนแบบจำลองประมาณค่า BMI ใน Stage 1 ส่วนที่สองคือฐานข้อมูลสุขภาพและโภชนาการแห่งชาติ CDC NHANES รอบปี 2003–2004 (Cycle C) [10] ซึ่งมีผู้เข้าร่วมสัมภาษณ์เริ่มต้น 10,122 ราย โดย 6,150 รายมีผลสแกน DEXA ทั้งตัว (DXDTOPF โดยคำนวณค่าเฉลี่ยจากการประมาณค่า 5 ชุดข้อมูลย่อยต่อบุคคลเพื่อป้องกันแถวซ้ำและการรั่วไหลระหว่างชุดข้อมูล) และการวัดสัดส่วนร่างกายรวมถึงรอบเอวครบถ้วน และเมื่อคัดเฉพาะผู้ใหญ่อายุ 20 ปีขึ้นไปได้ 3,588 ราย ส่งผลให้ได้กลุ่มประชากรสำหรับโรคเบาหวาน (DIQ010, หลังตัดผู้ตอบก้ำกึ่ง ปฏิเสธตอบ หรือไม่มีข้อมูล 52 ราย) รวมทั้งสิ้น 3,536 ราย (ผู้ป่วยจริง 365 ราย คิดเป็นความชุกร้อยละ 10.32) และกลุ่มประชากรสำหรับโรคความดันโลหิตสูง (BPQ020) รวมทั้งสิ้น 3,550 ราย (ผู้ป่วยจริง 1,208 ราย คิดเป็นความชุกร้อยละ 34.03) รายละเอียดการคัดกรองแสดงดังตารางที่ 1 ทั้งนี้ แม้ในรอบ 2003–2004 จะมีข้อมูลผลตรวจทางห้องปฏิบัติการ (HbA1c, น้ำตาลในเลือด และความดันที่วัดจริง) แต่งานวิจัยนี้มุ่งเน้นการคัดกรองเบื้องต้นจากประวัติการวินิจฉัยของแพทย์ (DIQ010 และ BPQ020) เพื่อจำลองการใช้งานในชุมชนที่ปราศจากอุปกรณ์ตรวจวัด ส่วนข้อมูล NHANES รอบ 2017–2018 (Cycle J) ซึ่งเป็นผู้เข้าร่วมคนละกลุ่ม ใช้เฉพาะการตรวจสอบภายนอกเชิงรองกับเกณฑ์ผลตรวจทางห้องปฏิบัติการ (HbA1c และความดันโลหิตที่วัดจริง) และไม่ได้ใช้ฝึกแบบจำลอง</p>

<!-- TABLE 1 -->
<div class="table-container" id="tbl_1">
  <div class="table-caption">ตารางที่ 1 แผนผังการคัดกรองและการแบ่งกลุ่มประชากรผู้ใหญ่ในฐานข้อมูล NHANES</div>
  <table class="paper-table dense-table" data-w="28.0,34.0,12.0,12.0,14.0" data-nw="2,3,4">
    <tr>
      <th>ขั้นตอนการคัดกรอง</th>
      <th>เกณฑ์พิจารณา</th>
      <th>คงเหลือ<br>(N)</th>
      <th>ผู้ป่วยจริง</th>
      <th>ความชุก</th>
    </tr>
    <tr><td>ประชากร NHANES เริ่มต้น (Cycle C)</td><td>ผู้เข้าร่วมสัมภาษณ์ปี 2003–2004</td><td>10,122</td><td>—</td><td>—</td></tr>
    <tr><td>ผลสแกน DEXA ทั้งตัว</td><td>DXDTOPF ไขมัน Android/Gynoid และรอบเอวครบถ้วน</td><td>6,150</td><td>—</td><td>—</td></tr>
    <tr><td>กรองตามช่วงอายุผู้ใหญ่</td><td>ผู้ใหญ่อายุ 20 ปีขึ้นไป</td><td>3,588</td><td>—</td><td>—</td></tr>
    <tr><td><b>กลุ่มโรคเบาหวาน (DIQ010)</b></td><td>ตอบชัดเจน (ตัด DIQ010 = 3, 7, 9, ว่าง)</td><td><b>3,536</b></td><td><b>365</b></td><td><b>10.32%</b></td></tr>
    <tr><td>— Training set (70%)</td><td>ชุดฝึกฝนแบบ Out-of-Fold</td><td>2,474</td><td>255</td><td>10.31%</td></tr>
    <tr><td>— Validation set (15%)</td><td>ชุดตรวจสอบหาจุดตัด (τ)</td><td>531</td><td>55</td><td>10.36%</td></tr>
    <tr><td>— Test set (15%)</td><td>ชุดทดสอบประเมินผลขั้นสุดท้าย</td><td>531</td><td>55</td><td>10.36%</td></tr>
    <tr><td><b>กลุ่มความดันโลหิตสูง (BPQ020)</b></td><td>ตอบชัดเจน (ตัด BPQ020 = 7, 9, ว่าง)</td><td><b>3,550</b></td><td><b>1,208</b></td><td><b>34.03%</b></td></tr>
    <tr><td>— Training set (70%)</td><td>ชุดฝึกฝนแบบ Out-of-Fold</td><td>2,484</td><td>846</td><td>34.06%</td></tr>
    <tr><td>— Validation set (15%)</td><td>ชุดตรวจสอบหาจุดตัด (τ)</td><td>533</td><td>181</td><td>33.96%</td></tr>
    <tr><td>— Test set (15%)</td><td>ชุดทดสอบประเมินผลขั้นสุดท้าย</td><td>533</td><td>181</td><td>33.96%</td></tr>
  </table>
  <div class="table-note">หมายเหตุ: ข้อมูล CDC NHANES รอบปี 2003–2004 (Cycle C) ในกลุ่มผู้ใหญ่อายุ 20 ปีขึ้นไป ส่วนรอบ 2017–2018 ใช้เฉพาะการตรวจสอบภายนอกเชิงรอง</div>
</div>

<h4>3.1.2 การแบ่งชุดข้อมูลและการป้องกันข้อมูลรั่วไหล (Data Splitting &amp; Out-of-Fold Protocol)</h4>
<p>ในส่วนของชุดข้อมูลภาพใบหน้า VisualBMI (N = 3,962) แบ่งออกเป็นชุดฝึกฝน (Training set) ร้อยละ 70 (2,773 ภาพ พร้อมทำ Data Augmentation 5 เท่าเป็น 13,865 ภาพ) ชุดตรวจสอบ (Validation set) ร้อยละ 10 (396 ภาพ) และชุดทดสอบ (Test set) ร้อยละ 20 (793 ภาพ) โดยแยกตามรหัสผู้ใช้เพื่อป้องกันข้อมูลรั่วไหล ส่วนข้อมูลตาราง NHANES ในแต่ละโรคถูกแบ่งด้วยวิธี Stratified Sampling เพื่อรักษาสัดส่วนความชุกของโรคออกเป็น Training set ร้อยละ 70, Validation set ร้อยละ 15 และ Test set ร้อยละ 15 โดยใช้การทำนายแบบ 5-Fold Out-of-Fold บนชุดฝึกฝนเพื่อส่งค่าทำนายจาก Stage 1.5 ไปยัง Stage 2 โดยไม่มีการซ้อนทับของรหัสผู้เข้าร่วม (SEQN) ระหว่างชุดฝึกฝนและชุดทดสอบ</p>

<h4>3.1.3 การสกัดคุณลักษณะใบหน้าและประมาณค่า BMI (Stage 1: ViT-H/14 &amp; MC Dropout)</h4>
<p>ภาพถ่ายใบหน้าจะถูกตรวจสอบคุณภาพผ่านโมดูล MediaPipe Face Guard เพื่อคัดกรองภาพที่ใบหน้าเอียงเกิน ±15° ออก จากนั้นส่งเข้าสู่โครงข่ายหลัก ViT-H/14 (Pre-trained บน ImageNet-21k) ที่ตรึงน้ำหนักไว้ ร่วมกับหัวทำนาย MLP 5 ชั้น (1280 → 640 → 320 → 160 → 80 → 1, GELU, Dropout 0.5) ฝึกฝนด้วยตัวปรับค่าเหมาะที่สุด AdamW (Learning Rate = 3×10<sup>-5</sup>, Weight Decay = 0.01, Batch Size = 24) ในขั้นตอนการอนุมานจะเปิดใช้งาน Monte Carlo Dropout สุ่มคำนวณ T = 25 รอบ หากค่าส่วนเบี่ยงเบนมาตรฐานของความไม่แน่นอนของแบบจำลอง (Epistemic Uncertainty) มีค่า σ &gt; 1.80 kg/m² ระบบจะปฏิเสธภาพนั้นและแนะนำให้ถ่ายใหม่</p>

<h4>3.1.4 การประมาณค่าเปอร์เซ็นต์ไขมันในร่างกาย (Stage 1.5: Tabular DEXA Regression)</h4>
<p>ค่าประมาณ BMI ที่ได้จาก Stage 1 จะถูกนำมารวมกับข้อมูลส่วนบุคคล 3 ตัวแปร ได้แก่ อายุ เพศ (ชาย = 1, หญิง = 0) และระดับกิจกรรมทางกาย 3 ระดับ (0 = เนือยนิ่ง, 1 = ปานกลาง จากตัวแปร PAD320, 2 = หนัก จากตัวแปร PAD200 ของแบบสอบถาม PAQ_C โดยตัดรหัสปฏิเสธตอบและไม่ทราบ [7, 9]) เพื่อป้อนเข้าสู่แบบจำลอง XGBoost Regressor สำหรับทำนายค่าเปอร์เซ็นต์ไขมันในร่างกายรวม (DEXA Total Body Fat) โดยปรับแต่งไฮเปอร์พารามิเตอร์ด้วยวิธี Randomized Search Cross-Validation ตามแนวทางของ Bergstra และ Bengio (2012) [20] ทั้งนี้ได้พัฒนาแบบจำลองเสริมที่รวมตัวแปรเส้นรอบเอว (Waist Circumference) เพื่อใช้เปรียบเทียบผลการดูดซับความแปรปรวน</p>

<h4>3.1.5 การสร้างแบบจำลองจำแนกความเสี่ยงและการปรับเทียบความน่าจะเป็น (Stage 2: Monotonic Calibrated XGBoost)</h4>
<p>ตัวแปรนำเข้าประกอบด้วย [อายุ, เพศ, BMI ที่ทำนาย, เปอร์เซ็นต์ไขมันที่ทำนาย, ระดับกิจกรรมทางกาย] จะถูกส่งเข้าสู่ตัวจำแนก XGBoost Classifier ที่ควบคุมความซับซ้อน (n_estimators = 120, max_depth = 3, learning_rate = 0.03, subsample = 0.80, colsample_bytree = 0.75, reg_alpha = 0.5, reg_lambda = 2.0) พร้อมบังคับเงื่อนไขความชันแบบโมโนโทนิก (+1) บนตัวแปรอายุ ค่า BMI เปอร์เซ็นต์ไขมัน และรอบเอว จากนั้นปรับเทียบความน่าจะเป็นด้วยฟังก์ชัน CalibratedClassifierCV(method='sigmoid', cv=5) ของ Scikit-learn [19] และค้นหาจุดตัด (τ) ที่เหมาะสมที่สุดบนชุดตรวจสอบ (Validation set) ด้วยเกณฑ์ F2-Score</p>

<h4>3.1.6 การประเมินผลแบบจำลองและการวิเคราะห์ความทนทานต่อสัญญาณรบกวน (Model Evaluation)</h4>
<p>การประเมินผลบนชุดทดสอบ (Test set) ใช้ตัวชี้วัดพื้นที่ใต้กราฟ ROC (AUC) ความไวหรือ Recall (Sensitivity) ความจำเพาะ (Specificity) ความแม่นยำ (Precision) คะแนน Brier Score และตาราง Confusion Matrix พร้อมทั้งจำลองการใส่สัญญาณรบกวนแบบเกาส์เซียน (Gaussian Noise) จำนวน 100 รอบ ที่ระดับความคลาดเคลื่อน σ<sub>sim</sub> ∈ {{2.14, 2.82, 4.40, 5.60}} kg/m² เพื่อทดสอบความทนทานต่อสภาพภาพถ่ายจริง</p>

<h2>4. ผลการดำเนินงาน</h2>
<h3>4.1 ผลการประมาณค่า BMI (Stage 1) และเปอร์เซ็นต์ไขมันร่างกาย DEXA (Stage 1.5)</h3>
<p>ผลการประเมินประสิทธิภาพของแบบจำลอง Stage 1 บนชุดทดสอบภาพใบหน้ามุมตรง (n = 412) ได้ค่า MAE เท่ากับ 2.14 kg/m² (RMSE = 2.82 kg/m², R² = 0.68) และบนชุดทดสอบรวมทุกสภาพแสง (N = 793) การคัดกรองความไม่แน่นอนของแบบจำลองที่เกณฑ์ σ &gt; 1.80 kg/m² (ปฏิเสธภาพที่มีความไม่แน่นอนสูงร้อยละ 4.2 ซึ่งมีค่า MAE สูงถึง 7.43 kg/m²) ช่วยลดค่า MAE ของชุดทดสอบรวมลงจาก 3.26 เหลือ 3.08 kg/m² สำหรับแบบจำลอง Stage 1.5 DEXA Regressor บนชุดทดสอบ (N = 699) พบว่าแบบจำลองหลักที่ไม่ใช้เส้นรอบเอวให้ค่า MAE เท่ากับ 2.91 เปอร์เซ็นต์พอยต์ (pp) และ R² = 0.83 ขณะที่แบบจำลองเสริมที่ใช้เส้นรอบเอวให้ค่า MAE ลดลงเป็น 2.70 pp และ R² = 0.85 ดังสรุปในตารางที่ 2 และแสดงแผนภาพการกระจายในรูปที่ 2</p>

<!-- TABLE 2 -->
<div class="table-container" id="tbl_2">
  <div class="table-caption">ตารางที่ 2 ผลการประเมินประสิทธิภาพการถดถอยของแบบจำลอง Stage 1 และ Stage 1.5 บนชุดทดสอบ (Test set)</div>
  <table class="paper-table dense-table" data-w="24.0,15.0,22.0,10.5,10.5,7.0,11.0" data-nw="3,4,5,6">
    <tr>
      <th>แบบจำลอง</th>
      <th>เป้าหมาย</th>
      <th>ตัวแปรนำเข้า</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>R²</th>
      <th>อัตรา<br>ปฏิเสธ</th>
    </tr>
    <tr>
      <td>Stage 1 ViT-H/14<br>+ MC Dropout</td>
      <td>BMI<br>(kg/m²)</td>
      <td>ภาพถ่ายใบหน้า<br>2 มิติ</td>
      <td>2.14</td>
      <td>2.82</td>
      <td>0.68</td>
      <td>4.2%</td>
    </tr>
    <tr>
      <td>Stage 1.5 DEXA<br>(แบบจำลองหลัก)</td>
      <td>ไขมันร่างกาย<br>(%)</td>
      <td>BMI, อายุ, เพศ,<br>กิจกรรมทางกาย</td>
      <td>2.91 pp</td>
      <td>3.64 pp</td>
      <td>0.83</td>
      <td>0.0%</td>
    </tr>
    <tr>
      <td>Stage 1.5 DEXA<br>(เสริมรอบเอว)</td>
      <td>ไขมันร่างกาย<br>(%)</td>
      <td>BMI, อายุ, เพศ,<br>กิจกรรม, รอบเอว</td>
      <td>2.70 pp</td>
      <td>3.43 pp</td>
      <td>0.85</td>
      <td>0.0%</td>
    </tr>
  </table>
  <div class="table-note">หมายเหตุ: Stage 1 รายงานค่า MAE/RMSE ในหน่วย kg/m² บนชุดควบคุมมุมตรง (n = 412) ส่วน Stage 1.5 รายงานในหน่วยเปอร์เซ็นต์พอยต์ (pp) บนชุดทดสอบ NHANES (ผู้ใหญ่ N = 699)</div>
</div>

<h3>4.2 ผลการจำแนกความเสี่ยงโรคเบาหวานและโรคความดันโลหิตสูง (Stage 2)</h3>
<p>ผลการประเมินประสิทธิภาพการจำแนกความเสี่ยงโรคเบาหวาน (DIQ010, N = 531) และโรคความดันโลหิตสูง (BPQ020, N = 533) บนชุดทดสอบที่ระดับความคลาดเคลื่อนอ้างอิง (σ<sub>sim</sub> = 2.14 kg/m²) สรุปได้ดังตารางที่ 3 และตารางที่ 4 พบว่าในโรคเบาหวาน หากใช้จุดตัดเริ่มต้นที่ τ = 0.500 แบบจำลองทั้งหมดจะมีค่า Recall (Sensitivity) เป็นร้อยละ 0.0 เนื่องจากความน่าจะเป็นหลังปรับเทียบสะท้อนความชุกจริงของโรคที่ต่ำ (ร้อยละ 10.32) แต่เมื่อปรับจุดตัดด้วยเกณฑ์ F2-Score ที่ τ = 0.060 แบบจำลองหลัก (Primary Monotonic XGBoost) ให้ค่า AUC สูงสุดที่ 0.762 และสามารถตรวจจับผู้ป่วยโรคเบาหวานได้ด้วยค่า Recall สูงถึงร้อยละ 90.9 (ตรวจพบ 50 จาก 55 ราย) ที่ความจำเพาะร้อยละ 48.7 ซึ่งชนะแบบจำลองพื้นฐาน Logistic Regression (AUC = 0.742, ΔAUC = +0.020, p = 0.015) อย่างมีนัยสำคัญทางสถิติ</p>

<!-- FIGURE 2 -->
<div class="fig-container" id="fig_2">
  <img src="fig1_stage15_scatter.png" style="object-fit: contain; width: 206pt; max-width: 206pt; height: 219pt;" alt="แผนภาพการกระจายระหว่างเปอร์เซ็นต์ไขมันในร่างกายที่ทำนายเทียบกับค่าตรวจวัด DEXA จริง">
  <div class="fig-caption">รูปที่ 2 แผนภาพการกระจายระหว่างเปอร์เซ็นต์ไขมันในร่างกายที่ทำนายเทียบกับค่าตรวจวัด DEXA จริงบนชุดทดสอบ (ผู้ใหญ่ N = 699)</div>
</div>

<!-- TABLE 3 -->
<div class="table-container" id="tbl_3">
  <div class="table-caption">ตารางที่ 3 ผลการเปรียบเทียบประสิทธิภาพการจำแนกความเสี่ยงโรคเบาหวาน (DIQ010) บนชุดทดสอบ (Test set)</div>
  <table class="paper-table dense-table" data-w="16.0,10.5,7.5,16.5,16.5,16.5,16.5" data-nw="1,2,3,4,5,6">
    <tr>
      <th>แบบจำลอง</th>
      <th>เกณฑ์</th>
      <th>จุดตัด</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Spec.</th>
      <th>Prec.</th>
    </tr>
    <tr><td>Logistic Reg.</td><td>เริ่มต้น</td><td>0.500</td><td>0.742<br><span style="white-space: nowrap;">(0.678–0.806)</span></td><td>0.0%<br><span style="white-space: nowrap;">(0.0–6.5%)</span></td><td>100.0%<br><span style="white-space: nowrap;">(99.2–100%)</span></td><td>0.0%</td></tr>
    <tr><td>Logistic Reg.</td><td>F2-Score</td><td>0.068</td><td>0.742<br><span style="white-space: nowrap;">(0.678–0.806)</span></td><td>89.1%<br><span style="white-space: nowrap;">(77.8–95.9%)</span></td><td>39.3%<br><span style="white-space: nowrap;">(34.9–43.8%)</span></td><td>14.5%<br><span style="white-space: nowrap;">(10.9–18.7%)</span></td></tr>
    <tr><td>Unconstrained XGB</td><td>เริ่มต้น</td><td>0.500</td><td>0.747<br><span style="white-space: nowrap;">(0.683–0.811)</span></td><td>0.0%<br><span style="white-space: nowrap;">(0.0–6.5%)</span></td><td>100.0%<br><span style="white-space: nowrap;">(99.2–100%)</span></td><td>0.0%</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>เริ่มต้น</td><td>0.500</td><td>0.762<br><span style="white-space: nowrap;">(0.701–0.823)</span></td><td>0.0%<br><span style="white-space: nowrap;">(0.0–6.5%)</span></td><td>100.0%<br><span style="white-space: nowrap;">(99.2–100%)</span></td><td>0.0%</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>Youden</td><td>0.082</td><td>0.762<br><span style="white-space: nowrap;">(0.701–0.823)</span></td><td>78.2%<br><span style="white-space: nowrap;">(65.0–88.2%)</span></td><td>59.0%<br><span style="white-space: nowrap;">(54.5–63.5%)</span></td><td>18.1%<br><span style="white-space: nowrap;">(13.4–23.6%)</span></td></tr>
    <tr><td><b>Primary (Ours)</b></td><td><b>F2-Score</b></td><td><b>0.060</b></td><td><b>0.762<br><span style="white-space: nowrap;">(0.701–0.823)</span></b></td><td><b>90.9%<br><span style="white-space: nowrap;">(80.0–96.7%)</span></b></td><td><b>48.7%<br><span style="white-space: nowrap;">(44.1–53.3%)</span></b></td><td><b>17.0%<br><span style="white-space: nowrap;">(13.0–21.8%)</span></b></td></tr>
    <tr><td>เสริมเส้นรอบเอว</td><td>เริ่มต้น</td><td>0.500</td><td>0.763<br><span style="white-space: nowrap;">(0.702–0.824)</span></td><td>0.0%<br><span style="white-space: nowrap;">(0.0–6.5%)</span></td><td>100.0%<br><span style="white-space: nowrap;">(99.2–100%)</span></td><td>0.0%</td></tr>
    <tr><td>เสริมเส้นรอบเอว</td><td>Youden</td><td>0.082</td><td>0.763<br><span style="white-space: nowrap;">(0.702–0.824)</span></td><td>78.2%<br><span style="white-space: nowrap;">(65.0–88.2%)</span></td><td>58.8%<br><span style="white-space: nowrap;">(54.3–63.3%)</span></td><td>18.0%<br><span style="white-space: nowrap;">(13.3–23.5%)</span></td></tr>
    <tr><td>เสริมเส้นรอบเอว</td><td>F2-Score</td><td>0.061</td><td>0.763<br><span style="white-space: nowrap;">(0.702–0.824)</span></td><td>89.1%<br><span style="white-space: nowrap;">(77.8–95.9%)</span></td><td>48.9%<br><span style="white-space: nowrap;">(44.4–53.5%)</span></td><td>16.8%<br><span style="white-space: nowrap;">(12.7–21.6%)</span></td></tr>
  </table>
  <div class="table-note">หมายเหตุ: ประเมินบน Test set (N = 531); ค่าในวงเล็บคือช่วงความเชื่อมั่น 95% CI ที่ σ<sub>sim</sub> = 2.14 kg/m²</div>
</div>

<p>สำหรับโรคความดันโลหิตสูง (BPQ020) ในตารางที่ 4 แบบจำลองหลักที่ไม่ใช้เส้นรอบเอวให้ค่า AUC เท่ากับ 0.757 และที่จุดตัด F2-Score (τ = 0.155) สามารถตรวจจับผู้ป่วยได้ด้วยค่า Recall ร้อยละ 90.6 (ตรวจพบ 164 จาก 181 ราย) ความจำเพาะร้อยละ 40.9 และความแม่นยำร้อยละ 44.1 ซึ่งสูงกว่า Logistic Regression (AUC = 0.740, ΔAUC = +0.017, p = 0.006) เส้นโค้ง ROC เปรียบเทียบประสิทธิภาพของทุกแบบจำลองแสดงดังรูปที่ 3 และผลการจำแนกแต่ละคลาสในรูปแบบ Confusion Matrix ณ จุดตัดทั้ง 3 เกณฑ์แสดงดังรูปที่ 5</p>

<!-- TABLE 4 -->
<div class="table-container" id="tbl_4">
  <div class="table-caption">ตารางที่ 4 ผลการเปรียบเทียบประสิทธิภาพการจำแนกความเสี่ยงโรคความดันโลหิตสูง (BPQ020) บนชุดทดสอบ (Test set)</div>
  <table class="paper-table dense-table" data-w="16.0,10.5,7.5,16.5,16.5,16.5,16.5" data-nw="1,2,3,4,5,6">
    <tr>
      <th>แบบจำลอง</th>
      <th>เกณฑ์</th>
      <th>จุดตัด</th>
      <th>AUC</th>
      <th>Recall</th>
      <th>Spec.</th>
      <th>Prec.</th>
    </tr>
    <tr><td>Logistic Reg.</td><td>เริ่มต้น</td><td>0.500</td><td>0.740<br><span style="white-space: nowrap;">(0.697–0.783)</span></td><td>48.1%<br><span style="white-space: nowrap;">(40.6–55.6%)</span></td><td>83.5%<br><span style="white-space: nowrap;">(79.2–87.2%)</span></td><td>60.0%<br><span style="white-space: nowrap;">(51.9–67.7%)</span></td></tr>
    <tr><td>Logistic Reg.</td><td>F2-Score</td><td>0.165</td><td>0.740<br><span style="white-space: nowrap;">(0.697–0.783)</span></td><td>89.5%<br><span style="white-space: nowrap;">(84.1–93.6%)</span></td><td>34.7%<br><span style="white-space: nowrap;">(29.7–39.9%)</span></td><td>41.3%<br><span style="white-space: nowrap;">(36.4–46.4%)</span></td></tr>
    <tr><td>Unconstrained XGB</td><td>เริ่มต้น</td><td>0.500</td><td>0.750<br><span style="white-space: nowrap;">(0.708–0.792)</span></td><td>49.2%<br><span style="white-space: nowrap;">(41.7–56.7%)</span></td><td>82.1%<br><span style="white-space: nowrap;">(77.7–85.9%)</span></td><td>58.6%<br><span style="white-space: nowrap;">(50.5–66.3%)</span></td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>เริ่มต้น</td><td>0.500</td><td>0.757<br><span style="white-space: nowrap;">(0.715–0.799)</span></td><td>50.3%<br><span style="white-space: nowrap;">(42.8–57.8%)</span></td><td>82.7%<br><span style="white-space: nowrap;">(78.3–86.5%)</span></td><td>59.9%<br><span style="white-space: nowrap;">(51.8–67.5%)</span></td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>Youden</td><td>0.339</td><td>0.757<br><span style="white-space: nowrap;">(0.715–0.799)</span></td><td>71.3%<br><span style="white-space: nowrap;">(64.1–77.7%)</span></td><td>67.0%<br><span style="white-space: nowrap;">(61.8–72.0%)</span></td><td>52.7%<br><span style="white-space: nowrap;">(46.2–59.0%)</span></td></tr>
    <tr><td><b>Primary (Ours)</b></td><td><b>F2-Score</b></td><td><b>0.155</b></td><td><b>0.757<br><span style="white-space: nowrap;">(0.715–0.799)</span></b></td><td><b>90.6%<br><span style="white-space: nowrap;">(85.3–94.4%)</span></b></td><td><b>40.9%<br><span style="white-space: nowrap;">(35.7–46.3%)</span></b></td><td><b>44.1%<br><span style="white-space: nowrap;">(39.0–49.3%)</span></b></td></tr>
    <tr><td>เสริมเส้นรอบเอว</td><td>เริ่มต้น</td><td>0.500</td><td>0.755<br><span style="white-space: nowrap;">(0.713–0.797)</span></td><td>49.7%<br><span style="white-space: nowrap;">(42.2–57.2%)</span></td><td>82.4%<br><span style="white-space: nowrap;">(78.0–86.2%)</span></td><td>59.2%<br><span style="white-space: nowrap;">(51.1–66.9%)</span></td></tr>
    <tr><td>เสริมเส้นรอบเอว</td><td>F2-Score</td><td>0.158</td><td>0.755<br><span style="white-space: nowrap;">(0.713–0.797)</span></td><td>88.4%<br><span style="white-space: nowrap;">(82.8–92.6%)</span></td><td>41.8%<br><span style="white-space: nowrap;">(36.6–47.1%)</span></td><td>43.8%<br><span style="white-space: nowrap;">(38.7–49.1%)</span></td></tr>
  </table>
  <div class="table-note">หมายเหตุ: ประเมินบน Test set (N = 533); ค่าในวงเล็บคือช่วงความเชื่อมั่น 95% CI ที่ σ<sub>sim</sub> = 2.14 kg/m²</div>
</div>

<!-- FIGURE 3 -->
<div class="fig-container" id="fig_3">
  <img src="fig2_roc_curves.png" style="object-fit: contain; width: 195pt; max-width: 195pt; height: 220pt;" alt="เส้นโค้ง ROC สำหรับการจำแนกโรคเบาหวานและโรคความดันโลหิตสูง">
  <div class="fig-caption">รูปที่ 3 เส้นโค้ง ROC สำหรับการจำแนกโรคเบาหวาน (DIQ010) และโรคความดันโลหิตสูง (BPQ020) บนชุดทดสอบ</div>
</div>

<h3>4.3 ผลการปรับเทียบความน่าจะเป็นและการศึกษาการตัดทอนองค์ประกอบ (Ablation Study)</h3>
<p>จากการตรวจสอบความน่าเชื่อถือของค่าความน่าจะเป็นด้วยแผนภาพ Calibration Curve ในรูปที่ 4 พบว่าแบบจำลอง XGBoost ที่ไม่ได้ปรับเทียบมีความคลาดเคลื่อนแบบรูปตัว S อย่างชัดเจน เมื่อผ่านการทำ Platt Scaling แล้ว เส้นกราฟความน่าจะเป็นสอดคล้องกับเส้นอุดมคติ ส่งผลให้ค่า Brier Score เท่ากับ 0.081 สำหรับโรคเบาหวาน (เทียบกับ 0.091 ของต้นไม้ที่ไม่ปรับเทียบและไม่ถ่วงน้ำหนักคลาส ขณะที่ค่า 0.138–0.142 เกิดจากการถ่วงน้ำหนักคลาสบวก ซึ่งการปรับเทียบด้วย Platt Scaling ช่วยแก้ไขความเอนเอียงดังกล่าว) และเท่ากับ 0.185 สำหรับโรคความดันโลหิตสูง นอกจากนี้ ผลการศึกษาการตัดทอนองค์ประกอบ (Ablation Study) ในตารางที่ 5 แสดงให้เห็นว่าการตัดตัวแปรเปอร์เซ็นต์ไขมันร่างกายออก (No Body Fat) ทำให้ค่า AUC ลดลงเหลือ 0.748 และค่า Recall ลดลงเหลือร้อยละ 83.6 ส่วนการตัดเงื่อนไขโมโนโทนิกออก (No Monotonicity) ทำให้เกิดความย้อนแย้งทางคลินิกถึง 14 กรณี</p>

<!-- FIGURE 4 -->
<div class="fig-container" id="fig_4">
  <img src="fig3_calibration.png" style="object-fit: contain; width: 145pt; max-width: 145pt; height: 160pt;" alt="แผนภาพความน่าเชื่อถือของการปรับเทียบความน่าจะเป็นก่อนและหลังทำ Platt Scaling">
  <div class="fig-caption">รูปที่ 4 แผนภาพความน่าเชื่อถือของการปรับเทียบความน่าจะเป็นก่อนและหลังทำ Platt Scaling บนชุดทดสอบ</div>
</div>

<!-- TABLE 5 -->
<div class="table-container" id="tbl_5">
  <div class="table-caption">ตารางที่ 5 ผลการศึกษาการตัดทอนองค์ประกอบ (Ablation Study) บนชุดทดสอบโรคเบาหวาน</div>
  <table class="paper-table dense-table" data-w="37.0,9.5,13.0,13.0,9.5,18.0" data-nw="1,2,3,4,5">
    <tr>
      <th>โครงสร้างแบบจำลอง</th>
      <th>AUC</th>
      <th>Recall (F2)</th>
      <th>Spec. (F2)</th>
      <th>Brier</th>
      <th>การละเมิด<br>โมโนโทนิก</th>
    </tr>
    <tr><td><b>ไปป์ไลน์ Face2Health เต็มรูปแบบ</b></td><td><b>0.762</b></td><td><b>90.9%</b></td><td><b>48.7%</b></td><td><b>0.081</b></td><td><b>0 ครั้ง</b></td></tr>
    <tr><td>ตัดเปอร์เซ็นต์ไขมัน (No Body Fat)</td><td>0.748</td><td>83.6%</td><td>51.3%</td><td>0.085</td><td>0 ครั้ง</td></tr>
    <tr><td>ไม่บังคับโมโนโทนิก (No Monotonicity)</td><td>0.747</td><td>85.5%</td><td>50.0%</td><td>0.084</td><td>14 ครั้ง</td></tr>
    <tr><td>ไม่บังคับโมโนโทนิกและไม่ปรับเทียบ</td><td>0.747</td><td>80.0%</td><td>52.5%</td><td>0.142</td><td>14 ครั้ง</td></tr>
  </table>
  <div class="table-note">หมายเหตุ: ประเมินบนชุดทดสอบโรคเบาหวาน (N = 531) ที่ σ<sub>sim</sub> = 2.14 kg/m²</div>
</div>

<h3>4.4 ผลการวิเคราะห์ความไวต่อสัญญาณรบกวนของภาพถ่าย (Noise Sensitivity &amp; Shock Absorber)</h3>
<p>จากการทดสอบกวาดค่าความคลาดเคลื่อนของ BMI (ΔBMI ∈ [-2.0, +2.0] kg/m²) ในตารางที่ 6 พบว่าเมื่อมีตัวแปรเส้นรอบเอวร่วมด้วย ความชันความไวของการทำนายไขมันร่างกายลดลงเหลือเพียง α = 0.39 pp ต่อ 1 kg/m² เทียบกับ α = 1.00 pp ต่อ 1 kg/m² ในแบบจำลองที่ไม่มีรอบเอว แสดงถึงคุณสมบัติการเป็นตัวดูดซับความแปรปรวน (Variance Shock Absorber) อย่างไรก็ตาม จากการสุ่ม Monte Carlo 100 รอบ ภายใต้สัญญาณรบกวนระดับห้องปฏิบัติการ (σ<sub>sim</sub> = 2.82 kg/m²) และระดับการใช้งานจริง (σ<sub>sim</sub> = 4.40 kg/m²) แบบจำลองหลักแบบไม่ใช้รอบเอว (Contactless / No Waist)ยังคงให้ค่า AUC เท่ากับ 0.757 ± 0.006 (Recall ร้อยละ 88.7) และ 0.745 ± 0.009 (Recall ร้อยละ 85.1) สำหรับโรคเบาหวาน ซึ่งสูงกว่าแบบจำลองที่ใช้เฉพาะข้อมูลประชากรศาสตร์พื้นฐาน (AUC = 0.712, p = 0.007) อย่างมีนัยสำคัญทางสถิติ อย่างไรก็ตาม ที่ σ = 5.60 kg/m² ค่า AUC ลดลงเหลือ 0.735 ± 0.011 (Recall ร้อยละ 81.3) และความแตกต่างจากแบบจำลองที่ใช้เฉพาะข้อมูลประชากรศาสตร์ไม่มีนัยสำคัญทางสถิติสำหรับโรคเบาหวาน (ΔAUC = +0.024, 95% CI: −0.002 ถึง +0.049, p = 0.068) เมื่อเทียบกับค่า BMI ที่รายงานด้วยตนเองแบบจำลอง (σ = 1.50 kg/m²) การประมาณจากใบหน้าไม่ต่างอย่างมีนัยสำคัญที่ σ = 2.82 kg/m² แต่แย่กว่าอย่างมีนัยสำคัญที่ σ ≥ 4.40 kg/m² (ΔAUC = −0.019 สำหรับโรคเบาหวานที่ σ = 4.40) การสุ่มความคลาดเคลื่อนแบบอิงข้อมูลจริง (Empirical Residual Resampling) ให้ Recall ร้อยละ 80.9–84.5 สำหรับโรคเบาหวาน และในกลุ่มที่ยังไม่ได้รับการวินิจฉัยตามเกณฑ์ HbA1c Recall ลดลงเหลือร้อยละ 76.9 (10/13 ราย) ระบบจึงเป็นเพียงทางเลือกสำรองเมื่อไม่มีเครื่องชั่งน้ำหนักหรือค่าที่รายงานด้วยตนเองที่เชื่อถือได้เท่านั้น</p>

<!-- TABLE 6 -->
<div class="table-container" id="tbl_6">
  <div class="table-caption">ตารางที่ 6 การเปลี่ยนแปลงของเปอร์เซ็นต์ไขมันร่างกายภายใต้การรบกวนค่า BMI</div>
  <table class="paper-table dense-table" data-w="20.0,28.0,20.0,32.0" data-nw="0,1,2,3">
    <tr>
      <th>การรบกวน<br>ΔBMI</th>
      <th>ค่าเฉลี่ย ΔBF<br>มีรอบเอว (pp)</th>
      <th>ความชัน α<br>(pp/kg/m²)</th>
      <th>ค่าเฉลี่ย ΔBF<br>ไม่มีรอบเอว (pp)</th>
    </tr>
    <tr><td>-2.0 kg/m²</td><td>-0.80 ± 0.80 pp</td><td>0.40</td><td>-2.02 ± 1.24 pp</td></tr>
    <tr><td>-1.0 kg/m²</td><td>-0.37 ± 0.50 pp</td><td>0.37</td><td>-1.01 ± 0.82 pp</td></tr>
    <tr><td>0.0 kg/m²</td><td>0.00 ± 0.00 pp</td><td>—</td><td>0.00 ± 0.00 pp</td></tr>
    <tr><td>+1.0 kg/m²</td><td>+0.39 ± 0.54 pp</td><td>0.39</td><td>+1.00 ± 0.83 pp</td></tr>
    <tr><td>+2.0 kg/m²</td><td>+0.77 ± 0.72 pp</td><td>0.38</td><td>+1.98 ± 1.14 pp</td></tr>
  </table>
</div>

<h3>4.5 การจัดลำดับความเสี่ยงทางคลินิก 4 ระดับ (4-Tier Clinical Triage)</h3>
<p>จากผลการจำแนกใน Confusion Matrix รูปที่ 5 ผู้วิจัยได้นำค่าความน่าจะเป็นที่ปรับเทียบแล้วมาจัดแบ่งเป็นระดับการคัดกรองทางคลินิก 4 ระดับ ดังแสดงในตารางที่ 7 พบว่าในกลุ่มความเสี่ยงเร่งด่วน (Urgent) มีความชุกของโรคเบาหวานจริงสูงถึงร้อยละ 39.13 (เข้มข้นขึ้น 3.8 เท่าจากความชุกพื้นฐาน) และความดันโลหิตสูงร้อยละ 59.87 ขณะที่กลุ่มเสี่ยงต่ำ (Low) มีค่าพยากรณ์ผลลบ (NPV) สูงถึงร้อยละ 97.89 สำหรับโรคเบาหวาน และร้อยละ 89.44 สำหรับโรคความดันโลหิตสูง ช่วยลดภาระการส่งตรวจทางห้องปฏิบัติการที่ไม่จำเป็นได้กว่าร้อยละ 44.6 ของผู้เข้ารับการคัดกรองทั้งหมด</p>

<!-- TABLE 7 -->
<div class="table-container" id="tbl_7" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">ตารางที่ 7 ตัวชี้วัดการจัดลำดับความเสี่ยงทางคลินิก 4 ระดับบนชุดทดสอบ (Test set)</div>
  <table class="paper-table dense-table" data-w="14.5,24.5,10.0,11.0,10.0,11.0,19.0" data-nw="0,1,2,3,4,5">
    <tr>
      <th>ระดับ<br>คัดกรอง</th>
      <th>ช่วงความน่าจะเป็น</th>
      <th>เบาหวาน<br>(N)</th>
      <th>ความชุก<br>เบาหวาน</th>
      <th>ความดันฯ<br>(N)</th>
      <th>ความชุก<br>ความดันฯ</th>
      <th>แนวปฏิบัติ<br>ทางคลินิก</th>
    </tr>
    <tr>
      <td>เสี่ยงต่ำ<br>(Low)</td>
      <td>เบาหวาน: p &lt; 6.0%<br>ความดันฯ: p &lt; 15.5%</td>
      <td>237<br>(44.6%)</td>
      <td>2.11%<br>(5/237)</td>
      <td>161<br>(30.2%)</td>
      <td>10.56%<br>(17/161)</td>
      <td>ตรวจสุขภาพ<br>ประจำปี</td>
    </tr>
    <tr>
      <td>เฝ้าระวัง<br><span style="white-space: nowrap;">(Watchful)</span></td>
      <td>เบาหวาน: 6.0–10.0%<br>ความดันฯ: 15.5–25.0%</td>
      <td>127<br>(23.9%)</td>
      <td>7.87%<br>(10/127)</td>
      <td>122<br>(22.9%)</td>
      <td>27.05%<br>(33/122)</td>
      <td>เจาะน้ำตาลปลายนิ้ว<br>วัดความดันซ้ำ</td>
    </tr>
    <tr>
      <td>ผลบวก<br>(Positive)</td>
      <td>เบาหวาน: 10.0–20.0%<br>ความดันฯ: 25.0–50.0%</td>
      <td>121<br>(22.8%)</td>
      <td>18.18%<br>(22/121)</td>
      <td>98<br>(18.4%)</td>
      <td>40.82%<br>(40/98)</td>
      <td>ตรวจเลือดในแล็บ<br>วัดความดันคลินิก</td>
    </tr>
    <tr>
      <td>เร่งด่วน<br>(Urgent)</td>
      <td>เบาหวาน: p ≥ 20.0%<br>ความดันฯ: p ≥ 50.0%</td>
      <td>46<br>(8.7%)</td>
      <td><b>39.13%<br>(18/46)</b></td>
      <td>152<br>(28.5%)</td>
      <td><b>59.87%<br>(91/152)</b></td>
      <td>ส่งต่อพบแพทย์<br>เฉพาะทาง</td>
    </tr>
  </table>
  <div class="table-note">หมายเหตุ: ประเมินบนชุดทดสอบ N = 531 (เบาหวาน) และ N = 533 (ความดันฯ) ที่ σ<sub>sim</sub> = 2.14 kg/m²</div>
</div>

<!-- FIGURE 5 -->
<div class="fig-container" id="fig_5">
  <img src="fig4_confusion_matrix.png" style="object-fit: contain; width: 185pt; max-width: 185pt; height: 264pt;" alt="Confusion Matrix ของการจำแนกโรคเบาหวานบนชุดทดสอบ">
  <div class="fig-caption">รูปที่ 5 Confusion Matrix ของการจำแนกโรคเบาหวานบนชุดทดสอบ (N = 531) ณ จุดตัดตัดสินใจทั้ง 3 เกณฑ์</div>
</div>

<h2>5. สรุปผล ข้อจำกัด และแนวทางในอนาคต</h2>
<h3>5.1 สรุปผลการวิจัย</h3>
<p>การศึกษาครั้งนี้ได้พัฒนากรอบการทำงาน Face2Health แบบแยกส่วน 3 ขั้นตอน ที่เชื่อมโยงการประมาณค่าดัชนีมวลกายจากภาพถ่ายใบหน้าด้วยแบบจำลอง ViT-H/14 ร่วมกับ Monte Carlo Dropout เข้ากับการประมาณค่าเปอร์เซ็นต์ไขมันร่างกาย DEXA และการจำแนกความเสี่ยงโรคด้วยแบบจำลอง Monotonic Calibrated XGBoost บนชุดข้อมูล CDC NHANES ผลการประเมินประสิทธิภาพบน Test set ยืนยันว่าการใช้จุดตัดที่เหมาะสมด้วยเกณฑ์ F2-Score ช่วยให้แบบจำลองหลักสามารถคัดกรองผู้ป่วยโรคเบาหวาน ได้ด้วยค่า Recall 0.9091 (AUC = 0.762, Brier Score = 0.081) และโรคความดันโลหิตสูงด้วยค่า Recall 0.9061 (AUC = 0.757, Brier Score = 0.185) โดยปราศจากการละเมิดเงื่อนไขความชันทางชีววิทยา ทั้งนี้ผลดังกล่าวเป็นผลที่ระดับสัญญาณรบกวนจำลองต่ำสุด (σ = 2.14 kg/m²) เมื่อจำลองสัญญาณรบกวนระดับการใช้งานจริง (σ = 4.40–5.60 kg/m²) ค่า Recall ลดลงเป็นร้อยละ 81–85 (โรคเบาหวาน) และร้อยละ 84–86 (โรคความดันโลหิตสูง) และที่ σ = 5.60 kg/m² ผลในโรคเบาหวานไม่มีนัยสำคัญทางสถิติเหนือแบบจำลองข้อมูลประชากรศาสตร์พื้นฐาน (p = 0.068) ระบบจึงเป็นเครื่องมือคัดกรองสำรองเบื้องต้นที่มีประโยชน์จำกัดภายใต้เงื่อนไข ไม่ใช่เครื่องมือวินิจฉัย และด้อยกว่าการใช้ค่า BMI ที่รายงานด้วยตนเองเมื่อมีข้อมูลดังกล่าว</p>

<h3>5.2 ข้อจำกัดของการศึกษา</h3>
<p>การศึกษานี้มีข้อจำกัดสำคัญที่ควรระบุอย่างโปร่งใสตามหลักวิชาการ ได้แก่ (1) <i>นิยามป้ายกำกับ:</i> อ้างอิงจากการรายงานประวัติการวินิจฉัยโดยแพทย์ (DIQ010/BPQ020) โมเดลจึงเรียนรู้ที่จะจำแนกผู้ที่ทราบผลแล้ว มิใช่กลุ่มผู้ป่วยที่ไม่เคยได้รับการวินิจฉัย (Undiagnosed Cases) ซึ่งเป็นเป้าหมายแท้จริงของการคัดกรองในชุมชน (2) <i>การจำลองสัญญาณรบกวน:</i> การใช้ Gaussian Noise โดยกำหนดค่า MAE เป็นความเบี่ยงเบนมาตรฐาน (σ) ทำให้ประมาณระดับ noise ต่ำกว่าความเป็นจริง (ตามทฤษฎี MAE ≈ 0.80σ ค่า σ จึงควรเป็นประมาณ 2.68 ไม่ใช่ 2.14 kg/m²) และความคลาดเคลื่อนจริงของโมเดลใบหน้ามีลักษณะ Regression to the Mean (R² = 0.68) ทำให้ประเมินค่าในกลุ่มคนอ้วนต่ำเกินไป อีกทั้งการลดทอนความคลาดเคลื่อนของค่าไขมันร่างกายเกิดขึ้นเด่นชัดเฉพาะเมื่อมีข้อมูลรอบเอวร่วมด้วย (3) <i>ตัวแปรไขมันร่างกาย:</i> ค่าไขมันจาก Stage 1.5 คำนวณจากตัวแปรชุดเดียวกับ Stage 2 ประโยชน์ที่ได้รับจึงเป็นการถ่ายทอดความรู้จากผลตรวจ DEXA มิใช่ข้อมูลใหม่จากภาพ และช่วงความเชื่อมั่น AUC มีความซ้อนทับกัน (4) <i>เกณฑ์คลินิกและ Calibration:</i> ยังไม่ได้เทียบกับเกณฑ์เวชปฏิบัติทั่วไป (เช่น คัดกรองผู้มีอายุ ≥ 35 ปี) และ Brier Skill Score เมื่อเทียบกับความชุกพื้นฐานอยู่ในระดับปานกลาง (~0.12 ในเบาหวาน และ ~0.18 ในความดันฯ) (5) <i>ข้อจำกัดของชุดข้อมูล:</i> DEXA ใน NHANES คัดผู้ที่หนักเกิน 300 ปอนด์ออก ส่งผลให้กลุ่ม BMI สูงมากมีตัวแทนน้อย และข้อมูลเป็นตัวแทนประชากรสหรัฐฯ ซึ่งอาจมีความสัมพันธ์ระหว่าง BMI กับไขมันร่างกายแตกต่างจากประชากรไทย และ (6) <i>จริยธรรมและความเป็นส่วนตัว:</i> ภาพถ่าย VisualBMI มาจากเว็บบอร์ดสาธารณะโดยไม่มีความยินยอมเพื่อการวิจัยทางคลินิก การนำไปใช้จริงจำเป็นต้องได้รับความยินยอมและปฏิบัติตามกฎหมาย PDPA อย่างเคร่งครัด งานนี้จึงเป็นเพียงการพิสูจน์แนวคิด (Proof of Concept) มิใช่เพื่อการวินิจฉัยโรคทางการแพทย์</p>

<h3>5.3 แนวทางในอนาคต</h3>
<p>เพื่อยกระดับประสิทธิภาพของ Face2Health งานวิจัยในอนาคตมีทิศทางสำคัญ 4 ประการ ได้แก่ (1) การทดสอบทางคลินิกแบบไปข้างหน้า (Prospective Validation) ในสถานพยาบาลปฐมภูมิในประเทศไทย เพื่อปรับเกณฑ์ให้สอดคล้องกับสรีระประชากรไทย (2) การผสานสัญญาณชีพจากวิดีโอใบหน้าด้วยเทคนิค rPPG เพื่อเพิ่มความแม่นยำในการคัดกรองโรคความดันโลหิตสูง (3) การบีบอัดแบบจำลอง (Model Quantization &amp; ONNX Runtime) เพื่อรองรับการทำงานแบบออฟไลน์บนตู้คัดกรองสุขภาพอัตโนมัติในพื้นที่ห่างไกล และ (4) การฝึกสอนแบบคำนึงถึงสัญญาณรบกวน (Noise-Aware Training) เพื่อเพิ่มความทนทานต่อภาพถ่ายคุณภาพต่ำในสภาพแวดล้อมจริง</p>

<h2>เอกสารอ้างอิง</h2>
<p class="ref-item">[1] World Health Organization. (2022). <i>Noncommunicable diseases: Progress monitor 2020 and fact sheet</i>. World Health Organization.</p>
<p class="ref-item">[2] International Diabetes Federation. (2021). <i>IDF diabetes atlas</i> (10th ed.). International Diabetes Federation.</p>
<p class="ref-item">[3] Coetzee, V., Perrett, D. I., &amp; Stephen, I. D. (2009). Facial adiposity: A reliable cue to health? <i>Perception</i>, 38(11), 1700–1711.</p>
<p class="ref-item">[4] Wen, L., &amp; Guo, G. (2013). A computational approach to body mass index prediction from face images. <i>Image and Vision Computing</i>, 31(5), 392–400.</p>
<p class="ref-item">[5] Lee, B. J., &amp; Kim, J. Y. (2014). Predicting visceral obesity based on facial characteristics. <i>BMC Complementary and Alternative Medicine</i>, 14(1), 248.</p>
<p class="ref-item">[6] He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep residual learning for image recognition. <i>Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)</i>, 770–778.</p>
<p class="ref-item">[7] Kocabey, E., Camurcu, M., Ofli, F., Aytar, Y., Marin, J., Torralba, A., &amp; Weber, I. (2017). Face-to-BMI: Using computer vision to infer body mass index on social media. <i>Proceedings of the 11th International AAAI Conference on Web and Social Media (ICWSM)</i>, 572–575.</p>
<p class="ref-item">[8] Stephen, I. D., Hiew, V., Coetzee, V., Tiddeman, B. P., &amp; Perrett, D. I. (2017). Facial shape analysis identifies valid cues to aspects of physiological health in Caucasian, Asian, and African populations. <i>Frontiers in Psychology</i>, 8, 1883.</p>
<p class="ref-item">[9] Jiang, M., Guo, G., &amp; Mu, G. (2020). Visual BMI estimation from face images using a label distribution based method. <i>Computer Vision and Image Understanding</i>, 197–198, 102985.</p>
<p class="ref-item">[10] Centers for Disease Control and Prevention. (2018). <i>National Health and Nutrition Examination Survey (NHANES): Body composition procedures manual and dual-energy X-ray absorptiometry (DXX) data documentation</i>. U.S. Department of Health and Human Services.</p>
<p class="ref-item">[11] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., &amp; Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. <i>Proceedings of the International Conference on Learning Representations (ICLR)</i>.</p>
<p class="ref-item">[12] Gal, Y., &amp; Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. <i>Proceedings of the 33rd International Conference on Machine Learning (ICML)</i>, 48, 1050–1059.</p>
<p class="ref-item">[13] Chen, T., &amp; Guestrin, C. (2016). XGBoost: A scalable tree boosting system. <i>Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>, 785–794.</p>
<p class="ref-item">[14] Zhou, Z.-H. (2012). <i>Ensemble methods: Foundations and algorithms</i>. CRC Press.</p>
<p class="ref-item">[15] Platt, J. C. (2000). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. In A. J. Smola, P. Bartlett, B. Schölkopf, &amp; D. Schuurmans (Eds.), <i>Advances in large margin classifiers</i> (pp. 61–74). MIT Press.</p>
<p class="ref-item">[16] Niculescu-Mizil, A., &amp; Caruana, R. (2005). Predicting good probabilities with supervised learning. <i>Proceedings of the 22nd International Conference on Machine Learning (ICML)</i>, 625–632.</p>
<p class="ref-item">[17] Youden, W. J. (1950). Index for rating diagnostic tests. <i>Cancer</i>, 3(1), 32–35.</p>
<p class="ref-item">[18] Elkan, C. (2001). The foundations of cost-sensitive learning. <i>Proceedings of the 17th International Joint Conference on Artificial Intelligence (IJCAI)</i>, 973–978.</p>
<p class="ref-item">[19] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., &amp; Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research</i>, 12, 2825–2830.</p>
<p class="ref-item">[20] Bergstra, J., &amp; Bengio, Y. (2012). Random search for hyper-parameter optimization. <i>Journal of Machine Learning Research</i>, 13, 281–305.</p>
"""

    def render_story_to_pdf(body_html_str, out_pdf_path):
        processed_top_html = insert_zws_to_html(PAGE1_TOP_HTML)
        processed_fn_html = insert_zws_to_html(PAGE1_FOOTNOTE_HTML)
        processed_body_html = prepare_tables(body_html_str, target_table_w=240.0)
        processed_body_html = insert_zws_to_html(processed_body_html)

        archive = pymupdf.Archive(RESULTS_DIR)
        writer = pymupdf.DocumentWriter(out_pdf_path)

        # A4 Page dimensions matching [THAI]-67076012-Short Paper.pdf (595.32 x 841.92 pt)
        a4_rect = pymupdf.paper_rect("a4")
        page_w, page_h = a4_rect.width, a4_rect.height

        # Page 1 margins (Full-width single column matching example paper Page 1)
        p1_margin_l = 60.0
        p1_margin_r = page_w - 60.0
        p1_top_y = 78.0

        # Pages 2+ margins (2-Column layout with 245pt columns)
        margin_l = 45.6
        margin_t = 50.0
        margin_b = 49.0
        col_w = 245.0
        col_gap = 13.0
        margin_r = margin_l + col_w + col_gap + col_w

        top_story = pymupdf.Story(html=processed_top_html, archive=archive)
        fn_story = pymupdf.Story(html=processed_fn_html, archive=archive)
        body_story = pymupdf.Story(html=processed_body_html, archive=archive)

        # --- Draw Page 1 (Single-Column Cover / Abstract / Keywords / Footnote) ---
        device = writer.begin_page(a4_rect)
        top_rect = pymupdf.Rect(p1_margin_l, p1_top_y, p1_margin_r, page_h - 190.0)
        top_story.place(top_rect)
        top_story.draw(device)

        fn_rect = pymupdf.Rect(p1_margin_l, page_h - 175.0, p1_margin_r, page_h - 55.0)
        fn_story.place(fn_rect)
        fn_story.draw(device)
        writer.end_page()

        # --- Draw Pages 2+ (Two-Column Layout for Chapters 1 to 5 + References) ---
        page_idx = 2
        max_pages = 30
        while page_idx <= max_pages:
            device = writer.begin_page(a4_rect)
            col1 = pymupdf.Rect(margin_l, margin_t, margin_l + col_w, page_h - margin_b)
            col2 = pymupdf.Rect(margin_l + col_w + col_gap, margin_t, margin_r, page_h - margin_b)

            more1, _ = body_story.place(col1)
            body_story.draw(device)

            if more1 != 0:
                more2, _ = body_story.place(col2)
                body_story.draw(device)
                has_more = (more2 != 0)
            else:
                has_more = False

            writer.end_page()
            if not has_more:
                break
            page_idx += 1

        writer.close()
        del writer
        return page_idx

    raw_pdf_path = os.path.join(RESULTS_DIR, "_raw_story_output_thai.pdf")
    total_pages = render_story_to_pdf(BODY_HTML, raw_pdf_path)

    doc_post = pymupdf.open(raw_pdf_path)
    try:
        doc_post.save(PDF_OUT, deflate=True)
        doc_post.close()
        del doc_post
        if os.path.exists(raw_pdf_path):
            try:
                os.remove(raw_pdf_path)
            except Exception:
                pass
        pdf_size = os.path.getsize(PDF_OUT)
        print(f"[+] คอมไพล์เอกสารวิจัยภาษาไทยสำเร็จ: {PDF_OUT} ({total_pages} หน้า, {pdf_size:,} bytes)")
    except Exception as err:
        print(f"[!] Primary save failed: {err}")
        alt_out = os.path.join(REPORTS_DIR, "paper_thai_temp.pdf")
        doc_post.save(alt_out, deflate=True)
        doc_post.close()
        del doc_post
        if os.path.exists(raw_pdf_path):
            try:
                os.remove(raw_pdf_path)
            except Exception:
                pass
        try:
            shutil.move(alt_out, PDF_OUT)
            pdf_size = os.path.getsize(PDF_OUT)
            print(f"[+] บันทึกทดแทนสำเร็จ: {PDF_OUT} ({total_pages} หน้า, {pdf_size:,} bytes)")
        except Exception:
            pdf_size = os.path.getsize(alt_out)
            print(f"[!] Warning: ไม่สามารถเขียนทับ '{PDF_OUT}' ได้ จึงบันทึกเป็น '{alt_out}' แทน")


if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=============================================================================")
    print(" กำลังสร้างและคอมไพล์เอกสารวิจัยวิชาการฉบับภาษาไทย (Face2Health Thai Paper)    ")
    print("=============================================================================")
    build_thai_pdf()
    print("\n[OK] เสร็จสิ้นกระบวนการคอมไพล์ paper_thai.pdf เรียบร้อยแล้ว")
