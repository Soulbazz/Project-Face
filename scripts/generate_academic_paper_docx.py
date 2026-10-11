"""
=============================================================================
Face2Health - Extended Academic Research Paper Pipeline (Exact Paper_2 Match)
=============================================================================
Applied Fixes:
  1. Applied exact text/shortened labels from Paper_2.pdf:
     - Table IV: Completely removed '— Remove' prefix ('No Body Fat', 'No Monotonicity', 'No Calibration').
     - Table I, II, III-A, III-B, V, VI: Applied exact shortened labels as edited in Paper_2.pdf.
  2. Fixed column overflow & width balance:
     - Inserted <br> breaks inside parentheses (95% CI and cases/counts) for compact vertical alignment.
     - Re-balanced column widths (data-w) to tightly fit inside 254pt column without any right-side overflow.
  3. Header & Title Styling:
     - Paper Title and Author Names set to font-weight: normal with 18px vertical spacing.
     - Section headings (H2) set to font-weight: normal.
=============================================================================
"""

import os
import io
import sys
import re
import pymupdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

PDF_OUT = os.path.join(REPORTS_DIR, "Paper.pdf")


def prepare_tables(html_content):
    """
    Parses <table> tags with data-w (widths) attributes,
    calculating precise point widths for MuPDF (254pt total column width) and injecting
    inline style rules directly onto cells to guarantee clean rendering without overflow.
    """
    COL_WIDTH_PT = 254.0

    def replace_table(match):
        table_open_tag = match.group(1)
        inner_content = match.group(2)
        
        w_match = re.search(r'data-w="([^"]+)"', table_open_tag)
        nw_match = re.search(r'data-nw="([^"]+)"', table_open_tag)
        
        widths = [float(x.strip()) for x in w_match.group(1).split(',')] if w_match else []
        nw_cols = [int(x.strip()) for x in nw_match.group(1).split(',')] if nw_match else []
        
        pt_widths = [(w / 100.0) * COL_WIDTH_PT for w in widths]
        
        rows = re.findall(r'(<tr.*?>.*?</tr>)', inner_content, re.DOTALL)
        new_rows = []
        
        for row_idx, row in enumerate(rows):
            cells = re.findall(r'(<t[hd].*?>)(.*?)(</t[hd]>)', row, re.DOTALL)
            new_cells = []
            for col_idx, (open_tag, content, close_tag) in enumerate(cells):
                styles = []
                if row_idx == 0 and col_idx < len(pt_widths):
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


def build_ieee_pdf():
    print("[*] Compiling Restored High-Quality Academic Paper PDF...")

    HEADER_HTML = """
<style>
.header-box { text-align: center; font-family: 'Times New Roman', serif; margin-bottom: 6px; }
.title { font-size: 16.5pt; font-style: italic; font-weight: normal; line-height: 1.12; margin-bottom: 18px; }
.authors-table { width: 100%; border-collapse: collapse; margin-bottom: 2px; border: none; }
.authors-table td { text-align: center; vertical-align: top; font-size: 8.8pt; line-height: 1.15; border: none; padding: 0 10px; width: 50%; }
.author-name { font-size: 9.8pt; font-weight: normal; margin-bottom: 1px; color: #000000; }
.author-inst { font-size: 8.2pt; font-style: italic; color: #000000; }
.author-loc { font-size: 8.2pt; font-style: normal; color: #000000; }
.author-mail { font-size: 8.2pt; font-style: normal; margin-top: 1px; color: #000000; }
</style>
<div class="header-box">
  <div class="title">Face2Health: A Decoupled Multimodal Machine Learning Framework for Screening Physician-Diagnosed Diabetes and Hypertension Risk</div>
  <table class="authors-table">
    <tr>
      <td>
        <div class="author-name">Phalathip Hemvut</div>
        <div class="author-inst">School of Information Technology</div>
        <div class="author-inst">King Mongkut’s Institute of Technology Ladkrabang</div>
        <div class="author-loc">Bangkok, Thailand</div>
        <div class="author-mail">67070255@it.kmitl.ac.th</div>
      </td>
      <td>
        <div class="author-name">Eua-Unggul Chaivivatporn</div>
        <div class="author-inst">School of Information Technology</div>
        <div class="author-inst">King Mongkut’s Institute of Technology Ladkrabang</div>
        <div class="author-loc">Bangkok, Thailand</div>
        <div class="author-mail">67070302@it.kmitl.ac.th</div>
      </td>
    </tr>
  </table>
</div>
"""

    BODY_HTML = """
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    margin: 0; padding: 0;
    font-family: 'Times New Roman', serif; font-size: 9.0pt;
    line-height: 1.10; text-align: justify; color: #000000;
    background-color: transparent;
}
.abstract-box {
    margin-bottom: 6px; font-size: 8.5pt; text-align: justify;
    line-height: 1.10; font-weight: bold; text-indent: 12pt;
    border: none; padding: 0;
}
.abstract-title { font-weight: bold; font-style: italic; }
.keywords {
    font-size: 8.5pt; margin-bottom: 8px; line-height: 1.10;
    font-weight: bold; font-style: italic; text-indent: 12pt;
}
.keywords-title { font-weight: bold; font-style: italic; }
h2 {
    font-size: 9.5pt; font-weight: normal; text-align: center;
    text-transform: uppercase; margin: 8px 0 3px 0; letter-spacing: 0.3px;
    break-after: avoid; page-break-after: avoid;
    background-color: transparent;
}
h3 {
    font-size: 9.0pt; font-weight: bold; font-style: italic;
    text-align: left; margin: 5px 0 2px 0; break-after: avoid; page-break-after: avoid;
    background-color: transparent;
}
p { margin: 0 0 3px 0; text-indent: 12pt; background-color: transparent; }

.table-container {
    break-inside: avoid;
    page-break-inside: avoid;
    margin: 5px 0;
    width: 100%;
}
.table-caption {
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 7.2pt; font-weight: bold; text-align: center;
    margin: 0 0 2px 0; text-transform: uppercase;
}
table.paper-table {
    break-inside: avoid;
    page-break-inside: avoid;
    width: 254pt;
    margin: 3px 0;
    border-collapse: collapse;
    font-size: 4.8pt;
    line-height: 1.05;
    text-align: center;
    table-layout: fixed;
}
table.dense-table {
    font-size: 4.1pt;
    line-height: 1.02;
}
table.paper-table th {
    border: 0.5pt solid #000000;
    font-weight: bold;
    padding: 1px 0.5px;
    background-color: transparent;
    color: #000000;
    word-wrap: break-word;
    overflow-wrap: break-word;
    vertical-align: middle;
}
table.paper-table td {
    border: 0.5pt solid #000000;
    padding: 1px 0.5px;
    color: #000000;
    word-wrap: break-word;
    overflow-wrap: break-word;
    vertical-align: middle;
}
.fig-container { break-inside: avoid; page-break-inside: avoid; text-align: center; margin: 6px 0; }
.fig-caption {
    break-inside: avoid; page-break-inside: avoid; font-size: 7.8pt;
    text-align: justify; margin-top: 2px; line-height: 1.08;
}
.equation {
    text-align: center; font-style: italic; margin: 3px 0;
    font-size: 8.5pt; break-inside: avoid; page-break-inside: avoid;
}
.ref-item {
    font-size: 7.0pt; margin-bottom: 2px; text-indent: -12pt;
    margin-left: 12pt; line-height: 1.05; text-align: left;
    break-inside: avoid; page-break-inside: avoid;
}
</style>

<div class="abstract-box">
  <span class="abstract-title">Abstract</span>—This paper presents <b>Face2Health</b>, a decoupled multimodal machine learning architecture engineered for accessible screening of physician-diagnosed Type 2 Diabetes Mellitus (DIQ010) and Essential Hypertension (BPQ020). The pipeline sequentially couples: (1) a Vision Transformer (ViT-H/14) fine-tuned on VisualBMI and Chicago Face Database (CFD) for Body Mass Index (BMI) estimation (MAE = 2.14 kg/m&sup2;, RMSE = 2.82 kg/m&sup2;, R&sup2; = 0.68) with 25-pass Monte Carlo Dropout for epistemic uncertainty rejection (&sigma; &gt; 1.80 kg/m&sup2;, 4.2% rejection rate) alongside scale-invariant facial morphometrics; (2) a tabular Extreme Gradient Boosting (XGBoost) regressor mapping estimated BMI, age, biological sex, and physical activity level to Dual-Energy X-ray Absorptiometry (DEXA)-derived Total Body Fat Percentage (R&sup2; = 0.74, MAE = 3.42% for non-invasive route; R&sup2; = 0.82, MAE = 2.85% with waist circumference); and (3) monotonically regularized, Platt-calibrated XGBoost classifiers estimating continuous posterior probabilities. Evaluated on the CDC NHANES adult cohort (2011–2018 cycles, aged 18–59 years, N = 3,540 for Diabetes, N = 3,554 for Hypertension) using out-of-fold cross-validation and noisy vision error simulation, the primary non-invasive pipeline (without waist circumference) achieves <b>90.91% sensitivity</b> for Diabetes (50/55 positive cases detected, 95% CI: 80.0%–96.7%) and <b>90.61% sensitivity</b> for Hypertension (164/181 positive cases detected, 95% CI: 85.3%–94.4%) under F2-optimized screening thresholds (&tau; = 0.060 for Diabetes, &tau; = 0.155 for Hypertension). We present comprehensive baseline comparisons against Logistic Regression, execute full ablation studies, detail participant inclusion flow, and establish an epidemiologically aligned 4-tier clinical triage protocol.
</div>

<div class="keywords">
  <span class="keywords-title">Keywords</span>—computer vision, vision transformer, MC dropout, non-invasive health screening, class imbalance, probability calibration, Platt scaling, XGBoost, monotonicity constraints, diabetes, hypertension, NHANES, DEXA.
</div>

<h2>I. INTRODUCTION & PROBLEM STATEMENT</h2>
<p>Non-communicable diseases (NCDs), led by Type 2 Diabetes Mellitus and Essential Hypertension, constitute the defining epidemiological crisis of the modern era, accounting for over 41 million deaths annually—equivalent to 74% of all mortalities worldwide [1]. Beyond mortality, sustained undetected hyperglycemia and arterial hypertension induce progressive microvascular and macrovascular pathology, culminating in diabetic nephropathy, proliferative retinopathy, peripheral neuropathy, ischemic stroke, and coronary heart disease. In middle-income and newly industrialized nations, population health surveys indicate that 30% to 45% of individuals living with diabetes and hypertension remain undiagnosed until secondary clinical complications necessitate emergency hospitalization [2].</p>

<p>Current gold-standard diagnostic pathways rely on invasive venous phlebotomy (fasting plasma glucose &ge; 126 mg/dL or HbA1c &ge; 6.5%) and clinical sphygmomanometry. While biochemically authoritative, these methods impose structural limitations requiring sterile consumables, certified medical technologists, and centralized laboratory infrastructure. In parallel, clinical anthropometry demonstrates that systemic adiposity and metabolic risk physically manifest in craniofacial soft tissue. Adipocyte hypertrophy within the superficial and deep buccal fat compartments, jowl fat expansion, and submental tissue thickening reflect central lipid deposition. The maturation of Vision Transformers (ViT) with self-attention, alongside gradient boosted decision trees (XGBoost), provides a computational framework capable of translating passive 2D digital portraits into physiological biomarkers.</p>

<p>However, cascading visual-to-tabular pipelines introduce three key algorithmic challenges: (1) multi-stage regression error propagation, where vision estimation noise affects subsequent metabolic models; (2) biological inconsistency, where collinear features lead unconstrained decision trees to form non-monotonic risk surfaces; and (3) probability output compression under class imbalance (~10.3% diabetes prevalence), where standard 0.50 decision thresholds yield 0.0% default clinical sensitivity. This paper formalizes, benchmarks, and resolves these operational dynamics on the CDC NHANES dataset using physician-diagnosed disease targets (DIQ010 and BPQ020).</p>

<h2>II. LITERATURE REVIEW & BENCHMARK ANALYSIS</h2>
<h3>A. Craniofacial Morphometry and Visceral Adiposity</h3>
<p>Craniofacial morphology serves as an external indicator of systemic adiposity due to the distribution of subcutaneous and deep facial fat pads. Landmark anatomical studies by Coetzee et al. [3] established that facial adiposity serves as a reliable cue for systemic health. L. Wen and G. Guo [4] demonstrated that computer-derived facial features correlate significantly with body mass index and metabolic parameters. Anatomically, lipid accumulation induces lateral and inferior displacement across the mandibular border. To capture these shifts, four scale-invariant geometric indices are defined: Lower Facial Width-to-Height Ratio (LFWR), Cheek-to-Jaw Width Ratio (CJWR), Perimeter-to-Area Ratio (PAR), and Facial Width-to-Height Ratio (FWHR). Lee and Kim [5] demonstrated that facial width measurements correlate significantly with abdominal obesity metrics.</p>

<h3>B. Facial Biomarker Estimation & Deep Learning Benchmarks</h3>
<p>Early facial health estimation relied on social media datasets. Kocabey et al. [6] introduced <i>Face-to-BMI</i> using public Reddit progress photographs, establishing initial CNN benchmarks (MAE ~2.8 kg/m&sup2;). Stephen et al. [7] demonstrated cross-cultural validity of facial shape cues across Caucasian, African, and Asian cohorts. Jiang, Guo, and Mu [8] extended facial adiposity models to metabolic risk prediction using deep residual networks [19]. However, these monolithic approaches map pixels directly to disease labels, lacking epistemic uncertainty estimation and intermediate biomarker transparency.</p>

<h3>C. Dual-Energy X-ray Absorptiometry (DEXA) & NHANES Cohort</h3>
<p>While Body Mass Index (BMI = weight/height&sup2;) is ubiquitous, it cannot differentiate skeletal muscle mass from adipose tissue. Dual-Energy X-ray Absorptiometry (DEXA) constitutes the diagnostic gold standard for body composition analysis, using photon attenuation to resolve bone mineral, lean soft tissue, and fat mass. The National Health and Nutrition Examination Survey (NHANES) conducted by the CDC represents a multi-ethnic dataset combining whole-body DEXA scans (total body fat percentage DXDTOPF), standardized anthropometry (BMX), laboratory biochemical assays (LBXGLU, LBXGH), and structured diagnostic interviews (DIQ010 for physician-diagnosed diabetes, BPQ020 for physician-diagnosed hypertension) across continuous 2011–2018 survey cycles for adult participants aged 18–59 years [16].</p>

<h3>D. Vision Transformers and Epistemic Uncertainty Quantification</h3>
<p>Dosovitskiy et al. [9] showed that Vision Transformers (ViT) model long-range spatial dependencies across distant facial landmarks using self-attention. Deterministic neural networks, however, suffer from overconfident mispredictions on out-of-distribution inputs. To address this, Monte Carlo Dropout (Gal & Ghahramani [10]) provides a Bayesian approximation of Gaussian process uncertainty by preserving active dropout masks during inference. Sampling T=25 forward stochastic passes yields the predictive mean &mu;(x*) and epistemic variance &sigma;&sup2;(x*):</p>
<div class="equation">&mu;(x*) = (1/T) &Sigma;<sub>t=1</sub><sup>T</sup> f<sub>W<sub>t</sub></sub>(x*), &nbsp;&nbsp;&nbsp;&nbsp; &sigma;&sup2;(x*) = [1/(T-1)] &Sigma;<sub>t=1</sub><sup>T</sup> [f<sub>W<sub>t</sub></sub>(x*) - &mu;(x*)]&sup2;</div>
<p>If &sigma;(x*) exceeds 1.80 kg/m&sup2;, the system rejects the input as an epistemic anomaly.</p>

<h3>E. Monotonic Gradient Boosting (XGBoost)</h3>
<p>Extreme Gradient Boosting (XGBoost) [11] is an optimized distributed gradient boosted decision tree algorithm [18]. Standard greedy tree construction can produce erratic step functions when handling collinear predictors. In clinical medicine, risk must adhere to physiological gradients (e.g., escalating age, BMI, or body fat cannot biologically decrease diabetes risk). XGBoost allows the enforcement of monotonicity constraints (c<sub>j</sub> = +1) on Age, BMI, Body Fat %, and Waist Circumference, while biological sex remains unconstrained (c<sub>j</sub> = 0).</p>

<h3>F. Platt Scaling & Probability Calibration</h3>
<p>Uncalibrated machine learning models output arbitrary ordinal scores. In epidemiological triage, output scores should represent true posterior probabilities. Platt Scaling [12] fits a sigmoid logistic function over validation margins z using maximum likelihood log-loss estimation: P(Y=1|z) = 1 / [1 + exp(A&middot;z + B)]. Calibration quality is commonly assessed with the Brier score [13]; calibrated posterior probabilities naturally reflect the empirical prevalence of the training cohort (~10.3% for diabetes), resulting in compressed probability ranges where standard 0.50 cutoffs fail.</p>

<h3>G. Decision Theory & Threshold Optimization</h3>
<p>In clinical disease screening, classification error costs are heavily asymmetric. The health cost of a False Negative (delayed diagnosis) exceeds that of a False Positive (requiring a low-cost blood or blood pressure re-test) [15]. Decision theory dictates adjusting classification thresholds &tau; below 0.50 [14]. We evaluate Youden's J Statistic (J(&tau;) = Sensitivity + Specificity - 1) and the F<sub>&beta;</sub> measure (&beta;=2, weighting Recall twice as heavily as Precision) to optimize screening sensitivity.</p>

<h2>III. METHODOLOGY & SYSTEM ARCHITECTURE</h2>
<h3>A. Process Overview & Cascading Architecture</h3>
<p>The Face2Health framework executes across three sequential, modular stages:</p>
<p>1) Stage 1 (Computer Vision & Morphometry): A frontal portrait is preprocessed via MediaPipe Face Mesh. If face pose exceeds &plusmn;15&deg; in pitch, yaw, or roll, the Face Guard module rejects the frame. Validated faces are processed by ViT-H/14 (fine-tuned on VisualBMI and Chicago Face Database). MC Dropout (25 passes) produces estimated BMI &mu; and epistemic uncertainty &sigma;. Simultaneously, landmark coordinates yield LFWR, CJWR, PAR, and FWHR ratios as supplementary morphometric features.</p>
<p>2) Stage 1.5 (Tabular DEXA Regression): Estimated BMI, age, biological sex (male=1, female=0), and physical activity level are passed to an XGBoost regressor trained on NHANES DEXA scans (N=2,477 training split using 5-fold out-of-fold predictions and hyperparameter optimization strategies [20]), achieving R&sup2; = 0.74 and MAE = 3.42% body fat for the primary non-invasive route.</p>
<p>3) Stage 2 (Monotonic Calibrated Risk Classification): Tabular features [Age, Sex, Predicted BMI, Predicted Body Fat, (Optional Waist)] are evaluated by monotonically regularized XGBoost classifiers calibrated via 5-fold internal Platt scaling [17] for physician-diagnosed Diabetes and Hypertension.</p>

<!-- TABLE I -->
<div class="table-container">
  <div class="table-caption">TABLE I. CDC NHANES ADULT COHORT PARTICIPANT FLOW & PARTITIONING (AGED 18–59 YEARS)</div>
  <table class="paper-table" data-w="26.8,34,12,12.4,11.2">
    <tr>
      <th>Filtering Step</th>
      <th>Criteria / Description</th>
      <th>Remaining (N)</th>
      <th>Positive Cases</th>
      <th>Base Prev.</th>
    </tr>
    <tr><td>Initial NHANES Cohort (2011–2018)</td><td>Continuous 4-cycle survey participants</td><td>39,156</td><td>—</td><td>—</td></tr>
    <tr><td>Age Filter (Adult Cohort)</td><td>Participants aged 18–59 years (DXA eligible)</td><td>14,210</td><td>—</td><td>—</td></tr>
    <tr><td>DEXA Examination Protocol</td><td>Valid whole-body DXA scan (DXDTOPF), &lt;450 lbs, &lt;6'5"</td><td>5,120</td><td>—</td><td>—</td></tr>
    <tr><td>Diabetes Cohort (DIQ010 Target)</td><td>Complete diagnostic questionnaire responses</td><td><b>3,540</b></td><td><b>366</b></td><td><b>10.34%</b></td></tr>
    <tr><td>— Train Split (70%)</td><td>Out-of-fold training partition</td><td>2,477</td><td>256</td><td>10.34%</td></tr>
    <tr><td>— Validation Split (15%)</td><td>Threshold optimization partition (&tau;)</td><td>532</td><td>55</td><td>10.34%</td></tr>
    <tr><td>— Held-Out Test Split (15%)</td><td>Unobserved final evaluation partition</td><td>531</td><td>55</td><td>10.36%</td></tr>
    <tr><td>Hypertension Cohort (BPQ020 Target)</td><td>Complete diagnostic questionnaire responses</td><td><b>3,554</b></td><td><b>1,207</b></td><td><b>33.96%</b></td></tr>
    <tr><td>— Train Split (70%)</td><td>Out-of-fold training partition</td><td>2,488</td><td>845</td><td>33.96%</td></tr>
    <tr><td>— Validation Split (15%)</td><td>Threshold optimization partition (&tau;)</td><td>533</td><td>181</td><td>33.96%</td></tr>
    <tr><td>— Held-Out Test Split (15%)</td><td>Unobserved final evaluation partition</td><td>533</td><td>181</td><td>33.96%</td></tr>
  </table>
</div>

<h3>B. Decoupled Evaluation & Out-of-Fold Protocol</h3>
<p>To ensure academic rigor, Stage 1.5 DEXA regressors were trained using 5-fold cross-validation on the 70% training split. Out-of-fold predictions were generated for training Stage 2 classifiers, eliminating intra-cohort data leakage. During Stage 2 test evaluation, Gaussian noise matching empirical ViT estimation error (&sigma; = 2.14 kg/m&sup2;) was injected into test BMI inputs to simulate real-world computer vision propagation.</p>

<h3>C. Sensitivity and Error Propagation Analysis</h3>
<p>Let predicted body fat &theta; be a function of estimated BMI b, age a, sex s, and waist w: &theta; = f(b, a, s, w). Empirical perturbation analysis on the test cohort indicates that when waist circumference is present, waist features absorb split variance (68.42% feature split importance), yielding an empirical error attenuation coefficient of &alpha; = 0.39 (&Delta;BF &approx; 0.39 &middot; &Delta;BMI). In the primary non-invasive model (without waist), biological sex commands 74.99% feature split importance, while vision estimation error propagates directly with &alpha; = 1.00 (&Delta;BF &approx; 1.00 &middot; &Delta;BMI).</p>

<h2>IV. EXPERIMENTAL RESULTS AND COMPARATIVE BENCHMARKS</h2>
<h3>A. Stage 1 ViT & Stage 1.5 DEXA Regression Performance</h3>
<p>Table II summarizes performance metrics across Stage 1 ViT vision estimation and Stage 1.5 DEXA body fat regression models evaluated on held-out test splits.</p>

<!-- TABLE II -->
<div class="table-container">
  <div class="table-caption">TABLE II. STAGE 1 VISION AND STAGE 1.5 DEXA REGRESSION BENCHMARKS</div>
  <table class="paper-table" data-w="25.4,10.4,23.6,10.4,10.4,4.4,10.6">
    <tr>
      <th>Model</th>
      <th>Target</th>
      <th>Input Features</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>R&sup2;</th>
      <th>Rejection</th>
    </tr>
    <tr><td>Stage 1 ViT-H/14 + MC Dropout</td><td>BMI (kg/m&sup2;)</td><td>2D Facial Image</td><td>2.14 kg/m&sup2;</td><td>2.82 kg/m&sup2;</td><td>0.68</td><td>4.2%<br>(&sigma;&gt;1.8)</td></tr>
    <tr><td>Stage 1.5 DEXA Regressor (Primary)</td><td>Body Fat %</td><td>BMI, Age, Sex, Physical Activity</td><td>3.42%</td><td>4.25%</td><td>0.74</td><td>0.0%</td></tr>
    <tr><td>Stage 1.5 DEXA Regressor (Auxiliary)</td><td>Body Fat %</td><td>BMI, Age, Sex, Waist</td><td>2.85%</td><td>3.62%</td><td>0.82</td><td>0.0%</td></tr>
  </table>
</div>

<!-- PAGE / COLUMN BREAK MARKER BEFORE IV.B FOR CLEAN COLUMN BALANCING -->
<div style="break-before: column; page-break-before: always;"></div>

<h3>B. Stage 2 Classification Benchmarks & Baseline Comparisons</h3>
<p>Evaluating the baseline Stage 2 diabetes classifier on the held-out test set (N=531) revealed predicted probabilities bounded by max(p) = 0.3776 due to low disease prevalence (~10.3%) and moderate discriminative capability (AUC = 0.762). Because no individual crossed the default 0.50 cutoff, the default model exhibited 0.00% sensitivity (55/55 false negatives). Shifting to the validation-optimized F2 screening threshold (&tau; = 0.060) on the primary non-invasive model (No Waist) achieved <b>90.91% sensitivity</b> (50/55 positive cases detected, 95% CI: 80.0%–96.7%). When waist circumference is optionally included, the F2 threshold (&tau; = 0.061) achieved <b>89.09% sensitivity</b> (49/55 cases detected, 95% CI: 77.8%–95.9%).</p>

<!-- TABLE III-A -->
<div class="table-container">
  <div class="table-caption">TABLE III-A. DIABETES MELLITUS CLASSIFICATION BENCHMARK (N=531, HELD-OUT TEST; VALUES IN PARENTHESES INDICATE 95% CI)</div>
  <table class="paper-table dense-table" data-w="18.6,11.2,9,14.1,14.1,14.1,14.1">
    <tr>
      <th>Model</th>
      <th>Rule</th>
      <th>Cutoff (&tau;)</th>
      <th>AUC</th>
      <th>Sens.</th>
      <th>Spec.</th>
      <th>Prec.</th>
    </tr>
    <tr><td>Logistic Reg.</td><td>Default</td><td>0.500</td><td>0.742<br>(0.678–0.806)</td><td>0.0%<br>(0.0–6.5%)</td><td>100.0%<br>(99.2–100%)</td><td>0.0%</td></tr>
    <tr><td>Logistic Reg.</td><td>Opt. (F2)</td><td>0.104</td><td>0.742<br>(0.678–0.806)</td><td>74.5%<br>(61.0–85.3%)</td><td>61.3%<br>(56.8–65.7%)</td><td>18.2%<br>(13.7–23.5%)</td></tr>
    <tr><td>XGBoost</td><td>Default</td><td>0.500</td><td>0.747<br>(0.683–0.811)</td><td>0.0%<br>(0.0–6.5%)</td><td>100.0%<br>(99.2–100%)</td><td>0.0%</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>Default</td><td>0.500</td><td>0.762<br>(0.701–0.823)</td><td>0.0%<br>(0.0–6.5%)</td><td>100.0%<br>(99.2–100%)</td><td>0.0%</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>Youden's J</td><td>0.082</td><td>0.762<br>(0.701–0.823)</td><td>78.2%<br>(65.0–88.2%)</td><td>59.0%<br>(54.5–63.5%)</td><td>18.1%<br>(13.4–23.6%)</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td><b>Opt. (F2)</b></td><td><b>0.060</b></td><td><b>0.762<br>(0.701–0.823)</b></td><td><b>90.9%<br>(80.0–96.7%)</b></td><td><b>48.7%<br>(44.1–53.3%)</b></td><td><b>17.1%<br>(13.0–21.9%)</b></td></tr>
    <tr><td>Auxiliary With Waist</td><td>Default</td><td>0.500</td><td>0.763<br>(0.702–0.824)</td><td>0.0%<br>(0.0–6.5%)</td><td>100.0%<br>(99.2–100%)</td><td>0.0%</td></tr>
    <tr><td>Auxiliary With Waist</td><td>Youden's J</td><td>0.082</td><td>0.763<br>(0.702–0.824)</td><td>78.2%<br>(65.0–88.2%)</td><td>58.8%<br>(54.3–63.3%)</td><td>18.0%<br>(13.3–23.5%)</td></tr>
    <tr><td>Auxiliary With Waist</td><td>Opt. (F2)</td><td>0.061</td><td>0.763<br>(0.702–0.824)</td><td>89.1%<br>(77.8–95.9%)</td><td>49.0%<br>(44.4–53.6%)</td><td>16.8%<br>(12.7–21.6%)</td></tr>
  </table>
</div>

<!-- TABLE III-B -->
<div class="table-container">
  <div class="table-caption">TABLE III-B. ESSENTIAL HYPERTENSION CLASSIFICATION BENCHMARK (N=533, HELD-OUT TEST; VALUES IN PARENTHESES INDICATE 95% CI)</div>
  <table class="paper-table dense-table" data-w="18.6,11.2,9,14.1,14.1,14.1,14.1">
    <tr>
      <th>Model Architecture</th>
      <th>Rule</th>
      <th>Cutoff (&tau;)</th>
      <th>AUC</th>
      <th>Sens.</th>
      <th>Spec.</th>
      <th>Prec.</th>
    </tr>
    <tr><td>Logistic Reg.</td><td>Default</td><td>0.500</td><td>0.740<br>(0.697–0.783)</td><td>48.1%<br>(40.6–55.6%)</td><td>83.5%<br>(79.2–87.2%)</td><td>60.0%<br>(51.9–67.7%)</td></tr>
    <tr><td>Logistic Reg.</td><td>Opt. (F2)</td><td>0.280</td><td>0.740<br>(0.697–0.783)</td><td>68.0%<br>(60.6–74.7%)</td><td>65.6%<br>(60.4–70.6%)</td><td>50.4%<br>(43.8–57.0%)</td></tr>
    <tr><td>Unconstrained XGB</td><td>Default</td><td>0.500</td><td>0.750<br>(0.708–0.792)</td><td>49.2%<br>(41.7–56.7%)</td><td>82.1%<br>(77.7–85.9%)</td><td>58.6%<br>(50.5–66.3%)</td></tr>
    <tr><td><b>Primary No Waist (Ours)</b></td><td>Default</td><td>0.500</td><td>0.757<br>(0.715–0.799)</td><td>50.3%<br>(42.8–57.8%)</td><td>82.7%<br>(78.3–86.5%)</td><td>59.9%<br>(51.8–67.5%)</td></tr>
    <tr><td><b>Primary No Waist (Ours)</b></td><td>Youden's J</td><td>0.339</td><td>0.757<br>(0.715–0.799)</td><td>71.3%<br>(64.1–77.7%)</td><td>67.0%<br>(61.8–72.0%)</td><td>52.6%<br>(46.2–59.0%)</td></tr>
    <tr><td><b>Primary No Waist (Ours)</b></td><td><b>Opt. (F2)</b></td><td><b>0.155</b></td><td><b>0.757<br>(0.715–0.799)</b></td><td><b>90.6%<br>(85.3–94.4%)</b></td><td><b>40.9%<br>(35.7–46.3%)</b></td><td><b>44.1%<br>(39.0–49.3%)</b></td></tr>
    <tr><td>Auxiliary With Waist</td><td>Default</td><td>0.500</td><td>0.755<br>(0.713–0.797)</td><td>49.7%<br>(42.2–57.2%)</td><td>82.4%<br>(78.0–86.2%)</td><td>59.2%<br>(51.1–66.9%)</td></tr>
    <tr><td>Auxiliary With Waist</td><td>Opt. (F2)</td><td>0.158</td><td>0.755<br>(0.713–0.797)</td><td>88.4%<br>(82.8–92.6%)</td><td>41.8%<br>(36.6–47.1%)</td><td>44.0%<br>(38.8–49.2%)</td></tr>
  </table>
</div>

<h3>C. Component Ablation Study</h3>
<p>To quantify individual contribution across pipeline components, systematic ablation experiments were executed on the Diabetes test cohort (N=531), summarizing AUC, Sensitivity, Specificity, Brier score, and Monotonicity violations in Table IV.</p>

<!-- TABLE IV -->
<div class="table-container">
  <div class="table-caption">TABLE IV. COMPONENT ABLATION STUDY ON DIABETES TEST COHORT (N=531)</div>
  <table class="paper-table" data-w="36.3,9.4,12.4,12.4,10.4,15">
    <tr>
      <th>Configuration</th>
      <th>AUC</th>
      <th>Sens (F2)</th>
      <th>Spec (F2)</th>
      <th>Brier</th>
      <th>Violations</th>
    </tr>
    <tr><td><b>Full Face2Health Pipeline (Ours)</b></td><td><b>0.762</b></td><td><b>90.9%</b></td><td>48.7%</td><td><b>0.078</b></td><td><b>0 (Guaranteed)</b></td></tr>
    <tr><td>No Body Fat</td><td>0.748</td><td>83.6%</td><td>51.2%</td><td>0.085</td><td>0 Violations</td></tr>
    <tr><td>No Monotonicity</td><td>0.747</td><td>85.5%</td><td>50.0%</td><td>0.084</td><td>14 Violations</td></tr>
    <tr><td>No Calibration</td><td>0.747</td><td>80.0%</td><td>52.5%</td><td>0.142</td><td>14 Violations</td></tr>
  </table>
</div>

<!-- SECTION IV.D + TABLE V IN A SINGLE NON-SPLITTING CONTAINER -->
<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <h3>D. Body Fat Error Attenuation Dynamics</h3>
  <p>Table V evaluates total body fat shifts under systematic BMI perturbation sweeps (&Delta;BMI &isin; [-2, +2] kg/m&sup2;), demonstrating the variance shock absorber effect under auxiliary waist features.</p>

  <div class="table-caption">TABLE V. BODY FAT SHIFT UNDER BMI PERTURBATION SWEEPS</div>
  <table class="paper-table" data-w="17.4,25.4,17.4,36.9">
    <tr>
      <th>Injected &Delta;BMI</th>
      <th>Mean &Delta;BF Waist</th>
      <th>Quotient (&alpha;)</th>
      <th>Mean &Delta;BF No Waist</th>
    </tr>
    <tr><td>-2.0 kg/m&sup2;</td><td>-0.80 &plusmn; 0.80%</td><td>0.40</td><td>-2.02 &plusmn; 1.24%</td></tr>
    <tr><td>-1.0 kg/m&sup2;</td><td>-0.37 &plusmn; 0.50%</td><td>0.37</td><td>-1.01 &plusmn; 0.82%</td></tr>
    <tr><td>0.0 kg/m&sup2;</td><td>0.00 &plusmn; 0.00%</td><td>—</td><td>0.00 &plusmn; 0.00%</td></tr>
    <tr><td>+1.0 kg/m&sup2;</td><td>+0.39 &plusmn; 0.54%</td><td>0.39</td><td>+1.00 &plusmn; 0.83%</td></tr>
    <tr><td>+2.0 kg/m&sup2;</td><td>+0.77 &plusmn; 0.72%</td><td>0.38</td><td>+1.98 &plusmn; 1.14%</td></tr>
  </table>
</div>

<h2>V. CLINICAL TRIAGE PROTOCOL & RISK SENSITIVITY ANALYSIS</h2>
<p>The calibrated posterior probabilities are operationalized into four clinical triage strata configured in weights/thresholds.json. Table VI details population breakdown and observed true disease prevalence across held-out test splits.</p>

<!-- TABLE VI -->
<div class="table-container">
  <div class="table-caption">TABLE VI. EPIDEMIOLOGICALLY ALIGNED 4-TIER CLINICAL TRIAGE STRATA METRICS</div>
  <table class="paper-table dense-table" data-w="12.4,18.3,11.4,11.4,12.4,12.4,17">
    <tr>
      <th>Triage Tier</th>
      <th>Probability Range</th>
      <th>Diabetes N</th>
      <th>Diabetes Prev.</th>
      <th>Hypertension Count N</th>
      <th>Hypertension Prev.</th>
      <th>Action Protocol</th>
    </tr>
    <tr>
      <td>Low Risk</td>
      <td>p &lt; 6.0% (Dia)<br>p &lt; 15.5% (HTN)</td>
      <td>232<br>(43.7%)</td>
      <td>2.15%<br>(5/232)</td>
      <td>161<br>(30.2%)</td>
      <td>10.56%<br>(17/161)</td>
      <td>Annual wellness checkup</td>
    </tr>
    <tr>
      <td>Watchful</td>
      <td>6.0% &le; p &lt; 10.0%<br>15.5% &le; p &lt; 25.0%</td>
      <td>132<br>(24.9%)</td>
      <td>7.58%<br>(10/132)</td>
      <td>122<br>(22.9%)</td>
      <td>27.05%<br>(33/122)</td>
      <td>Lifestyle counseling &amp; re-test</td>
    </tr>
    <tr>
      <td>Screen Positive</td>
      <td>10.0% &le; p &lt; 20.0%<br>25.0% &le; p &lt; 50.0%</td>
      <td>121<br>(22.8%)</td>
      <td>18.18%<br>(22/121)</td>
      <td>150<br>(28.1%)</td>
      <td>46.67%<br>(70/150)</td>
      <td>Actionable phlebotomy / BP trigger</td>
    </tr>
    <tr>
      <td>Urgent</td>
      <td>p &ge; 20.0% (Dia)<br>p &ge; 50.0% (HTN)</td>
      <td>46<br>(8.7%)</td>
      <td><b>39.13%<br>(18/46)</b></td>
      <td>100<br>(18.8%)</td>
      <td><b>61.00%<br>(61/100)</b></td>
      <td>Priority medical referral</td>
    </tr>
  </table>
</div>

<h2>VI. LIMITATIONS, PRIVACY, AND ETHICAL CONSIDERATIONS</h2>
<p>Face2Health is strictly an opportunistic triage tool, not a diagnostic instrument. Key limitations include: (1) target definitions rely on self-reported physician diagnoses (DIQ010/BPQ020) rather than active prospective laboratory testing; (2) potential domain shift between the US NHANES multi-ethnic training cohort and Asian/Southeast Asian populations regarding BMI-body fat relationships; (3) sensitivity to severe lighting extremes and head poses exceeding &plusmn;15&deg; (intercepted by Face Guard); and (4) decoupled pipeline validation due to the absence of paired facial photographs in NHANES. Biometric data handling is designed to comply with PDPA and GDPR guidelines, executing feature extraction locally on-device without storing raw facial images.</p>

<h2>VII. CONCLUSION & FUTURE WORK</h2>
<p>This study evaluated <b>Face2Health</b>, demonstrating that passive computer vision can be integrated into a calibrated triage instrument. By optimizing decision thresholds on validation partitions, the primary non-invasive pipeline (without waist) achieved 90.91% sensitivity for Diabetes and 90.61% for Hypertension on held-out NHANES test data. Future work focuses on: (1) prospective clinical validation in Southeast Asian outpatient clinics; (2) integrating remote photoplethysmography (rPPG) for optical pulse wave velocity estimation; and (3) deploying lightweight ONNX models for offline mobile triage.</p>

<h2>REFERENCES</h2>
<div class="ref-item">[1] World Health Organization, "Global status report on noncommunicable diseases 2020," Geneva, Switzerland: WHO, 2020.</div>
<div class="ref-item">[2] International Diabetes Federation, <i>IDF Diabetes Atlas</i>, 10th ed. Brussels, Belgium: IDF, 2021.</div>
<div class="ref-item">[3] V. Coetzee, D. I. Perrett, and I. D. Stephen, "Facial adiposity: A reliable cue to health?" <i>Perception</i>, vol. 38, no. 11, pp. 1700–1711, 2009.</div>
<div class="ref-item">[4] L. Wen and G. Guo, "A computational approach to body mass index prediction from face images," <i>Image and Vision Computing</i>, vol. 31, no. 5, pp. 392–400, 2013.</div>
<div class="ref-item">[5] B. J. Lee and J. Y. Kim, "Predicting visceral obesity based on facial characteristics," <i>BMC Complement. Altern. Med.</i>, vol. 14, p. 248, 2014.</div>
<div class="ref-item">[6] S. Kocabey, M. Cambronero, Y. Aslan, A. Birbeck, and A. Veit, "Face-to-BMI: Using computer vision to estimate body mass index on social media," in <i>Proc. 11th Int. AAAI Conf. Web and Social Media (ICWSM)</i>, 2017, pp. 572–575.</div>
<div class="ref-item">[7] I. D. Stephen et al., "Facial shape cues to health in Caucasian, African, and Asian populations," <i>PLoS ONE</i>, vol. 12, no. 7, e0180887, 2017.</div>
<div class="ref-item">[8] L. Jiang, G. Guo, and G. Mu, "Deep facial adiposity estimation and its application in health risk prediction," <i>Comput. Vis. Image Underst.</i>, vol. 197–198, p. 102985, 2020.</div>
<div class="ref-item">[9] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2021.</div>
<div class="ref-item">[10] Y. Gal and Z. Ghahramani, "Dropout as a bayesian approximation: Representing model uncertainty in deep learning," in <i>Proc. Int. Conf. Mach. Learn. (ICML)</i>, 2016, pp. 1050–1059.</div>
<div class="ref-item">[11] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.</i>, 2016, pp. 785–794.</div>
<div class="ref-item">[12] J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," <i>Adv. Large Margin Classif.</i>, vol. 10, no. 3, pp. 61–74, 1999.</div>
<div class="ref-item">[13] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in <i>Proc. 22nd Int. Conf. Mach. Learn. (ICML)</i>, 2005, pp. 625–632.</div>
<div class="ref-item">[14] W. J. Youden, "Index for rating diagnostic tests," <i>Cancer</i>, vol. 3, no. 1, pp. 32–35, 1950.</div>
<div class="ref-item">[15] C. Elkan, "The foundations of cost-sensitive learning," in <i>Proc. 17th Int. Joint Conf. Artif. Intell. (IJCAI)</i>, 2001, pp. 973–978.</div>
<div class="ref-item">[16] Centers for Disease Control and Prevention (CDC), "National Health and Nutrition Examination Survey (NHANES) Examination & Laboratory Protocols," U.S. Dept. of Health & Human Services, 2020.</div>
<div class="ref-item">[17] A. Géron, <i>Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow</i>, 2nd ed. Sebastopol, CA: O'Reilly Media, 2019.</div>
<div class="ref-item">[18] Z.-H. Zhou, <i>Ensemble Methods: Foundations and Algorithms</i>. Boca Raton, FL: CRC Press, 2012.</div>
<div class="ref-item">[19] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2016, pp. 770–778.</div>
<div class="ref-item">[20] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," <i>J. Mach. Learn. Res.</i>, vol. 13, pp. 281–305, 2012.</div>
"""

    processed_body_html = prepare_tables(BODY_HTML)

    raw_pdf_path = os.path.join(RESULTS_DIR, "_raw_story_output.pdf")
    archive = pymupdf.Archive(RESULTS_DIR)
    writer = pymupdf.DocumentWriter(raw_pdf_path)

    page_w, page_h = 612, 792
    margin_l = 45.4
    margin_t = 54
    margin_b = 54
    col_w = 254
    col_gap = 16
    margin_r = margin_l + col_w + col_gap + col_w
    header_h = 150
    top_y = margin_t + header_h + 6

    header_story = pymupdf.Story(html=HEADER_HTML)
    body_story = pymupdf.Story(html=processed_body_html, archive=archive)

    page_idx = 1
    max_pages = 8

    while page_idx <= max_pages:
        device = writer.begin_page(pymupdf.paper_rect("letter"))
        if page_idx == 1:
            header_rect = pymupdf.Rect(margin_l, margin_t, margin_r, margin_t + header_h)
            header_story.place(header_rect)
            header_story.draw(device)

            col1 = pymupdf.Rect(margin_l, top_y, margin_l + col_w, page_h - margin_b)
            col2 = pymupdf.Rect(margin_l + col_w + col_gap, top_y, margin_r, page_h - margin_b)
        else:
            col1 = pymupdf.Rect(margin_l, margin_t, margin_l + col_w, page_h - margin_b)
            col2 = pymupdf.Rect(margin_l + col_w + col_gap, margin_t, margin_r, page_h - margin_b)

        filled1, _ = body_story.place(col1)
        body_story.draw(device)

        if filled1 != 0:
            filled2, _ = body_story.place(col2)
            body_story.draw(device)
            has_more = (filled2 != 0)
        else:
            has_more = False

        writer.end_page()
        if not has_more:
            break
        page_idx += 1

    writer.close()
    del writer

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
        print(f"[+] Successfully compiled Restored Clean PDF: {PDF_OUT} ({page_idx} pages, {pdf_size:,} bytes)")
    except Exception:
        alt_out = os.path.join(REPORTS_DIR, "Paper_updated.pdf")
        doc_post.save(alt_out, deflate=True)
        doc_post.close()
        del doc_post
        if os.path.exists(raw_pdf_path):
            try:
                os.remove(raw_pdf_path)
            except Exception:
                pass
        pdf_size = os.path.getsize(alt_out)
        print(f"[!] Warning: ไม่สามารถบันทึกทับ '{PDF_OUT}' ได้เนื่องจากไฟล์ถูกเปิดค้างไว้อยู่")
        print(f"[+] บันทึกผลลัพธ์ใหม่ไว้ที่: '{alt_out}' ({page_idx} pages, {pdf_size:,} bytes) แทนเรียบร้อยแล้วครับ")


if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=============================================================================")
    print(" Compiling Restored High-Quality Academic Paper                              ")
    print("=============================================================================")
    build_ieee_pdf()
    print("\n[OK] Single production PDF compiled successfully.")