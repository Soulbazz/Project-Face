"""
=============================================================================
Face2Health - Extended Academic Research Paper Pipeline (Full Paper Edition)
=============================================================================
Peer Review & Quality Enhancements:
  1. Multi-Page Full Academic Conference Paper (max_pages = 10 support).
  2. Integration of 300-DPI visual assets into 254pt column format:
     - Figure 1: End-to-end Cascading Pipeline Architecture (fig0_pipeline_architecture.png)
     - Figure 2: Stage 1.5 DEXA Scatter Plot (fig1_stage15_scatter.png)
     - Figure 3: Stage 2 ROC Curves Comparison (fig2_roc_curves.png)
     - Figure 4: Stage 2 Calibration Reliability Diagram (fig3_calibration.png)
     - Figure 5: Stage 2 Confusion Matrix Heatmaps at F2 Decision Cutoffs (fig4_confusion_matrix.png)
  3. Strict Text & Mathematical Consistency:
     - Clarified MAE 2.14 kg/m2 (standardized clinical subset) vs 3.26 kg/m2 (unconstrained N=793)
     - Unified MC Dropout to T = 25 passes across text, abstract, and formulas
     - Randomized hyperparameter search [20] terminology
     - Couplings between epistemic input screening and downstream calibrated risk
     - Clarified N = 3,540 Primary Screening vs N = 364 Fasting Sub-sample Complete-Case (26 T2DM cases)
     - Zero-mean Gaussian perturbation scaled to nominal vision MAE (sigma_sim = 2.14 kg/m2)
     - Table IV: Row 4 labeled 'No Monot. & Calib.' with cumulative ablation description
     - Restored essential sentences and Limitation (4) Decoupled pipeline evaluation in Section VI
  4. Verified academic citations (numbered by first appearance, IEEE order):
     - [6] He et al. (ResNet, CVPR 2016)
     - [7] Kocabey et al. (ICWSM 2017)
     - [8] Stephen, Hiew, Coetzee, Tiddeman, and Perrett (Frontiers in Psychology 2017)
     - [9] M. Jiang, G. Guo, and G. Mu (CVIU 2020)
=============================================================================
"""

import os
import io
import sys
import re
import shutil
import pymupdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Ensure figures are regenerated if missing
figures = [
    os.path.join(RESULTS_DIR, "fig0_pipeline_architecture.png"),
    os.path.join(RESULTS_DIR, "fig1_stage15_scatter.png"),
    os.path.join(RESULTS_DIR, "fig2_roc_curves.png"),
    os.path.join(RESULTS_DIR, "fig3_calibration.png"),
    os.path.join(RESULTS_DIR, "fig4_confusion_matrix.png"),
]
if any(not os.path.exists(f) for f in figures):
    try:
        from regenerate_figures_fix import (
            generate_fig0_pipeline,
            generate_fig1_scatter,
            generate_fig2_roc,
            generate_fig3_calibration,
            generate_fig4_confusion_matrices,
        )
        generate_fig0_pipeline()
        generate_fig1_scatter()
        generate_fig2_roc()
        generate_fig3_calibration()
        generate_fig4_confusion_matrices()
    except Exception as e:
        print(f"[!] Note: Automatic figure generation call: {e}")

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
    print("[*] Compiling Extended High-Quality Academic Paper PDF...")

    HEADER_HTML = """
<style>
.header-box { text-align: center; font-family: 'Times New Roman', serif; margin-bottom: 0px; }
.title { font-family: 'Times New Roman', Times, serif; font-size: 20pt; font-style: normal; font-weight: normal; line-height: 22.5pt; margin-bottom: 6pt; text-align: center; }
.authors-table { width: 100%; border-collapse: collapse; margin-bottom: 0px; border: none; }
.authors-table td { text-align: center; vertical-align: top; font-size: 9.5pt; line-height: 11.2pt; border: none; padding: 0 5px; width: 50%; }
.author-name { font-size: 11pt; font-weight: normal; margin-bottom: 1px; color: #000000; text-align: center; }
.author-inst { font-size: 9.5pt; font-style: italic; line-height: 11.2pt; color: #000000; text-align: center; }
.author-loc { font-size: 9.5pt; font-style: normal; line-height: 11.2pt; color: #000000; text-align: center; }
.author-mail { font-size: 9.5pt; font-style: normal; line-height: 11.2pt; margin-top: 0px; color: #000000; text-align: center; }
.author-affil { font-size: 9.5pt; line-height: 11.2pt; text-align: center; }
</style>
<div class="header-box">
  <div class="title">Face2Health: Diabetes and Hypertension Risk Screening Using Vision Transformers and Gradient-Boosted Trees</div>
  <table class="authors-table">
    <tr>
      <td>
        <div class="author-name">Phalathip Hemvut</div>
        <div class="author-inst">School of Information Technology</div>
        <div class="author-inst">King Mongkut's Institute of Technology Ladkrabang</div>
        <div class="author-loc">Bangkok, Thailand</div>
        <div class="author-mail">67070255@kmitl.ac.th</div>
      </td>
      <td>
        <div class="author-name">Eua-Unggul Chaivivatporn</div>
        <div class="author-inst">School of Information Technology</div>
        <div class="author-inst">King Mongkut's Institute of Technology Ladkrabang</div>
        <div class="author-loc">Bangkok, Thailand</div>
        <div class="author-mail">67070302@kmitl.ac.th</div>
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
    font-family: 'Times New Roman', serif; font-size: 9.1pt;
    line-height: 1.00; text-align: justify; color: #000000;
    background-color: transparent;
}
.abstract-box {
    margin-bottom: 5px; font-size: 9.0pt; text-align: justify;
    line-height: 1.08; font-weight: bold; text-indent: 12pt;
    border: none; padding: 0;
}
.abstract-title { font-weight: bold; font-style: italic; }
.keywords {
    font-size: 9.0pt; margin-bottom: 6px; line-height: 1.08;
    font-weight: bold; font-style: normal; text-align: justify; text-indent: 12pt;
}
.keywords-title { font-weight: bold; font-style: italic; }
h2 {
    font-size: 10pt; font-weight: normal; text-align: center;
    text-transform: uppercase; margin: 8pt 0 4pt 0; letter-spacing: 0.3px;
    break-after: avoid; page-break-after: avoid;
    background-color: transparent;
}
h3 {
    font-size: 10pt; font-weight: normal; font-style: italic;
    text-align: left; margin: 5pt 0 2pt 0; break-after: avoid; page-break-after: avoid;
    background-color: transparent;
}
p { margin: 0 0 4.5pt 0; text-indent: 12pt; line-height: 11.5pt; background-color: transparent; text-align: justify; }

.table-container {
    break-inside: avoid;
    page-break-inside: avoid;
    margin: 2px 0;
    width: 100%;
}
.table-caption {
    break-inside: avoid;
    page-break-inside: avoid;
    break-after: avoid;
    page-break-after: avoid;
    font-size: 7.5pt; font-weight: bold; text-align: center;
    margin: 0 0 1.5px 0; text-transform: uppercase;
}
.table-note {
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 7.2pt;
    text-align: left;
    margin-top: 1.5px;
    line-height: 1.05;
    color: #000000;
}
table.paper-table tr {
    break-inside: avoid;
    page-break-inside: avoid;
}
table.paper-table {
    break-inside: avoid;
    page-break-inside: avoid;
    width: 254pt;
    margin: 3px 0;
    border-collapse: collapse;
    font-size: 7.0pt;
    line-height: 1.08;
    text-align: center;
    table-layout: fixed;
}
table.dense-table {
    font-size: 6.4pt;
    line-height: 1.05;
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
img {
    width: 248pt;
    max-width: 248pt;
    height: auto;
    display: block;
    margin: 0 auto;
}
.fig-container {
    break-inside: avoid;
    page-break-inside: avoid;
    text-align: center;
    margin: 3px 0;
    width: 100%;
}
.fig-caption {
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 8pt;
    text-align: justify;
    margin-top: 2px;
    line-height: 1.1;
}
.equation {
    text-align: center; font-style: italic; margin: 3px 0;
    font-size: 8.5pt; break-inside: avoid; page-break-inside: avoid;
}
.ref-item {
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 8pt;
    margin-left: 14pt;
    margin-right: 0pt;
    text-indent: -14pt;
    margin-bottom: 3.5pt;
    line-height: 10.8pt;
    text-align: left;
}
</style>

<div class="abstract-box">
  <span class="abstract-title"><b><i>Abstract</i>&mdash;</b></span>This paper presents Face2Health, a decoupled vision-to-tabular framework for pre-screening Physician-Diagnosed Diabetes Mellitus (DIQ010) and Hypertension (BPQ020) from 2D facial portraits. The pipeline sequentially couples: (1) a Vision Transformer (ViT-H/14) with a frozen backbone and a 5-layer MLP head for Body Mass Index (BMI) estimation, incorporating Monte Carlo Dropout uncertainty filtering; (2) a tabular XGBoost regressor mapping estimated anthropometrics to DEXA Total Body Fat Percentage; and (3) monotonically constrained, Platt-calibrated XGBoost classifiers for binary disease risk stratification. Evaluated on the CDC NHANES cohort under realistic pooled-to-in-the-wild vision estimation noise, facial screening provides statistically significant diagnostic gains over baseline demographics, serving as a zero-hardware fallback when physical scales or reliable self-reported metrics are unavailable.
</div>
<div class="keywords">
  <span class="keywords-title"><b><i>Keywords</i>&mdash;</b></span>Facial Biomarkers, Body Mass Index, Dual-Energy X-ray Absorptiometry (DEXA), Vision Transformers, Monte Carlo Dropout, Monotonic Gradient Boosting, Platt Scaling, Epidemiological Screening.
</div>

<h2>I. INTRODUCTION</h2>
<p>Non-communicable diseases (NCDs)—encompassing cardiovascular conditions, cancers, chronic respiratory diseases, and diabetes—constitute the defining epidemiological crisis of the modern era, accounting for over 41 million deaths annually, or 74% of all mortalities worldwide [1]. Among these, Diabetes Mellitus [2] and Hypertension [1] represent the primary modifiable metabolic drivers of premature morbidity. Beyond mortality, sustained undetected hyperglycemia and arterial hypertension induce progressive microvascular and macrovascular pathology, culminating in diabetic nephropathy, proliferative retinopathy, peripheral neuropathy, ischemic stroke, and coronary heart disease. In middle-income and newly industrialized nations, population health surveys indicate that 30% to 45% of individuals living with diabetes remain undiagnosed until secondary clinical complications necessitate emergency hospitalization [2].</p>

<p>Current gold-standard diagnostic pathways rely on invasive venous phlebotomy (fasting plasma glucose &ge; 126 mg/dL or HbA1c &ge; 6.5%) and clinical sphygmomanometry. While biochemically authoritative, these methods impose structural limitations requiring sterile consumables, certified medical technologists, and centralized laboratory infrastructure. In parallel, clinical anthropometry demonstrates that systemic adiposity and metabolic risk physically manifest in craniofacial soft tissue. Adipocyte hypertrophy within the superficial and deep buccal fat compartments, jowl fat expansion, and submental tissue thickening reflect central lipid deposition. The maturation of Vision Transformers (ViT) with self-attention, alongside gradient boosted decision trees (XGBoost), provides a computational framework capable of translating passive 2D digital portraits into physiological biomarkers.</p>

<p>However, cascading visual-to-tabular pipelines introduce three key algorithmic challenges: (1) multi-stage regression error propagation, where vision estimation noise affects subsequent metabolic models; (2) biological inconsistency, where collinear features lead unconstrained decision trees to form non-monotonic risk surfaces; and (3) probability output compression under class imbalance (~10.3% diabetes prevalence), where standard 0.50 decision thresholds yield 0.0% default clinical sensitivity. This paper formalizes, benchmarks, and resolves these operational dynamics on the CDC NHANES dataset using physician-diagnosed disease targets (DIQ010 and BPQ020).</p>

<h2>II. LITERATURE REVIEW</h2>
<h3>A. Craniofacial Morphometry and Visceral Adiposity</h3>
<p>Craniofacial morphology serves as an external indicator of systemic adiposity due to the distribution of subcutaneous and deep facial fat pads. Perceptual and empirical studies by Coetzee et al. [3] established that perceived facial adiposity serves as a reliable visual cue linked to body mass index and cardiovascular health. L. Wen and G. Guo [4] demonstrated that computer-derived facial features correlate significantly with body mass index and metabolic parameters. Anatomically, lipid accumulation induces lateral and inferior displacement across the mandibular border. To capture these shifts, three scale-invariant geometric ratios—Lower Facial Width-to-Height Ratio (LFWR), Cheek-to-Jaw Width Ratio (CJWR), and Facial Width-to-Height Ratio (FWHR)—alongside a scale-normalized mandibular Perimeter-to-Area descriptor (PAR, computed after normalizing landmark coordinates by inter-ocular distance) are defined. Lee and Kim [5] demonstrated that inter-ear and mandibular width distances serve as significant predictors of visceral and abdominal obesity.</p>

<h3>B. Facial Biomarker Estimation and Deep Learning Benchmarks</h3>
<p>Early facial health estimation relied on deep convolutional architectures [6] and social media datasets. Kocabey et al. [7] introduced <i>Face-to-BMI</i> using public Reddit progress photographs, establishing initial CNN benchmarks (MAE ~2.8 kg/m&sup2;). Stephen et al. [8] demonstrated cross-cultural validity of facial shape cues across Caucasian, African, and Asian cohorts. Jiang, Guo, and Mu [9] advanced facial adiposity estimation by modeling BMI uncertainty through label distribution learning. However, existing vision methods predominantly terminate at anthropometric prediction without coupling epistemic input screening with calibrated downstream chronic disease risk stratification.</p>

<h3>C. Dual-Energy X-ray Absorptiometry (DEXA) and NHANES Cohort</h3>
<p>While Body Mass Index (BMI = weight/height&sup2;) is ubiquitous, it cannot differentiate skeletal muscle mass from adipose tissue. Dual-Energy X-ray Absorptiometry (DEXA) constitutes the diagnostic gold standard for body composition analysis, using photon attenuation to resolve bone mineral, lean soft tissue, and fat mass. The National Health and Nutrition Examination Survey (NHANES) conducted by the CDC represents a multi-ethnic dataset combining whole-body DEXA scans (total body fat percentage DXDTOPF), standardized anthropometry (BMX), laboratory biochemical assays (LBXGLU, LBXGH), and structured diagnostic interviews (DIQ010 for physician-diagnosed diabetes, BPQ020 for physician-diagnosed hypertension) across CDC NHANES survey cycles—primarily the continuous 2003–2004 cycle (Cycle C) for DEXA body composition and diagnostic interviews, alongside the 2017–2018 cycle (Cycle J) for biochemical laboratory validation—for adult participants aged 18–59 years [10].</p>
<p>In the NHANES DEXA examination cohorts, whole-body DXA scans (DXDTOPF) were administered to designated Mobile Examination Center (MEC) participants, contracting the adult cohort aged 18–59 years to participants with complete, non-imputed whole-body scan status (DXAEXSTS = 1, indicating scan completed without major artifacts) alongside standard physical exclusions (pregnancy, scanner weight limits &gt;450 lbs, height &gt;6'5", or recent nuclear medicine or radiographic contrast barium administration within 7 days). Next, enforcing a unified complete-covariate filtering protocol (requiring valid BMI, standing waist circumference, and complete multi-domain Physical Activity Questionnaire [PAQ] sub-items without imputation to enable direct paired head-to-head benchmarking between the primary non-invasive and auxiliary waist models on the identical cohort) excluded 1,492 incomplete records (29.1% of DXA-eligible participants)—specifically, 1,180 participants excluded due to a strict listwise complete-case requirement across all PAQ sub-items without zero-imputation (requiring valid responses across vigorous/moderate occupational and recreational items PAQ605, PAQ620, PAQ650, PAQ665, and sedentary duration PAD680, while excluding refusal/don't-know codes 7 and 9) and 312 participants missing standing waist circumference or measured BMI (BMX; 1,180 + 312 = 1,492)—yielding 3,628 eligible participants (with this complete-case restriction acknowledged as a methodological limitation). Progression from 3,628 to final disease cohorts narrows to N = 3,540 for Diabetes (366 positive cases; 10.34% prevalence) after excluding 88 cases with self-reported borderline status (DIQ010 = 3), and N = 3,554 for Hypertension (1,207 positive cases; 33.96% prevalence) after excluding 74 records with incomplete BPQ020 interview responses (precisely accounting for the 14-participant discrepancy between cohorts: 3,628 - 88 = 3,540 and 3,628 - 74 = 3,554). Crucially, variables restricted to morning fasted sub-samples (contracting samples to N = 2,426) are Fasting Plasma Glucose (LBXGLU) and oral glucose tests, whereas HbA1c (LBXGH) and sphygmomanometry cover the broader MEC cohort. We adopt DIQ010 and BPQ020 as primary screening targets, with composite endpoints incorporating thresholded laboratory HbA1c (&ge; 6.5%) and measured blood pressure (&ge; 140/90 mmHg) serving as parallel secondary validation targets.</p>

<h3>D. Vision Transformers and Epistemic Uncertainty Quantification</h3>
<p>Unlike conventional convolutional residual networks (ResNet) [6] constrained by localized receptive fields, Dosovitskiy et al. [11] demonstrated that Vision Transformers (ViT) model long-range spatial dependencies across image patches through self-attention; we adapt this architecture here to capture structural relationships across distant facial regions. Pre-trained on ImageNet-21k, our ViT-H/14 architecture retains a frozen feature extraction backbone while adapting a specialized regression head. Deterministic neural networks, however, suffer from overconfident mispredictions on out-of-distribution inputs. To address this, Monte Carlo Dropout (Gal &amp; Ghahramani [12]) provides a Bayesian approximation of Gaussian process uncertainty by preserving active dropout masks during inference across the prediction head. Sampling T = 25 forward stochastic passes yields the predictive mean &mu;(x*) and epistemic variance &sigma;&sup2;(x*):</p>
<div class="equation">&mu;(x*) = (1/T) &Sigma;<sub>t=1</sub><sup>T</sup> f<sub>W<sub>t</sub></sub>(x*), &nbsp;&nbsp;&nbsp;&nbsp; &sigma;&sup2;(x*) = [1/(T-1)] &Sigma;<sub>t=1</sub><sup>T</sup> [f<sub>W<sub>t</sub></sub>(x*) - &mu;(x*)]&sup2;</div>
<p>If &sigma;(x*) exceeds 1.80 kg/m&sup2;, the system rejects the input as an epistemic anomaly.</p>

<h3>E. Monotonic Gradient Boosting (XGBoost)</h3>
<p>Extreme Gradient Boosting (XGBoost) [13] is an optimized distributed gradient boosted decision tree algorithm [14]. Standard greedy tree construction can produce erratic step functions when handling collinear predictors. In clinical medicine, risk must adhere to physiological gradients (e.g., escalating age, BMI, or body fat cannot biologically decrease diabetes risk). XGBoost allows the enforcement of monotonicity constraints (c<sub>j</sub> = +1) on Age, BMI, Body Fat %, and Waist Circumference, while biological sex remains unconstrained (c<sub>j</sub> = 0).</p>

<h3>F. Platt Scaling and Probability Calibration</h3>
<p>Uncalibrated machine learning models output arbitrary ordinal scores. In epidemiological triage, output scores should represent true posterior probabilities. Platt Scaling [15] fits a sigmoid logistic function over validation margins z using 5-fold cross-validated maximum likelihood log-loss estimation: P(Y=1|z) = 1 / [1 + exp(A&middot;z + B)]. Calibration quality is assessed with the Brier score [16]; calibrated posterior probabilities naturally reflect the empirical prevalence of the training cohort (~10.3% for diabetes), resulting in compressed probability ranges where standard 0.50 cutoffs fail.</p>

<h3>G. Decision Theory and Threshold Optimization</h3>
<p>In clinical disease screening, classification error costs are heavily asymmetric. Decision theory dictates adjusting classification thresholds &tau; below 0.50 to optimize screening sensitivity, evaluating Youden's J Statistic (J(&tau;) = Sensitivity + Specificity - 1) [17] and the F<sub>&beta;</sub> measure (&beta;=2, weighting Recall twice as heavily as Precision). In our intended pre-screening setting, we assume that missed positive cases (False Negatives) carry greater clinical cost than additional low-cost confirmatory tests (False Positives), motivating cost-sensitive threshold selection below 0.50 [18].</p>

<h2>III. METHODOLOGY</h2>
<h3>A. Process Overview and Cascading Architecture</h3>
<p>The Face2Health framework executes across three sequential, modular stages to screen metabolic conditions in the CDC NHANES cohort [10], integrating standard machine learning toolkits [19] and tree ensemble foundations [14]. Figure 1 depicts the end-to-end cascade, and each stage is detailed below:</p>

<!-- FIGURE 1 (PIPELINE ARCHITECTURE) -->
<div class="fig-container" style="break-inside: avoid; page-break-inside: avoid;">
  <img src="fig0_pipeline_architecture.png" style="width: 248pt; max-width: 248pt; height: 356pt;" alt="Face2Health Cascading Triage Pipeline Architecture">
  <div class="fig-caption">Fig. 1. End-to-end architecture of the Face2Health pre-screening framework.</div>
</div>
<p>1) <b>Stage 1 (Computer Vision and Morphometry):</b> A frontal portrait is preprocessed via MediaPipe Face Mesh. If face pose exceeds &plusmn;15&deg; in pitch, yaw, or roll, or if lighting/blur variances fail boundary checks, the Face Guard rejects the frame. Validated faces are processed by ViT-H/14 configured with a frozen ImageNet-21k pre-trained backbone and a trained 5-layer MLP regression head. The training dataset comprises 3,962 facial images sourced from the curated VisualBMI benchmark dataset [7], partitioned strictly into 2,773 training (70%), 396 validation (10%), and 793 held-out testing (20%) samples partitioned by unique source profile identifiers to prevent known identity overlap across splits (though complete deduplication across anonymous Reddit uploads in VisualBMI cannot be guaranteed). The training split undergoes 5x offline training augmentation (13,865 samples). The frozen ViT-H/14 backbone connects directly to a trainable 5-layer MLP head (1280 &rarr; 640 &rarr; 320 &rarr; 160 &rarr; 80 &rarr; 1, GELU activations, Dropout 0.5) trained with AdamW (learning rate = 3&times;10<sup>-5</sup>, weight decay = 0.01, batch size = 24, AMP mixed precision, early stopping patience = 5 on validation MAE). The held-out test partition (N = 793) comprises n = 412 well-aligned frontal portraits (controlled frontal subset: achieving MAE = 2.14 kg/m&sup2;, RMSE = 2.82 kg/m&sup2;, R&sup2; = 0.68; sample BMI SD = 4.98 kg/m&sup2;) and n = 381 unconstrained in-the-wild portraits from VisualBMI (unconstrained subset: MAE = 4.47 kg/m&sup2;, RMSE = 5.62 kg/m&sup2;), pooling to MAE = 3.26 kg/m&sup2; (RMSE = 4.40 kg/m&sup2;, R&sup2; = 0.735; sample BMI SD = 8.55 kg/m&sup2;) pre-rejection. Across BMI strata on N = 793, prediction error exhibits pronounced heteroscedasticity: normal/mild overweight (BMI &lt; 25 kg/m&sup2;, n = 284) achieves MAE = 2.41 kg/m&sup2;, overweight to moderate obesity (BMI 25–35 kg/m&sup2;, n = 362) yields MAE = 3.18 kg/m&sup2;, and severe obesity (BMI &gt; 35 kg/m&sup2;, n = 147) exhibits MAE = 5.10 kg/m&sup2; (pooling to exactly [284(2.41) + 362(3.18) + 147(5.10)] / 793 = 3.26 kg/m&sup2;). Applying the MC Dropout rejection threshold (&sigma; &gt; 1.80 kg/m&sup2;, selected at the 95th percentile of validation set uncertainty, rejecting 4.2% rather than 5.0% on the held-out test split due to slightly lower upper-tail uncertainty dispersion in the test partition) exploits the Spearman rank correlation between predictive uncertainty &sigma; and absolute BMI error |y&#770; - y| (&rho; = 0.38, p &lt; 0.001 on N = 793): the top 2.0% most uncertain frames (16 frames) exhibit true MAE = 8.21 kg/m&sup2;, while the subsequent 2.2% (17 frames, spanning 2.0% to 4.2%) exhibit MAE = 6.69 kg/m&sup2;, yielding a combined weighted error of [16(8.21) + 17(6.69)] / 33 = 7.43 kg/m&sup2; (RMSE = 8.62 kg/m&sup2;, 2.4-fold higher than accepted frames) across all 33 rejected frames (4.2%). Consequently, filtering at 0%, 2.0% (16 frames), 4.2% (33 frames, n = 760 accepted), and 10.0% (79 frames) progressively reduces pooled test MAE from 3.26 &rarr; 3.16 &rarr; 3.08 &rarr; 2.95 kg/m&sup2; (RMSE = 4.12 kg/m&sup2; at 4.2%), whereas random 4.2% rejection leaves MAE unchanged at 3.26 kg/m&sup2;. In downstream NHANES evaluation, individuals with severe obesity (BMI &gt; 40 kg/m&sup2;, 7.8% of cohort) were fully retained in all sensitivity calculations; in operational deployment, routing both &sigma; &gt; 1.80 and extreme estimated BMI &gt; 40 kg/m&sup2; to manual physical measurement yields an aggregate referral rate of approximately 11.7% (1 - [0.958 &times; 0.922] = 11.67%, assuming statistical independence). In parallel, landmark geometry extracts LFWR (r = 0.42), CJWR (r = -0.38), PAR (r = 0.46), and FWHR (r = 0.35; all p &lt; 0.001, Pearson correlation with ground-truth BMI on N = 793) as visual explainability proxies for clinical dashboard feedback.</p>
<p>2) <b>Stage 1.5 (Tabular DEXA Regression):</b> Estimated BMI, age, biological sex (male=1, female=0), and physical activity level (encoded from NHANES PAQ physical activity items PAD200 and PAD320 into a 3-tier ordinal scale: 0 = Sedentary, 1 = Moderate [moderate activity], 2 = Active/Vigorous [vigorous activity]) are passed to an XGBoost regressor trained on NHANES DEXA scans using target-specific training partitions (N = 2,477 for Diabetes, N = 2,488 for Hypertension; utilizing 5-fold out-of-fold predictions for training folds and randomized hyperparameter search [20]), achieving R&sup2; = 0.74 and MAE = 3.42 percentage points (pp) body fat for the primary non-invasive route (R&sup2; = 0.82, MAE = 2.85 pp with waist circumference) while ensuring no participant overlap by SEQN between training and evaluation splits.</p>
<p>3) <b>Stage 2 (Monotonic Calibrated Risk Classification):</b> In the contactless primary route, the model requires zero physical instruments, pairing the facial image with three simple self-reported responses: age, biological sex (male=1, female=0), and 3-tier physical activity level (0 = Sedentary, 1 = Moderate, 2 = Active/Vigorous). Tabular predictors [Age, Sex, Predicted BMI, Predicted Body Fat %, Physical Activity, (Optional Waist)] are evaluated by regularized XGBoost classifiers: n_estimators = 120, max_depth = 3 (preventing volatile step-function overfitting), learning_rate = 0.03, subsample = 0.80, colsample_bytree = 0.75 (mitigating BMI-waist collinearity), reg_alpha = 0.5, reg_lambda = 2.0, min_child_weight = 5, with monotone constraints (+1) on Age, BMI, Waist, and Body Fat %. Probability calibration is executed via CalibratedClassifierCV(method='sigmoid', cv=5) [19] for Physician-Diagnosed Diabetes Mellitus (DIQ010) and Physician-Diagnosed Hypertension (BPQ020).</p>

<!-- TABLE I -->
<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">TABLE I. NHANES ADULT COHORT PARTICIPANT FLOW AND PARTITIONING</div>
  <table class="paper-table" data-w="26.8,34,12,12.4,11.2">
    <tr>
      <th>Filtering Step</th>
      <th>Criteria / Description</th>
      <th>Remaining (N)</th>
      <th>Positive Cases</th>
      <th>Base Prev.</th>
    </tr>
    <tr><td>Initial NHANES Cohort (2003–2004 / 2017–2018)</td><td>Survey participants across DEXA cycles</td><td>39,156</td><td>—</td><td>—</td></tr>
    <tr><td>Age Filter (Adult Cohort)</td><td>Participants aged 18–59 years (DXA eligible)</td><td>14,210</td><td>—</td><td>—</td></tr>
    <tr><td>DEXA Examination Protocol</td><td>Valid whole-body DXA scan (DXDTOPF), &lt;450 lbs, &lt;6'5"</td><td>5,120</td><td>—</td><td>—</td></tr>
    <tr><td>Diabetes Cohort (DIQ010 Target)</td><td>Non-pregnant, complete covariates, definitive (excl. DIQ010=3)</td><td><b>3,540</b></td><td><b>366</b></td><td><b>10.34%</b></td></tr>
    <tr><td>— Train Split (70%)</td><td>Out-of-fold training partition</td><td>2,477</td><td>256</td><td>10.34%</td></tr>
    <tr><td>— Validation Split (15%)</td><td>Threshold optimization partition (&tau;)</td><td>532</td><td>55</td><td>10.34%</td></tr>
    <tr><td>— Held-Out Test Split (15%)</td><td>Unobserved final evaluation partition</td><td>531</td><td>55</td><td>10.36%</td></tr>
    <tr><td>Hypertension Cohort (BPQ020 Target)</td><td>Non-pregnant, complete covariates, definitive diagnosis</td><td><b>3,554</b></td><td><b>1,207</b></td><td><b>33.96%</b></td></tr>
    <tr><td>— Train Split (70%)</td><td>Out-of-fold training partition</td><td>2,488</td><td>845</td><td>33.96%</td></tr>
    <tr><td>— Validation Split (15%)</td><td>Threshold optimization partition (&tau;)</td><td>533</td><td>181</td><td>33.96%</td></tr>
    <tr><td>— Held-Out Test Split (15%)</td><td>Unobserved final evaluation partition</td><td>533</td><td>181</td><td>33.96%</td></tr>
  </table>
  <div class="table-note">Note: CDC NHANES 2003–2004 (Cycle C) primary cohort and 2017–2018 (Cycle J) validation cycles, adult cohort aged 18–59 years.</div>
</div>

<h3>B. Decoupled Evaluation and Out-of-Fold Protocol</h3>
<p>Partitioning into Train (70%), Validation (15%), and Held-Out Test (15%) splits was executed via target-stratified partitioning independently for each disease to preserve empirical baseline prevalence (10.34% for Diabetes, 33.96% for Hypertension) across all subsets. In the primary pipeline, Stage 1.5 DEXA regressors are trained on target-specific training partitions (N = 2,477 for Diabetes, N = 2,488 for Hypertension) using 5-fold out-of-fold predictions to supply inputs to Stage 2, ensuring no participant overlap by SEQN between training and evaluation splits. To further evaluate cross-cohort robustness, a secondary sensitivity audit strictly excluded all 359 SEQN participants overlapping between the Diabetes training set and the Hypertension test set, creating a fully disjoint joint training cohort of N = 2,118; refitting Stage 1.5 on this trimmed cohort yielded negligible divergence (&Delta;MAE &le; 0.04 percentage points (pp) in body fat, &Delta;AUC &le; 0.002 in Stage 2, with identical validation cutoffs &tau; = 0.060 and &tau; = 0.155), confirming that cross-target test leakage does not influence benchmark performance. In the empirical NHANES adult cohort (N = 3,540, BMI SD = 7.18 kg/m&sup2;), the population distribution comprises 31.0% normal/underweight (BMI &lt; 25 kg/m&sup2;, 1,097 participants), 51.0% overweight to moderate obesity (BMI 25–35 kg/m&sup2;, 1,805 participants), and 18.0% severe obesity (BMI &gt; 35 kg/m&sup2;, 638 participants). Weighting the stratum-specific Stage 1 MAEs by NHANES cohort proportions yields an expected population MAE of 0.31(2.41) + 0.51(3.18) + 0.18(5.10) = 3.29 kg/m&sup2; (assuming comparable within-stratum error distributions between the vision test set and NHANES). Because a zero-mean Gaussian perturbation with &sigma;<sub>sim</sub> = 4.40 kg/m&sup2; produces a theoretical MAE of &radic;(2/&pi;)&middot;&sigma;<sub>sim</sub> &approx; 3.51 kg/m&sup2; (while &sigma;<sub>sim</sub> = 2.82 kg/m&sup2; produces MAE &approx; 2.25 kg/m&sup2;), realistic deployment error in NHANES aligns closely with the pooled regime (&sigma;<sub>sim</sub> = 4.40 kg/m&sup2;). Downstream Stage 2 classifiers were trained on unperturbed out-of-fold Stage 1.5 predictions. To assess real-world inference resilience across diverse imaging conditions, we conducted a parametric sensitivity sweep by injecting zero-mean Gaussian perturbation across four empirically grounded error levels corresponding to Stage 1 benchmarks: &sigma;<sub>sim</sub> &isin; {2.14 (controlled frontal MAE), 2.82 (controlled frontal RMSE), 4.40 (pooled test RMSE), 5.60 (unconstrained VisualBMI RMSE)} kg/m&sup2;. In addition, models were benchmarked against noise-free ground-truth BMI inputs (anthropometric upper bound). In this decoupled flow, Stage 1.5 acts as a DEXA-guided feature engineering module that maps [BMI, Age, Sex, Physical Activity] onto an estimated body fat percentage scale prior to Stage 2 classification.</p>

<h3>C. Sensitivity and Error Propagation Analysis</h3>
<p>Let predicted body fat &theta; be a function of estimated BMI b, age a, biological sex s, physical activity p_a, and waist w: &theta; = f(b, a, s, p_a, w). In the auxiliary model with waist circumference, tree feature split importance attributes 74.99% to biological sex, 14.20% to waist circumference, 6.28% to BMI, 2.94% to age, and 1.59% to physical activity. When users provide auxiliary ground-truth waist measurements, waist circumference acts as an anatomical anchor, dampening upstream BMI estimation noise down to a sensitivity slope of &alpha; = 0.39 percentage points per 1 kg/m&sup2; BMI perturbation [&Delta;BF &approx; 0.39 &middot; &Delta;BMI], compared with approximately 1.00 percentage point per 1 kg/m&sup2; in the primary route without waist [&Delta;BF &approx; 1.00 &middot; &Delta;BMI]. Importantly, in terminal disease classification, the primary route without waist achieves equivalent discriminability without requiring physical measuring tape hardware.</p>



<h2>IV. EXPERIMENTS AND RESULTS</h2>
<h3>A. Stage 1 ViT and Stage 1.5 DEXA Regression Performance</h3>
<p>Table II summarizes performance metrics across Stage 1 ViT vision estimation (reporting the controlled frontal VisualBMI subset, n = 412; with 4.2% uncertainty rejection evaluated across the pooled test split N = 793) and Stage 1.5 DEXA body fat regression models evaluated on held-out test splits (evaluated on measured, unperturbed NHANES BMI inputs). Figure 2 illustrates the empirical scatter correlation between predicted body fat and gold-standard DEXA whole-body absorptiometry across held-out testing subjects (N = 531) evaluated on measured, unperturbed NHANES BMI inputs for both the primary non-invasive model without waist (MAE = 3.42 percentage points [pp], RMSE = 4.25 pp, R&sup2; = 0.74) and the auxiliary route with waist (MAE = 2.85 pp, RMSE = 3.62 pp, R&sup2; = 0.82).</p>

<!-- TABLE II -->
<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">TABLE II. STAGE 1 VISION AND STAGE 1.5 DEXA REGRESSION BENCHMARKS</div>
  <table class="paper-table" data-w="25.4,10.4,23.6,10.4,10.4,4.4,10.6">
    <tr style="break-inside: avoid; page-break-inside: avoid;">
      <th>Model</th>
      <th>Target</th>
      <th>Input Features</th>
      <th>MAE</th>
      <th>RMSE</th>
      <th>R&sup2;</th>
      <th>Rejection</th>
    </tr>
    <tr style="break-inside: avoid; page-break-inside: avoid;"><td>Stage 1 ViT-H/14 + MC Dropout</td><td>BMI (kg/m&sup2;)</td><td>2D Facial Image</td><td>2.14 kg/m&sup2;</td><td>2.82 kg/m&sup2;</td><td>0.68</td><td>4.2% (pooled)<br>(&sigma;&gt;1.8)</td></tr>
    <tr style="break-inside: avoid; page-break-inside: avoid;"><td>Stage 1.5 DEXA Regressor (Primary)</td><td>Body Fat %</td><td>BMI, Age, Sex, Physical Activity</td><td>3.42 pp</td><td>4.25 pp</td><td>0.74</td><td>0.0%</td></tr>
    <tr style="break-inside: avoid; page-break-inside: avoid;"><td>Stage 1.5 DEXA Regressor (Auxiliary)</td><td>Body Fat %</td><td>BMI, Age, Sex, Physical<br>Activity, Waist</td><td>2.85 pp</td><td>3.62 pp</td><td>0.82</td><td>0.0%</td></tr>
  </table>
  <div class="table-note">Note: Stage 1 evaluated on the controlled frontal VisualBMI subset (n = 412; 4.2% pooled rejection). Stage 1.5 evaluated on measured, unperturbed NHANES BMI inputs (body fat metrics in percentage points [pp]).</div>
</div>

<h3>B. Stage 2 Classification Benchmarks and Baseline Comparisons</h3>

<p><b>Baseline and Threshold Analysis:</b> Evaluating the baseline Stage 2 diabetes classifier on the held-out test set (N=531) revealed predicted probabilities bounded by max(p) = 0.3776 due to low disease prevalence (~10.3%) and moderate discriminative capability (AUC = 0.762). Because no individual crossed the default 0.50 cutoff, the default model exhibited 0.00% sensitivity (55/55 false negatives, as shown in Fig. 5a, vs. 78.2% sensitivity at Youden's J cutoff &tau; = 0.082 in Fig. 5b). Shifting to the validation-optimized F2 screening threshold (&tau; = 0.060; Fig. 5c and Table III) on the primary non-invasive model (No Waist) at the lower-error reference (&sigma;<sub>sim</sub> = 2.14 kg/m&sup2;, corresponding to Gaussian MAE &approx; 1.71 kg/m&sup2;; evaluated at fixed seed = 42 for integer case reporting) achieved <b>90.91% sensitivity</b> (50/55 positive cases detected, 95% CI: 80.0%&ndash;96.7%) at 48.7% specificity and 17.0% precision (and <b>90.61% sensitivity</b> [164/181 cases] at 40.9% specificity for hypertension at &tau; = 0.155 in Table IV). When waist circumference is optionally included, the F2 threshold (&tau; = 0.061) achieved <b>89.09% sensitivity</b> (49/55 cases detected, 95% CI: 77.8%&ndash;95.9%). As illustrated by the ROC curves in Fig. 3, compared against Baseline Logistic Regression evaluated on the exact same input features, Primary Monotonic XGBoost achieved higher overall discrimination for both diabetes (AUC = 0.762 vs 0.742; &Delta;AUC = +0.020, 95% paired bootstrap CI: +0.004 to +0.036, p = 0.015) and hypertension (AUC = 0.757 vs 0.740; &Delta;AUC = +0.017, 95% paired bootstrap CI: +0.005 to +0.029, p = 0.006).</p>

<p><b>Parametric and Residual Noise Sweeps:</b> Because the controlled frontal subset RMSE (2.82 kg/m&sup2;) was measured on a sub-cohort with narrower BMI dispersion (SD = 4.98 kg/m&sup2;) than our empirical NHANES cohort (SD = 7.18 kg/m&sup2; across N = 3,540, where 18.0% of adults exhibit BMI &gt; 35 kg/m&sup2;), realistic deployment error centers in the pooled-to-unconstrained range (&sigma;<sub>sim</sub> = 4.40–5.60 kg/m&sup2;), bounded below by controlled laboratory capture (&sigma;<sub>sim</sub> = 2.82 kg/m&sup2;). Across 100 Monte Carlo noise draws (seeds 0–99, reporting mean &plusmn; SD alongside 95% CIs for &Delta;AUC vs the demographic-only baseline [Age + Sex + PA: AUC = 0.712 for diabetes, 0.708 for hypertension] across 100 draws &times; 1,000 paired bootstrap replicates, computed via percentile aggregation across the pooled 100,000 draw-bootstrap replicate differences): at nominal reference &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;, Face2Health yields &Delta;AUC = +0.050 (95% CI: +0.029 to +0.070, p &lt; 0.001) for diabetes and &Delta;AUC = +0.049 (95% CI: +0.031 to +0.066, p &lt; 0.001) for hypertension; at laboratory RMSE noise &sigma;<sub>sim</sub> = 2.82 kg/m&sup2;, 100-draw evaluation achieves Sens = 88.7% &plusmn; 2.4%, Spec = 47.3% &plusmn; 1.8%, and AUC = 0.757 &plusmn; 0.006 for diabetes (&Delta;AUC = +0.045, 95% CI: +0.023 to +0.067, p &lt; 0.001) and Sens = 89.2% &plusmn; 1.6%, Spec = 39.6% &plusmn; 1.5%, and AUC = 0.752 &plusmn; 0.005 for hypertension (&Delta;AUC = +0.044, 95% CI: +0.026 to +0.063, p &lt; 0.001); at pooled test noise &sigma;<sub>sim</sub> = 4.40 kg/m&sup2;, performance reaches Sens = 85.1% &plusmn; 3.2%, Spec = 45.6% &plusmn; 2.1%, and AUC = 0.745 &plusmn; 0.009 for diabetes (&Delta;AUC = +0.033, 95% CI: +0.009 to +0.057, p = 0.007) and Sens = 86.4% &plusmn; 2.1%, Spec = 38.2% &plusmn; 1.8%, and AUC = 0.741 &plusmn; 0.007 for hypertension (&Delta;AUC = +0.033, 95% CI: +0.014 to +0.053, p = 0.001); and under unconstrained social media noise &sigma;<sub>sim</sub> = 5.60 kg/m&sup2;, performance contracts to Sens = 81.3% &plusmn; 3.9%, Spec = 43.9% &plusmn; 2.4%, and AUC = 0.735 &plusmn; 0.011 for diabetes (&Delta;AUC = +0.024, 95% CI: -0.002 to +0.049, p = 0.068) and Sens = 83.7% &plusmn; 2.5%, Spec = 37.0% &plusmn; 2.0%, and AUC = 0.732 &plusmn; 0.009 for hypertension (&Delta;AUC = +0.025, 95% CI: +0.003 to +0.046, p = 0.026).</p>


<div style="break-before: column; page-break-before: always;"></div>
<!-- FIGURE 1 -->
<div class="fig-container" style="break-inside: avoid; page-break-inside: avoid;">
  <img src="fig1_stage15_scatter.png" style="width: 248pt; max-width: 248pt; height: 251pt;" alt="Stage 1.5 DEXA Scatter Plot">
  <div class="fig-caption">Fig. 2. Predicted vs. DEXA total body fat percentage (N = 531): (a) without waist circumference; (b) with waist circumference.</div>
</div>

<p>As a retrospective test-ROC operating point comparison (distinct from the fixed validation-threshold results above), when evaluating test specificity at the empirical threshold yielding matched 90.9% diabetes sensitivity (50/55) and 90.6% hypertension sensitivity (164/181) against the demographic-only baseline (specificity 31.1% [95% CI: 26.9%–35.5%] for diabetes and 28.4% [95% CI: 23.8%–33.4%] for hypertension), facial estimates yield paired specificity gains (&Delta;Spec) across noise regimes of 3.1 to 17.6 percentage points: for diabetes, matched specificity reaches 48.7% (&Delta;Spec = +17.6 pp, 95% paired CI: +12.4 to +22.8 pp) at &sigma;=2.14, 45.6% (&Delta;Spec = +14.5 pp, CI: +9.1 to +19.8 pp) at &sigma;=2.82, 39.7% (&Delta;Spec = +8.6 pp, CI: +2.9 to +14.2 pp) at &sigma;=4.40, and 35.5% (&Delta;Spec = +4.4 pp, 95% paired CI: -1.3 to +10.1 pp, p = 0.13) at &sigma;=5.60; for hypertension, matched specificity reaches 40.9% (&Delta;Spec = +12.5 pp, CI: +7.8 to +17.2 pp) at &sigma;=2.14, 38.6% (&Delta;Spec = +10.2 pp, CI: +5.4 to +15.0 pp) at &sigma;=2.82, 34.1% (&Delta;Spec = +5.7 pp, CI: +0.8 to +10.6 pp) at &sigma;=4.40, and 31.5% (&Delta;Spec = +3.1 pp, 95% paired CI: -1.8 to +8.0 pp, p = 0.21) at &sigma;=5.60—demonstrating that at &sigma;<sub>sim</sub> = 5.60 kg/m&sup2;, matched-sensitivity specificity gains (&Delta;Spec) become statistically non-significant for both conditions (p = 0.13 for diabetes; p = 0.21 for hypertension) and overall &Delta;AUC becomes non-significant for diabetes (+0.024, p = 0.068), while hypertension retains only marginal overall &Delta;AUC significance (+0.025, 95% CI: +0.003 to +0.046, p = 0.026).</p>

<p>Under BMI-decile stratified empirical residual resampling across 100 Monte Carlo draws (seeds 0–99), drawing from the pooled test set (N = 793 residuals) yielded AUC = 0.744 &plusmn; 0.008 (Sens = 84.5% &plusmn; 2.9%, Spec = 45.2% &plusmn; 2.0%) for diabetes and AUC = 0.740 &plusmn; 0.006 (Sens = 86.2% &plusmn; 1.9%, Spec = 38.1% &plusmn; 1.7%) for hypertension, whereas resampling unconstrained social media residuals (n = 381 VisualBMI) yielded AUC = 0.734 &plusmn; 0.010 (Sens = 80.9% &plusmn; 3.6%, Spec = 43.7% &plusmn; 2.3%) for diabetes and AUC = 0.731 &plusmn; 0.008 (Sens = 83.4% &plusmn; 2.4%, Spec = 36.9% &plusmn; 1.9%) for hypertension; this BMI-stratified empirical residual resampling provides a more realistic representation than fixed Gaussian perturbation by capturing amplified vision estimation error within severe obesity (BMI &gt; 35 kg/m&sup2;), where metabolic disease prevalence is heavily concentrated. While fixing clean validation cutoffs (&tau; = 0.060 / 0.155) under &sigma;<sub>sim</sub> = 5.60 kg/m&sup2; reduces sensitivity, noise-aware threshold recalibration on perturbed validation sets (&tau; = 0.048 / 0.132) restores ~90% sensitivity (89.1% at 36.6% specificity for diabetes; 89.5% at 32.7% specificity for hypertension). Compared against a simulated self-reported BMI baseline (&sigma;<sub>self</sub> = 1.50 kg/m&sup2;, yielding 100-draw mean AUC = 0.764 for diabetes and 0.759 for hypertension; modeled as zero-mean Gaussian noise without accounting for systematic high-BMI under-reporting bias), for diabetes (and hypertension), facial estimation shows no statistically significant difference compared to simulated self-report under ideal laboratory noise at &sigma;<sub>sim</sub> = 2.82 kg/m&sup2; (diabetes &Delta;AUC = -0.007, 95% CI: -0.020 to +0.005; hypertension &Delta;AUC = -0.007, 95% CI: -0.017 to +0.003), but becomes significantly less accurate under pooled noise at &sigma;<sub>sim</sub> = 4.40 kg/m&sup2; (diabetes &Delta;AUC = -0.019, 95% CI: -0.037 to -0.002; hypertension &Delta;AUC = -0.018, 95% CI: -0.032 to -0.004) and unconstrained noise at &sigma;<sub>sim</sub> = 5.60 kg/m&sup2; (diabetes &Delta;AUC = -0.029, 95% CI: -0.050 to -0.010; hypertension &Delta;AUC = -0.027, 95% CI: -0.044 to -0.010). Consequently, camera-based estimation is positioned strictly as a fallback when neither physical scales nor reliable self-reported anthropometrics are obtainable.</p>


<p><b>Secondary Composite Label Validation:</b> Secondary sensitivity analyses evaluated the primary models (fixed at &tau; = 0.060 and &tau; = 0.155) against objective composite diagnostic criteria: (1) for composite diabetes (DIQ010 = 1 OR HbA1c &ge; 6.5%; N = 531, 68 positive cases, 12.81% prevalence), the model achieved AUC = 0.758 (95% CI: 0.701–0.815) and 88.2% F2 sensitivity (60/68 cases) at 49.5% specificity (229/463 true negatives, as 10 of the 13 undiagnosed cases were already flagged positive at &tau; = 0.060 while 3 shifted from negative controls); in the undiagnosed subgroup (negative self-reported diagnosis [recoded binary DIQ010 = 0; raw code 2] with HbA1c &ge; 6.5%, comprising 13/68 [19.1%] of diabetic cases), sensitivity contracted to 76.9% (10/13 cases, 95% CI: 46.2%–95.0%); (2) for composite hypertension (BPQ020 = 1 OR measured SBP &ge; 140 / DBP &ge; 90 mmHg; N = 533, 212 positive cases, 39.77% prevalence), the model achieved AUC = 0.752 (95% CI: 0.710–0.794) and 89.2% F2 sensitivity (189/212 cases) at 43.0% specificity (138/321 true negatives, reflecting 25 of 31 undiagnosed cases captured among prior false positives and 6 shifting from true negatives), with undiagnosed sensitivity contracting to 80.6% (25/31 cases, 95% CI: 62.5%–92.5%, representing 14.6% [31/212] of cases; vs 90.9% and 90.6% under diagnosed targets), where confidence intervals remain wide due to modest subgroup sample sizes and survey single-occasion HbA1c or blood pressure values represent preliminary screening indicators requiring clinical confirmation; and (3) re-incorporating self-reported borderline status (DIQ010 = 3; 11 test cases, N = 542) into negative controls yielded AUC = 0.755 (95% CI: 0.693–0.817).</p>

<div style="break-before: column; page-break-before: always;"></div>
<!-- TABLE III-A -->
<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">TABLE III. DIABETES (DIQ010) CLASSIFICATION BENCHMARK</div>
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
    <tr><td>Logistic Reg.</td><td>Opt. (F2)</td><td>0.068</td><td>0.742<br>(0.678–0.806)</td><td>89.1%<br>(77.8–95.9%)</td><td>39.3%<br>(34.9–43.8%)</td><td>14.5%<br>(10.9–18.7%)</td></tr>
    <tr><td>XGBoost</td><td>Default</td><td>0.500</td><td>0.747<br>(0.683–0.811)</td><td>0.0%<br>(0.0–6.5%)</td><td>100.0%<br>(99.2–100%)</td><td>0.0%</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>Default</td><td>0.500</td><td>0.762<br>(0.701–0.823)</td><td>0.0%<br>(0.0–6.5%)</td><td>100.0%<br>(99.2–100%)</td><td>0.0%</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td>Youden's J</td><td>0.082</td><td>0.762<br>(0.701–0.823)</td><td>78.2%<br>(65.0–88.2%)</td><td>59.0%<br>(54.5–63.5%)</td><td>18.1%<br>(13.4–23.6%)</td></tr>
    <tr><td><b>Primary (Ours)</b></td><td><b>Opt. (F2)</b></td><td><b>0.060</b></td><td><b>0.762<br>(0.701–0.823)</b></td><td><b>90.9%<br>(80.0–96.7%)</b></td><td><b>48.7%<br>(44.1–53.3%)</b></td><td><b>17.0%<br>(13.0–21.8%)</b></td></tr>
    <tr><td>Auxiliary With Waist</td><td>Default</td><td>0.500</td><td>0.763<br>(0.702–0.824)</td><td>0.0%<br>(0.0–6.5%)</td><td>100.0%<br>(99.2–100%)</td><td>0.0%</td></tr>
    <tr><td>Auxiliary With Waist</td><td>Youden's J</td><td>0.082</td><td>0.763<br>(0.702–0.824)</td><td>78.2%<br>(65.0–88.2%)</td><td>58.8%<br>(54.3–63.3%)</td><td>18.0%<br>(13.3–23.5%)</td></tr>
    <tr><td>Auxiliary With Waist</td><td>Opt. (F2)</td><td>0.061</td><td>0.763<br>(0.702–0.824)</td><td>89.1%<br>(77.8–95.9%)</td><td>48.9%<br>(44.4–53.5%)</td><td>16.8%<br>(12.7–21.6%)</td></tr>
  </table>
  <div class="table-note">Note: N = 531 held-out test split. Values in parentheses indicate 95% CI. Single-draw reference benchmark at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;, seed = 42.</div>
</div>

<!-- TABLE III-B -->
<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">TABLE IV. HYPERTENSION (BPQ020) CLASSIFICATION BENCHMARK</div>
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
    <tr><td>Logistic Reg.</td><td>Default</td><td>0.500</td><td>0.740<br>(0.697–0.783)</td><td>48.1%<br>(40.6–55.6%)</td><td>83.5%<br>(79.2–87.2%)</td><td>60.0%<br>(51.9–67.7%)</td></tr>
    <tr><td>Logistic Reg.</td><td>Opt. (F2)</td><td>0.165</td><td>0.740<br>(0.697–0.783)</td><td>89.5%<br>(84.1–93.6%)</td><td>34.7%<br>(29.7–39.9%)</td><td>41.3%<br>(36.4–46.4%)</td></tr>
    <tr><td>Unconstrained XGB</td><td>Default</td><td>0.500</td><td>0.750<br>(0.708–0.792)</td><td>49.2%<br>(41.7–56.7%)</td><td>82.1%<br>(77.7–85.9%)</td><td>58.6%<br>(50.5–66.3%)</td></tr>
    <tr><td><b>Primary (No Waist)</b></td><td>Default</td><td>0.500</td><td>0.757<br>(0.715–0.799)</td><td>50.3%<br>(42.8–57.8%)</td><td>82.7%<br>(78.3–86.5%)</td><td>59.9%<br>(51.8–67.5%)</td></tr>
    <tr><td><b>Primary (No Waist)</b></td><td>Youden's J</td><td>0.339</td><td>0.757<br>(0.715–0.799)</td><td>71.3%<br>(64.1–77.7%)</td><td>67.0%<br>(61.8–72.0%)</td><td>52.7%<br>(46.2–59.0%)</td></tr>
    <tr><td><b>Primary (No Waist)</b></td><td><b>Opt. (F2)</b></td><td><b>0.155</b></td><td><b>0.757<br>(0.715–0.799)</b></td><td><b>90.6%<br>(85.3–94.4%)</b></td><td><b>40.9%<br>(35.7–46.3%)</b></td><td><b>44.1%<br>(39.0–49.3%)</b></td></tr>
    <tr><td>Auxiliary With Waist</td><td>Default</td><td>0.500</td><td>0.755<br>(0.713–0.797)</td><td>49.7%<br>(42.2–57.2%)</td><td>82.4%<br>(78.0–86.2%)</td><td>59.2%<br>(51.1–66.9%)</td></tr>
    <tr><td>Auxiliary With Waist</td><td>Opt. (F2)</td><td>0.158</td><td>0.755<br>(0.713–0.797)</td><td>88.4%<br>(82.8–92.6%)</td><td>41.8%<br>(36.6–47.1%)</td><td>43.8%<br>(38.7–49.1%)</td></tr>
  </table>
  <div class="table-note">Note: N = 533 held-out test split. Values in parentheses indicate 95% CI. Single-draw reference benchmark at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;, seed = 42.</div>
</div>





<h3>C. Probability Calibration and Component Ablation</h3>
<p>Figure 4 illustrates the reliability calibration curves before and after Platt Sigmoid Scaling. Uncalibrated tree outputs exhibit classical S-shaped reliability distortion. Platt calibration shifts predicted probabilities to track empirical observation lines tightly, achieving low Brier scores of 0.081 for Diabetes and 0.185 for Hypertension at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;; under realistic pooled-to-unconstrained noise (&sigma;<sub>sim</sub> = 4.40–5.60 kg/m&sup2;), 100-draw mean Brier scores modestly increase to 0.084–0.086 for diabetes (remaining well below the uncalibrated class-weighted diabetes baseline of 0.138–0.142 at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2; and 0.145–0.149 at matching noise, as well as the constant-prevalence reference of 0.093) and to 0.191–0.195 for hypertension (remaining below the uncalibrated hypertension baseline of 0.218 at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2; [0.225–0.229 at matching noise] and the constant-prevalence reference of 0.224).</p>

<p>To quantify individual component contributions, ablation experiments were executed on the Diabetes test cohort (N=531), summarizing AUC, Sensitivity, Specificity, Brier score, and Monotonicity violations in Table V. The `No Body Fat` configuration evaluates Stage 2 supplied directly with [Age, Sex, Predicted BMI, Physical Activity] without the intermediate DEXA regressor (achieving AUC = 0.748, F2 sensitivity = 83.6%, Brier = 0.085). Incorporating Predicted Body Fat % yields a modest numerical increase to AUC = 0.762 and matched-sensitivity specificity of 48.7% vs 42.0% for No Body Fat (&Delta;AUC = +0.014, 95% CI: -0.004 to +0.032, p = 0.12). Because Predicted Body Fat % is deterministically derived from [BMI, Age, Sex, Physical Activity]—variables already supplied to Stage 2—this step does not introduce external sensors, but rather is hypothesized to act as DEXA-guided non-linear feature engineering for shallow Stage 2 trees; given the overlapping confidence intervals (p = 0.12), its incremental discriminative contribution is statistically modest. Because minor sensitivity differences at validation F2 cutoffs across Table V rows (e.g., 85.5% vs 80.0%, representing 3–4 cases) co-occur with threshold shifts, comparative model utility is most transparently evaluated via &Delta;AUC and matched-sensitivity specificity. Removing monotonicity constraints produces 14 clinically impermissible risk inversions (pairwise grid instances where predicted disease risk paradoxically decreases as BMI or body fat increases). Regarding probability calibration, because uncalibrated trees trained without class weighting already achieve Brier = 0.091 (compared to 0.081 after calibration), the primary practical utility of Platt scaling in our pipeline is correcting the severe probability inflation (Brier = 0.138–0.142) induced when training trees with positive class weighting (scale_pos_weight).</p>

<!-- FIGURE 2 -->
<div class="fig-container" style="break-inside: avoid; page-break-inside: avoid;">
  <img src="fig2_roc_curves.png" style="width: 248pt; max-width: 248pt; height: 288pt;" alt="Stage 2 ROC Curves Benchmark">
  <div class="fig-caption">Fig. 3. ROC curves for diabetes (DIQ010, top) and hypertension (BPQ020, bottom) at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;.</div>
</div>

<div style="break-before: column; page-break-before: always;"></div>
<!-- TABLE V -->
<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">TABLE V. COMPONENT ABLATION STUDY ON THE DIABETES TEST COHORT</div>
  <table class="paper-table" data-w="36.3,9.4,12.4,12.4,10.4,15">
    <tr>
      <th>Configuration</th>
      <th>AUC</th>
      <th>Sens (F2)</th>
      <th>Spec (F2)</th>
      <th>Brier</th>
      <th>Violations</th>
    </tr>
    <tr><td><b>Full Face2Health Pipeline (Ours)</b></td><td><b>0.762</b></td><td><b>90.9%</b></td><td>48.7%</td><td><b>0.081</b></td><td><b>0 (By Constraint)</b></td></tr>
    <tr><td>No Body Fat</td><td>0.748</td><td>83.6%</td><td>51.3%</td><td>0.085</td><td>0 Violations</td></tr>
    <tr><td>No Monotonicity</td><td>0.747</td><td>85.5%</td><td>50.0%</td><td>0.084</td><td>14 Violations</td></tr>
    <tr><td>No Monot. &amp; Calib.</td><td>0.747</td><td>80.0%</td><td>52.5%</td><td>0.142</td><td>14 Violations</td></tr>
  </table>
  <div class="table-note">Note: N = 531 held-out test split; single-draw reference benchmark at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;, seed = 42.</div>
</div>

<!-- FIGURE 3 -->
<div class="fig-container" style="break-inside: avoid; page-break-inside: avoid;">
  <img src="fig3_calibration.png" style="width: 248pt; max-width: 248pt; height: 283pt;" alt="Stage 2 Calibration Reliability Diagram">
  <div class="fig-caption">Fig. 4. Calibration reliability diagrams before and after Platt scaling (evaluated at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;).</div>
</div>

<h3>D. Body Fat Error Attenuation Dynamics</h3>
<p>Table VI evaluates total body fat shifts under systematic BMI perturbation sweeps (&Delta;BMI &isin; [-2, +2] kg/m&sup2;), demonstrating the variance shock absorber effect under auxiliary waist features, which stabilizes body fat estimates when physical tape measurement is available. While auxiliary waist measurements substantially attenuate intermediate body fat perturbation (&alpha; = 0.39 vs 1.00), terminal disease classification AUC remains virtually identical between the auxiliary and primary routes (0.763 vs 0.762), confirming that the contactless primary route suffices for risk ranking.</p>



<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">TABLE VI. BODY FAT SHIFT UNDER BMI PERTURBATION SWEEPS</div>
  <table class="paper-table" data-w="17.4,25.4,17.4,36.9">
    <tr>
      <th>Injected &Delta;BMI</th>
      <th>Mean &Delta;BF Waist (pp)</th>
      <th>Slope &alpha;<br>(pp/kg/m&sup2;)</th>
      <th>Mean &Delta;BF No Waist (pp)</th>
    </tr>
    <tr><td>-2.0 kg/m&sup2;</td><td>-0.80 &plusmn; 0.80 pp</td><td>0.40</td><td>-2.02 &plusmn; 1.24 pp</td></tr>
    <tr><td>-1.0 kg/m&sup2;</td><td>-0.37 &plusmn; 0.50 pp</td><td>0.37</td><td>-1.01 &plusmn; 0.82 pp</td></tr>
    <tr><td>0.0 kg/m&sup2;</td><td>0.00 &plusmn; 0.00 pp</td><td>—</td><td>0.00 &plusmn; 0.00 pp</td></tr>
    <tr><td>+1.0 kg/m&sup2;</td><td>+0.39 &plusmn; 0.54 pp</td><td>0.39</td><td>+1.00 &plusmn; 0.83 pp</td></tr>
    <tr><td>+2.0 kg/m&sup2;</td><td>+0.77 &plusmn; 0.72 pp</td><td>0.38</td><td>+1.98 &plusmn; 1.14 pp</td></tr>
  </table>
</div>



<h2>V. CLINICAL TRIAGE ANALYSIS</h2>
<p>The calibrated posterior probabilities are operationalized into four clinical triage strata bounded at the lower end by the validation F2 screening thresholds (&tau; = 0.060 and &tau; = 0.155) and subdivided into illustrative, heuristic risk-communication intervals (10%/20% for diabetes; 25%/50% for hypertension; reflecting discrete probability clustering inherent to shallow depth-3 decision trees after sigmoid mapping). Table VII details population breakdown and observed true disease prevalence across held-out test splits.</p>



<!-- FIGURE 4 -->
<div class="fig-container" style="break-inside: avoid; page-break-inside: avoid;">
  <img src="fig4_confusion_matrix.png" style="width: 248pt; max-width: 248pt; height: 357pt;" alt="Stage 2 Confusion Matrix Heatmaps">
  <div class="fig-caption">Fig. 5. Confusion matrices for the diabetes test cohort (N = 531) at three decision thresholds (&sigma;<sub>sim</sub> = 2.14 kg/m&sup2;).</div>
</div>

<!-- TABLE VII -->
<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">
  <div class="table-caption">TABLE VII. 4-TIER CLINICAL TRIAGE STRATA METRICS</div>
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
      <td>237<br>(44.6%)</td>
      <td>2.11%<br>(5/237)</td>
      <td>161<br>(30.2%)</td>
      <td>10.56%<br>(17/161)</td>
      <td>Annual wellness checkup</td>
    </tr>
    <tr>
      <td>Watchful</td>
      <td>6.0% &le; p &lt; 10.0%<br>15.5% &le; p &lt; 25.0%</td>
      <td>127<br>(23.9%)</td>
      <td>7.87%<br>(10/127)</td>
      <td>122<br>(22.9%)</td>
      <td>27.05%<br>(33/122)</td>
      <td>Point-of-care capillary / BP check</td>
    </tr>
    <tr>
      <td>Screen Positive</td>
      <td>10.0% &le; p &lt; 20.0%<br>25.0% &le; p &lt; 50.0%</td>
      <td>121<br>(22.8%)</td>
      <td>18.18%<br>(22/121)</td>
      <td>98<br>(18.4%)</td>
      <td>40.82%<br>(40/98)</td>
      <td>Venous fasting lab / clinical BP</td>
    </tr>
    <tr>
      <td>Urgent</td>
      <td>p &ge; 20.0% (Dia)<br>p &ge; 50.0% (HTN)</td>
      <td>46<br>(8.7%)</td>
      <td><b>39.13%<br>(18/46)</b></td>
      <td>152<br>(28.5%)</td>
      <td><b>59.87%<br>(91/152)</b></td>
      <td>Priority medical referral</td>
    </tr>
  </table>
  <div class="table-note">Note: Single-draw reference benchmark at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;, seed = 42.</div>
</div>



<p>The clinical progression across the four tiers validates screening utility: the Urgent stratum exhibits 39.13% true diabetes and 59.87% hypertension prevalence—a 3.8-fold enrichment (95% CI: 2.6–5.2x) over the 10.36% baseline (18.5-fold [95% CI: 7.2–47.4x] vs Low Risk), and a 1.8-fold enrichment (95% CI: 1.5–2.1x) over the 33.96% baseline. Screening at the validation F2 cutoff routes screenees across the top three tiers (Watchful, Screen Positive, Urgent; totaling 294/531 [55.4%] for diabetes and 372/533 [69.8%] for hypertension) into stepped confirmatory testing (where the illustrative &ge;50% hypertension tier encompasses 28.5% [152/533] of the cohort due to the high 33.96% baseline disease prevalence, serving to prioritize clinic scheduling rather than acute emergency dispatch), capturing 50/55 (90.9%) and 164/181 (90.6%) positive cases. Alternatively, resource-constrained primary clinics directing confirmatory testing solely to the top two tiers (Screen Positive + Urgent; 167/531 [31.5%] for diabetes, 250/533 [46.9%] for hypertension) retain substantial case capture: 40/55 cases (72.7% sensitivity) and 131/181 cases (72.4% sensitivity), respectively. In frontline opportunistic triage, accepting a moderate False Positive burden (PPV = 17.0% for Diabetes, 44.1% for Hypertension) is clinically justified given safe, low-cost secondary testing (capillary glucose or repeated sphygmomanometry). At the F2 cutoff, the diabetes model achieves a Negative Predictive Value (NPV) of 97.89% (232/237) with positive likelihood ratio LR+ = 1.77 and negative likelihood ratio LR- = 0.19, placing 44.6% (237/531) of screenees below the diabetes referral cutoff at a false omission rate (FOR = 1 - NPV) of only 2.11% (5/237; overall false negative miss rate 9.09% [5/55]). In the Watchful tier (7.87% prevalence), performing secondary point-of-care capillary checks on 127 screenees captures 10 additional cases (incremental Number Needed to Screen NNS = 12.7). For hypertension, the F2 cutoff achieves NPV = 89.44% (144/161) with LR+ = 1.53 and LR- = 0.23, placing 30.2% (161/533) of screenees below the hypertension referral cutoff at a miss rate of 9.39% (17/181).</p>

<h2 style="break-inside: avoid; page-break-inside: avoid;">VI. DISCUSSION</h2>
<p>The clinical and architectural value of Face2Health resides in two foundational strengths: First, passive facial screening enables zero-hardware camera-assisted triage (eliminating physical scales and tape measures, requiring only a portrait alongside three brief self-reported items: age, sex, and physical activity level) in remote or resource-constrained environments. Second, Stage 1.5 functions as a DEXA-guided feature engineering step mapping existing demographic and anthropometric inputs onto a physiological body fat scale; while this representation yields a modest numerical improvement in matched-sensitivity specificity (48.7% vs 42.0% at 90.9% sensitivity) and AUC (0.762 vs 0.748), the overall discriminative difference does not reach statistical significance (&Delta;AUC = +0.014, p = 0.12), indicating that primary risk stratification is driven predominantly by BMI, age, sex, and physical activity.</p>
<p>Face2Health is strictly an opportunistic pre-screening triage tool, not a standalone diagnostic instrument. Key limitations include: (1) Target definitions rely on self-reported physician diagnoses (DIQ010/BPQ020), introducing potential label noise, as undiagnosed cases (accounting for 19.1% [13/68] of composite diabetes cases and 14.6% [31/212] of composite hypertension cases in our held-out NHANES test partitions, and up to 30%–45% in developing regions [2]) are recorded as negative controls under self-reported labels, alongside potential reverse causality if diagnosed individuals adopted post-diagnosis lifestyle modifications; however, parallel secondary validation against composite endpoints incorporating thresholded laboratory HbA1c (&ge; 6.5%) and measured blood pressure (&ge; 140/90 mmHg) showed moderate sensitivity retention (76.9%–80.6%) albeit with wider confidence intervals due to small undiagnosed subgroup sizes; (2) Cross-ethnic morphological shifts: the NHANES cohort represents US demographics; validating craniofacial adiposity markers in diverse Southeast Asian populations is critical before localized deployment; (3) Facial obstruction: severe facial asymmetry, intense makeup, or extreme lighting conditions require enforcement of the Face Guard protocol to block unreliable predictions; (4) Decoupled pipeline evaluation: because CDC NHANES does not release participant facial photographs due to federal privacy regulations, Stage 1 and Stage 2 could not be evaluated on paired images and clinical records from the exact same individuals, necessitating the decoupled noise-injection protocol. Furthermore, downstream Stage 2 classifiers were trained on clean out-of-fold Stage 1.5 predictions without simulated noise and subsequently evaluated across test noise regimes; thus, noise-aware training augmentation during model fitting represents an important avenue to enhance downstream resilience. While parametric noise injection sweeps demonstrated that screening performance degrades gracefully but materially as imaging error escalates, real-world facial estimation errors may exhibit heteroscedastic or demographic dependencies; moreover, estimating expected population MAE in NHANES (3.29 kg/m&sup2;) relies on the untestable assumption that within-stratum imaging error distributions from VisualBMI generalize directly to NHANES participants; and (5) Unweighted survey design: our evaluation utilizes unweighted survey partitions rather than complex survey sample weights, as the primary objective is internal algorithmic benchmarking and comparative machine learning validation within the observed cohort rather than generating official US population point prevalence estimates. Because VisualBMI [7] comprises publicly posted social media photographs subject to self-reporting noise and demographic skew, future clinical translation requires consented prospective cohorts and explicit subgroup fairness auditing across skin tones, sexes, and age strata. All clinical and demographic records analyzed in this study are publicly accessible via the CDC NHANES portal (2003–2004 and 2017–2018 cycles).</p>

<h2>VII. CONCLUSION AND FUTURE WORK</h2>
<p>This study established <b>Face2Health</b>, demonstrating that passive computer vision coupled with calibrated Gradient-Boosted Trees provides modest pre-screening utility as a zero-hardware fallback when physical scales or self-reported anthropometrics are unavailable. Across held-out NHANES test partitions under realistic pooled-to-unconstrained Gaussian noise (&sigma;<sub>sim</sub> = 4.40–5.60 kg/m&sup2;) and BMI-stratified empirical residual resampling, the primary contactless pipeline achieves 80.9%–85.1% sensitivity for Diabetes and 83.4%–86.4% sensitivity for Hypertension (compared to 88.7% / 89.2% under laboratory RMSE &sigma;<sub>sim</sub> = 2.82 kg/m&sup2; and 90.91% / 90.61% in reference tables at &sigma;<sub>sim</sub> = 2.14 kg/m&sup2;), retaining modest but statistically significant discriminative gains over demographics up to &sigma;<sub>sim</sub> = 4.40 kg/m&sup2; while attenuating to statistical non-significance for diabetes at &sigma;<sub>sim</sub> = 5.60 kg/m&sup2;. Future work includes: (1) prospective multicenter clinical trials in Southeast Asian community clinics; (2) integrating remote photoplethysmography (rPPG) for non-contact arterial pulse wave analysis; (3) deploying lightweight ONNX models for mobile edge triage kiosks; and (4) noise-augmented Stage 2 training to mitigate accuracy degradation under unconstrained low-quality imaging.</p>

<div style="break-before: column; page-break-before: always;"></div>
<h2>REFERENCES</h2>
<div class="ref-item">[1] World Health Organization, <i>Noncommunicable Diseases: Progress Monitor 2020</i>, Geneva, Switzerland: WHO, 2020, and WHO Fact Sheet, "Noncommunicable diseases," 2022.</div>
<div class="ref-item">[2] International Diabetes Federation, <i>IDF Diabetes Atlas</i>, 10th ed. Brussels, Belgium: IDF, 2021.</div>
<div class="ref-item">[3] V. Coetzee, D. I. Perrett, and I. D. Stephen, "Facial adiposity: A reliable cue to health?" <i>Perception</i>, vol. 38, no. 11, pp. 1700–1711, 2009.</div>
<div class="ref-item">[4] L. Wen and G. Guo, "A computational approach to body mass index prediction from face images," <i>Image and Vision Computing</i>, vol. 31, no. 5, pp. 392–400, 2013.</div>
<div class="ref-item">[5] B. J. Lee and J. Y. Kim, "Predicting visceral obesity based on facial characteristics," <i>BMC Complement. Altern. Med.</i>, vol. 14, p. 248, 2014.</div>
<div class="ref-item">[6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, 2016, pp. 770–778.</div>
<div class="ref-item">[7] E. Kocabey, M. Camurcu, F. Ofli, Y. Aytar, J. Marin, A. Torralba, and I. Weber, "Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media," in <i>Proc. 11th Int. AAAI Conf. Web and Social Media (ICWSM)</i>, 2017, pp. 572–575.</div>
<div class="ref-item">[8] I. D. Stephen, V. Hiew, V. Coetzee, B. P. Tiddeman, and D. I. Perrett, "Facial Shape Analysis Identifies Valid Cues to Aspects of Physiological Health in Caucasian, Asian, and African Populations," <i>Front. Psychol.</i>, vol. 8, p. 1883, 2017.</div>
<div class="ref-item">[9] M. Jiang, G. Guo, and G. Mu, "Visual BMI estimation from face images using a label distribution based method," <i>Comput. Vis. Image Underst.</i>, vol. 197–198, p. 102985, 2020.</div>
<div class="ref-item">[10] Centers for Disease Control and Prevention (CDC), <i>National Health and Nutrition Examination Survey (NHANES): Body Composition Procedures Manual</i> and Dual-Energy X-ray Absorptiometry (DXX) Data Documentation (2003–2004 and 2017–2018 cycles), U.S. Dept. of Health &amp; Human Services.</div>
<div style="break-before: column; page-break-before: always;"></div>
<div class="ref-item">[11] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2021.</div>
<div class="ref-item">[12] Y. Gal and Z. Ghahramani, "Dropout as a bayesian approximation: Representing model uncertainty in deep learning," in <i>Proc. Int. Conf. Mach. Learn. (ICML)</i>, 2016, pp. 1050–1059.</div>
<div class="ref-item">[13] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.</i>, 2016, pp. 785–794.</div>
<div class="ref-item">[14] Z.-H. Zhou, <i>Ensemble Methods: Foundations and Algorithms</i>. Boca Raton, FL: CRC Press, 2012.</div>
<div class="ref-item">[15] J. C. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," in <i>Advances in Large Margin Classifiers</i>, A. J. Smola, P. Bartlett, B. Schölkopf, and D. Schuurmans, Eds. Cambridge, MA: MIT Press, 2000, pp. 61–74.</div>
<div class="ref-item">[16] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in <i>Proc. 22nd Int. Conf. Mach. Learn. (ICML)</i>, 2005, pp. 625–632.</div>
<div class="ref-item">[17] W. J. Youden, "Index for rating diagnostic tests," <i>Cancer</i>, vol. 3, no. 1, pp. 32–35, 1950.</div>
<div class="ref-item">[18] C. Elkan, "The foundations of cost-sensitive learning," in <i>Proc. 17th Int. Joint Conf. Artif. Intell. (IJCAI)</i>, 2001, pp. 973–978.</div>
<div class="ref-item">[19] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," <i>J. Mach. Learn. Res.</i>, vol. 12, pp. 2825–2830, 2011.</div>
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
    header_h = 128
    top_y = margin_t + header_h + 10

    header_story = pymupdf.Story(html=HEADER_HTML)
    body_story = pymupdf.Story(html=processed_body_html, archive=archive)

    page_idx = 1
    max_pages = 10

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
        print(f"[+] Successfully compiled Extended Academic Paper PDF: {PDF_OUT} ({page_idx} pages, {pdf_size:,} bytes)")
    except Exception as err:
        print(f"[!] Primary save failed: {err}")
        # Try saving to alt out if Paper.pdf is open, then try to overwrite
        alt_out = os.path.join(REPORTS_DIR, "Paper_temp.pdf")
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
            print(f"[+] Successfully replaced: {PDF_OUT} ({page_idx} pages, {pdf_size:,} bytes)")
        except Exception:
            pdf_size = os.path.getsize(alt_out)
            print(f"[!] Warning: Could not overwrite '{PDF_OUT}'. Saved to '{alt_out}'.")


if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=============================================================================")
    print(" Compiling Full Extended Academic Paper (IEEE Format)                        ")
    print("=============================================================================")
    build_ieee_pdf()
    print("\n[OK] Production Full Academic Paper compiled successfully.")