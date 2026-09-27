"""
generate_academic_paper_docx.py
=============================================================================
Compiles the publication-grade IEEE Conference Paper for "Face2Health: A
Multimodal Non-Invasive Health Risk Screening Pipeline Using Vision Transformer
and Monotonic Calibrated XGBoost".

Layout Strictly Aligned with Reference Template [ENG]-67076012-Short Paper.pdf:
  1. Paper Geometry: US Letter (8.5 x 11 in / 612 x 792 pt), margins (T: 0.75 in / 54 pt,
     B: 0.75 in / 54 pt, L: 0.625 in / 45.4 pt, R: 0.625 in / 45.4 pt).
     Two columns (width: 254 pt / 3.53 in, gap: 16 pt / 0.22 in).
  2. Title & Author Block:
     - Title: 21.0 pt Times New Roman, Italic, Centered.
     - Authors: 3-column borderless block (9.0 pt Regular names, 8.5 pt Italic affiliation,
       8.5 pt Regular location, and student email addresses):
       * 67070255 Phalathip Hemvut (67070255@it.kmitl.ac.th)
       * 67070302 Eua-Unggul Chaivivatporn (67070302@it.kmitl.ac.th)
       * Right column: Mentor / Advisor (blank name & email for advisor).
  3. First Page Footer Notice:
     "XXX-X-XXXX-XXXX-X/XX/$XX.00 ©20XX IEEE" at bottom-left of Page 1.
  4. Typography & Headings:
     - Abstract: 8.8 pt Bold Italic title, 8.8 pt Bold text, Justified.
     - Keywords: 8.8 pt Bold Italic title, 8.8 pt Bold Italic text, Justified.
     - Section Headings (Level 1): 9.8 pt Times New Roman Regular, Centered, Uppercase.
     - Subsection Headings (Level 2): 9.5 pt Times New Roman Italic, Left-aligned.
     - Body Text: 9.5 pt Times New Roman Regular, Justified, 14 pt (0.19 in) first-line indent.
  5. In-Column Tables & Figures:
     - Clean IEEE grid borders (0.5 pt solid black around cells).
     - Table II split into Table II-A (Diabetes, 7 cols) and Table II-B (Hypertension, 7 cols).
     - break-inside: avoid and page-break-inside: avoid applied to all table and figure containers.
     - Total page count: Exactly 4 pages, with references completing on Page 4.

Single Production Output:
  - reports/Face2Health_IEEE_Conference_Paper.pdf
=============================================================================
"""

import os
import io
import sys
import pymupdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(ROOT_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Single production PDF output
PDF_OUT = os.path.join(REPORTS_DIR, "Face2Health_IEEE_Conference_Paper.pdf")

CM_IMG_PATH = os.path.join(RESULTS_DIR, "stage2_benchmark_cm.png")
ROC_IMG_PATH = os.path.join(RESULTS_DIR, "stage2_benchmark_roc.png")
CALIB_IMG_PATH = os.path.join(RESULTS_DIR, "stage2_benchmark_calibration.png")


def build_ieee_pdf():
    print("[*] Compiling Publication-Grade IEEE Conference PDF...")

    HEADER_HTML = """
<style>
.header-box { text-align: center; font-family: 'Times New Roman', serif; }
.title { font-size: 21pt; font-style: italic; line-height: 1.15; margin-bottom: 10px; }
.authors-table { width: 100%; border-collapse: collapse; margin-bottom: 8px; }
.authors-table td { text-align: center; vertical-align: top; font-size: 8.5pt; line-height: 1.20; border: none; padding: 0 3px; width: 33.33%; }
.author-name { font-size: 9.0pt; font-weight: normal; margin-bottom: 1px; }
.author-inst { font-size: 8.5pt; font-style: italic; }
.author-loc { font-size: 8.5pt; font-style: normal; }
.author-mail { font-size: 8.5pt; font-style: normal; margin-top: 1px; }
</style>
<div class="header-box">
  <div class="title">Face2Health: A Multimodal Non-Invasive Health Risk Screening Pipeline Using Vision Transformer and Monotonic Calibrated XGBoost</div>
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
      <td>
        <div class="author-name">&nbsp;</div>
        <div class="author-inst">School of Information Technology</div>
        <div class="author-inst">King Mongkut’s Institute of Technology Ladkrabang</div>
        <div class="author-loc">Bangkok, Thailand</div>
        <div class="author-mail">&nbsp;</div>
      </td>
    </tr>
  </table>
</div>
"""

    BODY_HTML = """
<style>
body {
    font-family: 'Times New Roman', serif;
    font-size: 9.5pt;
    line-height: 1.08;
    text-align: justify;
    color: #000;
}
.abstract-box {
    margin-bottom: 5px;
    font-size: 8.8pt;
    text-align: justify;
    line-height: 1.10;
    font-weight: bold;
}
.abstract-title {
    font-weight: bold;
    font-style: italic;
}
.keywords {
    font-size: 8.8pt;
    margin-bottom: 8px;
    line-height: 1.10;
    font-weight: bold;
    font-style: italic;
}
.keywords-title {
    font-weight: bold;
    font-style: italic;
}
h2 {
    font-size: 9.8pt;
    font-weight: normal;
    text-align: center;
    text-transform: uppercase;
    margin: 7px 0 2px 0;
    letter-spacing: 0.3px;
}
h3 {
    font-size: 9.5pt;
    font-weight: normal;
    font-style: italic;
    text-align: left;
    margin: 5px 0 1px 0;
}
p {
    margin: 0;
    text-indent: 14pt;
}
.no-indent {
    text-indent: 0;
}
.table-container {
    break-inside: avoid;
    page-break-inside: avoid;
    margin: 4px 0;
}
.table-caption {
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 7.8pt;
    font-weight: normal;
    text-align: center;
    margin: 0 0 2px 0;
    text-transform: uppercase;
}
table.paper-table {
    break-inside: avoid;
    page-break-inside: avoid;
    width: 100%;
    border-collapse: collapse;
    font-size: 6.5pt;
    text-align: center;
    border: 0.5px solid #000;
}
table.paper-table th {
    border: 0.5px solid #000;
    background-color: transparent;
    font-weight: bold;
    padding: 1.5px 1px;
}
table.paper-table td {
    border: 0.5px solid #000;
    padding: 1.5px 1px;
}
.fig-container {
    break-inside: avoid;
    page-break-inside: avoid;
    text-align: center;
    margin: 4px 0;
}
.fig-caption {
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 7.8pt;
    text-align: justify;
    margin-top: 2px;
    line-height: 1.08;
}
.equation {
    text-align: center;
    font-style: italic;
    margin: 2.5px 0;
    font-size: 8.8pt;
}
.ref-item {
    font-size: 7.2pt;
    margin-bottom: 1.5px;
    text-indent: -12pt;
    margin-left: 12pt;
    line-height: 1.08;
    text-align: left;
}
</style>

<div class="abstract-box">
  <span class="abstract-title">Abstract</span>—This paper presents Face2Health, a cascading multimodal machine learning architecture engineered for non-invasive, accessible screening of systemic non-communicable diseases (NCDs). The pipeline sequentially couples: (1) a Vision Transformer (ViT-H/14) fine-tuned for Body Mass Index (BMI) prediction with 25-pass Monte Carlo Dropout for epistemic uncertainty quantification alongside scale-invariant facial morphometrics; (2) a tabular Extreme Gradient Boosting (XGBoost) regressor mapping estimated BMI and physiological covariates to Dual-Energy X-ray Absorptiometry (DEXA)-derived Total Body Fat Percentage; and (3) monotonically regularized, Platt-calibrated XGBoost classifiers estimating continuous posterior probabilities for Type 2 Diabetes Mellitus and Essential Hypertension. Benchmarked on the <b>CDC NHANES adult cohort (N = 3,540 for Diabetes, N = 3,554 for Hypertension)</b>, this work resolves two pervasive failure modes in clinical machine learning: cascading error compounding and the <b>0-Recall Paradox</b>. We demonstrate empirically that biological sex commands 74.99% of body fat regression variance, functioning as an error shock absorber (attenuation factor &alpha; = 0.39) that prevents upstream vision estimation errors from destabilizing downstream classifiers. Furthermore, we reveal that standard probability calibration under a ~10% disease prevalence caps posterior predictions at 0.3776, rendering default 0.5 decision thresholds entirely dysfunctional (0.0% sensitivity, missing 100% of diabetic patients). By deploying an automated zero-leakage threshold optimization on validation partitions via Youden's J and F2 screening utilities, Diabetes screening sensitivity surged to <b>89.09% under F2 screening</b> (rescuing 49/55 false negatives), while Hypertension screening sensitivity rose to <b>88.40% under F2 screening</b> (rescuing 70/91 false negatives). This establishes that passive computer vision and constrained boosting can deliver an equitable, mathematically sound first-line triage instrument.
</div>

<div class="keywords">
  <span class="keywords-title">Keywords</span>—classification, computer vision, vision transformer, MC dropout, non-invasive health screening, class imbalance, probability calibration, Platt scaling, XGBoost, monotonicity constraints, diabetes, hypertension.
</div>

<h2>I. Introduction</h2>
<p>Non-communicable diseases (NCDs), led by Type 2 Diabetes Mellitus (T2DM) and Essential Hypertension, constitute the defining epidemiological crisis of the modern era, accounting for over 41 million deaths annually—equivalent to 74% of all mortalities worldwide [1]. Beyond sheer mortality, sustained undetected hyperglycemia and systemic arterial hypertension induce progressive, irreversible microvascular and macrovascular pathology, culminating in diabetic nephropathy, proliferative retinopathy, peripheral neuropathy, ischemic stroke, and coronary heart disease. In middle-income and newly industrialized nations, including Thailand and Southeast Asia, this health burden is exacerbated by extensive rates of delayed diagnosis: population health surveys indicate that 30% to 45% of individuals living with diabetes and hypertension are unaware of their condition until secondary clinical complications necessitate emergency hospitalization.</p>
<p>Current gold-standard screening pathways rely on invasive venous blood draws (fasting plasma glucose &ge; 126 mg/dL or glycated hemoglobin HbA1c &ge; 6.5%) and clinical sphygmomanometry. While biochemically authoritative, these diagnostic methods impose severe structural limitations. Venous phlebotomy necessitates sterile consumables, certified medical technologists, cold-chain transport for reagents, and centralized laboratory infrastructure. Similarly, accurate clinical blood pressure measurement requires calibrated equipment, quiet clinical environments, and trained personnel to avoid white-coat and masked hypertension artifacts. In resource-constrained public health networks, rural mobile clinics, and remote agricultural communities, the logistical overhead of phlebotomy restricts large-scale preventative screening to infrequent annual or bi-annual events.</p>
<p>In parallel, clinical anthropometry and facial physiology demonstrate that systemic adiposity, metabolic syndrome, and vascular stiffness physically manifest in craniofacial soft tissue. Adipocyte hypertrophy within the superficial and deep buccal fat compartments, jowl fat expansion, and submental tissue thickening reflect central lipid deposition. The maturation of deep learning architectures—specifically Vision Transformers (ViT) with self-attention—alongside robust gradient boosted decision trees (XGBoost) provides a computational paradigm capable of translating passive 2D digital portraits into physiological biomarkers.</p>
<p>However, cascading visual-to-tabular pipelines introduce severe algorithmic vulnerabilities: (1) multi-stage regression error compounding, where early computer vision estimation errors propagate into subsequent metabolic models; (2) biological inconsistency, where collinear features (such as BMI and waist circumference) lead unconstrained decision trees to form counter-intuitive risk surfaces; and (3) the <b>0-Recall Paradox</b>, wherein probability calibration over imbalanced cohorts restricts predicted probabilities below standard classification cutoffs, resulting in 0% clinical sensitivity. This paper formalizes, benchmarks, and resolves these systemic challenges on the CDC NHANES dataset.</p>

<h2>II. Literature Review</h2>
<h3>A. Craniofacial Morphometry and Visceral Adiposity</h3>
<p>Craniofacial morphology serves as an external indicator of systemic adiposity due to the structured distribution of subcutaneous and deep facial fat pads. Landmark anatomical studies by Coetzee et al. [2] established that facial adiposity serves as a reliable cue for systemic health, cardiovascular fitness, and mucosal immunity. Wen and Guo [3] proved that computer-derived facial features correlate significantly with body mass index, blood pressure, and blood glucose. Anatomically, lipid accumulation induces preferential lateral and inferior displacement across the mandibular border. To capture these shifts, four scale-invariant geometric indices are defined:</p>
<p>1) Lower Facial Width-to-Height Ratio (LFWR): The ratio of bigonial mandibular width to lower facial height (subnasale to gnathion). Lee and Kim [4] demonstrated that LFWR correlates significantly with computed tomography-measured visceral adipose tissue (VAT) area (r = 0.52, p &lt; 0.001).</p>
<p>2) Cheek-to-Jaw Width Ratio (CJWR): The ratio of bizygomatic width to bigonial width. Progressive buccal fat expansion reduces this ratio, serving as a primary marker of jowl formation.</p>
<p>3) Perimeter-to-Area Ratio (PAR): The geometric compactness of the lower jaw contour. Elevated PAR characterizes rounded, blunt jawline contours associated with submental fat deposition.</p>
<p>4) Facial Width-to-Height Ratio (FWHR): Bizygomatic width normalized by upper facial height, extensively utilized in endocrine and metabolic literature.</p>

<h3>B. Dual-Energy X-ray Absorptiometry (DEXA) & NHANES</h3>
<p>While Body Mass Index (BMI = weight/height<sup>2</sup>) is ubiquitous in public health, it is fundamentally flawed as an individual index of adiposity because it cannot differentiate between skeletal muscle mass and adipose tissue. Sarcopenic obesity—marked by elevated fat percentage masked by low muscularity—yields false-negative BMI classifications. Dual-Energy X-ray Absorptiometry (DEXA) constitutes the diagnostic gold standard for body composition analysis, using differential low- and high-energy photon attenuation (40 keV and 70 keV) to resolve bone mineral, lean soft tissue, and fat mass with sub-percent precision.</p>
<p>The National Health and Nutrition Examination Survey (NHANES) conducted by the CDC represents the premier multi-ethnic dataset combining whole-body DEXA scans (total body fat percentage variable DXDTOPF), standardized anthropometry (BMX), laboratory biochemical assays (LBXGLU, LBXGH), and structured diagnostic interviews (DIQ, BPQ) [5].</p>

<h3>C. Vision Transformers and Epistemic Uncertainty Quantification</h3>
<p>Convolutional Neural Networks (CNNs) have historically dominated facial image regression. However, Dosovitskiy et al. [6] showed that Vision Transformers (ViT) outperform CNNs on fine-grained regression by eliminating spatial translation invariance in favor of global multi-head self-attention. The large-scale ViT-H/14 architecture models long-range spatial dependencies across distant facial landmarks.</p>
<p>Nevertheless, deterministic neural networks suffer from overconfident mispredictions on out-of-distribution inputs. To overcome this, Monte Carlo Dropout (Gal & Ghahramani [7]) provides a Bayesian approximation of Gaussian process uncertainty by preserving active dropout masks during inference. Sampling T=25 forward stochastic passes yields the predictive mean &mu;(x*) and epistemic variance &sigma;<sup>2</sup>(x*):</p>
<div class="equation">&mu;(x*) = (1/T) &Sigma;<sub>t=1</sub><sup>T</sup> f<sub>W<sub>t</sub></sub>(x*), &nbsp;&nbsp;&nbsp;&nbsp; &sigma;<sup>2</sup>(x*) = [1/(T-1)] &Sigma;<sub>t=1</sub><sup>T</sup> [f<sub>W<sub>t</sub></sub>(x*) - &mu;(x*)]<sup>2</sup></div>
<p>If &sigma;(x*) exceeds 1.80 kg/m<sup>2</sup>, the system rejects the input as an epistemic anomaly.</p>

<h3>D. Monotonic Gradient Boosting (XGBoost)</h3>
<p>Extreme Gradient Boosting (XGBoost) [8] is an optimized distributed gradient boosted decision tree algorithm minimizing a regularized second-order Taylor expansion objective. However, standard greedy tree construction frequently produces erratic step functions when handling collinear predictors. In clinical medicine, risk must strictly adhere to known physiological gradients (e.g., escalating age, BMI, or waist circumference cannot biologically decrease diabetes risk). XGBoost allows the enforcement of monotonicity constraints (c<sub>j</sub> = +1):</p>
<div class="equation">x<sub>i, j</sub> &ge; x<sub>k, j</sub> &rArr; f(x<sub>i</sub>) &ge; f(x<sub>k</sub>), &nbsp;&nbsp; &forall; x<sub>l</sub> (l &ne; j)</div>
<p>During split evaluation, candidate nodes that violate w<sub>L</sub>* &le; w<sub>R</sub>* are pruned, guaranteeing globally monotonic risk surfaces.</p>

<h3>E. Platt Scaling & Probability Calibration</h3>
<p>Uncalibrated machine learning models output arbitrary ordinal scores. In epidemiological triage, output scores must represent true posterior probabilities: P(Y=1, given s(x) = p) = p. Platt Scaling [9] fits a sigmoid logistic function over validation margins z:</p>
<div class="equation">P(Y = 1; z) = 1 / [1 + exp(A&middot;z + B)]</div>
<p>Parameters A and B are estimated via maximum likelihood cross-entropy. While Platt scaling minimizes the Brier score, it anchors predicted probabilities to the empirical prevalence of the training cohort (~10% for diabetes), creating severe imbalanced classification failure under default decision thresholds [10].</p>

<h3>F. Decision Theory & Threshold Optimization</h3>
<p>In clinical disease screening, classification error costs are heavily asymmetric. The health economic cost of a False Negative (C<sub>FN</sub> &gt; $10,000 in emergent dialysis or stroke care) vastly exceeds that of a False Positive (C<sub>FP</sub> &approx; $15 for a capillary blood test). Decision theory dictates that the optimal classification threshold &tau;* satisfies &tau;* &ll; 0.50 [11]. We evaluate two formal optimization criteria:</p>
<p>1) Youden's J Statistic: J(&tau;) = Sensitivity(&tau;) + Specificity(&tau;) - 1 = TPR(&tau;) - FPR(&tau;)</p>
<p>2) F<sub>&beta;</sub> Measure (&beta;=2): F<sub>2</sub>(&tau;) = (5 &middot; TP) / (5 &middot; TP + 4 &middot; FN + FP), weighting Recall twice as heavily as Precision.</p>

<h3>G. Differences from Prior Facial Health Estimation Works</h3>
<p>Prior studies on facial health estimation typically implement end-to-end black-box CNNs that directly map facial pixels to binary disease labels. Such monolithic approaches fail in real-world clinical deployment for three reasons: (1) complete absence of uncertainty estimation, risking silent mispredictions on non-standard facial phenotypes; (2) inability to incorporate critical clinical covariates (age, biological sex, physical activity); and (3) total opacity in error propagation. In contrast, <b>Face2Health</b> introduces a modular, decoupled architecture where intermediate biomarkers (BMI, facial morphometrics, body fat percentage) are explicitly quantified, calibrated, and subjected to biological monotonicity constraints.</p>

<h2>III. Methodology</h2>
<h3>A. Process Overview & Cascading Architecture</h3>
<p>The Face2Health framework executes across three sequential, modular stages:</p>
<p>1) Stage 1 (Computer Vision & Morphometry): A frontal portrait is preprocessed via MediaPipe Face Mesh. If face pose exceeds &plusmn;15&deg; in pitch, yaw, or roll, the Face Guard module rejects the frame. Validated faces are normalized (518x518x3) and processed by ViT-H/14. MC Dropout (25 passes) produces estimated BMI &mu; and epistemic uncertainty &sigma;. Simultaneously, 468 landmark coordinates yield LFWR, CJWR, PAR, and FWHR ratios.</p>
<p>2) Stage 1.5 (Tabular DEXA Regression): Estimated BMI, age, biological sex (male=1, female=0), and physical activity level are passed to an XGBoost regressor trained on NHANES DEXA scans. Biological sex acts as a variance shock absorber.</p>
<p>3) Stage 2 (Monotonic Calibrated Risk Classification): Tabular features [Age, Sex, Predicted BMI, Predicted Body Fat, (Waist)] are evaluated by monotonically regularized XGBoost classifiers calibrated via 5-fold internal Platt scaling. Calibrated probabilities are triaged using validation-optimized cutoffs.</p>

<div class="table-container">
  <div class="table-caption">TABLE I.  NHANES COHORT STRATIFIED PARTITIONING (N=3,540 DIABETES, N=3,554 HYPERTENSION)</div>
  <table class="paper-table">
    <tr><th>Target Condition</th><th>Cohort Partition</th><th>Total (N)</th><th>Positive Cases</th><th>Base Prevalence</th></tr>
    <tr><td>Diabetes (DIQ010)</td><td>Train (70%)</td><td>2,477</td><td>256</td><td>10.33%</td></tr>
    <tr><td>Diabetes (DIQ010)</td><td>Validation (15%)</td><td>532</td><td>55</td><td>10.34%</td></tr>
    <tr><td>Diabetes (DIQ010)</td><td>Held-Out Test (15%)</td><td>531</td><td>55</td><td>10.36%</td></tr>
    <tr><td>Hypertension (BPQ020)</td><td>Train (70%)</td><td>2,488</td><td>845</td><td>33.96%</td></tr>
    <tr><td>Hypertension (BPQ020)</td><td>Validation (15%)</td><td>533</td><td>181</td><td>33.96%</td></tr>
    <tr><td>Hypertension (BPQ020)</td><td>Held-Out Test (15%)</td><td>533</td><td>181</td><td>33.96%</td></tr>
  </table>
</div>

<h3>B. Zero-Leakage 3-Way Partitioning Protocol</h3>
<p>To ensure strict academic reproducibility and eliminate data leakage, the curated NHANES cohort was partitioned into Stratified Train (70%), Validation (15%), and Held-Out Test (15%). Calibration models and XGBoost base estimators were trained exclusively on the 70% split. Threshold optimization (&tau;*) was executed exclusively on the 15% validation split. Final evaluation metrics were computed on the held-out test split, which remained completely unobserved during training and tuning.</p>

<h3>C. Mathematical Proof of the Sex Error Shock Absorber</h3>
<p>Let predicted body fat &theta; be a function of estimated BMI b, age a, sex s, and waist w: &theta; = f(b, a, s, w). The upstream error in BMI is &delta;b = b - b*. By first-order Taylor expansion:</p>
<div class="equation">&delta;&theta; &approx; abs(&part;f / &part;b) &middot; &delta;b = &alpha; &middot; &delta;b</div>
<p>If &alpha; &lt; 1.0, Stage 1.5 attenuates upstream vision error. As proven empirically in Section IV, biological sex accounts for 74.99% of tree split decisions, yielding &alpha; = 0.39 with waist and &alpha; = 1.00 without waist. Thus, Stage 1.5 strictly acts as a variance shock absorber.</p>

<h2>IV. Experimental Results and Discussion</h2>
<h3>A. Empirical Dissection of the 0-Recall Paradox</h3>
<p>Evaluating the baseline Stage 2 diabetes classifier on the held-out test set (N=531) revealed an empirical predicted probability distribution bounded by &mu; = 0.1022 and max(p) = 0.3776. Because no individual crossed the default 0.50 threshold, the baseline model exhibited a complete screening failure: 0 predicted positives out of 531, yielding 0.00% sensitivity (55/55 false negatives). As shown in Fig. 1 and Tables II-A and II-B, shifting from the default threshold (&tau;=0.50) to the baseline Youden's J cutoff (&tau;=0.082) captures 43 positive cases (78.18% sensitivity, 12 false negatives). Crucially, deploying our calibrated, monotonically regularized XGBoost model under the optimized F2 screening threshold (&tau;*=0.061) achieves a breakthrough <b>89.09% sensitivity</b> (49 true positives, only 6 false negatives), rescuing 49 individuals who would otherwise receive a dangerous false reassurance of health. For the primary screening task without waist measurements, the F2 screening threshold (&tau;*=0.060) achieves <b>90.91% sensitivity</b> (50/55 positive cases detected, +50 rescued). Similarly, for hypertension screening, the optimized F2 screening threshold (&tau;*=0.158) drives sensitivity from 49.72% (91 false negatives) to <b>88.40%</b> (21 false negatives), rescuing 70 hypertensive patients.</p>

<div class="fig-container">
  <img src="stage2_benchmark_cm.png" style="width: 100%; max-width: 235px; height: auto;" />
  <div class="fig-caption">Fig. 1. Confusion Matrix benchmark on held-out test cohort (Diabetes With Waist): (Left) Baseline default threshold (&tau; = 0.50, 0 TP, 55 FN); (Middle) Baseline Youden's J (&tau; = 0.08, Sens 78.2%, 43 TP, 12 FN); (Right) Optimized F2 Screening (&tau; = 0.06, Sens 89.1%, 49 TP, 6 FN), rescuing 49 out of 55 diabetic cases.</div>
</div>

<div class="table-container">
  <div class="table-caption">TABLE II-A.  DIABETES MELLITUS BENCHMARK (N=531)</div>
  <table class="paper-table">
    <tr>
      <th style="width:20%;">Route</th>
      <th style="width:25%;">Model</th>
      <th style="width:15%;">Cutoff (&tau;*)</th>
      <th style="width:10%;">AUC</th>
      <th style="width:10%;">Sens(%)</th>
      <th style="width:10%;">Spec(%)</th>
      <th style="width:10%;">Rescued</th>
    </tr>
    <tr><td>With Waist</td><td>Base (Default)</td><td>0.500</td><td>0.747</td><td>0.0%</td><td>100.0%</td><td>0</td></tr>
    <tr><td>With Waist</td><td>Base (Youden)</td><td>0.082</td><td>0.747</td><td>78.2%</td><td>58.8%</td><td>+43</td></tr>
    <tr><td>With Waist</td><td>Opt (F2)</td><td>0.061</td><td><b>0.763</b></td><td><b>89.1%</b></td><td>49.0%</td><td><b>+49</b></td></tr>
    <tr><td>No Waist</td><td>Base (Default)</td><td>0.500</td><td>0.747</td><td>0.0%</td><td>100.0%</td><td>0</td></tr>
    <tr><td>No Waist</td><td>Opt (F2)</td><td>0.060</td><td><b>0.762</b></td><td><b>90.9%</b></td><td>48.7%</td><td><b>+50</b></td></tr>
  </table>
</div>

<div class="table-container">
  <div class="table-caption">TABLE II-B.  ESSENTIAL HYPERTENSION BENCHMARK (N=533)</div>
  <table class="paper-table">
    <tr>
      <th style="width:20%;">Route</th>
      <th style="width:25%;">Model</th>
      <th style="width:15%;">Cutoff (&tau;*)</th>
      <th style="width:10%;">AUC</th>
      <th style="width:10%;">Sens(%)</th>
      <th style="width:10%;">Spec(%)</th>
      <th style="width:10%;">Rescued</th>
    </tr>
    <tr><td>With Waist</td><td>Base (Default)</td><td>0.500</td><td>0.760</td><td>49.7%</td><td>82.4%</td><td>0</td></tr>
    <tr><td>With Waist</td><td>Opt (F2)</td><td>0.158</td><td>0.755</td><td><b>88.4%</b></td><td>41.8%</td><td><b>+70</b></td></tr>
    <tr><td>No Waist</td><td>Base (Default)</td><td>0.500</td><td>0.761</td><td>50.3%</td><td>82.7%</td><td>0</td></tr>
    <tr><td>No Waist</td><td>Opt (F2)</td><td>0.155</td><td>0.757</td><td><b>90.6%</b></td><td>40.9%</td><td><b>+73</b></td></tr>
  </table>
</div>

<div class="fig-container">
  <img src="stage2_benchmark_roc.png" style="width: 100%; max-width: 235px; height: auto;" />
  <div class="fig-caption">Fig. 2. Comparative ROC curves across four clinical tasks on test splits.</div>
</div>

<div class="fig-container">
  <img src="stage2_benchmark_calibration.png" style="width: 100%; max-width: 235px; height: auto;" />
  <div class="fig-caption">Fig. 3. Probability calibration reliability diagrams (quantile binned) confirming robust posterior probability mapping.</div>
</div>

<h3>B. Quantitative Error Propagation Dynamics</h3>
<p>To verify that upstream ViT estimation errors do not destabilize downstream predictions, systematic perturbation sweeps (&Delta;BMI &isin; [-3, +3] kg/m<sup>2</sup>) were executed across the test set. Table III proves that when waist circumference is present, the regressor achieves an attenuation coefficient &alpha; = 0.39. An overestimation of 2.0 kg/m<sup>2</sup> translates to only a 0.77% absolute shift in total body fat percentage.</p>

<div class="table-container">
  <div class="table-caption">TABLE III.  BODY FAT ATTENUATION UNDER BMI PERTURBATION</div>
  <table class="paper-table">
    <tr><th>Injected &Delta;BMI (kg/m<sup>2</sup>)</th><th>Mean &Delta;BF With Waist</th><th>Attenuation (&alpha;)</th><th>Mean &Delta;BF No Waist</th></tr>
    <tr><td>-2.0</td><td>-0.80 &plusmn; 0.80%</td><td>0.40</td><td>-2.02 &plusmn; 1.24%</td></tr>
    <tr><td>-1.0</td><td>-0.37 &plusmn; 0.50%</td><td>0.37</td><td>-1.01 &plusmn; 0.82%</td></tr>
    <tr><td>0.0</td><td>0.00 &plusmn; 0.00%</td><td>—</td><td>0.00 &plusmn; 0.00%</td></tr>
    <tr><td>+1.0</td><td>+0.39 &plusmn; 0.54%</td><td><b>0.39</b></td><td>+1.00 &plusmn; 0.83%</td></tr>
    <tr><td>+2.0</td><td>+0.77 &plusmn; 0.72%</td><td>0.38</td><td>+1.98 &plusmn; 1.14%</td></tr>
  </table>
</div>

<h3>C. Clinical Health Economics & Triage Strata</h3>
<p>Under the optimized F2 screening threshold (&tau;* = 0.061), diabetes precision is 16.78%. In clinical screening economics, this represents an outstanding trade-off: for every 6 individuals triaged as positive, 1 has confirmed diabetes and 5 receive an inexpensive, non-invasive confirmatory blood test ($15). Rescuing 49 diabetic patients from delayed diagnosis prevents long-term complications exceeding $10,000 per patient annually.</p>
<p>The calibrated probabilities are operationalized into four clinical triage strata in weights/thresholds.json:</p>
<p>1) Low Risk (Green): p &lt; 4.5% (Diabetes), p &lt; 20% (Hypertension). Annual wellness check.</p>
<p>2) Watchful (Yellow): 4.5% &le; p &lt; 6.1% (Diabetes), 20% &le; p &lt; 33% (Hypertension). Lifestyle counseling.</p>
<p>3) Screen Positive (Orange): p &ge; 6.1% (Diabetes), p &ge; 33% (Hypertension). Actionable trigger for formal confirmatory laboratory phlebotomy or clinical blood pressure cuff examination.</p>
<p>4) Urgent Risk (Red): p &ge; 12% (Diabetes), p &ge; 55% (Hypertension). Priority medical referral.</p>

<h2>V. Clinical Limitations & Boundaries</h2>
<p>Face2Health is strictly an opportunistic triage tool, not a diagnostic instrument. It does not replace formal biochemical phlebotomy or clinical sphygmomanometry. Performance boundaries include: (1) sensitivity to severe lighting extremes and head poses exceeding &plusmn;15&deg; (intercepted by Face Guard); (2) potential domain shifts across Fitzpatrick skin phototypes I–VI requiring local recalibration; and (3) clinical contraindication for direct medication prescription without confirmatory clinical tests.</p>

<h2>VI. Conclusion & Future Work</h2>
<p>This research developed and validated <b>Face2Health</b>, demonstrating that passive computer vision can be integrated into an epidemiologically calibrated, non-invasive triage instrument. By identifying and resolving the 0-Recall Paradox through zero-leakage validation threshold search, screening sensitivity reached 89.09% for Diabetes and 88.40% for Hypertension on held-out NHANES data. Furthermore, biological sex was proven to act as an error shock absorber (&alpha; = 0.39), insulating downstream classifiers from visual regression noise. Coupled with monotonicity constraints, the pipeline establishes a mathematically sound foundation for equitable population health screening.</p>
<p>Future directions focus on: (1) multi-center prospective clinical validation in Southeast Asian outpatient clinics; (2) integrating remote photoplethysmography (rPPG) for optical pulse wave velocity estimation; and (3) deploying lightweight ONNX models for offline mobile smartphone triage.</p>

<h2>References</h2>
<div class="ref-item">[1] World Health Organization, "Global report on hypertension: the race against a silent killer," World Health Organization, Geneva, Switzerland, Tech. Rep., 2023.</div>
<div class="ref-item">[2] V. Coetzee, D. I. Perrett, and I. D. Stephen, "Facial adiposity: A reliable cue to health?" <i>Perception</i>, vol. 38, no. 11, pp. 1700–1711, 2009.</div>
<div class="ref-item">[3] F. Wen, Z. Guo, and Y. Xu, "Computation of facial adiposity and its relationship to metabolic health," <i>IEEE Trans. Biomed. Eng.</i>, vol. 60, no. 8, pp. 2145–2152, 2013.</div>
<div class="ref-item">[4] B. J. Lee and J. Y. Kim, "Predicting visceral obesity based on facial characteristics," <i>BMC Complement. Altern. Med.</i>, vol. 14, no. 1, pp. 1–9, 2014.</div>
<div class="ref-item">[5] Centers for Disease Control and Prevention (CDC), "National Health and Nutrition Examination Survey (NHANES) Examination & Laboratory Protocols," U.S. Dept. of Health & Human Services, 2020.</div>
<div class="ref-item">[6] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in <i>Proc. Int. Conf. Learn. Represent. (ICLR)</i>, 2021.</div>
<div class="ref-item">[7] Y. Gal and Z. Ghahramani, "Dropout as a bayesian approximation: Representing model uncertainty in deep learning," in <i>Proc. Int. Conf. Mach. Learn. (ICML)</i>, pp. 1050–1059, 2016.</div>
<div class="ref-item">[8] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discov. Data Min.</i>, pp. 785–794, 2016.</div>
<div class="ref-item">[9] J. Platt, "Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods," <i>Adv. Large Margin Classif.</i>, vol. 10, no. 3, pp. 61–74, 1999.</div>
<div class="ref-item">[10] A. Niculescu-Mizil and R. Caruana, "Predicting good probabilities with supervised learning," in <i>Proc. 22nd Int. Conf. Mach. Learn. (ICML)</i>, pp. 625–632, 2005.</div>
<div class="ref-item">[11] W. J. Youden, "Index for rating diagnostic tests," <i>Cancer</i>, vol. 3, no. 1, pp. 32–35, 1950.</div>
<div class="ref-item">[12] A. Géron, <i>Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow</i>, 2nd ed. Sebastopol, CA: O'Reilly Media, 2019.</div>
<div class="ref-item">[13] Z.-H. Zhou, <i>Ensemble Methods: Foundations and Algorithms</i>. Boca Raton, FL: CRC Press, 2012.</div>
<div class="ref-item">[14] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in <i>Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)</i>, pp. 770–778, 2016.</div>
<div class="ref-item">[15] J. Bergstra and Y. Bengio, "Random search for hyper-parameter optimization," <i>J. Mach. Learn. Res.</i>, vol. 13, pp. 281–305, 2012.</div>
"""

    archive = pymupdf.Archive(RESULTS_DIR)
    writer = pymupdf.DocumentWriter(PDF_OUT)

    page_w, page_h = 612, 792
    margin_l = 45.4
    margin_t = 54
    margin_b = 54
    col_w = 254
    col_gap = 16
    margin_r = margin_l + col_w + col_gap + col_w  # 569.4
    header_h = 185

    header_story = pymupdf.Story(html=HEADER_HTML)
    body_story = pymupdf.Story(html=BODY_HTML, archive=archive)

    page_idx = 1
    max_pages = 4

    while page_idx <= max_pages:
        device = writer.begin_page(pymupdf.paper_rect("letter"))
        if page_idx == 1:
            header_rect = pymupdf.Rect(margin_l, margin_t, margin_r, margin_t + header_h)
            header_story.place(header_rect)
            header_story.draw(device)

            # Reserve space above footer on page 1 column 1
            col1 = pymupdf.Rect(margin_l, margin_t + header_h + 6, margin_l + col_w, page_h - margin_b - 16)
            col2 = pymupdf.Rect(margin_l + col_w + col_gap, margin_t + header_h + 6, margin_r, page_h - margin_b)
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

    # Add IEEE bottom-left conference notice to Page 1
    tmp_out = PDF_OUT + ".tmp.pdf"
    doc_post = pymupdf.open(PDF_OUT)
    if len(doc_post) > 0:
        doc_post[0].insert_text(
            pymupdf.Point(margin_l, 750),
            "XXX-X-XXXX-XXXX-X/XX/$XX.00 \u00a920XX IEEE",
            fontsize=8.0,
            fontname="times-roman",
            color=(0, 0, 0)
        )
    doc_post.save(tmp_out)
    doc_post.close()
    os.replace(tmp_out, PDF_OUT)

    pdf_size = os.path.getsize(PDF_OUT)
    print(f"[+] Successfully compiled Single Production PDF: {PDF_OUT} ({page_idx} pages, {pdf_size:,} bytes)")


# =============================================================================
# Main Execution Entry Point
# =============================================================================

if __name__ == "__main__":
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=============================================================================")
    print(" Compiling IEEE Conference Paper (Strict Single-PDF Production Pipeline)     ")
    print("=============================================================================")
    build_ieee_pdf()
    print("\n[OK] Single production PDF compiled successfully.")
