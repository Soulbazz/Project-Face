import os

paper_script = os.path.join(os.path.dirname(__file__), 'generate_academic_paper_docx.py')
with open(paper_script, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Section III.A: remove margin-bottom on 1) and 2), and remove break-before
assert '<p style="margin-bottom: 7pt;">1) <b>Stage 1 (Computer Vision and Morphometry):</b>' in text
text = text.replace(
    '<p style="margin-bottom: 7pt;">1) <b>Stage 1 (Computer Vision and Morphometry):</b>',
    '<p>1) <b>Stage 1 (Computer Vision and Morphometry):</b>',
    1
)

assert '<p style="margin-bottom: 10pt;">2) <b>Stage 1.5 (Tabular DEXA Regression):</b>' in text
text = text.replace(
    '<p style="margin-bottom: 10pt;">2) <b>Stage 1.5 (Tabular DEXA Regression):</b>',
    '<p>2) <b>Stage 1.5 (Tabular DEXA Regression):</b>',
    1
)

break_sec3 = '<div style="break-before: column; page-break-before: always;"></div>\n<p>3) <b>Stage 2 (Monotonic Calibrated Risk Classification):</b>'
assert break_sec3 in text
text = text.replace(
    break_sec3,
    '<p>3) <b>Stage 2 (Monotonic Calibrated Risk Classification):</b>',
    1
)

# 2. Section IV: remove break-before
break_sec4 = '<h2>IV. EXPERIMENTS AND RESULTS</h2>\n<div style="break-before: column; page-break-before: always;"></div>'
assert break_sec4 in text
text = text.replace(
    break_sec4,
    '<h2>IV. EXPERIMENTS AND RESULTS</h2>',
    1
)

# 3. Explicit dimensions for all 5 figures so MuPDF NEVER squeezes any image
fig0_old = '<img src="fig0_pipeline_architecture.png" alt="Face2Health Cascading Triage Pipeline Architecture">'
assert fig0_old in text
text = text.replace(fig0_old, '<img src="fig0_pipeline_architecture.png" style="width: 248pt; max-width: 248pt; height: 356pt;" alt="Face2Health Cascading Triage Pipeline Architecture">', 1)

fig1_old = '<img src="fig1_stage15_scatter.png" alt="Stage 1.5 DEXA Scatter Plot">'
assert fig1_old in text
text = text.replace(fig1_old, '<img src="fig1_stage15_scatter.png" style="width: 248pt; max-width: 248pt; height: 251pt;" alt="Stage 1.5 DEXA Scatter Plot">', 1)

fig2_old = '<img src="fig2_roc_curves.png" style="width: 190pt;" alt="Stage 2 ROC Curves Benchmark">'
assert fig2_old in text
text = text.replace(fig2_old, '<img src="fig2_roc_curves.png" style="width: 248pt; max-width: 248pt; height: 288pt;" alt="Stage 2 ROC Curves Benchmark">', 1)

fig3_old = '<img src="fig3_calibration.png" alt="Stage 2 Calibration Reliability Diagram">'
assert fig3_old in text
text = text.replace(fig3_old, '<img src="fig3_calibration.png" style="width: 248pt; max-width: 248pt; height: 283pt;" alt="Stage 2 Calibration Reliability Diagram">', 1)

fig4_old = '<img src="fig4_confusion_matrix.png" alt="Stage 2 Confusion Matrix Heatmaps">'
assert fig4_old in text
text = text.replace(fig4_old, '<img src="fig4_confusion_matrix.png" style="width: 248pt; max-width: 248pt; height: 357pt;" alt="Stage 2 Confusion Matrix Heatmaps">', 1)

# 4. Table III: ensure it starts cleanly at the top of Page 6 Col 1 without splitting across Page 5 & 6
table3_old = '<!-- TABLE III-A -->\n<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">\n  <div class="table-caption">TABLE III. DIABETES (DIQ010) CLASSIFICATION BENCHMARK</div>'
assert table3_old in text
table3_new = '<!-- TABLE III-A -->\n<div style="break-before: column; page-break-before: always;"></div>\n<div class="table-container" style="break-inside: avoid; page-break-inside: avoid;">\n  <div class="table-caption">TABLE III. DIABETES (DIQ010) CLASSIFICATION BENCHMARK</div>'
text = text.replace(table3_old, table3_new, 1)

# 5. Table IV improvements: header 'Model' instead of 'Model Architecture', 'Primary (No Waist)'
text = text.replace('<th>Model Architecture</th>', '<th>Model</th>', 1)
text = text.replace('Primary No Waist (Ours)', 'Primary (No Waist)')

# 6. CSS refinements for Table captions, notes, and references
assert 'font-size: 8pt; font-weight: bold; text-align: center;\n    margin: 0 0 2px 0; text-transform: uppercase;' in text
text = text.replace(
    'font-size: 8pt; font-weight: bold; text-align: center;\n    margin: 0 0 2px 0; text-transform: uppercase;',
    'font-size: 7.5pt; font-weight: bold; text-align: center;\n    margin: 0 0 1.5px 0; text-transform: uppercase;',
    1
)

assert 'font-size: 7.8pt;\n    text-align: left;\n    margin-top: 2px;\n    line-height: 1.1;' in text
text = text.replace(
    'font-size: 7.8pt;\n    text-align: left;\n    margin-top: 2px;\n    line-height: 1.1;',
    'font-size: 7.2pt;\n    text-align: left;\n    margin-top: 1.5px;\n    line-height: 1.05;',
    1
)

ref_css_old = '''.ref-item {
    break-inside: avoid;
    page-break-inside: avoid;
    font-size: 8pt; margin-left: 15pt; text-indent: -15pt; margin-bottom: 3.5pt;
    line-height: 10.8pt; text-align: justify;
}'''
assert ref_css_old in text
ref_css_new = '''.ref-item {
    break-inside: avoid;
    page-break-inside: avoid;
    max-width: 238pt;
    font-size: 8pt; margin-left: 15pt; text-indent: -15pt; margin-bottom: 3.5pt;
    line-height: 10.8pt; text-align: justify;
}'''
text = text.replace(ref_css_old, ref_css_new, 1)

# 7. Balanced References on Page 9
assert '<h2>REFERENCES</h2>' in text
text = text.replace('<h2>REFERENCES</h2>', '<div style="break-before: column; page-break-before: always;"></div>\n<h2>REFERENCES</h2>', 1)

ref10_target = '<div class="ref-item">[10] Centers for Disease Control and Prevention (CDC), <i>National Health and Nutrition Examination Survey (NHANES): Body Composition Procedures Manual</i> and Dual-Energy X-ray Absorptiometry (DXX) Data Documentation (2003–2004 and 2017–2018 cycles), U.S. Dept. of Health &amp; Human Services.</div>\n<div class="ref-item">[11]'
assert ref10_target in text
ref10_new = '<div class="ref-item">[10] Centers for Disease Control and Prevention (CDC), <i>National Health and Nutrition Examination Survey (NHANES): Body Composition Procedures Manual</i> and Dual-Energy X-ray Absorptiometry (DXX) Data Documentation (2003–2004 and 2017–2018 cycles), U.S. Dept. of Health &amp; Human Services.</div>\n<div style="break-before: column; page-break-before: always;"></div>\n<div class="ref-item">[11]'
text = text.replace(ref10_target, ref10_new, 1)

with open(paper_script, 'w', encoding='utf-8') as f:
    f.write(text)

print('[SUCCESS] All target replacements verified and applied successfully!')

