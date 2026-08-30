import matplotlib.pyplot as plt

# ข้อมูลจากโมเดล
models = ['Basic Model\n(No Waist)', 'Full Model\n(With Waist)']
r2_scores = [0.7340, 0.7434]
mae_scores = [3.62, 3.57]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# ----------------------------------------------------
# 1. กราฟฝั่งซ้าย: R-squared (สเกล 0 ถึง 1.0 ที่ถูกต้อง)
# ----------------------------------------------------
bars1 = ax1.bar(models, r2_scores, color=['#8cb3d9', '#4da6ff'], edgecolor='black', width=0.5)
ax1.set_title('Model Confidence (R-squared)\n[Higher is Better]', fontsize=14, pad=15)
ax1.set_ylim(0, 1.0)  # กำหนดให้ถูกต้อง: R-squared มีค่าเต็มคือ 1.0
ax1.set_ylabel('R-squared Score', fontsize=12)

for bar in bars1:
    yval = bar.get_height()
    # วางตัวเลขไว้เหนือแท่งกราฟพอดี ไม่ลอยเกินไป
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.03, f"{yval:.4f}", 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

# ----------------------------------------------------
# 2. กราฟฝั่งขวา: MAE (สเกล 0 ถึง 5.0)
# ----------------------------------------------------
bars2 = ax2.bar(models, mae_scores, color=['#ffb3b3', '#ff6666'], edgecolor='black', width=0.5)
ax2.set_title('Mean Absolute Error (MAE)\n[Lower is Better]', fontsize=14, pad=15)
ax2.set_ylim(0, 5.0)  # กำหนดให้ถูกต้อง: MAE ใช้สเกลเปอร์เซ็นต์ Error สูงสุดที่ 5%
ax2.set_ylabel('Error Percentage (%)', fontsize=12)

for bar in bars2:
    yval = bar.get_height()
    # ปรับตำแหน่งตัวหนังสือไม่ให้ชนขอบบนของกราฟ
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"±{yval:.2f}%", 
             ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
output_name = 'honest_model_comparison_chart.png'
plt.savefig(output_name, dpi=300)
plt.show()
print(f"✅ บันทึกกราฟสำเร็จ! ไฟล์ชื่อ: {output_name}")