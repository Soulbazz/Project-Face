import pandas as pd
import matplotlib.pyplot as plt

# สมมติว่าตอนเทรนคุณมีบันทึก history เป็น csv หรือ log
# ถ้าไม่มี ลองเช็คในโฟลเดอร์ logs หรือ check points ที่โมเดลสร้างไว้
history = pd.read_csv('training_history.csv') # ปรับชื่อไฟล์ให้ตรงกับที่คุณมี

plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()