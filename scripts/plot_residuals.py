import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('stage1_residuals.csv')

plt.figure(figsize=(10, 6))
plt.scatter(df['y_true'], df['resid'], alpha=0.5)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('True BMI')
plt.ylabel('Residual (True - Pred)')
plt.title('Residual Analysis')
plt.savefig('residual_plot.png')
print("Residual plot saved as residual_plot.png")