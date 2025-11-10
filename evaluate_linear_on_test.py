import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. LOAD TEST DATA
# =========================
# Chỉnh lại path nếu cấu trúc thư mục khác
X_test = pd.read_csv("archive/test/x_test.csv")          # (nếu cần dùng full features)
X_test_BIC = pd.read_csv("archive/test/x_test_BIC.csv")  # dùng cho model linear (baseline)
y_test = pd.read_csv("archive/test/y_test.csv").squeeze()

print("Test shape:", X_test_BIC.shape, "y_test shape:", y_test.shape)

# =========================
# 2. LOAD LẠI MÔ HÌNH LINEAR
# =========================
linear_best = joblib.load("./models/linear_best_model.pkl")
print("\n✅ Đã load mô hình linear_best_model.pkl")

# =========================
# 3. DỰ ĐOÁN TRÊN TẬP TEST
# =========================
# Lưu ý: mô hình linear của bạn được train với X_train_BIC,
# nên ở tập test cần dùng X_test_BIC để dự đoán
y_test_pred = linear_best.predict(X_test_BIC)

# =========================
# 4. TÍNH TOÁN CÁC METRICS
# =========================
mae = mean_absolute_error(y_test, y_test_pred)
rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
r2 = r2_score(y_test, y_test_pred)

print("\n=== Hiệu suất mô hình Linear trên TEST set ===")
print(f"Test MAE:  {mae:,.4f}")
print(f"Test RMSE: {rmse:,.4f}")
print(f"Test R²:   {r2:,.4f}")

# (Tuỳ chọn) Lưu metrics ra file CSV
metrics_test = pd.DataFrame(
    {
        "MAE": [mae],
        "RMSE": [rmse],
        "R2": [r2],
    },
    index=["linear_baseline_test"]
)
metrics_test.to_csv("model_metrics_test_linear.csv")
print("\n💾 Đã lưu metrics vào 'model_metrics_test_linear.csv'")

# (Tuỳ chọn) Lưu y_true và y_pred ra file CSV để soi thêm
pred_df = pd.DataFrame(
    {
        "y_true": y_test,
        "y_pred": y_test_pred,
        "residual": y_test - y_test_pred,
    }
)
pred_df.to_csv("linear_test_predictions.csv", index=False)
print("💾 Đã lưu dự đoán test vào 'linear_test_predictions.csv'")

# =========================
# 5. VẼ HÌNH ĐÁNH GIÁ TRÊN TEST
# =========================
os.makedirs("figures_test", exist_ok=True)

# 5.1: Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.4)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual y (test)")
plt.ylabel("Predicted y")
plt.title("Predicted vs Actual - Linear model (TEST)")
plt.tight_layout()
plt.savefig("figures_test/linear_test_pred_vs_actual.png", dpi=200)
# plt.show()

# 5.2: Residuals vs Predicted
residuals = y_test - y_test_pred
plt.figure(figsize=(6, 4))
plt.scatter(y_test_pred, residuals, alpha=0.4)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted y (test)")
plt.ylabel("Residual (y_true - y_pred)")
plt.title("Residuals vs Predicted - Linear model (TEST)")
plt.tight_layout()
plt.savefig("figures_test/linear_test_residuals.png", dpi=200)
# plt.show()

# 5.3: Histogram residuals
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=30)
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Residual distribution - Linear model (TEST)")
plt.tight_layout()
plt.savefig("figures_test/linear_test_residual_hist.png", dpi=200)
# plt.show()

print("\n📊 Đã vẽ và lưu các hình đánh giá vào thư mục 'figures_test/'")
