import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# 1. LOAD DATA
# =========================

# Chỉnh lại path nếu cần
X_train = pd.read_csv("archive/train/x_train.csv")
y_train = pd.read_csv("archive/train/y_train.csv").squeeze()

X_val = pd.read_csv("archive/val/x_val.csv")
y_val = pd.read_csv("archive/val/y_val.csv").squeeze()

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# =========================
# 2. KHAI BÁO 4 MODEL
# =========================
baseline = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LinearRegression())
    ]
)

ridge = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("model", Ridge(alpha=1.0, random_state=42))
    ]
)

lasso = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("model", Lasso(alpha=0.001, random_state=42, max_iter=10000))
    ]
)

elastic_net = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=False)),
        ("model", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, max_iter=10000))
    ]
)

models = {
    "baseline_linear": baseline,
    "ridge": ridge,
    "lasso": lasso,
    "elasticnet": elastic_net,
}

# =========================
# 3. TRAIN + EVALUATE
# =========================

results = []          # lưu lại bảng metrics
predictions = {}      # lưu y_pred để vẽ hình

for name, model in models.items():
    print(f"\n=== Training model: {name} ===")
    model.fit(X_train, y_train)
    
    y_val_pred = model.predict(X_val)
    predictions[name] = y_val_pred
    
    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
    r2 = r2_score(y_val, y_val_pred)
    
    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    })
    
    print(f"Validation MAE:  {mae:,.4f}")
    print(f"Validation RMSE: {rmse:,.4f}")
    print(f"Validation R²:   {r2:,.4f}")

# Đưa về DataFrame cho dễ xem
results_df = pd.DataFrame(results).set_index("model")
print("\n=== Summary metrics on validation set ===")
print(results_df)

# Lưu ra CSV nếu muốn chèn vào report
results_df.to_csv("model_metrics_val.csv")

# =========================
# 4. VẼ HÌNH SO SÁNH
# =========================

# Tạo thư mục lưu hình (nếu muốn)
import os
os.makedirs("figures", exist_ok=True)

# 4.1: Predicted vs Actual cho từng model
for name, y_pred in predictions.items():
    plt.figure(figsize=(6, 6))
    plt.scatter(y_val, y_pred, alpha=0.4)
    # đường y = x (dự đoán hoàn hảo)
    min_val = min(y_val.min(), y_pred.min())
    max_val = max(y_val.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("Actual y (validation)")
    plt.ylabel("Predicted y")
    plt.title(f"Predicted vs Actual - {name}")
    plt.tight_layout()
    plt.savefig(f"figures/{name}_pred_vs_actual.png", dpi=200)
    # plt.show()  # bật dòng này nếu muốn xem trực tiếp

# 4.2: Residual plot (residual vs predicted) cho từng model
for name, y_pred in predictions.items():
    residuals = y_val - y_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted y")
    plt.ylabel("Residual (y_true - y_pred)")
    plt.title(f"Residuals vs Predicted - {name}")
    plt.tight_layout()
    plt.savefig(f"figures/{name}_residuals.png", dpi=200)
    # plt.show()

# 4.3: Histogram residuals cho từng model
for name, y_pred in predictions.items():
    residuals = y_val - y_pred
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual distribution - {name}")
    plt.tight_layout()
    plt.savefig(f"figures/{name}_residual_hist.png", dpi=200)
    # plt.show()

# 4.4: Bar chart so sánh MAE, RMSE, R2 giữa các model
models_order = results_df.index.tolist()

plt.figure(figsize=(6, 4))
plt.bar(models_order, results_df["MAE"])
plt.ylabel("MAE (lower is better)")
plt.title("Model comparison - MAE")
plt.tight_layout()
plt.savefig("figures/compare_MAE.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.bar(models_order, results_df["RMSE"])
plt.ylabel("RMSE (lower is better)")
plt.title("Model comparison - RMSE")
plt.tight_layout()
plt.savefig("figures/compare_RMSE.png", dpi=200)

plt.figure(figsize=(6, 4))
plt.bar(models_order, results_df["R2"])
plt.ylabel("R² (higher is better)")
plt.title("Model comparison - R²")
plt.tight_layout()
plt.savefig("figures/compare_R2.png", dpi=200)

print("\nAll figures saved to 'figures/' folder and metrics saved to 'model_metrics_val.csv'.")
