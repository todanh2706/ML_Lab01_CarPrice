import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ========================================
# 1️⃣ Đọc dữ liệu
# ========================================
X_train = pd.read_csv("archive/train/x_train.csv")
y_train = pd.read_csv("archive/train/y_train.csv").squeeze()

print("✅ Loaded training data:", X_train.shape, "target:", y_train.shape)

# ========================================
# 2️⃣ Tạo grid chung
# ========================================
alpha_grid = np.logspace(-4, 3, 20)  # từ 1e-4 đến 1e2
l1_grid = [0.1, 0.3, 0.5, 0.7, 0.9]  # cho ElasticNet

# ========================================
# 3️⃣ Hàm tiện ích chạy GridSearchCV
# ========================================
def run_grid_search(model, param_grid, name):
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", model)
    ])

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    print(f"\n🔹 {name} Results")
    print("--------------------------")
    print("Best Params:", search.best_params_)
    print(f"Best CV R²: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


# ========================================
# 4️⃣ Ridge
# ========================================
ridge_params = {"model__alpha": alpha_grid}
best_ridge, ridge_best_params, ridge_best_score = run_grid_search(
    Ridge(), ridge_params, "Ridge"
)

# ========================================
# 5️⃣ Lasso
# ========================================
lasso_params = {"model__alpha": alpha_grid}
best_lasso, lasso_best_params, lasso_best_score = run_grid_search(
    Lasso(max_iter=10000), lasso_params, "Lasso"
)

# ========================================
# 6️⃣ ElasticNet
# ========================================
elastic_params = {
    "model__alpha": alpha_grid,
    "model__l1_ratio": l1_grid
}
best_enet, enet_best_params, enet_best_score = run_grid_search(
    ElasticNet(max_iter=10000, random_state=42),
    elastic_params,
    "ElasticNet"
)

# ========================================
# 7️⃣ Tổng hợp kết quả
# ========================================
print("\n===============================")
print("🏁 TỔNG KẾT HIỆU NĂNG 3 MÔ HÌNH")
print("===============================")
print(f"Ridge:      R² = {ridge_best_score:.4f}, alpha = {ridge_best_params['model__alpha']:.4g}")
print(f"Lasso:      R² = {lasso_best_score:.4f}, alpha = {lasso_best_params['model__alpha']:.4g}")
print(f"ElasticNet: R² = {enet_best_score:.4f}, alpha = {enet_best_params['model__alpha']:.4g}, l1_ratio = {enet_best_params['model__l1_ratio']}")

# # ========================================
# # 8️⃣ (Tuỳ chọn) Lưu mô hình tốt nhất
# # ========================================
# import joblib

# joblib.dump(best_ridge, "ridge_best_model.pkl")
# joblib.dump(best_lasso, "lasso_best_model.pkl")
# joblib.dump(best_enet, "elasticnet_best_model.pkl")

# print("\n💾 Đã lưu mô hình tốt nhất của từng loại vào file .pkl")
