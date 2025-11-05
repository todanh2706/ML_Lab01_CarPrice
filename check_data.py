import argparse
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """
    Tính VIF cho từng feature. Cần statsmodels.
    """
    if not HAS_STATSMODELS:
        print("⚠️ statsmodels chưa được cài, bỏ qua phần VIF.")
        return pd.DataFrame({"feature": X.columns, "VIF": np.nan})

    vif_data = []
    values = X.values
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(values, i)
        vif_data.append((X.columns[i], vif))
    return pd.DataFrame(vif_data, columns=["feature", "VIF"])


def analyze_dataset(
    csv_path: str,
    target_col: str,
    corr_thresh: float = 0.8,
    weak_corr_thresh: float = 0.1,
):
    print("📂 Đang đọc dữ liệu từ:", csv_path)
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"Cột target '{target_col}' không tồn tại trong dataset.")

    # Chỉ lấy các cột số để phân tích
    num_df = df.select_dtypes(include=[np.number]).copy()

    if target_col not in num_df.columns:
        raise ValueError(
            f"Cột target '{target_col}' không phải kiểu số trong dữ liệu numeric.\n"
            f"Hãy encode hoặc chuyển target về dạng số trước."
        )

    X = num_df.drop(columns=[target_col])
    y = num_df[target_col]

    print("\n📊 Thông tin chung:")
    print(f"- Số dòng (n): {len(df)}")
    print(f"- Số feature numeric (p, không tính target): {X.shape[1]}")
    print(f"- Tên các feature: {list(X.columns)}")

    if X.shape[1] < 2:
        print("\n❌ Dataset có quá ít feature để phân tích đa cộng tuyến.")
        return

    # =========================
    # 1. Kiểm tra đa cộng tuyến
    # =========================
    print("\n🔍 1) Phân tích đa cộng tuyến (multicollinearity)")

    corr_matrix = X.corr()
    high_corr_pairs = []

    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_ij = corr_matrix.iloc[i, j]
            if abs(corr_ij) >= corr_thresh:
                high_corr_pairs.append((cols[i], cols[j], corr_ij))

    if high_corr_pairs:
        print(f"- Số cặp feature có |corr| >= {corr_thresh}: {len(high_corr_pairs)}")
        print("  Một vài cặp tiêu biểu:")
        for pair in high_corr_pairs[:10]:
            print(f"  • {pair[0]} – {pair[1]}: corr = {pair[2]:.3f}")
    else:
        print(f"- Không có cặp feature nào có |corr| >= {corr_thresh}.")

    vif_df = compute_vif(X)
    if HAS_STATSMODELS:
        print("\n- Top 10 feature có VIF cao nhất:")
        print(vif_df.sort_values("VIF", ascending=False).head(10))

    has_high_corr = len(high_corr_pairs) > 0
    has_high_vif = HAS_STATSMODELS and (vif_df["VIF"] > 5).any()

    has_multicollinearity = has_high_corr or has_high_vif

    if has_multicollinearity:
        print("\n✅ Nhận xét:")
        print("  → Dataset có dấu hiệu đa cộng tuyến ⇒ hợp lý để thử Ridge / ElasticNet.")
    else:
        print("\nℹ️ Nhận xét:")
        print("  → Đa cộng tuyến không rõ ràng. Ridge/ElasticNet vẫn dùng được,")
        print("    nhưng lập luận nên tập trung vào regularization chống overfitting hơn là đa cộng tuyến.")

    # ======================================
    # 2. Kiểm tra các feature yếu / nhiễu
    # ======================================
    print("\n🔍 2) Phân tích tương quan feature–target (feature importance sơ bộ)")

    ft_corr = X.corrwith(y)
    ft_corr_sorted = ft_corr.abs().sort_values(ascending=True)

    weak_features = ft_corr_sorted[ft_corr_sorted < weak_corr_thresh].index.tolist()
    strong_features = ft_corr_sorted[ft_corr_sorted >= weak_corr_thresh].index.tolist()

    print(f"- Số feature có |corr(feature, target)| < {weak_corr_thresh}: {len(weak_features)}")
    if weak_features:
        print("  → Các feature yếu (tương quan thấp với target):")
        print("   ", weak_features)

    print(f"- Số feature có tương quan tương đối với target: {len(strong_features)}")
    if strong_features:
        print("  → Một vài feature mạnh:")
        for feat in strong_features[-5:]:
            print(f"    • {feat}: corr = {ft_corr[feat]:.3f}")

    # Định nghĩa "nhiều feature yếu": ví dụ ≥ 20% số feature
    if len(weak_features) >= max(1, 0.2 * X.shape[1]):
        has_many_weak = True
        print("\n✅ Nhận xét:")
        print("  → Có khá nhiều feature yếu ⇒ hợp lý để thử Lasso / ElasticNet để tự động chọn lọc biến.")
    else:
        has_many_weak = False
        print("\nℹ️ Nhận xét:")
        print("  → Không có quá nhiều feature yếu. Lasso vẫn có thể dùng,")
        print("    nhưng lập luận feature selection sẽ không quá mạnh.")

    # =========================================
    # 3. Độ tuyến tính tổng thể (Linear baseline)
    # =========================================
    print("\n🔍 3) Kiểm tra sơ bộ độ tuyến tính (Linear Regression baseline)")

    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    r2_mean = scores.mean()
    r2_std = scores.std()

    print(f"- R² trung bình (5-fold CV) của Linear Regression: {r2_mean:.3f} ± {r2_std:.3f}")

    if r2_mean >= 0.3:
        has_linear_signal = True
        print("✅ Nhận xét:")
        print("  → Có tín hiệu tuyến tính đáng kể ⇒ phù hợp để dùng các mô hình họ Linear (OLS/Ridge/Lasso/ElasticNet).")
    else:
        has_linear_signal = False
        print("⚠️ Nhận xét:")
        print("  → Quan hệ tuyến tính yếu (R² thấp). Vẫn có thể làm bài tập,")
        print("    nhưng khi viết báo cáo nên chú ý giải thích là dữ liệu nhiều nhiễu hoặc không tuyến tính.")

    # =========================================
    # 4. Kết luận tổng hợp cho bài báo cáo
    # =========================================
    print("\n📌 4) Kết luận tổng hợp (gợi ý cho bài báo cáo)")

    print("- Tóm tắt dấu hiệu:")
    print(f"  • Đa cộng tuyến (multicollinearity): {'CÓ' if has_multicollinearity else 'KHÔNG RÕ'}")
    print(f"  • Nhiều feature yếu / nhiễu: {'CÓ' if has_many_weak else 'ÍT'}")
    print(f"  • Tín hiệu tuyến tính (Linear R² >= 0.3): {'CÓ' if has_linear_signal else 'YẾU'}")

    print("\n- Gợi ý mô hình & lập luận:")

    # Linear Regression
    if has_linear_signal:
        print("  ✅ Linear Regression (OLS):")
        print("     → Dùng làm baseline vì dữ liệu có quan hệ tuyến tính tương đối với target.")
    else:
        print("  ⚠️ Linear Regression (OLS):")
        print("     → Vẫn dùng được làm baseline, nhưng cần ghi chú R² thấp, dữ liệu nhiều nhiễu/phi tuyến.")

    # Ridge
    if has_multicollinearity:
        print("  ✅ Ridge Regression:")
        print("     → Dùng để xử lý đa cộng tuyến (corr cao hoặc VIF lớn) giữa các feature.")
    else:
        print("  ℹ️ Ridge Regression:")
        print("     → Có thể dùng như một hình thức regularization chống overfitting,")
        print("       nhưng lập luận về đa cộng tuyến sẽ yếu.")

    # Lasso
    if has_many_weak:
        print("  ✅ Lasso Regression:")
        print("     → Hợp lý để tự động loại bỏ những feature yếu (corr thấp với target).")
    else:
        print("  ℹ️ Lasso Regression:")
        print("     → Vẫn có thể thử, nhưng số feature yếu không nhiều,")
        print("       nên hiệu ứng feature selection có thể không rõ rệt.")

    # ElasticNet
    if has_multicollinearity and has_many_weak:
        print("  ✅ ElasticNet:")
        print("     → Dữ liệu vừa có đa cộng tuyến vừa có nhiều feature yếu, rất hợp lý để dùng ElasticNet (kết hợp L1 + L2).")
    elif has_multicollinearity or has_many_weak:
        print("  ✅ ElasticNet:")
        print("     → Có một trong hai vấn đề (đa cộng tuyến hoặc nhiều feature yếu),")
        print("       ElasticNet vẫn là lựa chọn trung hòa giữa Ridge và Lasso.")
    else:
        print("  ℹ️ ElasticNet:")
        print("     → Có thể dùng như mô hình lai giữa Ridge và Lasso,")
        print("       nhưng cần nhấn mạnh góc độ regularization hơn là xử lý vấn đề cụ thể trong dữ liệu.")

    print("\n🎓 Gợi ý viết trong báo cáo:")
    print("  → Dựa vào các thống kê trên, bạn có thể giải thích:")
    print("    - Vì sao chọn Linear làm baseline.")
    print("    - Vì sao thử Ridge (đa cộng tuyến) hoặc/ và Lasso (loại bỏ biến yếu).")
    print("    - Vì sao ElasticNet là lựa chọn kết hợp, rồi so sánh kết quả 4 mô hình để chọn mô hình cuối cùng.")


def main():
    parser = argparse.ArgumentParser(
        description="Phân tích nhanh dataset để xem có phù hợp cho Linear / Ridge / Lasso / ElasticNet không."
    )
    parser.add_argument("csv_path", help="Đường dẫn tới file CSV dữ liệu")
    parser.add_argument(
        "--target",
        required=True,
        help="Tên cột target (biến cần dự đoán)",
    )
    parser.add_argument(
        "--corr_thresh",
        type=float,
        default=0.8,
        help="Ngưỡng |corr| để coi là đa cộng tuyến (mặc định: 0.8)",
    )
    parser.add_argument(
        "--weak_corr_thresh",
        type=float,
        default=0.1,
        help="Ngưỡng |corr(feature, target)| < threshold để coi là feature yếu (mặc định: 0.1)",
    )

    args = parser.parse_args()
    analyze_dataset(
        csv_path=args.csv_path,
        target_col=args.target,
        corr_thresh=args.corr_thresh,
        weak_corr_thresh=args.weak_corr_thresh,
    )


if __name__ == "__main__":
    main()
