import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches 

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


# ========================
# Các hàm vẽ biểu đồ
# ========================

# def plot_corr_heatmap(corr_matrix: pd.DataFrame,
#                       save_path: str = "corr_features_heatmap.png",
#                       title_suffix: str = ""):
#     fig, ax = plt.subplots(figsize=(8, 6))
#     im = ax.imshow(corr_matrix.values, vmin=0, vmax=1)

#     # Ticks & labels
#     ax.set_xticks(np.arange(len(corr_matrix.columns)))
#     ax.set_yticks(np.arange(len(corr_matrix.columns)))
#     ax.set_xticklabels(corr_matrix.columns, rotation=90)
#     ax.set_yticklabels(corr_matrix.columns)

#     ax.set_title("Ma trận |correlation| giữa các feature" + title_suffix)
#     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close(fig)
#     print(f"💾 Đã lưu heatmap tương quan feature tại: {save_path}")

def plot_corr_heatmap(
    corr_matrix: pd.DataFrame,
    high_corr_pairs=None,
    corr_thresh: float = None,
    save_path: str = "corr_features_heatmap.png",
    title_suffix: str = "",
):
    """
    Vẽ heatmap tương quan giữa các feature.
    - Nếu high_corr_pairs được truyền vào: highlight các feature/cặp có |corr| cao.
    """
    cols = corr_matrix.columns

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix.values, vmin=-1, vmax=1)

    # Ticks & labels
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=90)
    ax.set_yticklabels(cols)

    # Đặt title
    if corr_thresh is not None:
        ax.set_title(
            f"Ma trận tương quan giữa các feature (ngưỡng |corr| cao: {corr_thresh})"
        )
    else:
        ax.set_title("Ma trận tương quan giữa các feature" + title_suffix)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ===== Highlight các feature có corr cao =====
    if high_corr_pairs:
        # 1) Các feature có ít nhất một cặp corr cao
        problematic_feats = set()
        for f1, f2, c in high_corr_pairs:
            problematic_feats.add(f1)
            problematic_feats.add(f2)

        # Đổi màu tick label thành đỏ cho các feature "problematic"
        for label in ax.get_xticklabels():
            if label.get_text() in problematic_feats:
                label.set_color("red")
                label.set_fontweight("bold")
        for label in ax.get_yticklabels():
            if label.get_text() in problematic_feats:
                label.set_color("red")
                label.set_fontweight("bold")

        # 2) Vẽ khung đỏ quanh các ô corr cao
        for f1, f2, c in high_corr_pairs:
            i = cols.get_loc(f1)
            j = cols.get_loc(f2)

            # ma trận đối xứng nên highlight cả (i,j) và (j,i)
            for (row, col) in [(i, j), (j, i)]:
                rect = patches.Rectangle(
                    (col - 0.5, row - 0.5),  # (x, y)
                    1, 1,                    # width, height
                    fill=False,
                    edgecolor="red",
                    linewidth=1.5,
                )
                ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Đã lưu heatmap tương quan feature tại: {save_path}")


def plot_vif_bar(vif_df: pd.DataFrame,
                 top_n: int = 20,
                 save_path: str = "vif_top_features.png"):
    if vif_df["VIF"].isna().all():
        print("⚠️ Không có VIF hợp lệ để vẽ.")
        return

    vif_sorted = vif_df.sort_values("VIF", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(vif_sorted))
    ax.bar(x_pos, vif_sorted["VIF"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(vif_sorted["feature"], rotation=45, ha="right")
    ax.set_ylabel("VIF")
    ax.set_title(f"Top {len(vif_sorted)} feature có VIF cao nhất")

    # Ngưỡng VIF ~ 5 thường được dùng để nghi ngờ đa cộng tuyến
    ax.axhline(5, linestyle="--")
    ax.text(0, 5, " VIF = 5", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Đã lưu biểu đồ VIF tại: {save_path}")


def plot_feature_target_corr(ft_corr: pd.Series,
                             weak_corr_thresh: float,
                             top_n: int = 30,
                             save_path: str = "feature_target_corr.png"):
    """
    Vẽ bar chart corr(feature, target) (lấy top_n feature theo |corr|).
    """
    corr_df = ft_corr.to_frame(name="corr")
    corr_df["abs_corr"] = corr_df["corr"].abs()
    corr_sorted = corr_df.sort_values("abs_corr", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 4))
    x_pos = np.arange(len(corr_sorted))
    ax.bar(x_pos, corr_sorted["corr"])

    ax.set_xticks(x_pos)
    ax.set_xticklabels(corr_sorted.index, rotation=45, ha="right")
    ax.set_ylabel("corr(feature, target)")
    ax.set_title(f"Tương quan feature–target (top {len(corr_sorted)} theo |corr|)")

    # Vẽ ngưỡng feature yếu
    ax.axhline(weak_corr_thresh, linestyle="--")
    ax.axhline(-weak_corr_thresh, linestyle="--")
    ax.text(0, weak_corr_thresh, f"  +{weak_corr_thresh}", va="bottom")
    ax.text(0, -weak_corr_thresh, f"  -{weak_corr_thresh}", va="top")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Đã lưu biểu đồ corr(feature, target) tại: {save_path}")


def plot_r2_cv(scores: np.ndarray,
               save_path: str = "linear_cv_r2.png"):
    """
    Vẽ bar chart R² cho từng fold + đường trung bình.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    folds = np.arange(1, len(scores) + 1)
    ax.bar(folds, scores)

    mean_r2 = scores.mean()
    ax.axhline(mean_r2, linestyle="--")
    ax.text(1, mean_r2, f"  mean R² = {mean_r2:.3f}", va="bottom")

    ax.set_xlabel("Fold")
    ax.set_ylabel("R²")
    ax.set_title("Kết quả R² của Linear Regression (k-fold CV)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"💾 Đã lưu biểu đồ R² CV tại: {save_path}")


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
    # 1. Đa cộng tuyến
    # =========================
    print("\n🔍 1) Phân tích đa cộng tuyến (multicollinearity)")

    # corr_matrix = abs(X.corr())
    # high_corr_pairs = []

    # cols = corr_matrix.columns
    # for i in range(len(cols)):
    #     for j in range(i + 1, len(cols)):
    #         corr_ij = corr_matrix.iloc[i, j]
    #         if abs(corr_ij) >= corr_thresh:
    #             high_corr_pairs.append((cols[i], cols[j], corr_ij))

    # # Vẽ heatmap cho ma trận tương quan
    # plot_corr_heatmap(
    #     corr_matrix,
    #     save_path="corr_features_heatmap.png",
    #     title_suffix=f" (ngưỡng |corr| cao: {corr_thresh})"
    # )

    corr_matrix = X.corr()
    high_corr_pairs = []

    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_ij = corr_matrix.iloc[i, j]
            if abs(corr_ij) >= corr_thresh:
                high_corr_pairs.append((cols[i], cols[j], corr_ij))

    # Vẽ heatmap cho ma trận tương quan + highlight
    plot_corr_heatmap(
        corr_matrix,
        high_corr_pairs=high_corr_pairs,
        corr_thresh=corr_thresh,
        save_path="corr_features_heatmap.png",
    )

    if high_corr_pairs:
        print(f"- Số cặp feature có |corr| >= {corr_thresh}: {len(high_corr_pairs)}")
        print("  (xem thêm trong heatmap để minh hoạ đa cộng tuyến)")
    else:
        print(f"- Không có cặp feature nào có |corr| >= {corr_thresh} (xem heatmap).")

    vif_df = compute_vif(X)
    if HAS_STATSMODELS:
        plot_vif_bar(vif_df, save_path="vif_top_features.png")

    has_high_corr = len(high_corr_pairs) > 0
    has_high_vif = HAS_STATSMODELS and (vif_df["VIF"] > 5).any()
    has_multicollinearity = has_high_corr or has_high_vif

    # =========================
    # 2. Feature yếu / nhiễu
    # =========================
    print("\n🔍 2) Tương quan feature–target")

    ft_corr = X.corrwith(y)
    ft_corr_sorted = ft_corr.abs().sort_values(ascending=True)

    weak_features = ft_corr_sorted[ft_corr_sorted < weak_corr_thresh].index.tolist()
    strong_features = ft_corr_sorted[ft_corr_sorted >= weak_corr_thresh].index.tolist()

    print(f"- Số feature có |corr(feature, target)| < {weak_corr_thresh}: {len(weak_features)}")
    print(f"- Số feature có tương quan ≥ {weak_corr_thresh}: {len(strong_features)}")

    # Vẽ bar chart corr(feature, target)
    plot_feature_target_corr(
        ft_corr,
        weak_corr_thresh=weak_corr_thresh,
        top_n=30,
        save_path="feature_target_corr.png"
    )

    if len(weak_features) >= max(1, 0.2 * X.shape[1]):
        has_many_weak = True
    else:
        has_many_weak = False

    # =========================
    # 3. Linear baseline (CV)
    # =========================
    print("\n🔍 3) Linear Regression baseline (k-fold CV)")

    model = LinearRegression()
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    r2_mean = scores.mean()
    r2_std = scores.std()

    print(f"- R² trung bình (5-fold CV): {r2_mean:.3f} ± {r2_std:.3f}")
    plot_r2_cv(scores, save_path="linear_cv_r2.png")

    has_linear_signal = r2_mean >= 0.3

    # =========================
    # 4. Kết luận tổng hợp
    # =========================
    print("\n📌 4) Gợi ý lập luận dựa trên các biểu đồ:")

    print(f"  • Đa cộng tuyến (multicollinearity): {'CÓ' if has_multicollinearity else 'KHÔNG RÕ'}")
    print(f"  • Nhiều feature yếu / nhiễu: {'CÓ' if has_many_weak else 'ÍT'}")
    print(f"  • Tín hiệu tuyến tính (Linear R² >= 0.3): {'CÓ' if has_linear_signal else 'YẾU'}")

    print("\n👉 Khi viết báo cáo, bạn có thể dùng các hình:")
    print("  1) Hình ma trận tương quan giữa các feature (corr_features_heatmap.png):")
    print("     - Chỉ ra các cụm feature tương quan cao ⇒ lý do thử Ridge / ElasticNet.")
    if HAS_STATSMODELS:
        print("  2) Hình VIF top feature (vif_top_features.png):")
        print("     - Giải thích đa cộng tuyến qua VIF lớn ⇒ Ridge hợp lý.")
    print("  3) Hình tương quan feature–target (feature_target_corr.png):")
    print("     - Nhóm feature mạnh vs yếu ⇒ Lasso / ElasticNet dùng để chọn lọc biến.")
    print("  4) Hình R² theo fold (linear_cv_r2.png):")
    print("     - Cho thấy Linear Regression có (hoặc không có) tín hiệu tuyến tính ⇒")
    print("       là baseline hợp lý để so sánh với Ridge / Lasso / ElasticNet.")


def main():
    parser = argparse.ArgumentParser(
        description="Phân tích nhanh dataset để xem có phù hợp cho Linear / Ridge / Lasso / ElasticNet không (kèm biểu đồ)."
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
