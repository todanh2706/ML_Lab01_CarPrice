import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings

warnings.filterwarnings("ignore")

def forward_selection_bic(X, y):
    """
    Thực hiện Forward Stepwise Selection (Lựa chọn Tiến)
    dựa trên tiêu chí bic.
    
    Yêu cầu: X đã được thêm cột hằng số (intercept).
    """
    print("--- Bắt đầu Lựa chọn Thuộc tính Tiến (Forward Selection) bằng bic ---")
    
    remaining_features = list(X.columns)
    remaining_features.remove('const') # Bỏ hằng số khỏi danh sách
    
    # Bắt đầu với mô hình rỗng (chỉ có hằng số)
    selected_features = ['const']
    
    # Tính BIC của mô hình rỗng
    current_best_bic = sm.OLS(y, X[selected_features]).fit().bic
    print(f"Mô hình cơ sở (chỉ intercept): bic = {current_best_bic:.2f}")

    while remaining_features:
        best_feature_to_add = None
        best_new_bic = current_best_bic # BIC tốt nhất của vòng lặp này

        # Thử thêm từng thuộc tính còn lại
        for feature in remaining_features:
            model_features = selected_features + [feature]
            try:
                model = sm.OLS(y, X[model_features]).fit()
                new_bic = model.bic
            except Exception as e:
                # Bỏ qua thuộc tính nếu có lỗi (ví dụ: tương quan hoàn hảo)
                # print(f"  [LỖI] Không thể thêm {feature}: {e}")
                continue
            
            # Nếu bic mới tốt hơn (thấp hơn) bic tốt nhất hiện tại
            if new_bic < best_new_bic:
                best_new_bic = new_bic
                best_feature_to_add = feature
        
        # Sau khi thử hết, kiểm tra xem có cải thiện không
        if best_feature_to_add:
            # Nếu CÓ, thêm thuộc tính vào danh sách
            selected_features.append(best_feature_to_add)
            remaining_features.remove(best_feature_to_add) # Xóa khỏi danh sách ban đầu
            current_best_bic = best_new_bic
            print(f"  + Thêm '{best_feature_to_add}', bic mới = {current_best_bic:.2f}")
        else:
            # Nếu KHÔNG, dừng lại
            print("\nKhông thuộc tính nào cải thiện bic. Dừng lựa chọn.")
            break
            
    print("\n--- Lựa chọn hoàn tất ---")
    return selected_features


def prepare_data(df):
    """
    Chuẩn bị dữ liệu: Chọn thuộc tính, mã hóa, và dọn dẹp.
    Đây là bước bắt buộc để bic có thể chạy.
    """
    print("Đang chuẩn bị dữ liệu (Feature Engineering)...")
    
    # 1. Chọn biến mục tiêu
    y = df['avg_salary']
    
    # 2. Loại bỏ các cột không phải là thuộc tính
    # Loại bỏ ID, văn bản tự do, và chính biến mục tiêu
    cols_to_drop = [
        'Unnamed: 0', 'Job Title', 'Salary Estimate', 'Job Description',
        'Company Name', 'Location', 'Headquarters', 'Competitors',
        'company_txt', 'min_salary', 'max_salary', 'avg_salary'
    ]
    
    # Xử lý các cột có thể không tồn tại
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    X = df.drop(columns=existing_cols_to_drop)

    # 3. Dọn dẹp nhanh
    # Thay thế '-1' (chuỗi) trong 'Sector' và các cột khác
    X.replace('-1', 'Other', inplace=True)
    X.replace(-1, 0, inplace=True) # Thay -1 (số)

    # 4. Mã hóa One-Hot các cột 'object' còn lại
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)
    
    # 5. Xử lý NaNs
    # statsmodels không chấp nhận NaNs
    X_processed.fillna(0, inplace=True) 
    
    print(f"Chuẩn bị xong. Tổng số thuộc tính để lựa chọn: {len(X_processed.columns)}")
    return X_processed, y


# === SCRIPT CHÍNH ===
def main():
    DATA_PATH = './archive/eda_data.csv'

    # 1. Tải dữ liệu
    if not os.path.exists(DATA_PATH):
        print(f"[LỖI] Không tìm thấy file tại đường dẫn: {DATA_PATH}")
        return
        
    print(f"Đang tải dữ liệu từ {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Chuẩn bị và dọn dẹp dữ liệu
    X_processed, y = prepare_data(df)
    
    # 3. Thêm hằng số (Intercept)
    # Bắt buộc cho statsmodels OLS
    X_with_const = sm.add_constant(X_processed, has_constant='add')
    
    # 4. Chạy hàm lựa chọn
    final_selected_features = forward_selection_bic(X_with_const, y)
    
    # 5. In kết quả cuối cùng
    print(f"\n========================================================")
    print(f"CÁC THUỘC TÍNH PHÙ HỢP NHẤT ĐƯỢC CHỌN (theo bic):")
    print(f"========================================================")
    
    # In ra danh sách
    for f in final_selected_features:
        print(f)

if __name__ == "__main__":
    main()