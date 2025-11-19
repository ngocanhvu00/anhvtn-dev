# app_predict_motobike.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import os

###### Giao diện Streamlit ######
st.set_page_config(page_title="Dự đoán giá xe máy", layout="centered")
### cho version cũ
st.image('xe_may_cu.jpg', use_column_width=True)
### cho version mới
# st.image("xe_may_cu.jpg", use_container_width=True)
st.title("Dự đoán giá xe máy (Model pickle)")
st.markdown("Ứng dụng đọc `data_motobikes.xlsx` (nếu có) để tạo các dropdown cho các thuộc tính categorical.")

# ---------- Helpers ----------
def to_number_from_str(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.integer, np.floating)):
        return float(s)
    s = str(s)
    s = s.replace('₫', '').replace('VND','').replace('vnd','')
    s = s.replace(',', '').replace('.', '').replace(' ', '')
    s = re.sub(r'[^\d\-]', '', s)
    if s in ['', '-', None]:
        return np.nan
    try:
        return float(s)
    except:
        return np.nan

def format_vnd(x):
    try:
        x = float(x)
    except:
        return str(x)
    return f"{int(round(x)):,} ₫".replace(',', '.')

def load_model(path='motobike_price_prediction_model.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file model: {path}")
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def infer_expected_features(model):
    try:
        if hasattr(model, 'named_steps'):
            for name, step in model.named_steps.items():
                if step is None:
                    continue
                if hasattr(step, 'feature_names_in_'):
                    return list(step.feature_names_in_)
                if hasattr(step, 'transformers') or hasattr(step, 'transformers_'):
                    pre = step
                    cols = []
                    for trans in getattr(pre, 'transformers', getattr(pre, 'transformers_', [])):
                        if len(trans) >= 3:
                            cols_spec = trans[2]
                            if isinstance(cols_spec, (list, tuple)):
                                cols.extend(list(cols_spec))
                            elif isinstance(cols_spec, str):
                                cols.append(cols_spec)
                    if cols:
                        return cols
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
            final = list(model.named_steps.items())[-1][1]
            if hasattr(final, 'feature_names_in_'):
                return list(final.feature_names_in_)
    except Exception:
        pass
    return ['mileage_km', 'age', 'brand', 'model', 'condition', 'bike_type','origin','engine_capacity','min_price','max_price']

# ---------- Load model ----------
try:
    model = load_model('motobike_price_prediction_model.pkl')
    st.success("Đã load model: motobike_price_prediction_model.pkl")
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Lỗi khi load model: {e}")
    st.stop()

# ---------- Try load original dataset to populate categorical options ----------
df = None
for p in ['data_motobikes.xlsx']:
    try:
        if p.endswith('.csv') and os.path.exists(p):
            df = pd.read_csv(p)
            st.info(f"Đã đọc dữ liệu từ: {p}")
            break
        if p.endswith('.xlsx') and os.path.exists(p):
            df = pd.read_excel(p, engine='openpyxl')
            st.info(f"Đã đọc dữ liệu từ: {p}")
            break
    except Exception as e:
        # continue trying other paths
        pass

if df is None:
    st.warning("Không tìm thấy file dữ liệu gốc (data_motobikes.xlsx). App sẽ dùng selectbox rỗng — bạn có thể upload file hoặc đặt file vào thư mục app.")
else:
    st.write(f"Dữ liệu mẫu: {df.shape[0]} hàng, {df.shape[1]} cột")

# ---------- Determine expected features and split numeric/categorical ----------
expected_features = infer_expected_features(model)
st.write("**Các feature model mong đợi (tự suy đoán):**")
st.write(expected_features)

# Heuristics cho numeric vs categorical
numeric_guess = {'mileage_km','mileage','km','số km','số km đã đi','age','năm','year','Trọng lượng','weight','Giá','Khoảng giá min','Khoảng giá max','origin','engine_capacity','min_price','max_price'}
numeric_features = [c for c in expected_features if any(k.lower() in c.lower() for k in numeric_guess)]
categorical_features = [c for c in expected_features if c not in numeric_features]

if not numeric_features:
    numeric_features = ['mileage_km','age']

st.write("Numeric:", numeric_features)
st.write("Categorical (sẽ hiện selectbox nếu có dữ liệu):", categorical_features)
st.markdown("---")

# ---------- Build UI: nếu có df, lấy unique values cho categorical ----------
select_options = {}
for cat in categorical_features:
    opts = []
    if df is not None and cat in df.columns:
        # lấy giá trị unique, bỏ NA và chuyển sang str
        raw_vals = df[cat].dropna().astype(str).str.strip()
        # loại bỏ chuỗi rỗng
        raw_vals = raw_vals[raw_vals != '']
        # sort và lấy unique
        unique_vals = sorted(raw_vals.unique().tolist())
        opts = ['(Không chọn)'] + unique_vals + ['Khác...']
    else:
        # fallback: rỗng
        opts = ['(Không chọn)', 'Khác...']
    select_options[cat] = opts

# ---------- Input form ----------
st.header("Nhập thông tin xe để dự đoán")
with st.form("input_form"):
    user_inputs = {}
    # numeric inputs (text boxes so user có thể paste "15.000 km")
    cols = st.columns(2)
    for i, feat in enumerate(numeric_features):
        col = cols[i % 2]
        default_val = "15000" if 'mileage_km' in feat.lower() or 'km' in feat.lower() else "2"
        with col:
            user_inputs[feat] = st.text_input(f"{feat} (số hoặc chuỗi):", value=default_val, key=f"num_{feat}")
    # categorical inputs: nếu có df và giá trị unique -> selectbox, nếu chọn 'Khác...' -> text_input hiện thêm
    for feat in categorical_features:
        opts = select_options.get(feat, ['(Không chọn)', 'Khác...'])
        sel = st.selectbox(f"{feat}:", opts, key=f"sel_{feat}")
        if sel == 'Khác...':
            txt = st.text_input(f"Nhập giá trị khác cho {feat} (hoặc để trống):", value="", key=f"text_{feat}")
            user_inputs[feat] = txt if txt.strip() != "" else np.nan
        elif sel == '(Không chọn)':
            user_inputs[feat] = np.nan
        else:
            user_inputs[feat] = sel
    submitted = st.form_submit_button("Dự đoán giá")

if submitted:
    # build df for prediction and clean numeric fields
    input_dict = {}
    for feat in numeric_features:
        val = user_inputs.get(feat, '')
        parsed = to_number_from_str(val)
        # special: nếu user nhập year vào age, convert to age
        if feat.lower() in ['age'] and not np.isnan(parsed) and parsed > 100:
            parsed = 2025 - parsed
        if np.isnan(parsed):
            try:
                parsed = float(val)
            except:
                parsed = np.nan
        input_dict[feat] = parsed
    for feat in categorical_features:
        input_dict[feat] = user_inputs.get(feat, np.nan)

    X_pred = pd.DataFrame([input_dict])
    st.write("### Dữ liệu gửi vào model (sau làm sạch cơ bản)")
    st.dataframe(X_pred.T)

    # Reindex theo expected_features nếu có thể
    try:
        X_for_model = X_pred.reindex(columns=expected_features)
    except Exception:
        X_for_model = X_pred

    # Predict
    try:
        y_pred = model.predict(X_for_model)
        pred_val = float(y_pred[0])
        st.success("Dự đoán hoàn tất")
        st.metric(label="Giá dự đoán", value=format_vnd(pred_val))
        st.write(f"Giá (số thực): {pred_val:.2f}")
    except Exception as e:
        st.error("Có lỗi khi gọi model.predict(). Kiểm tra xem tên cột và định dạng dữ liệu có khớp với lúc train không.")
        st.exception(e)
        st.write("Gợi ý:")
        st.write("- Nếu model yêu cầu các cột khác tên, chỉnh `expected_features` hoặc rebuild pipeline để ăn input thô.")
        st.write("- Nếu selectbox không có giá trị mong muốn, hãy upload/đặt file CSV gốc có tên cột đúng (subset_100motobikes.csv).")

st.markdown("---")
st.caption("Ghi chú: app cố gắng đọc file dữ liệu gốc trong cùng thư mục. Nếu bạn muốn dropdown chứa các giá trị cụ thể, hãy đảm bảo cột trong CSV có cùng tên như tên feature mà model mong đợi.")
