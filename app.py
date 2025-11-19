
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
from function_preprocessing_motorbike import preprocess_motobike_data
from build_model_price_anomaly_detection import detect_outliers

st.set_page_config(page_title="Motorbike Price & Anomaly App", layout="wide")

MODEL_PATH = "motobike_price_prediction_model.pkl"
TRAINING_DATA = "data_motobikes.xlsx"  # optional, used to compute brand_meanprice & grouping to match train

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
# def build_training_helpers(path=TRAINING_DATA):
#     """Try to load training data to reconstruct brand/model grouping and brand_meanprice.
#     Returns a dict with maps and thresholds. If file not present, return None.
#     """
#     if not os.path.exists(path):
#         return None
#     try:
#         df_train = preprocess_motobike_data(path)
#         # brand grouping threshold from preprocess function
#         brand_counts = df_train['brand'].value_counts()
#         rare_brands = set(brand_counts[brand_counts < 50].index)

#         # model grouping by brand
#         model_group_maps = {}
#         for bg, g in df_train.groupby('brand_grouped'):
#             counts = g['model'].value_counts()
#             rare = set(counts[counts < 100].index)
#             model_group_maps[bg] = rare

#         brand_mean_map = df_train.groupby('brand')['brand_meanprice'].first().to_dict()

#         return {
#             'rare_brands': rare_brands,
#             'model_group_maps': model_group_maps,
#             'brand_mean_map': brand_mean_map
#         }
#     except Exception as e:
#         return None

def build_training_helpers(path=TRAINING_DATA):
    """
    Load training data & build grouping rules + statistical thresholds
    (p10/p90, residual mean/std) for anomaly detection.
    """
    if not os.path.exists(path):
        return None

    try:
        df_train = preprocess_motobike_data(path)

        # =============== 1) BRAND GROUPING ==================
        brand_counts = df_train['brand'].value_counts()
        rare_brands = set(brand_counts[brand_counts < 50].index)

        # model grouping by brand_grouped
        model_group_maps = {}
        for bg, g in df_train.groupby('brand_grouped'):
            counts = g['model'].value_counts()
            rare_models = set(counts[counts < 100].index)
            model_group_maps[bg] = rare_models

        # mean price for brand
        brand_mean_map = df_train.groupby('brand')['brand_meanprice'].first().to_dict()

        # =============== 2) PRICE P10/P90 BY SEGMENT ==================
        seg_price_stats = (
            df_train.groupby('segment')['price']
                    .quantile([0.10, 0.90])
                    .unstack(level=1)
                    .rename(columns={0.10:'p10', 0.90:'p90'})
        ).reset_index()

        seg_price_map = seg_price_stats.set_index('segment').to_dict('index')
        # format: seg_price_map[segment] = {'p10':..., 'p90':...}

        # =============== 3) RESIDUAL STATS BY SEGMENT ==================

        # Load model
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)

        # Define cols
        cat_cols = ['segment','bike_type','origin','engine_capacity']
        num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']

        # Build matrix
        X = df_train[cat_cols + num_cols]
        # y = df['log_price']

        # Predict price
        df_train['price_hat'] = np.expm1(model.predict(X))
        df_train['resid'] = df_train['price'] - df_train['price_hat']  # price_hat từ preprocess

        seg_resid_stats = (
            df_train.groupby('segment')['resid']
                    .agg(['mean', 'std'])
                    .rename(columns={'mean': 'resid_mean', 'std': 'resid_std'})
        ).reset_index()

        seg_resid_map = seg_resid_stats.set_index('segment').to_dict('index')
        # format: seg_resid_map[seg] = {'resid_mean':..., 'resid_std':...}

        return {
            'rare_brands': rare_brands,
            'model_group_maps': model_group_maps,
            'brand_mean_map': brand_mean_map,
            'seg_price_map': seg_price_map,
            'seg_resid_map': seg_resid_map
        }

    except Exception as e:
        print("Error building helpers:", e)
        return None


helpers = build_training_helpers(TRAINING_DATA)
model = load_model(MODEL_PATH)

st.title("Motorbike Price Prediction & Anomaly Detection")
st.markdown("Ứng dụng cho phép: 1) Dự đoán giá xe máy (nhập tay hoặc upload file) 2) Phát hiện xe bất thường (upload file)")

page = st.sidebar.selectbox("Chọn chức năng", ["Dự đoán giá", "Phát hiện bất thường"])

# --- PREDICTION PAGE ---
@st.cache_data
def load_reference_data():
    return preprocess_motobike_data(TRAINING_DATA)

df_ref = load_reference_data()
brand_list = sorted(df_ref['brand_grouped'].dropna().unique())
model_list = sorted(df_ref['model_grouped'].dropna().unique())
bike_type_list = sorted(df_ref['bike_type'].dropna().unique())
origin_list = sorted(df_ref['origin'].dropna().unique())
engine_capacity_list = sorted(df_ref['engine_capacity'].dropna().unique())

if page == "Dự đoán giá":
    st.header("Dự đoán giá xe máy")

    mode = st.radio("Chọn cách input:", ["Nhập tay một xe", "Upload file (Excel/CSV) để dự đoán nhiều xe)"])

    if mode == "Nhập tay một xe":
        col1, col2 = st.columns(2)
        with col1:
            brand = st.selectbox("Thương hiệu (brand)", options=brand_list)
            model_name = st.selectbox("Dòng xe (model)", options=model_list)
            bike_type = st.selectbox("Loại xe (bike_type)", options=bike_type_list)
            origin = st.selectbox("Xuất xứ (origin)", options=origin_list)
        with col2:
            engine_capacity = st.selectbox("Dung tích (engine_capacity)", options=engine_capacity_list)
            registration_year = st.number_input("Năm đăng ký", min_value=1980, max_value=2025, value=2019)
            mileage_km = st.number_input("Số km đã đi", min_value=0, value=10000)
            min_price = st.number_input("Khoảng giá min (VND)", min_value=0, value=0)
            max_price = st.number_input("Khoảng giá max (VND)", min_value=0, value=0)

        if st.button("Chạy dự đoán"):
            if model is None:
                st.error(f"Không tìm thấy model tại '{MODEL_PATH}'. Vui lòng đảm bảo file tồn tại.")
            else:
                # create df
                df_in = pd.DataFrame([{ 
                    'brand': brand,
                    'model': model_name,
                    'bike_type': bike_type,
                    'origin': origin,
                    'engine_capacity': engine_capacity,
                    'registration_year': registration_year,
                    'mileage_km': mileage_km,
                    'min_price': min_price if min_price>0 else np.nan,
                    'max_price': max_price if max_price>0 else np.nan
                }])

                # compute age
                current_year = 2025
                df_in['age'] = current_year - pd.to_numeric(df_in['registration_year'], errors='coerce')

                # apply grouping using helpers if available
                if helpers is not None:
                    # brand_grouped
                    if df_in.at[0, 'brand'] in helpers['rare_brands']:
                        df_in['brand_grouped'] = 'Hãng khác'
                    else:
                        df_in['brand_grouped'] = df_in['brand']

                    # model_grouped
                    bg = df_in.at[0, 'brand_grouped']
                    rare_models = helpers['model_group_maps'].get(bg, set())
                    if df_in.at[0, 'model'] in rare_models:
                        df_in['model_grouped'] = 'Dòng khác'
                    else:
                        df_in['model_grouped'] = df_in['model']

                    # segment
                    df_in['segment'] = df_in['brand_grouped'] + '_' + df_in['model_grouped']

                    # brand_meanprice
                    df_in['brand_meanprice'] = helpers['brand_mean_map'].get(df_in.at[0,'brand'], np.nan)
                else:
                    # fallback simple
                    df_in['brand_grouped'] = df_in['brand']
                    df_in['model_grouped'] = df_in['model']
                    df_in['segment'] = df_in['brand'] + '_' + df_in['model']
                    df_in['brand_meanprice'] = np.nan
                    st.warning("Không tìm thấy data huấn luyện (data_motobikes.xlsx). App sẽ dùng fallback — brand_meanprice có thể là NaN, dự đoán có thể không chính xác.")

                # ensure columns order and types as model training
                cat_cols = ['segment','bike_type','origin','engine_capacity']
                num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']
                X_pred = df_in[cat_cols + num_cols]

                # predict
                try:
                    log_hat = model.predict(X_pred)
                    price_hat = np.expm1(log_hat)
                    df_in['predicted_price'] = price_hat

                    st.success(f"Giá dự đoán: {int(price_hat[0]):,} VND")
                    st.dataframe(df_in[['brand','model','bike_type','origin','engine_capacity','predicted_price']]) # chỉnh in df ở đây

                except Exception as e:
                    st.exception(e)

    else:
        st.subheader("Upload file để dự đoán nhiều xe (Excel/CSV)")
        uploaded_file = st.file_uploader("Chọn file (xlsx/csv)", type=['xlsx','csv'])
        if uploaded_file is not None:
            # save to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                df_proc = preprocess_motobike_data(tmp_path)
            except Exception as e:
                st.error(f"Lỗi khi tiền xử lý file: {e}")
                df_proc = None

            if df_proc is not None:
                if model is None:
                    st.error(f"Không tìm thấy model tại '{MODEL_PATH}'.")
                else:
                    cat_cols = ['segment','bike_type','origin','engine_capacity']
                    num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']
                    X = df_proc[cat_cols + num_cols]
                    df_proc['predicted_price'] = np.expm1(model.predict(X))

                    st.write("Kết quả (một vài dòng):")
                    st.dataframe(df_proc.head(50))

                    csv = df_proc.to_csv(index=False).encode('utf-8')
                    st.download_button("Tải kết quả (CSV)", data=csv, file_name="predictions.csv", mime='text/csv')


# --- ANOMALY PAGE ---
else:
    st.header("Phát hiện xe bất thường")

    mode_anom = st.radio(
        "Chọn cách kiểm tra:",
        ["Nhập tay 1 xe", "Upload file / dùng file mặc định"],
        horizontal=True
    )

    # ============================================================
    # MODE 1: NHẬP TAY 1 XE
    # ============================================================
    if mode_anom == "Nhập tay 1 xe":

        st.subheader("Nhập thông tin xe cần kiểm tra")

        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Thương hiệu (brand)", options=brand_list)
            model_name = st.selectbox("Dòng xe (model)", options=model_list)
            bike_type = st.selectbox("Loại xe (bike_type)", options=bike_type_list)
            origin = st.selectbox("Xuất xứ (origin)", options=origin_list)
            engine_capacity = st.selectbox("Dung tích (engine_capacity)", options=engine_capacity_list)
        with col2:
            registration_year = st.number_input("Năm đăng ký", min_value=1980, max_value=2025, value=2019)
            mileage_km = st.number_input("Số km đã đi", min_value=0, value=10000)
            min_price = st.number_input("Khoảng giá min (VND)", min_value=0, value=0)
            max_price = st.number_input("Khoảng giá max (VND)", min_value=0, value=0)
            price = st.number_input("Giá niêm yết", min_value=0, value=20000000)

        model_path_input = st.text_input("Đường dẫn model (.pkl)", value=MODEL_PATH)

        if st.button("Kiểm tra xe này có bất thường không?"):
            with st.spinner("Đang kiểm tra..."):

                # Tạo 1 DataFrame duy nhất
                df_in = pd.DataFrame([{
                    "brand": brand,
                    "model": model_name,
                    "bike_type": bike_type,
                    "origin": origin,
                    "engine_capacity": engine_capacity,
                    "registration_year": registration_year,
                    "mileage_km": mileage_km,
                    "min_price" : min_price,
                    "max_price" : max_price,
                    "price": price
                }])
                # compute age
                current_year = 2025
                df_in['age'] = current_year - pd.to_numeric(df_in['registration_year'], errors='coerce')

                # apply grouping using helpers if available
                if helpers is not None:
                    # brand_grouped
                    if df_in.at[0, 'brand'] in helpers['rare_brands']:
                        df_in['brand_grouped'] = 'Hãng khác'
                    else:
                        df_in['brand_grouped'] = df_in['brand']

                    # model_grouped
                    bg = df_in.at[0, 'brand_grouped']
                    rare_models = helpers['model_group_maps'].get(bg, set())
                    if df_in.at[0, 'model'] in rare_models:
                        df_in['model_grouped'] = 'Dòng khác'
                    else:
                        df_in['model_grouped'] = df_in['model']

                    # segment
                    df_in['segment'] = df_in['brand_grouped'] + '_' + df_in['model_grouped']

                    # brand_meanprice
                    df_in['brand_meanprice'] = helpers['brand_mean_map'].get(df_in.at[0,'brand'], np.nan)
                else:
                    # fallback simple
                    df_in['brand_grouped'] = df_in['brand']
                    df_in['model_grouped'] = df_in['model']
                    df_in['segment'] = df_in['brand'] + '_' + df_in['model']
                    df_in['brand_meanprice'] = np.nan
                    st.warning("Không tìm thấy data huấn luyện (data_motobikes.xlsx). App sẽ dùng fallback — brand_meanprice có thể là NaN, dự đoán có thể không chính xác.")

                try:
                    # Gọi detect_outliers cho 1 xe duy nhất
                    df_all, anomaly = detect_outliers(df_in, model_path_input, input_is_df=True, helpers=helpers)

                    if len(anomaly) > 0:
                        st.error("🚨 Xe này **BẤT THƯỜNG** theo mô hình phát hiện outlier.")
                        st.dataframe(anomaly)
                    else:
                        st.success("Xe này **KHÔNG bất thường** theo mô hình.")

                except Exception as e:
                    st.exception(e)

    # ============================================================
    # MODE 2: UPLOAD FILE HOẶC DÙNG FILE DEFAULT
    # ============================================================
    else:
        st.subheader("Upload file hoặc dùng file mặc định")

        uploaded_file_anom = st.file_uploader("Chọn file (xlsx/csv)", type=['xlsx','csv'], key='anom')
        use_default = st.checkbox("Dùng file mặc định data_motobikes.xlsx", value=False)

        model_path_input = st.text_input("Đường dẫn model (.pkl)", value=MODEL_PATH)

        if st.button("Chạy phát hiện bất thường (nhiều xe)"):
            if not use_default and uploaded_file_anom is None:
                st.error("Vui lòng upload file hoặc chọn dùng mặc định.")
            else:
                if use_default:
                    excel_path = TRAINING_DATA
                else:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file_anom.name)[1]) as tmp:
                        tmp.write(uploaded_file_anom.getvalue())
                        excel_path = tmp.name

                if not os.path.exists(model_path_input):
                    st.error(f"Không tìm thấy model tại '{model_path_input}'.")
                else:
                    with st.spinner("Đang chạy detect_outliers ..."):
                        try:
                            df_all, anomaly = detect_outliers(excel_path, model_path_input)

                            st.success(f"Hoàn tất. Tổng {len(df_all):,} bản ghi, phát hiện {len(anomaly):,} bất thường ({100*len(anomaly)/len(df_all):.2f}%).")

                            if len(anomaly) > 0:
                                st.subheader("Một vài bản ghi bất thường:")
                                st.dataframe(anomaly.head(50))
                                csv = anomaly.to_csv(index=False).encode('utf-8')
                                st.download_button("Tải outliers_detected.csv", data=csv, file_name="outliers_detected.csv", mime='text/csv')
                            else:
                                st.info("Không tìm thấy bản ghi bất thường.")

                        except Exception as e:
                            st.exception(e)



st.sidebar.markdown("---")
st.sidebar.markdown("App demo")
