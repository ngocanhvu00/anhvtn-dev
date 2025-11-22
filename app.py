
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
import pytz
from datetime import datetime
from streamlit.components.v1 import html
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from function_preprocessing_motorbike import preprocess_motobike_data
from build_model_price_anomaly_detection import detect_outliers

st.set_page_config(page_title="Motorbike Price & Anomaly App", layout="wide")

# Bắt đầu đoạn code cần thêm để áp dụng justify (căn đều)
html_code = """
<style>
/* Chọn tất cả các thành phần chứa văn bản chính của Streamlit
   (như st.markdown, st.write, st.header, st.subheader, st.text, v.v.)
   và áp dụng căn đều (text-align: justify;) */
.stMarkdown, .stText, .stHtml, .stHeader, .stSubheader, .stTitle, .stPageLink, .css-selector-cho-cac-phan-tu-khac {
    text-align: justify;
    text-justify: inter-word; /* Dành cho các trình duyệt IE/Edge */
}
/* Một số component như st.write/st.markdown sẽ được bọc trong class 'stMarkdown'
   và class này có thể được bọc trong các div khác. Ta cần selector mạnh hơn. */
div.stMarkdown p, div.stMarkdown, div[data-testid="stText"] {
    text-align: justify;
    text-justify: inter-word;
}
</style>
"""
st.components.v1.html(html_code, height=0)

# 1. BẮT ĐẦU: CSS ĐỂ LOẠI BỎ GIỚI HẠN CHIỀU RỘNG TỐI ĐA CỦA STREAMLIT
# Đặt max-width rất lớn (ví dụ: 2000px) để cho phép hình ảnh hiển thị rộng hơn
html_code_width = """
<style>
/* Loại bỏ giới hạn max-width của khối nội dung chính */
.main .block-container {
    max-width: 2000px !important; 
    padding-left: 1rem;
    padding-right: 1rem;
}
</style>
"""
st.components.v1.html(html_code_width, height=0)
# KẾT THÚC: CSS ĐỂ LOẠI BỎ GIỚI HẠN CHIỀU RỘNG TỐI ĐA

MODEL_PATH = "motobike_price_prediction_model.pkl"
TRAINING_DATA = "data_motobikes.xlsx"  # optional, used to compute brand_meanprice & grouping to match train

@st.cache_resource
def load_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def build_training_helpers(path=TRAINING_DATA):
    """
    Load training data & build grouping rules + statistical thresholds
    (p10/p90, residual mean/std) for anomaly detection.
    """
    if not os.path.exists(path):
        return None

    try:
        df_train = preprocess_motobike_data(path)
        # =============== LOAD MODELS ===============
        with open("unsup_scaler.pkl", "rb") as f:
            scaler = pickle.load(f)

        with open("kmeans_model.pkl", "rb") as f:
            kmeans = pickle.load(f)

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
# st.markdown("Ứng dụng cho phép: 1) Dự đoán giá xe máy (nhập tay hoặc upload file) 2) Phát hiện xe bất thường (upload file)")
# st.image("xe_may_cu.jpg", caption="Xe máy cũ")
st.image("xe_may_cu.jpg", caption="Xe máy cũ", width=1000)

# page = st.sidebar.selectbox("Chọn chức năng", ["Dự đoán giá", "Phát hiện bất thường"])
menu = ["Giới thiệu", "Bài toán nghiệp vụ", "Đánh giá mô hình và Báo cáo", "Dự đoán giá", "Phát hiện xe bất thường"]
page = st.sidebar.selectbox('Menu', menu)


@st.cache_data
def load_reference_data():
    return preprocess_motobike_data(TRAINING_DATA)

df_ref = load_reference_data()
brand_list = sorted(df_ref['brand_grouped'].dropna().unique())
model_list = sorted(df_ref['model_grouped'].dropna().unique())
bike_type_list = sorted(df_ref['bike_type'].dropna().unique())
origin_list = sorted(df_ref['origin'].dropna().unique())
engine_capacity_list = sorted(df_ref['engine_capacity'].dropna().unique())

if page == 'Giới thiệu':

    st.subheader("[Trang chủ](https://www.chotot.com/)")
    
    st.header('Giới thiệu dự án')
    st.markdown('''Đây là dự án xây dựng hệ thống hỗ trợ **định giá xe máy cũ** và **phát hiện tin đăng bất thường** trên nền tảng *Chợ Tốt* - trong khóa đồ án tốt nghiệp Data Science and Machine Learning 2024 lớp DL07_K308 của nhóm 6. \nThành viên nhóm gồm có:
        \n1. Vũ Thị Ngọc Anh \n2. Nguyễn Phạm Quỳnh Anh''')
    
    st.header('Mục tiêu của dự án')
    # st.text('''1. Tạo mô hình đề xuất xe máy tương tự đối với mẫu xe được chọn hoặc từ khóa tìm kiếm do người dùng cung cấp.\n2. Phân khúc thị trường xe máy''')
    st.write("""
    Mục tiêu của dự án:
    - **Tăng cường minh bạch** thị trường xe máy cũ thông qua dự báo giá hợp lý.
    - **Phát hiện các tin đăng bất thường**, giúp lọc ra xe có giá hoặc thông tin sai lệch.
    - **Hỗ trợ người dùng** đưa ra quyết định mua/bán chính xác và tin cậy hơn.
    - **Tối ưu hóa quy trình kiểm duyệt** thông tin trên nền tảng giao dịch xe máy.
    """)

    st.header('Phân công công việc')

    st.write("""
        - Xử lý dữ liệu: Ngọc Anh và Quỳnh Anh
        - Dự đoán giá xe theo phương pháp ML truyền thống: Ngọc Anh và Quỳnh Anh
        - Dự đoán giá xe theo PySpark: Ngọc Anh
        - Phát hiện giá bất thường: Ngọc Anh
        - Làm slide: Ngọc Anh và Quỳnh Anh
        - Giao diện streamlit: Ngọc Anh

        """)
elif page == 'Bài toán nghiệp vụ':
    st.subheader("[Trang chủ](https://www.chotot.com/)")

    st.markdown("""

        ### Vấn đề nghiệp vụ
        - Giá niêm yết không đồng nhất, khó xác định giá thị trường.
        - Nhiều tin đăng có giá bất thường gây nhiễu dữ liệu.
        - Kiểm duyệt thủ công tốn thời gian và không nhất quán.
        - Cần một hệ thống dự báo giá và cảnh báo bất thường tự động.

        ---

        ### Bài toán đặt ra
        - Xây dựng mô hình **Price Prediction**.
        - Thiết kế mô hình **Anomaly Detection** (ML-based + Rule-based).
        - Tối ưu quy trình kiểm duyệt và nâng cao chất lượng tin đăng.

        ---

        ### Phạm vi triển khai
        - Tối ưu & chuẩn hóa dữ liệu thô.
        - Tạo đặc trưng cho mô hình dự đoán giá.
        - Huấn luyện mô hình **Regression** để ước lượng giá thị trường.
        - Xây dựng hệ thống gắn cờ bất thường gồm:
        - **Model-based score** (ML)
        - **Business rule score** (Rule-based)
        - Triển khai giao diện demo bằng **Streamlit**.

    """)

    # st.header("###Thu thập dữ liệu")

    st.markdown("""  
    ### Thu thập dữ liệu    
    Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).  

    Bộ dữ liệu bao gồm các thông tin sau:

    - **id**: số thứ tự của sản phẩm trong bộ dữ liệu  
    - **Tiêu đề**: tựa đề bài đăng bán sản phẩm  
    - **Giá**: giá bán của xe máy  
    - **Khoảng giá min**: giá sàn ước tính của xe máy  
    - **Khoảng giá max**: giá trần ước tính của xe máy  
    - **Địa chỉ**: địa chỉ giao dịch (phường, quận, thành phố Hồ Chí Minh)  
    - **Mô tả chi tiết**: mô tả thêm về sản phẩm — đặc điểm nổi bật, tình trạng, thông tin khác  
    - **Thương hiệu**: hãng sản xuất (Honda, Yamaha, Piaggio, SYM…)  
    - **Dòng xe**: dòng xe cụ thể (Air Blade, Vespa, Exciter, LEAD, Vario, …)  
    - **Năm đăng ký**: năm đăng ký lần đầu của xe  
    - **Số km đã đi**: số kilomet xe đã vận hành  
    - **Tình trạng**: tình trạng hiện tại (ví dụ: đã sử dụng)  
    - **Loại xe**: Xe số, Tay ga, Tay côn/Moto  
    - **Dung tích xe**: dung tích xi-lanh (ví dụ: Dưới 50cc, 50–100cc, 100–175cc, …)  
    - **Xuất xứ**: quốc gia sản xuất (Việt Nam, Đài Loan, Nhật Bản, ...)  
    - **Chính sách bảo hành**: thông tin bảo hành nếu có  
    - **Trọng lượng**: trọng lượng ước tính của xe  
    - **Href**: đường dẫn tới bài đăng sản phẩm  
    """)


elif page == 'Đánh giá mô hình và Báo cáo':    
    st.subheader("[Trang chủ](https://www.chotot.com/)")  

    # df_home = preprocess_motobike_data(TRAINING_DATA)
    # st.subheader("Dữ liệu xe máy cũ (10 mẫu)")
    # st.dataframe(df_home.head(10))
    # st.subheader("Quy trình thực hiện")
    st.subheader("I. Thống kê mô tả sơ bộ")

    # st.markdown("""
    # **1. Thống kê mô tả sơ bộ** 
    # """)
    st.markdown("""        
    Bộ dữ liệu gồm **7.208 tin đăng** với **18 thuộc tính** (thương hiệu, dòng xe, số km, năm đăng ký, giá niêm yết, mô tả…) được thu thập từ nền tảng **Chợ Tốt** (trước ngày 01/07/2025).  
                """)
    # --- Vẽ biểu đồ ---

    # # Hiển thị 4 biểu đồ dạng lưới 2x2
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.image("brand_grouped_count.png")
    #     st.image("age_bin_stats.png")

    # with col2:
    #     st.image("price_bin_stats.png")
    #     st.image("mileage_bin_stats.png")

    # Đặt chiều rộng cho từng hình ảnh là 500px
    # Tổng chiều rộng 2 cột sẽ là 1000px
    image_width = 500
    
    # Hiển thị 4 biểu đồ dạng lưới 2x2
    col1, col2 = st.columns(2)
    with col1:
        st.image("brand_grouped_count.png", width=image_width) # Thêm width=500
        st.image("age_bin_stats.png", width=image_width)       # Thêm width=500

    with col2:
        st.image("price_bin_stats.png", width=image_width)     # Thêm width=500
        st.image("mileage_bin_stats.png", width=image_width)   # Thêm width=500

    st.subheader("II. Mô hình dự đoán giá xe máy")

    st.markdown("""

    ##### Lựa chọn thuộc tính           
    Để xây dựng mô hình dự đoán giá xe máy, chúng tôi đã chọn lọc các thuộc tính đầu vào (input features) có tính chất dự báo cao, bao gồm: **Thương hiệu, Dòng xe, Tuổi xe, Số km đã đi, Loại xe, Dung tích xe, Xuất xứ, Khoảng giá min,** và **Khoảng giá max**.
                         
    ##### Đánh giá mô hình
                
    Chúng tôi thử nghiệm nhiều mô hình machine learning, bao gồm **Random Forest, SVR, Gradient Boosting, Decision Tree** và **Linear Regression**. Trong số đó, **Random Forest** cho kết quả vượt trội nhất, thể hiện rõ qua bảng dưới đây:
    ##### 📊 So sánh hiệu quả các mô hình

    | Mô hình              | R²       | MAE (VNĐ)        | RMSE (VNĐ)       |
    |---------------------|----------|------------------|------------------|
    | **Random Forest**    | 0.888230 | 4,381,802        | 7,635,801        |
    | **SVR**              | 0.871969 | 4,607,752        | 8,172,413        |
    | **Gradient Boosting**| 0.851320 | 4,884,985        | 8,806,793        |
    | **Decision Tree**    | 0.813617 | 5,319,813        | 9,860,408        |
    | **Linear Regression**| 0.731268 | 6,343,373        | 11,840,010       |
    
    """)
    st.image("actual_vs_predicted.png")
    st.markdown("""
    => Kết quả đánh giá cho thấy **Random Forest là mô hình có hiệu suất tốt nhất**. Do đó, chúng tôi lựa chọn Random Forest làm mô hình chính cho bài toán **dự đoán giá xe máy**.
                """)
    
    st.subheader("III. Mô hình phát hiện xe bất thường")

    # st.markdown("""
    #     ###### Hệ thống phát hiện bất thường được xây dựng dựa trên **hai nhóm tiêu chí**: **Điểm số từ mô hình học máy** (`score_model_based`) và Điểm số từ logic nghiệp vụ** (`score_business_based`) 
    #         """)
    # st.markdown("""
    #     ###### Hai nhóm tiêu chí này được kết hợp nhằm đảm bảo việc phát hiện bất thường vừa **khách quan theo mô hình**, vừa **phù hợp thực tế kinh doanh**.   
    #             """)
    st.markdown("""
        ###### Hệ thống phát hiện bất thường được xây dựng dựa trên **hai nhóm tiêu chí**:

        * **Điểm số từ mô hình học máy** (`score_model_based`): Đảm bảo việc phát hiện bất thường mang tính **khách quan theo mô hình**.
        * **Điểm số từ logic nghiệp vụ** (`score_business_based`): Đảm bảo việc phát hiện bất thường **phù hợp thực tế kinh doanh**.

        Hai nhóm tiêu chí này được kết hợp nhằm mang lại kết quả phát hiện bất thường toàn diện và đáng tin cậy.
        """)

    st.markdown("""
        #### 1. Tiêu chí đánh dấu bất thường theo Logic Học máy (`score_model_based`)

        Hệ thống sử dụng **bốn tiêu chí** chính dựa trên mô hình thống kê và học máy để gán điểm bất thường:

        ---

        ##### 1.1. **`flag_resid` – Dựa trên phần dư (Residual Z-score)**
        * **Ngưỡng**: Được đặt là **3**.
        * **Đánh dấu bất thường**: Nếu **Residual Z-score > 3**, `flag_resid = 1`.
        * **Bình thường**: Nếu không, `flag_resid = 0`.

        ---

        ##### 1.2. **`flag_minmax` – Dựa trên khoảng giá hợp lý**
        * **Đánh dấu bất thường**: Nếu **giá niêm yết** nằm **ngoài khoảng giá Min-Max** được khai báo, `flag_minmax = 1`.
        * **Bình thường**: Nếu không, `flag_minmax = 0`.

        ---

        ##### 1.3. **`flag_p10p90` – Dựa trên Phân vị theo Phân khúc**
        * **Cơ sở**: Xác định **Phân vị 10% (P10)** và **90% (P90)** của giá xe trong từng phân khúc.
        * **Đánh dấu bất thường**: Nếu giá trị nằm **ngoài khoảng P10–P90**, `flag_p10p90 = 1`.
        * **Bình thường**: Nếu không, `flag_p10p90 = 0`.

        ---

        ##### 1.4. **`flag_unsup` – Tổng hợp từ Học máy không giám sát**
        * **Mô hình**: Kết hợp kết quả từ ba mô hình chính: **Isolation Forest, Local Outlier Factor, và KMeans**.
        * **Tiêu chí KMeans**: Điểm bất thường có số điểm trong cụm nhỏ hơn 10% tổng thể hoặc nằm trong 5% điểm xa tâm cụm nhất.
        * **Đánh dấu bất thường**: Nếu **hai trong ba** mô hình trên đánh dấu bất thường, `flag_unsup = 1`.

        ---

        ##### 📈 Tính toán `score_model_based`
        Điểm logic theo mô hình (`score_model_based`) là tổng có trọng số của 4 tiêu chí trên, trong đó **`flag_resid`** có **trọng số 0.4**, và các tiêu chí còn lại có trọng số **0.2**.

        ---

        #### 2. Tiêu chí đánh dấu bất thường theo Logic Nghiệp vụ (`score_business_based`)

        Tiêu chí này tập trung vào sự bất thường của mối quan hệ giữa **Số km đã đi** và **Tuổi xe**:

        * **Nghi vấn Tua công-tơ-mét (Quá thấp)**: Nếu **Số km đã đi < 200 * Tuổi xe**.
        * **Số km cao bất thường (Khai thác/Khai báo sai)**: Nếu **Số km đã đi > 20000 * Tuổi xe**.

        ---

        #### 3. Tổng hợp và Đánh dấu cuối cùng

        * **Điểm tổng hợp cuối cùng (`final_score`)** là tổng của hai điểm: **`score_model_based`** và **`score_business_based`**.
        * **Đánh dấu Bất thường**: Xe có tổng điểm **lớn hơn 50** sẽ được đánh dấu là **Bất thường**.
        """)

    st.markdown("##### Ví dụ 10 mẫu xe bất thường được phát hiện:")
    df_anomaly = pd.read_csv("outliers_detected_full.csv")
    st.dataframe(df_anomaly.sort_values('final_score', ascending=False).head(10))
    
elif page == "Dự đoán giá":

    # --- PREDICTION PAGE ---

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
                # ===============================
                # 1) Load file raw
                # ===============================
                if uploaded_file.name.endswith(".csv"):
                    df_raw = pd.read_csv(tmp_path)
                else:
                    df_raw = pd.read_excel(tmp_path)
                    df_raw = df_raw.rename(columns={
                        'Giá': 'price',
                        'Khoảng giá min': 'min_price',
                        'Khoảng giá max': 'max_price',
                        'Thương hiệu': 'brand',
                        'Dòng xe': 'model',
                        'Năm đăng ký': 'registration_year',
                        'Số Km đã đi': 'mileage_km',
                        'Tình trạng': 'condition',
                        'Loại xe': 'bike_type',
                        'Dung tích xe': 'engine_capacity',
                        'Xuất xứ': 'origin',
                        'Chính sách bảo hành': 'warranty_policy',
                        'Trọng lượng': 'weight'
                    })

                # ===============================
                # 2) Chỉ giữ đúng các cột cần thiết
                # KHÔNG CLEAN nữa để KHÔNG lệch pipeline nhập tay
                # ===============================
                needed_cols = [
                    'brand', 'model', 'bike_type', 'origin', 'engine_capacity',
                    'registration_year', 'mileage_km', 'min_price', 'max_price'
                ]

                df = df_raw[needed_cols].copy()

                # ===============================
                # 3) Chuyển NaN min/max về NaN (giống nhập tay)
                # ===============================
                df['min_price'] = df['min_price'].replace(0, np.nan)
                df['max_price'] = df['max_price'].replace(0, np.nan)

                # ===============================
                # 4) Tính age giống hệt nhập tay
                # ===============================
                current_year = 2025
                df['age'] = current_year - pd.to_numeric(df['registration_year'], errors='coerce')

                # ===============================
                # 5) Apply grouping EXACT như nhập tay
                # ===============================
                if helpers is not None:
                    # brand_grouped
                    df['brand_grouped'] = df['brand'].apply(
                        lambda b: 'Hãng khác' if b in helpers['rare_brands'] else b
                    )

                    # model_grouped theo từng brand_grouped
                    def map_model(row):
                        bg = row['brand_grouped']
                        rare_models = helpers['model_group_maps'].get(bg, set())
                        return 'Dòng khác' if row['model'] in rare_models else row['model']

                    df['model_grouped'] = df.apply(map_model, axis=1)

                    # segment
                    df['segment'] = df['brand_grouped'] + '_' + df['model_grouped']

                    # brand_meanprice
                    df['brand_meanprice'] = df['brand'].map(helpers['brand_mean_map'])

                else:
                    # fallback
                    df['brand_grouped'] = df['brand']
                    df['model_grouped'] = df['model']
                    df['segment'] = df['brand'] + '_' + df['model']
                    df['brand_meanprice'] = np.nan
                    st.warning("Không có helpers, dự đoán có thể không chính xác.")

                # ===============================
                # 6) Chuẩn bị dữ liệu để predict
                # ===============================
                cat_cols = ['segment','bike_type','origin','engine_capacity']
                num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']

                X = df[cat_cols + num_cols]

                # ===============================
                # 7) Predict
                # ===============================
                df['predicted_price'] = np.expm1(model.predict(X))

                # ===============================
                # 8) Show result
                # ===============================
                st.write("Kết quả (10 dòng đầu):")
                st.dataframe(df.head(10))

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Tải kết quả (CSV)", data=csv, file_name="predictions.csv", mime='text/csv')

            except Exception as e:
                st.error(f"Lỗi xử lý file: {e}")


# --- ANOMALY PAGE ---
else:
    st.header("Phát hiện xe bất thường")

    # Tạo 2 TAB
    tab_user, tab_admin = st.tabs(["👤 User kiểm tra xe", "🛠 Admin kiểm tra dữ liệu"])

    # ======================================
    # 1) TAB USER
    # ======================================
    with tab_user:

        # st.subheader("Nhập tay 1 xe để kiểm tra")

        # Hàm lưu request user vào file Excel
        # def save_user_request(df_input):
        #     save_path = "user_submissions.xlsx"
        #     if os.path.exists(save_path):
        #         old = pd.read_excel(save_path)
        #         new = pd.concat([old, df_input], ignore_index=True)
        #     else:
        #         new = df_input.copy()

        #     new.to_excel(save_path, index=False)

        # Hàm lưu request user vào file Excel
        def save_user_request(df_input):
            save_path = "user_submissions.xlsx"
            
            # Tạo bản sao để tránh thay đổi DataFrame gốc (df_in)
            df_save = df_input.copy() 

            # 1. Kiểm tra xem cột 'post_time' có tồn tại không
            if 'post_time' in df_save.columns:
                # 2. Nếu cột là timezone-aware (có múi giờ), chuyển nó thành timezone-unaware
                if df_save['post_time'].dt.tz is not None:
                    # .dt.tz_localize(None) sẽ loại bỏ thông tin múi giờ (GMT+7)
                    # Dữ liệu ngày giờ vẫn giữ nguyên giá trị theo giờ địa phương (GMT+7)
                    df_save['post_time'] = df_save['post_time'].dt.tz_localize(None)

            if os.path.exists(save_path):
                old = pd.read_excel(save_path)
                new = pd.concat([old, df_save], ignore_index=True)
            else:
                new = df_save.copy()

            # Đoạn này sẽ chạy trơn tru vì cột ngày giờ đã là timezone-unaware
            new.to_excel(save_path, index=False)

        # ============================
        # 1.1 Nhập tay
        # ============================
        st.subheader("Nhập thông tin xe cần rao bán")
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox("Thương hiệu", brand_list)
            model_name = st.selectbox("Dòng xe", model_list)
            bike_type = st.selectbox("Loại xe", bike_type_list)
            origin = st.selectbox("Xuất xứ", origin_list)
            engine_capacity = st.selectbox("Dung tích", engine_capacity_list)

        with col2:
            registration_year = st.number_input("Năm đăng ký", 1980, 2025, 2019)
            mileage_km = st.number_input("Số km đã đi", 0, value=10000)
            min_price = st.number_input("Khoảng giá min", 0)
            max_price = st.number_input("Khoảng giá max", 0)
            price = st.number_input("Giá niêm yết", 0, value=20000000)
        
        # Thêm ngày giờ đăng tin
        col_d, col_t = st.columns(2)

        # with col_d:
        #     post_date = st.date_input("Ngày đăng tin", value=pd.Timestamp.now().date())

        # with col_t:
        #     post_time = st.time_input("Giờ đăng tin", value=pd.Timestamp.now().time())

        # # Gộp thành datetime
        # post_datetime = pd.to_datetime(str(post_date) + " " + str(post_time))

        with col_d:
            # Bạn có thể giữ nguyên giá trị mặc định là giờ hiện tại
            post_date = st.date_input("Ngày đăng tin", value=pd.Timestamp.now(tz=pytz.timezone('Asia/Ho_Chi_Minh')).date())

        with col_t:
            post_time = st.time_input("Giờ đăng tin", value=pd.Timestamp.now(tz=pytz.timezone('Asia/Ho_Chi_Minh')).time())

        # Gộp thành datetime và gán múi giờ:
        # 1. Tạo đối tượng datetime thô (naive datetime) từ date và time input
        naive_datetime = pd.to_datetime(str(post_date) + " " + str(post_time))

        # 2. Định nghĩa múi giờ Asia/Ho_Chi_Minh (GMT+7)
        vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

        # 3. Gán múi giờ cho đối tượng datetime
        post_datetime = vietnam_tz.localize(naive_datetime)

        # chuẩn bị key cho session_state
        if "last_df_in" not in st.session_state:
            st.session_state["last_df_in"] = None
        if "last_anomaly" not in st.session_state:
            st.session_state["last_anomaly"] = None
        if "checked" not in st.session_state:
            st.session_state["checked"] = False

        if st.button("Kiểm tra"):
            df_in = pd.DataFrame([{
                "brand": brand,
                "model": model_name,
                "bike_type": bike_type,
                "origin": origin,
                "engine_capacity": engine_capacity,
                "registration_year": registration_year,
                "mileage_km": mileage_km,
                "min_price": min_price,
                "max_price": max_price,
                "price": price
            }])

            df_in["age"] = 2025 - df_in["registration_year"]
            df_in["post_time"] = post_datetime

            # Mapping using helpers
            if helpers is not None:
                if df_in.at[0, 'brand'] in helpers['rare_brands']:
                    df_in['brand_grouped'] = 'Hãng khác'
                else:
                    df_in['brand_grouped'] = df_in['brand']

                rare_models = helpers['model_group_maps'].get(df_in.at[0, 'brand_grouped'], set())
                if df_in.at[0, 'model'] in rare_models:
                    df_in['model_grouped'] = 'Dòng khác'
                else:
                    df_in['model_grouped'] = df_in['model']

                df_in["segment"] = df_in["brand_grouped"] + "_" + df_in["model_grouped"]
                df_in["brand_meanprice"] = helpers["brand_mean_map"].get(df_in.at[0,"brand"], np.nan)
            else:
                df_in["brand_grouped"] = df_in["brand"]
                df_in["model_grouped"] = df_in["model"]
                df_in["segment"] = df_in["brand"] + "_" + df_in["model"]
                df_in["brand_meanprice"] = np.nan

            try:
                df_all, anomaly = detect_outliers(df_in, model_path=MODEL_PATH, input_is_df=True, helpers=helpers)

                # lưu tạm vào session để dùng sau khi user xác nhận
                st.session_state["last_df_in"] = df_in
                st.session_state["last_anomaly"] = anomaly
                st.session_state["checked"] = True

            except Exception as e:
                st.exception(e)

        # Nếu đã có kết quả kiểm tra trong session_state thì hiển thị
        if st.session_state.get("checked", False):
            df_in = st.session_state["last_df_in"]
            anomaly = st.session_state["last_anomaly"]

            if anomaly is None:
                st.info("Không có kết quả kiểm tra.")
            else:
                if len(anomaly) > 0:
                    # xác định reason dựa trên score như yêu cầu (model/business)
                    # note: detect_outliers đã tính score_model_based, score_business_based
                    r = []
                    if anomaly["score_model_based"].iloc[0] >= 50:
                        r.append("Mô hình cảnh báo phát hiện")
                    if anomaly["flag_mileage_low"].iloc[0] == 1:
                        r.append("Logic nghiệp vụ (Số km đã đi thấp bất thường)")
                    if anomaly["flag_mileage_high"].iloc[0] == 1:
                        r.append("Logic nghiệp vụ (Số km đã đi cao bất thường)")
                    reason_text = " + ".join(r) if r else "Không xác định"

                    st.error(f"🚨 Xe này BẤT THƯỜNG — do {reason_text}")
                    # st.dataframe(anomaly)

                    # hỏi user: có muốn đăng không? + nút xác nhận lưu
                    choice = st.radio("Xe này bất thường, bạn vẫn muốn đăng tin không?", ["Không", "Có"], horizontal=True, key="confirm_post_radio")

                    if st.button("Xác nhận"):
                        if choice == "Có":
                            # chuẩn bị bản lưu: loại bỏ cột nội bộ trước khi lưu
                            # df_save = df_in.copy()
                            # cols_to_drop = ["brand_grouped", "model_grouped", "segment", "brand_meanprice"]
                            # df_save = df_save.drop(columns=[c for c in cols_to_drop if c in df_save.columns])
                            save_user_request(df_in) # save đủ thông tin
                            st.success("Đã đăng tin.")
                            # reset flags
                            st.session_state["last_df_in"] = None
                            st.session_state["last_anomaly"] = None
                            st.session_state["checked"] = False
                        else:
                            st.info("Bạn đã chọn không đăng tin này.")
                            # reset session
                            st.session_state["last_df_in"] = None
                            st.session_state["last_anomaly"] = None
                            st.session_state["checked"] = False

                else:
                    st.success("Xe này KHÔNG bất thường")
                    # Show nút lưu nếu user muốn (optional) — tự lưu hoặc cho user bấm
                    if st.button("Đăng tin"):
                        # df_save = df_in.copy()
                        # cols_to_drop = ["brand_grouped", "model_grouped", "segment", "brand_meanprice"]
                        # df_save = df_save.drop(columns=[c for c in cols_to_drop if c in df_save.columns])
                        save_user_request(df_in)
                        st.success("Đã đăng tin.")
                        st.session_state["last_df_in"] = None
                        st.session_state["last_anomaly"] = None
                        st.session_state["checked"] = False




    # ======================================
    # 2) TAB ADMIN
    # ======================================
    with tab_admin:

        st.subheader("Chế độ kiểm tra dành cho Admin")

        mode_admin = st.radio(
            "Chọn cách kiểm tra:",
            ["Dữ liệu user nhập hôm nay", "Upload file"],
            horizontal=True
        )

        save_path = "user_submissions.xlsx"

        # ============================================================
        # MODE 1: KIỂM TRA DỮ LIỆU USER NHẬP HÔM NAY
        # ============================================================
        if mode_admin == "Dữ liệu user nhập hôm nay":

            st.subheader("Danh sách tin user đã gửi")

            if os.path.exists(save_path):
                df_user = pd.read_excel(save_path)

                cols_to_hide = ["brand_grouped", "model_grouped", "segment", "brand_meanprice"]
                df_user_display = df_user.drop(columns=[c for c in cols_to_hide if c in df_user.columns])

                st.dataframe(df_user_display.sort_values(by='post_time', ascending=False))

                if st.button("Chạy kiểm tra bất thường (User submissions)"):
                    try:
                        df_all, anomaly = detect_outliers(
                            df_user,
                            model_path=MODEL_PATH,
                            input_is_df=True,
                            helpers=helpers
                        )

                        st.success(f"Phát hiện {len(anomaly)} bất thường")
                        anomaly_print = anomaly.copy()
                        cols_to_drop = ['brand_grouped', 'model_grouped', 'segment', 'brand_meanprice','price_hat','resid','resid_median','resid_std','resid_z','flag_resid','p10','p90'
]
                        anomaly_print = anomaly_print.drop(columns=[c for c in cols_to_drop if c in anomaly_print.columns])
                        st.dataframe(anomaly_print.sort_values(by='post_time', ascending=False).head(20))

                        # === BẮT ĐẦU THÊM NÚT TẢI XUỐNG ===
                        if len(anomaly) > 0:
                            # 1. Tạo tên file có ngày giờ
                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_name = f"anomaly_detection_user_{now}.csv"
                            
                            # 2. Chuyển DataFrame sang CSV
                            # Loại bỏ múi giờ khỏi cột 'post_time' trước khi tải xuống nếu cần (đảm bảo không lỗi)
                            df_output = anomaly_print.copy()
                            if 'post_time' in df_output.columns and df_output['post_time'].dt.tz is not None:
                                df_output['post_time'] = df_output['post_time'].dt.tz_localize(None)

                            csv = df_output.to_csv(index=False).encode('utf-8')
                            
                            # 3. Tạo nút tải xuống
                            st.download_button(
                                label="Tải kết quả bất thường (CSV)",
                                data=csv,
                                file_name=file_name,
                                mime='text/csv'
                            )
                        # === KẾT THÚC THÊM NÚT TẢI XUỐNG ===

                    except Exception as e:
                        st.exception(e)

            else:
                st.info("⚠ Chưa có user nào gửi dữ liệu.")


        # ============================================================
        # MODE 2: ADMIN UPLOAD FILE KIỂM TRA
        # ============================================================
        else:
            st.subheader("Upload file để Admin kiểm tra")

            file_admin = st.file_uploader(
                "Chọn file dữ liệu cần kiểm tra (xlsx/csv)",
                type=["xlsx", "csv"],
                key="admin_upload_file"
            )

            if st.button("Chạy kiểm tra file Admin"):
                if file_admin is None:
                    st.error("Vui lòng upload file trước!")
                else:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=os.path.splitext(file_admin.name)[1]
                    ) as tmp:
                        tmp.write(file_admin.getvalue())
                        excel_path = tmp.name

                    try:
                        df_in = preprocess_motobike_data(excel_path)
                        df_all, anomaly = detect_outliers(
                            df_in, 
                            model_path=MODEL_PATH, 
                            input_is_df=True, 
                            helpers=helpers
                        )

                        st.success(
                            f"Hoàn tất kiểm tra. Tổng {len(df_in)} bản ghi — phát hiện {len(anomaly)} bất thường."
                        )
                        # st.dataframe(anomaly.head(20))
                        anomaly_print = anomaly.copy()
                        cols_to_drop = ['brand_grouped', 'model_grouped', 'segment', 'brand_meanprice','price_hat','resid','resid_median','resid_std','resid_z','flag_resid','p10','p90'
]
                        anomaly_print = anomaly_print.drop(columns=[c for c in cols_to_drop if c in anomaly_print.columns])
                        st.dataframe(anomaly_print.head(20))

                        # === BẮT ĐẦU THÊM NÚT TẢI XUỐNG ===
                        if len(anomaly) > 0:
                            # 1. Tạo tên file có ngày giờ
                            now = datetime.now().strftime("%Y%m%d_%H%M%S")
                            file_name = f"anomaly_detection_admin_{now}.csv"
                            
                            # 2. Chuyển DataFrame sang CSV
                            df_output = anomaly_print.copy()
                            # Nếu cột post_time có, hãy loại bỏ múi giờ (để tránh lỗi)
                            if 'post_time' in df_output.columns and df_output['post_time'].dt.tz is not None:
                                df_output['post_time'] = df_output['post_time'].dt.tz_localize(None)

                            csv = df_output.to_csv(index=False).encode('utf-8')
                            
                            # 3. Tạo nút tải xuống
                            st.download_button(
                                label="Tải kết quả bất thường (CSV)",
                                data=csv,
                                file_name=file_name,
                                mime='text/csv'
                            )
                        # === KẾT THÚC THÊM NÚT TẢI XUỐNG ===

                    except Exception as e:
                        st.exception(e)

st.sidebar.markdown("---")
st.sidebar.markdown("Ứng dụng cho phép: 1) Dự đoán giá xe máy 2) Phát hiện xe bất thường (nhập tay hoặc upload file)")
