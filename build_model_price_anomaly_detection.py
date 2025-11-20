import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import KMeans
from function_preprocessing_motorbike import *
import pickle

def detect_outliers(data_input, model_path, input_is_df=False, helpers=None):
    """
    Detect anomalies cho xe máy.
    - Nếu input_is_df = False: data_input là đường dẫn file Excel/CSV.
    - Nếu input_is_df = True: data_input là DataFrame (1 hoặc nhiều xe).
    """
    # 0. THRESHOLDS
    TH = {
        # Business rule (mileage)
        "min_km_per_year": 2000,
        "max_km_per_year": 20000,

        # Residual threshold
        "resid_z_threshold": 3,

        # Unsupervised
        "if_contam": 0.2,
        "lof_contam": 0.2,
        "kmeans_clusters": 4,
        "kmeans_small_ratio": 0.1,

        # Final scoring
        "score_resid": 0.4,
        "score_minmax": 0.2,
        "score_p10p90": 0.2,
        "score_unsup": 0.2,
        "score_threshold": 50
    }


    # 1. LOAD DỮ LIỆU
    if input_is_df:
        # Trường hợp nhập tay 1 xe → đã là DataFrame
        df = data_input.copy()

    else:
        # Trường hợp đọc file Excel hoặc CSV
        df = preprocess_motobike_data(data_input)

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Define cols
    cat_cols = ['segment','bike_type','origin','engine_capacity']
    num_cols = ['age','mileage_km','min_price','max_price','brand_meanprice']

    # Build matrix
    X = df[cat_cols + num_cols]
    # y = df['log_price']

    # Predict price
    df['price_hat'] = np.expm1(model.predict(X))

    # Residual z-score by segment
    df['resid'] = df['price'] - df['price_hat']

    # seg_stats = (
    #     df.groupby('segment')['resid']
    #       .agg(['median', 'std'])
    #       .reset_index()
    #       .rename(columns={'median':'resid_median', 'std':'resid_std'})
    # )
    if helpers is not None and df['segment'].iloc[0] in helpers['seg_resid_map']:
        seg = df['segment'].iloc[0]
        df['resid_median'] = helpers['seg_resid_map'][seg]['resid_mean']
        df['resid_std']    = helpers['seg_resid_map'][seg]['resid_std']
    else:
        # fallback khi không có helpers
        seg_stats = (
                df.groupby('segment')['resid']
                .agg(['median', 'std'])
                .reset_index()
                .rename(columns={'median':'resid_median', 'std':'resid_std'})
            )
        df = df.merge(seg_stats.rename(columns={'median':'resid_median','std':'resid_std'}), on='segment')


    # df = df.merge(seg_stats, on='segment', how='left')

    df['resid_z'] = (df['resid'] - df['resid_median']) / (df['resid_std'] + 1e-9)
    df['flag_resid'] = (df['resid_z'].abs() > TH["resid_z_threshold"]).astype(int)

    # Min/Max & P10/P90 violations
    # seg_price_stats = (
    #     df.groupby('segment')['price']
    #       .quantile([0.10, 0.90])
    #       .unstack(level=1)
    #       .reset_index()
    #       .rename(columns={0.10:'p10', 0.90:'p90'})
    # )
    # df = df.merge(seg_price_stats, on='segment', how='left')

    if helpers is not None and df['segment'].iloc[0] in helpers['seg_price_map']:
        seg = df['segment'].iloc[0]
        df['p10'] = helpers['seg_price_map'][seg]['p10']
        df['p90'] = helpers['seg_price_map'][seg]['p90']
    else:
        # fallback
        seg_price_stats = (
            df.groupby('segment')['price']
              .quantile([0.10, 0.90])
              .unstack(level=1)
              .reset_index()
              .rename(columns={0.10:'p10', 0.90:'p90'})
        )
        df = df.merge(seg_price_stats, on='segment', how='left')


    df['flag_minmax'] = ((df['price'] < df['min_price']) |
                         (df['price'] > df['max_price'])).astype(int)
    df['flag_p10p90'] = ((df['price'] < df['p10']) |
                         (df['price'] > df['p90'])).astype(int)

    # business rule số km đã đi
    df['flag_mileage_low'] = (
        df['mileage_km'] < df['age'] * TH["min_km_per_year"]
    ).astype(int)

    df['flag_mileage_high'] = (
        df['mileage_km'] > df['age'] * TH["max_km_per_year"]
    ).astype(int)

    # Nếu 1 trong 2 rule vi phạm → flag_br = 1
    df['flag_br'] = (
        (df['flag_mileage_low'] == 1) | 
        (df['flag_mileage_high'] == 1)
    ).astype(int)

    # Nếu dữ liệu quá ít (< 10 bản ghi) -> tắt unsupervised
    if len(df) < 10:
        df['flag_if'] = 0
        df['flag_lof'] = 0
        df['flag_kmeans'] = 0
        df['flag_unsup'] = 0
    else:

        # Unsupervised models
        unsup_feats = ['age','mileage_km','resid_z']
        X_unsup = df[unsup_feats].fillna(0).values

        scaler = StandardScaler()
        Xu = scaler.fit_transform(X_unsup)

        # --- Isolation Forest ---
        if_model = IsolationForest(
            n_estimators=200,
            contamination=TH["if_contam"],
            random_state=42
        )
        if_model.fit(Xu)
        df['flag_if'] = (if_model.predict(Xu) == -1).astype(int)

        # --- KMeans ---
        n_clusters = TH["kmeans_clusters"]
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(Xu)
        df['kmeans_cluster'] = cluster_labels

        cluster_stats = []
        n_total = len(df)
        outlier_flags = np.zeros(n_total, dtype=int)

        for k in range(n_clusters):
            pts = Xu[cluster_labels == k]
            size = len(pts)
            center = pts.mean(axis=0)
            dist = np.linalg.norm(pts - center, axis=1)
            r95 = np.percentile(dist, 95)

            cluster_stats.append({
                'cluster': k,
                'size': size,
                'center': center,
                'r95': r95
            })

            # Outlier rules
            if size / n_total < TH["kmeans_small_ratio"]:
                # small cluster
                outlier_flags[cluster_labels == k] = 1
            else:
                # by distance
                idx = np.where(cluster_labels == k)[0]
                outlier_idx = idx[dist > r95]
                outlier_flags[outlier_idx] = 1

        df['flag_kmeans'] = outlier_flags

        # --- LOF ---
        lof = LocalOutlierFactor(n_neighbors=20, contamination=TH["lof_contam"])
        lof_fit = lof.fit_predict(Xu)
        df['flag_lof'] = (lof_fit == -1).astype(int)

        # Combined unsupervised flag
        df['flag_unsup'] = (df[['flag_if','flag_lof','flag_kmeans']].sum(axis=1) > 1).astype(int)

    # Final scoring
    df['final_score'] = 100 * (
        TH["score_resid"]*df['flag_resid'] +
        TH["score_minmax"]*df['flag_minmax'] +
        TH["score_p10p90"]*df['flag_p10p90'] +
        TH["score_unsup"] *df['flag_unsup']
    )

    df['is_outlier'] = ((df['final_score'] > TH["score_threshold"]) | (df['flag_br'] == 1)).astype(int)

    # Xuất kết quả
    anomaly = df[df['is_outlier'] == 1].copy()
    anomaly.to_csv("outliers_detected.csv", index=False)

    print("=== Tổng thể ===")
    print(f"Tổng bản ghi: {len(df):,}")
    print(f"Số bản ghi bất thường: {len(anomaly):,} "
          f"({100*len(anomaly)/len(df):.2f}%)")
    print("Đã xuất 'outliers_detected.csv'.")

    return df, anomaly

# df, anomaly = detect_outliers(
#     "data_motobikes.xlsx",
#     "motobike_price_prediction_model.pkl"
# )
