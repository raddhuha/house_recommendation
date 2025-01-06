import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
import plotly.express as px

# atur judul halaman dan dekripsi
st.set_page_config(page_title="Sistem Rekomendasi Rumah", layout="wide")
st.title("Sistem Rekomendasi Ketersediaan Rumah")
st.write("Aplikasi ini membantu Anda menemukan rumah berdasarkan kriteria tertentu.")

@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        dataset = pd.read_csv('dataset_rumah_jateng_diy.csv')
        return dataset
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan. Pastikan file CSV tersedia.")
        return None

def preprocess_data(data):
    """Preprocess the dataset by handling missing values and duplicates"""
    if data is None:
        return None
    
    # buat salinan data untuk menghindari perubahan data asli
    processed_data = data.copy()
    
    # menangani missing values
    numeric_columns = processed_data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        processed_data[col].fillna(processed_data[col].median(), inplace=True)
    
    # Handle kategori missing values
    categorical_columns = processed_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
    
    # drop duplikat
    processed_data.drop_duplicates(inplace=True)
    
    # hapus fitur yang tidak diperlukan
    unused_features = ['id', 'label_bintang']
    processed_data.drop(columns=[col for col in unused_features if col in processed_data.columns], inplace=True)
    
    return processed_data

def get_recommendations(data, user_input, use_kriteria):
    """Persiapan data untuk clustering dan rekomendasi"""
    # persiapan clustering
    data_for_clustering = data.copy()
    
    # inisialisasi label encoders dan kolom kategori
    label_encoders = {}
    categorical_columns = data_for_clustering.select_dtypes(include=['object']).columns
    
    # emcode data kategori
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data_for_clustering[col] = label_encoders[col].fit_transform(data_for_clustering[col].astype(str))
    
    # scaling data numerik
    numerical_columns = data_for_clustering.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    data_for_clustering[numerical_columns] = scaler.fit_transform(data_for_clustering[numerical_columns])
    
    # aplikasikan KMeans clustering
    n_clusters = min(5, len(data_for_clustering))
    kmeans = KMeans(n_clusters=n_clusters, random_state=1000, n_init=10)
    clusters = kmeans.fit_predict(data_for_clustering)
    
    # user input
    user_data = pd.DataFrame([user_input])
    
    # transform user input
    for col in categorical_columns:
        if col in user_data.columns and user_input[col]:
            user_data[col] = label_encoders[col].transform([user_input[col]])
        else:
            user_data[col] = 0
    
    # scaling user input
    user_data_scaled = scaler.transform(user_data[numerical_columns])
    
    # prediksi cluster user
    user_cluster = kmeans.predict(user_data_scaled)[0]
    # Get houses from the same cluster
    cluster_mask = (clusters == user_cluster)
    recommendations = data[cluster_mask].copy()

    """Calculate similarity score within the cluster"""
    # Numeric features similarity
    feature_weights = {
        'harga': 0.3,
        'jumlah_kamar_tidur': 0.15,
        'jumlah_kamar_mandi': 0.15,
        'jumlah_lantai': 0.1,
        'luas_lahan': 0.15,
        'luas_bangunan': 0.15
    }
    
    def calculate_similarity(row):
        score = 0
        for feature, weight in feature_weights.items():
            if feature in row and feature in user_input:
                max_val = data[feature].max()
                min_val = data[feature].min()
                range_val = max_val - min_val if max_val != min_val else 1
                normalized_diff = abs(float(row[feature]) - float(user_input[feature])) / range_val
                score += (1 - normalized_diff) * weight
        return score
    recommendations['similarity_score'] = recommendations.apply(calculate_similarity, axis=1)
    recommendations = recommendations.sort_values('similarity_score', ascending=False)
    recommendations = recommendations.drop('similarity_score', axis=1)
    
    return recommendations

def main():
    # load data
    data = load_data()
    if data is None:
        return
    
    processed_data = preprocess_data(data)
    
    # Sidebar 
    st.sidebar.title("Filter Kriteria")
    
    # Budget input
    st.subheader("Masukkan Budget")
    
    budget_max = st.number_input("Budget (dalam juta):", min_value=0, step=1)
    
    # tambahan criteria
    use_kriteria = st.checkbox("Gunakan kriteria tambahan")
    
    if use_kriteria:
        with st.sidebar:
            kamar_tidur = st.slider("Jumlah kamar tidur:", 0, 6, (0, 6))
            kamar_mandi = st.slider("Jumlah kamar mandi:", 0, 6, (0, 6))
            jumlah_lantai = st.slider("Jumlah lantai:", 1, 3, (1, 3))
            luas_lahan = st.slider("Luas lahan (m2):", 0, 500, (0, 500))
            luas_bangunan = st.slider("Luas bangunan (m2):", 0, 500, (0, 500))
            
            lokasi_sekitar = st.selectbox(
                "Lokasi sekitar:",
                [""] + sorted(processed_data['lokasi_sekitar'].unique().tolist())
            )
            
            lokasi_rumah = st.selectbox(
                "Lokasi rumah:",
                [""] + sorted(processed_data['lokasi_rumah'].unique().tolist())
            )
            
            fasilitas_sekitar = st.selectbox(
                "Fasilitas sekitar:",
                [""] + sorted(processed_data['fasilitas_sekitar'].unique().tolist())
            )
    
    if st.button("Cek Ketersediaan"):
        # Create filter conditions
        conditions = (
            
            (processed_data["harga"].astype(float) <= budget_max * 1_000_000)
        )
        
        if use_kriteria:
            additional_conditions = (
                (processed_data["jumlah_kamar_tidur"] >= kamar_tidur[0]) &
                (processed_data["jumlah_kamar_tidur"] <= kamar_tidur[1]) &
                (processed_data["jumlah_kamar_mandi"] >= kamar_mandi[0]) &
                (processed_data["jumlah_kamar_mandi"] <= kamar_mandi[1]) &
                (processed_data["jumlah_lantai"] >= jumlah_lantai[0]) &
                (processed_data["jumlah_lantai"] <= jumlah_lantai[1]) &
                (processed_data["luas_lahan"] >= luas_lahan[0]) &
                (processed_data["luas_lahan"] <= luas_lahan[1]) &
                (processed_data["luas_bangunan"] >= luas_bangunan[0]) &
                (processed_data["luas_bangunan"] <= luas_bangunan[1])
            )
            
            if lokasi_sekitar:
                additional_conditions &= (processed_data["lokasi_sekitar"] == lokasi_sekitar)
            if lokasi_rumah:
                additional_conditions &= (processed_data["lokasi_rumah"] == lokasi_rumah)
            if fasilitas_sekitar:
                additional_conditions &= (processed_data["fasilitas_sekitar"] == fasilitas_sekitar)
            
            conditions &= additional_conditions
        
        # filter data
        hasil = processed_data[conditions]
        
        if not hasil.empty:
            st.success(f"Ditemukan {len(hasil)} rumah yang sesuai dengan kriteria Anda.")
            st.dataframe(hasil)
            
            # visualisasi distribusi harga
            fig = px.histogram(hasil, x="harga", title="Distribusi Harga Rumah yang Sesuai")
            st.plotly_chart(fig)
            
        else:
            st.warning("Tidak ditemukan rumah yang sesuai dengan kriteria Anda.")
            st.write("Berikut adalah rekomendasi rumah dengan kriteria yang mirip:")
            
            # user input
            user_input = {
                "harga": budget_max * 1_000_000,
                "jumlah_kamar_tidur": kamar_tidur[0] if use_kriteria else 0,
                "jumlah_kamar_mandi": kamar_mandi[0] if use_kriteria else 0,
                "jumlah_lantai": jumlah_lantai[0] if use_kriteria else 1,
                "luas_lahan": luas_lahan[0] if use_kriteria else 0,
                "luas_bangunan": luas_bangunan[0] if use_kriteria else 0,
                "lokasi_sekitar": lokasi_sekitar if lokasi_sekitar else processed_data["lokasi_sekitar"].mode()[0],
                "lokasi_rumah": lokasi_rumah if lokasi_rumah else processed_data["lokasi_rumah"].mode()[0],
                "fasilitas_sekitar": fasilitas_sekitar if fasilitas_sekitar else processed_data["fasilitas_sekitar"].mode()[0]
            }
            
            # mendapatkan rekomendasi
            recommendations = get_recommendations(processed_data, user_input, use_kriteria)
            
            # menunjukkan 10 rekomendasi
            st.dataframe(recommendations.head(10))
            
            # visualisasi rekomendasi
            fig = px.scatter(recommendations.head(10), 
                           x="harga", 
                           y="luas_bangunan",
                           color="lokasi_rumah",
                           title="Rekomendasi Rumah Berdasarkan Harga dan Luas Bangunan")
            st.plotly_chart(fig)

if __name__ == "__main__":
    main()
