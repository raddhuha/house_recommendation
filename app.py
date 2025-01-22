import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

# Set judul halaman dan dekripsi
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
    """Get house recommendations using PCA, K-Means clustering and similarity scoring"""
    # Prepare data for clustering
    data_for_clustering = data.copy()
    
    # Initialize encoders and scaler
    label_encoders = {}
    categorical_columns = data_for_clustering.select_dtypes(include=['object']).columns
    
    # Encode categorical features
    for col in categorical_columns:
        label_encoders[col] = LabelEncoder()
        data_for_clustering[col] = label_encoders[col].fit_transform(data_for_clustering[col].astype(str))
    
    # Scale numerical features
    numerical_columns = data_for_clustering.select_dtypes(include=['float64', 'int64']).columns
    scaler = MinMaxScaler()
    data_for_clustering[numerical_columns] = scaler.fit_transform(data_for_clustering[numerical_columns])
    
    # Apply PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(data_for_clustering)
    
    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        features_pca,
        columns=['Principal Component 1', 'Principal Component 2']
    )
    
    # Apply K-Means clustering on PCA results
    n_clusters = min(9, len(data_for_clustering))
    kmeans = KMeans(n_clusters=n_clusters, random_state=1000, n_init=10)
    clusters = kmeans.fit_predict(features_pca)
    
    # Visualize clustering results on PCA dimensions
    pca_df['Cluster'] = clusters
    
    # Prepare user input
    user_data = pd.DataFrame([user_input])
    
    # Transform user input
    for col in categorical_columns:
        if col in user_data.columns and user_input[col]:
            user_data[col] = label_encoders[col].transform([user_input[col]])
        else:
            user_data[col] = 0
    
    # Scale user input
    user_data_scaled = scaler.transform(user_data[numerical_columns])
    
    # Transform user input using PCA
    user_pca = pca.transform(user_data_scaled)
    
    # Get user's cluster
    user_cluster = kmeans.predict(user_pca)[0]
    st.write(f"Input Anda termasuk dalam Cluster {user_cluster}")
    
    # Get houses from the same cluster
    cluster_mask = (clusters == user_cluster)
    recommendations = data[cluster_mask].copy()
    
    # Calculate similarity scores within the cluster
    feature_weights = {
        'harga': 0.3,
        'lokasi_rumah': 0.2,    
        'lokasi_sekitar': 0.1,
        'fasilitas_sekitar': 0.05,
        'luas_bangunan': 0.05,
        'luas_lahan': 0.05,
        'jumlah_kamar_tidur': 0.15,
        'jumlah_kamar_mandi': 0.15,
        'jumlah_lantai': 0.1
    }
    
    def calculate_similarity(row):
        score = 0
        for feature, weight in feature_weights.items():
            if feature in row and feature in user_input:
                if feature in categorical_columns:
                    score += weight if str(row[feature]) == str(user_input[feature]) else 0
                else:
                    max_val = data[feature].max()
                    min_val = data[feature].min()
                    range_val = max_val - min_val if max_val != min_val else 1
                    normalized_diff = abs(float(row[feature]) - float(user_input[feature])) / range_val
                    score += (1 - normalized_diff) * weight
        return score
    
    recommendations['similarity_score'] = recommendations.apply(calculate_similarity, axis=1)
    recommendations = recommendations.sort_values('similarity_score', ascending=False)
    recommendations['Persentase_Kemiripan'] = (recommendations['similarity_score'] / sum(feature_weights.values()) * 100).round(2)
    recommendations = recommendations.drop('similarity_score', axis=1)
    
    return recommendations, pca_df, clusters
    
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
    budget = st.number_input("Budget (dalam juta):", min_value=0, step=1)
    
    # tambahan criteria
    use_kriteria = st.checkbox("Gunakan kriteria tambahan")
    
    lokasi_sekitar = ""
    lokasi_rumah = ""
    fasilitas_sekitar = ""
    
    if use_kriteria:
        with st.sidebar:
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
            kamar_tidur = st.slider("Jumlah kamar tidur:", 0, 6, (0, 6))
            kamar_mandi = st.slider("Jumlah kamar mandi:", 0, 6, (0, 6))
            jumlah_lantai = st.slider("Jumlah lantai:", 1, 3, (1, 3))
            luas_lahan = st.slider("Luas lahan (m2):", 0, 500, (0, 500))
            luas_bangunan = st.slider("Luas bangunan (m2):", 0, 500, (0, 500))
    
    if st.button("Cek Rekomendasi"):
        # Create filter conditions
        conditions = (
            (processed_data["harga"].astype(float) <= budget * 1_000_000)
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
            st.success(f"Rekomendasi rumah berdasarkan kriteria Anda.")
            st.write("Berikut adalah rekomendasi rumah dengan kriteria yang mirip (dari cluster yang sama):")
            user_input = {
                "harga": budget * 1_000_000,
                "jumlah_kamar_tidur": kamar_tidur[0] if use_kriteria else 0,
                "jumlah_kamar_mandi": kamar_mandi[0] if use_kriteria else 0,
                "jumlah_lantai": jumlah_lantai[0] if use_kriteria else 1,
                "luas_lahan": luas_lahan[0] if use_kriteria else 0,
                "luas_bangunan": luas_bangunan[0] if use_kriteria else 0,
                "lokasi_sekitar": lokasi_sekitar if lokasi_sekitar else processed_data["lokasi_sekitar"].mode()[0],
                "lokasi_rumah": lokasi_rumah if lokasi_rumah else processed_data["lokasi_rumah"].mode()[0],
                "fasilitas_sekitar": fasilitas_sekitar if fasilitas_sekitar else processed_data["fasilitas_sekitar"].mode()[0]
            }
            
            recommendations, pca_df, clusters = get_recommendations(processed_data, user_input, use_kriteria)
            
            if not recommendations.empty:
                st.subheader("Top 10 Rekomendasi")
                display_cols = ['Persentase_Kemiripan', 'harga', 'lokasi_rumah', 'lokasi_sekitar', 'jumlah_kamar_tidur', 'jumlah_kamar_mandi', 
                                'jumlah_lantai', 'luas_lahan', 'luas_bangunan', 'fasilitas_sekitar']
                st.dataframe(recommendations[display_cols].head(10))
                
                # Visualisasi rekomendasi pada dimensi PCA
                pca_recommendations = pca_df[pca_df.index.isin(recommendations.head(10).index)]
                fig = px.scatter(
                    pca_df,
                    x='Principal Component 1',
                    y='Principal Component 2',
                    color='Cluster',
                    title='Posisi Cluster Rekomendasi Rumah'
                )
                
                # Highlight recommended houses
                fig.add_scatter(
                    x=pca_recommendations['Principal Component 1'],
                    y=pca_recommendations['Principal Component 2'],
                    mode='markers',
                    marker=dict(size=15, symbol='star', color='yellow'),
                    name='Rekomendasi'
                )
                
                st.plotly_chart(fig)
            else:
                st.error("Tidak dapat menemukan rekomendasi yang sesuai.")
            
        else:
            st.warning("Tidak ditemukan rumah yang sesuai dengan kriteria Anda.")
            st.write("Berikut adalah rekomendasi rumah dengan kriteria yang mirip (dari cluster yang sama):")
            
            user_input = {
                "harga": budget * 1_000_000,
                "jumlah_kamar_tidur": kamar_tidur[0] if use_kriteria else 0,
                "jumlah_kamar_mandi": kamar_mandi[0] if use_kriteria else 0,
                "jumlah_lantai": jumlah_lantai[0] if use_kriteria else 1,
                "luas_lahan": luas_lahan[0] if use_kriteria else 0,
                "luas_bangunan": luas_bangunan[0] if use_kriteria else 0,
                "lokasi_sekitar": lokasi_sekitar if lokasi_sekitar else processed_data["lokasi_sekitar"].mode()[0],
                "lokasi_rumah": lokasi_rumah if lokasi_rumah else processed_data["lokasi_rumah"].mode()[0],
                "fasilitas_sekitar": fasilitas_sekitar if fasilitas_sekitar else processed_data["fasilitas_sekitar"].mode()[0]
            }
            
            recommendations, pca_df, clusters = get_recommendations(processed_data, user_input, use_kriteria)
            
            if not recommendations.empty:
                st.subheader("Top 10 Rekomendasi")
                display_cols = ['Persentase_Kemiripan', 'harga', 'lokasi_rumah', 'lokasi_sekitar', 'jumlah_kamar_tidur', 'jumlah_kamar_mandi', 
                                'jumlah_lantai', 'luas_lahan', 'luas_bangunan', 'fasilitas_sekitar']
                st.dataframe(recommendations[display_cols].head(10))
                
                # Visualisasi rekomendasi pada dimensi PCA
                pca_recommendations = pca_df[pca_df.index.isin(recommendations.head(10).index)]
                fig = px.scatter(
                    pca_df,
                    x='Principal Component 1',
                    y='Principal Component 2',
                    color='Cluster',
                    title='Posisi Cluster Rekomendasi Rumah'
                )
                
                # Highlight recommended houses
                fig.add_scatter(
                    x=pca_recommendations['Principal Component 1'],
                    y=pca_recommendations['Principal Component 2'],
                    mode='markers',
                    marker=dict(size=15, symbol='star', color='yellow'),
                    name='Rekomendasi'
                )
                
                st.plotly_chart(fig)
            else:
                st.error("Tidak dapat menemukan rekomendasi yang sesuai.")

if __name__ == "__main__":
    main()
