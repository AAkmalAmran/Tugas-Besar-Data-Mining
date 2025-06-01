
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import plotly.express as px
from PIL import Image
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler # Diganti dari sklearn.discriminant_analysis

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    /* Remove default top margin and padding */
    .block-container {
        padding-top: 0rem;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #333333;
    }
    .subheader {
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #555555;
    }
    .scrollable-summary {
        height: 420px;
        overflow-y: auto;
        border: 1px solid #ccc;
        padding: 10px;
        background-color: #f9f9f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Judul Dashboard
st.markdown("---------------------------------------------")

# Menggunakan Tabs untuk memisahkan dua jenis model
tab1, tab2 = st.tabs(["🔍 Unsupervised Learning", "🎯 Supervised Learning"])

# --- Tab Unsupervised Learning ---
with tab1:

    data_encoded_columns = joblib.load('data_encoded_columns.pkl')

    with open('kmeans_model.pkl', 'rb') as model_file:
        kmeans_model_unsupervised = pickle.load(model_file) 

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler_unsupervised = pickle.load(scaler_file) 

    # Load PCA data from Excel
    pca_2d_df = pd.read_excel('pca_2d.xlsx')

    # Load cluster_analysis data from Excel
    cluster_analysis = pd.read_excel('cluster_analysis.xlsx')

    # Main page layout
    st.markdown('<h1 class="title">Pengelompokan Game Berdasarkan Segmentasi Pasar menggunakan Metode K-Means Clustering</h1>', unsafe_allow_html=True)

    st.subheader("Cluster Visualization") 
    fig = px.scatter(
        pca_2d_df, 
        x='PCA1', 
        y='PCA2', 
        color='Cluster', 
        title="Clusters Visualized with PCA", 
        labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2'},
        template='plotly'
    )
    st.plotly_chart(fig)

    data_summary = {
        "Aspek": [
            "Target Usia Dominan",
            "Harga Dominan",
            "Genre Populer",
            "Multiplayer",
            "Game Mode"
        ],
        "Cluster 0": [
            "Kids dan Teens",
            "2 (terjangkau) dan 3 (tinggi)",
            "Puzzle, Simulation, Sports, Adventure",
            "Ya (Yes)",
            "Online"
        ],
        "Cluster 1": [
            "Teens dan Adults",
            "3 (tinggi)",
            "Shooter, Strategy, RPG, Adventure",
            "Ya (Yes)",
            "Offline"
        ],
        "Cluster 2": [
            "All Ages dan Kids", 
            "3 (tinggi) dan 2 (menengah)",
            "Puzzle, RPG, Adventure, Shooter, Party",
            "Tidak (No)",
            "Online"
        ],
        "Cluster 3": [
            "All Ages",
            "3 (tinggi) dan 2 (menengah)",
            "Adventure, Party, RPG, Simulation, Action",
            "Tidak (No)",
            "Offline"
        ]
    }

    # Membuat DataFrame dari data
    df_summary = pd.DataFrame(data_summary)

    st.subheader("Ringkasan Segmentasi Pasar per Cluster") 
    st.table(df_summary)

    st.markdown("---")
    st.markdown("Kluster 0:")
    st.markdown("Game online multiplayer (sosial/kompetitif) untuk anak & remaja dengan genre dominan Puzzle, Simulasi, Sports, Adventure. Harga bervariasi (terjangkau & tinggi), dipasarkan sebagai platform online multiplayer yang murah.")
    st.markdown("Kluster 1:")
    st.markdown("Game premium offline dengan gameplay kompleks untuk remaja & dewasa, didominasi genre Shooter, Strategy, RPG, Adventure. Harga tinggi dan mendukung multiplayer (kemungkinan besar lokal/offline).")
    st.markdown("Kluster 2:")
    st.markdown("Game online single-player kasual naratif untuk semua usia & anak. Genre populer meliputi Puzzle, RPG, Adventure, Shooter, Party. Harga cenderung menengah hingga tinggi dan tidak ada multiplayer.")
    st.markdown("Kluster 3:")
    st.markdown("Game offline santai yang cocok untuk keluarga dan semua usia. Genre utama adalah Adventure, Party, RPG, Simulation, Action. Harga berada di kisaran menengah hingga tinggi dan tidak ada fitur multiplayer.")
    st.markdown("---")


    st.subheader("Cluster Analysis Table")
    st.dataframe(cluster_analysis.head(), height=212)

    col_1, col_2 = st.columns(2)

    with col_1:
        st.markdown('<h2 class="subheader">Input Data untuk K-Means Clustering</h2>', unsafe_allow_html=True)
        numeric_features_unsupervised = ['Price'] 
        categorical_options_unsupervised = { 
            'Age Group Targeted': ['All Ages', 'Adults', 'Teens', 'Kids'],
            'Genre': ['Action', 'Adventure', 'Fighting', 'Party', 'Puzzle', 'RPG', 'Shooter', 'Simulation', 'Sports', 'Strategy'],
            'Multiplayer': ['Yes', 'No'],
            'Game Mode': ['Online', 'Offline']
        }

        user_inputs_unsupervised = {} 

        # Numeric input
        for feature in numeric_features_unsupervised:
            user_inputs_unsupervised[feature] = st.number_input(f"{feature} (USD)", value=0.0, key=f"unsupervised_{feature}")
        # One-hot encoding for categorical input
        for feature, options in categorical_options_unsupervised.items():
            selected_value = st.selectbox(f"{feature} (Unsupervised)", options, key=f"unsupervised_{feature}_select")
            for option in options:
                col_name = f"{feature}_{option}"
                user_inputs_unsupervised[col_name] = 1 if selected_value == option else 0

        # Buat DataFrame dan reindex sesuai urutan kolom saat training
        input_df_unsupervised = pd.DataFrame([user_inputs_unsupervised])
        input_df_unsupervised = input_df_unsupervised.reindex(columns=data_encoded_columns, fill_value=0)

        if st.button("Cluster Me", key="cluster_me_button"):
            # Pastikan scaler dan kmeans model yang benar digunakan
            cluster_id = int(kmeans_model_unsupervised.predict(scaler_unsupervised.transform(input_df_unsupervised))[0])
            st.session_state['cluster_id'] = cluster_id
            st.success(f"You belong to Cluster {cluster_id}.")

    with col_2:
        cluster_id = st.session_state.get('cluster_id', None)
        if cluster_id is not None:
            if cluster_id == 0:
                st.markdown('<h2 class="subheader">Interpretasi Segmentasi Pasar</h2>', unsafe_allow_html=True)
                st.write("")
                st.markdown("## Segmentasi Pasar untuk Cluster 0:")
                st.write("") 
                st.markdown("### Target utama: Anak-anak dan remaja")
                st.write("")  
                st.markdown("### Karakteristik pengguna:")
                st.markdown("""
                * Menginginkan game online multiplayer
                * Menyukai game dengan harga terjangkau
                * Tertarik pada genre ringan, kasual, dan mudah dimainkan
                """)
                st.write("") 
                st.markdown("### Strategi pemasaran yang cocok:")
                st.markdown("""
                * Fokus pada media sosial atau platform yang sering digunakan anak-anak dan remaja (TikTok, Instagram, YouTube)
                * Tawarkan sistem komunitas atau fitur sosial dalam game
                * Promosikan keunggulan multiplayer online dan harga ekonomis
                """)
            elif cluster_id == 1:
                st.markdown('<h2 class="subheader">Interpretasi Segmentasi Pasar</h2>', unsafe_allow_html=True)
                st.write("")
                st.markdown("## Segmentasi Pasar untuk Cluster 1:")
                st.write("") 
                st.markdown("### Target utama: Dewasa dan remaja")
                st.write("") 
                st.markdown("### Karakteristik pengguna:")
                st.markdown("""
                * Lebih menyukai game **offline** dengan dukungan multiplayer lokal
                * Bersedia membayar lebih untuk game dengan **harga tinggi (3)**
                * Tertarik pada genre yang **kompleks dan strategis** seperti Shooter, RPG, Strategy, dan Adventure
                """)
                st.write("") 
                st.markdown("### Strategi pemasaran yang cocok:")
                st.markdown("""
                * Fokus pada **komunitas gamer serius** seperti forum game, YouTube review RPG/strategi, atau komunitas offline
                * Soroti **kualitas gameplay mendalam, cerita menarik, dan nilai replayability tinggi**
                * Gunakan model **premium (berbayar penuh)** daripada sistem mikrotransaksi atau freemium
                """)
            elif cluster_id == 2:
                st.markdown('<h2 class="subheader">Interpretasi Segmentasi Pasar</h2>', unsafe_allow_html=True)
                st.write("")
                st.markdown("## Segmentasi Pasar untuk Cluster 2:")
                st.write("") 
                st.markdown("### Target utama: Campuran antara anak-anak, remaja, dan dewasa, dengan distribusi cukup merata")
                st.write("")  
                st.markdown("### Karakteristik pengguna:")
                st.markdown("""
                * Menyukai game **online**, namun **tidak mendukung multiplayer** (single-player online)
                * Didominasi oleh game dengan **harga tinggi (3)**, meskipun terdapat beberapa harga menengah (2)
                * Genre yang muncul cukup beragam, seperti **Puzzle, RPG, Adventure, Shooter, Fighting, dan Party**, namun semuanya dimainkan secara individu
                """)
                st.write("")  
                st.markdown("### Strategi pemasaran yang cocok:")
                st.markdown("""
                * Fokus pada promosi ke pengguna yang **menyukai pengalaman bermain sendiri namun tetap online**, seperti kompetisi skor, pencapaian, dan konten yang bisa diunduh
                * Soroti elemen gameplay yang **immersif, naratif, atau menantang**, karena pengguna ini mencari kepuasan pribadi dalam bermain
                * Gunakan platform online seperti **Steam, PlayStation Store, atau mobile app store**, dengan penekanan pada **konten tambahan dan replayability**
                """)
            elif cluster_id == 3:
                st.markdown('<h2 class="subheader">Interpretasi Segmentasi Pasar</h2>', unsafe_allow_html=True)
                st.write("")
                st.markdown("## Segmentasi Pasar untuk Cluster 3:")
                st.write("")  
                st.markdown("### Target utama: All Ages mendominasi, diikuti oleh Kids, Adults, dan Teens")
                st.write("")  
                st.markdown("### Karakteristik pengguna:")
                st.markdown("""
                * Lebih menyukai game **offline single-player** (tanpa multiplayer dan tanpa koneksi internet)
                * Banyak game berada pada **harga tinggi (3)**, menunjukkan kecenderungan pada produk dengan nilai lebih
                * Genre yang mendominasi adalah **Adventure**, diikuti oleh **Party, RPG, Simulation, dan Action**, yang menunjukkan preferensi pada eksplorasi dan hiburan kasual
                """)
                st.write("") 
                st.markdown("### Strategi pemasaran yang cocok:")
                st.markdown("""
                * Cocok untuk **keluarga dan pengguna kasual** yang ingin bermain tanpa tergantung koneksi internet
                * Soroti fitur seperti **cerita yang menarik, gameplay santai, dan aksesibilitas lintas usia**
                * Dapat dipasarkan melalui **bundle keluarga** atau **edukatif ringan**, serta promosi di **platform offline-friendly** seperti konsol atau PC non-online
                """)


# --- Tab Supervised Learning ---
with tab2:
    st.markdown('<h1 class="title">Prediksi Rating Game (Naive Bayes)</h1>', unsafe_allow_html=True)

    try:
        nb_model = joblib.load('naive_bayes_model.pkl')
    except FileNotFoundError:
        st.error("Model Naive Bayes (naive_bayes_model.pkl) tidak ditemukan. Pastikan file model ada di direktori yang sama.")
        nb_model = None
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat model: {e}")
        nb_model = None

    if nb_model is not None:
        st.sidebar.header("Input Fitur Prediksi Rating")

        # Definisikan opsi dan mapping untuk input supervised learning
        age_options_nb = {'All Ages': 0, 'Adults': 1, 'Teens': 2, 'Kids': 3}
        platform_options_nb = {'Mobile': 0, 'PC': 1, 'PlayStation': 2, 'Xbox': 3, 'Nintendo Switch': 4}
        genre_options_nb = {'Action': 0, 'Adventure': 1, 'Fighting': 2, 'Party': 3, 'Puzzle': 4, 'RPG': 5, 'Shooter': 6, 'Simulation': 7, 'Sports': 8, 'Strategy': 9}
        graphic_options_nb = {'Low': 0, 'Medium': 1, 'High': 2, 'Ultra': 3}
        sound_story_options_nb = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
        yes_no_options_nb = {'No': 0, 'Yes': 1}
        game_mode_options_nb = {'Offline': 0, 'Online': 1}

        # Urutan fitur sesuai dengan saat training model Naive Bayes
        # X = df_downsampled.drop(columns=['User Rating', 'user_rating_class'])
        # Urutan kolom X: ['Age Group Targeted', 'Price', 'Platform', 'Requires Special Device', 
        # 'Genre', 'Multiplayer', 'Game Length (Hours)', 'Graphics Quality', 
        # 'Soundtrack Quality', 'Story Quality', 'Game Mode', 'Min Number of Players']

        input_nb = {}
        input_nb['Age Group Targeted'] = st.sidebar.selectbox("Target Usia (Prediksi)", list(age_options_nb.keys()), key="nb_age")
        input_nb['Price'] = st.sidebar.number_input("Harga (Prediksi)", min_value=0.0, value=30.0, step=0.01, key="nb_price")
        input_nb['Platform'] = st.sidebar.selectbox("Platform (Prediksi)", list(platform_options_nb.keys()), key="nb_platform")
        input_nb['Requires Special Device'] = st.sidebar.selectbox("Membutuhkan Perangkat Khusus? (Prediksi)", list(yes_no_options_nb.keys()), key="nb_special_device")
        input_nb['Genre'] = st.sidebar.selectbox("Genre (Prediksi)", list(genre_options_nb.keys()), key="nb_genre")
        input_nb['Multiplayer'] = st.sidebar.selectbox("Multiplayer? (Prediksi)", list(yes_no_options_nb.keys()), key="nb_multiplayer")
        input_nb['Game Length (Hours)'] = st.sidebar.number_input("Durasi Game (Jam) (Prediksi)", min_value=0.0, value=20.0, step=0.1, key="nb_length")
        input_nb['Graphics Quality'] = st.sidebar.selectbox("Kualitas Grafis (Prediksi)", list(graphic_options_nb.keys()), key="nb_graphics")
        input_nb['Soundtrack Quality'] = st.sidebar.selectbox("Kualitas Soundtrack (Prediksi)", list(sound_story_options_nb.keys()), key="nb_soundtrack")
        input_nb['Story Quality'] = st.sidebar.selectbox("Kualitas Cerita (Prediksi)", list(sound_story_options_nb.keys()), key="nb_story")
        input_nb['Game Mode'] = st.sidebar.selectbox("Mode Game (Prediksi)", list(game_mode_options_nb.keys()), key="nb_game_mode")
        input_nb['Min Number of Players'] = st.sidebar.number_input("Minimal Pemain (Prediksi)", min_value=1, value=1, step=1, key="nb_min_players")

        if st.sidebar.button("Prediksi Rating Game", key="predict_rating_button"):
            # Konversi input ke format numerik yang sesuai dengan model
            feature_vector_nb = np.array([
                age_options_nb[input_nb['Age Group Targeted']],
                input_nb['Price'],
                platform_options_nb[input_nb['Platform']],
                yes_no_options_nb[input_nb['Requires Special Device']],
                genre_options_nb[input_nb['Genre']],
                yes_no_options_nb[input_nb['Multiplayer']],
                input_nb['Game Length (Hours)'],
                graphic_options_nb[input_nb['Graphics Quality']],
                sound_story_options_nb[input_nb['Soundtrack Quality']],
                sound_story_options_nb[input_nb['Story Quality']],
                game_mode_options_nb[input_nb['Game Mode']],
                input_nb['Min Number of Players']
            ]).reshape(1, -1)

            try:
                prediction_nb = nb_model.predict(feature_vector_nb)
                prediction_proba_nb = nb_model.predict_proba(feature_vector_nb)

                # Mapping hasil prediksi kembali ke label string
                label_mapping_reverse = {0: 'Poor', 1: 'Average', 2: 'Good'}
                predicted_class_label = label_mapping_reverse[int(prediction_nb[0][0])]

                st.subheader("Hasil Prediksi Rating:")
                st.success(f"Prediksi Kelas Rating: **{predicted_class_label}**")

                st.subheader("Probabilitas Prediksi per Kelas:")
                proba_df = pd.DataFrame({
                    "Kelas": [label_mapping_reverse[i] for i in nb_model.classes_],
                    "Probabilitas": prediction_proba_nb[0]
                })
                st.dataframe(proba_df)

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

        st.markdown("---")
        st.subheader("Evaluasi Model Naive Bayes (dari notebook)")
        # Anda bisa menambahkan gambar confusion matrix atau metrik lain di sini jika diinginkan
        # Contoh: Menampilkan metrik yang sudah ada
        st.markdown("""
        **Akurasi Model (dari data test di notebook):** 0.8759

        **Classification Report (dari data test di notebook):**
        ```
                      precision    recall  f1-score   support

                   0       0.93      0.88      0.90      2500
                   1       0.78      0.86      0.82      2430
                   2       0.93      0.89      0.91      2437

            accuracy                           0.88      7367
           macro avg       0.88      0.88      0.88      7367
        weighted avg       0.88      0.88      0.88      7367
        ```

        **Rata-rata ROC AUC (OVR Weighted) dari Cross-Validation:** 0.9819
        """)

        # Menampilkan gambar ROC Curve yang sudah dibuat
        try:
            roc_image = Image.open('multiclass_roc_curve.png') # Ganti dengan nama file gambar ROC Anda
            st.image(roc_image, caption='Multiclass ROC Curve (dari notebook)')
        except FileNotFoundError:
            st.warning("Gambar Multiclass ROC Curve (multiclass_roc_curve.png) tidak ditemukan.")
