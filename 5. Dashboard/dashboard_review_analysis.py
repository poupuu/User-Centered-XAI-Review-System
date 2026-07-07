import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import lime
import lime.lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Review Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk tema e-commerce yang modern
st.markdown("""
<style>
    .main {
        background-color: #0f1116;
        color: #ffffff;
    }
    
    /* Header dan judul */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: #ffffff;
        border-left: 4px solid #667eea;
        padding-left: 1rem;
    }
    
    /* Card styling seperti e-commerce */

    .review-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #333;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .analysis-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #333;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        height: 100%;
    }
    
    .info-card {
        background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid #444;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    
    /* Prediksi box dengan gradient */
    .prediction-box {
        background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 100%);
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    
    .sentiment-prediction {
        border-left-color: #00D4AA;
    }
    
    .fakereal-prediction {
        border-left-color: #FFD700;
    }
    
    /* Metric boxes */
    .metric-card {
        background: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #444;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Feature tags seperti e-commerce */
    .feature-tag {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.3rem;
        font-size: 0.85rem;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .negative-tag {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
    }
    
    .positive-tag {
        background: linear-gradient(135deg, #00D4AA 0%, #00b894 100%);
    }
    
    .neutral-tag {
        background: linear-gradient(135deg, #FFD700 0%, #f9ca24 100%);
    }
    
    /* Probability bars */
    .probability-bar {
        background: #333;
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    
    .negative-bar { border-left-color: #ff6b6b; }
    .neutral-bar { border-left-color: #FFD700; }
    .positive-bar { border-left-color: #00D4AA; }
    
    /* Score categories */
    .score-badge {
        font-size: 0.75rem;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin-left: 0.5rem;
        font-weight: 600;
    }
    
    .low-badge { background: #00b894; color: white; }
    .moderate-badge { background: #f9ca24; color: black; }
    .high-badge { background: #e17055; color: white; }
    .critical-badge { background: #d63031; color: white; }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #1e1e1e;
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background: #2d2d2d !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
    }
    
    /* Number input styling */
    .stNumberInput input {
        background: #2d2d2d !important;
        color: white !important;
        border: 1px solid #444 !important;
        border-radius: 10px !important;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load dataset"""
    try:
        df = pd.read_csv("Dummy_yelp_dataset_with_text.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

@st.cache_resource
def load_sentiment_model():
    """Load sentiment model dengan semua artifacts"""
    try:
        artifacts = joblib.load('Sentiment_Analysis_XGBoost.pkl')
        return artifacts
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

@st.cache_resource
def load_fakereal_model():
    """Load fake/real model dengan semua artifacts"""
    try:
        artifacts = joblib.load('FakeReal_XGBoost.pkl')
        return artifacts
    except Exception as e:
        st.error(f"Error loading fake/real model: {e}")
        return None

def tokens_to_string(tokens):
    """Convert tokens to string"""
    if isinstance(tokens, list):
        return ' '.join(tokens)
    elif isinstance(tokens, str):
        try:
            tokens_list = eval(tokens)
            if isinstance(tokens_list, list):
                return ' '.join(tokens_list)
        except:
            return str(tokens)
    return str(tokens)

def prepare_sentiment_features(text, stars_review, artifacts):
    """Prepare features for sentiment analysis"""
    tfidf_vectorizer = artifacts['tfidf_vectorizer']
    scaler = artifacts['scaler']
    
    # Prepare text
    text_processed = tokens_to_string(text)
    text_processed = str(text_processed) if pd.notna(text_processed) else ""
    
    # TF-IDF transformation
    text_tfidf = tfidf_vectorizer.transform([text_processed])
    
    # Prepare numerical features
    numerical_values = np.array([[stars_review]])
    numerical_scaled = scaler.transform(numerical_values)
    
    # Combine features
    combined_features = hstack([text_tfidf, numerical_scaled])
    return csr_matrix(combined_features)

def prepare_fakereal_features(text, stars_review, daily_spike, hourly_spike, max_similarity, artifacts):
    """Prepare features for fake/real analysis"""
    tfidf_vectorizer = artifacts['tfidf_vectorizer']
    scaler = artifacts['scaler']
    
    # Prepare text
    text_processed = tokens_to_string(text)
    text_processed = str(text_processed) if pd.notna(text_processed) else ""
    
    # TF-IDF transformation
    text_tfidf = tfidf_vectorizer.transform([text_processed])
    
    # Prepare numerical features
    numerical_values = np.array([[stars_review, daily_spike, hourly_spike, max_similarity]])
    numerical_scaled = scaler.transform(numerical_values)
    
    # Combine features
    combined_features = hstack([text_tfidf, numerical_scaled])
    return csr_matrix(combined_features)


def get_lime_explanation_for_text(text, stars_review, artifacts, model_type='sentiment'):
    """
    Mendapatkan LIME explanation khusus untuk kata-kata yang ada dalam review
    
    Args:
        text: Review text yang sudah di-lemmatized
        stars_review: Rating stars
        artifacts: Model artifacts
        model_type: 'sentiment' atau 'fakereal'
    """
    try:
        # Persiapkan features
        if model_type == 'sentiment':
            features = prepare_sentiment_features(text, stars_review, artifacts)
            model = artifacts['model']
            training_data = artifacts.get('lime_training_data')
            feature_names = artifacts['feature_names']
            class_names = artifacts['class_names']
        else:
            # Untuk fakereal, kita butuh lebih banyak parameter
            # Dalam konteks ini, kita hanya pakai yang penting
            features = prepare_fakereal_features(
                text, stars_review, 
                0.5, 1.0, 0.3,  # Default values, akan di-override nanti
                artifacts
            )
            model = artifacts['model']
            training_data = artifacts.get('lime_training_data')
            feature_names = artifacts['feature_names']
            class_names = artifacts['class_names']
        
        # Konversi ke dense array untuk LIME
        instance_dense = features.toarray().reshape(1, -1)[0]
        
        # Buat LIME explainer dengan SEMUA training data
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=training_data,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=42,
            discretize_continuous=False
        )
        
        # Dapatkan explanation dari LIME
        exp = explainer.explain_instance(
            instance_dense,
            model.predict_proba,
            num_features=50,  # Ambil lebih banyak fitur dulu
            top_labels=len(class_names)
        )
        
        # Prediksi model untuk instance ini
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        
        # Dapatkan semua explanations untuk prediksi yang dipilih
        explanation_list = exp.as_list(label=prediction)
        
        # Filter: Hanya ambil kata-kata yang ADA dalam review text ini
        # Cara: Cek apakah kata tersebut ada dalam text yang sudah di-lemmatized
        lemmatized_text = tokens_to_string(text).lower()
        
        filtered_explanations = []
        for feature_name, weight in explanation_list:
            # Feature name dari LIME akan berupa "f1234" atau nama aktual
            # Untuk TF-IDF features, kita perlu mapping ke kata sebenarnya
            
            # Cari kata sebenarnya dari feature name
            # Jika feature_name adalah indeks (seperti "f1234"), cari di feature_names
            if feature_name.startswith('f') and feature_name[1:].isdigit():
                idx = int(feature_name[1:])
                if idx < len(feature_names):
                    actual_feature = feature_names[idx]
                else:
                    continue
            else:
                actual_feature = feature_name
            
            # Skip jika ini adalah numerical feature (bukan kata)
            numerical_features = artifacts.get('numerical_features', [])
            if actual_feature in numerical_features:
                filtered_explanations.append((actual_feature, weight, 'numerical'))
                continue
            
            # Cek apakah kata ini ada dalam review text
            # Convert to lowercase dan split untuk matching yang lebih baik
            if actual_feature.lower() in lemmatized_text:
                # Atau cek dengan n-gram
                words_in_text = lemmatized_text.split()
                # Cek untuk unigram
                if actual_feature.lower() in words_in_text:
                    filtered_explanations.append((actual_feature, weight, 'text'))
                # Cek untuk bigram/trigram (jika feature mengandung spasi)
                elif ' ' in actual_feature:
                    if actual_feature.lower() in lemmatized_text:
                        filtered_explanations.append((actual_feature, weight, 'text_ngram'))
        
        # Urutkan berdasarkan absolute weight dan ambil top 10
        filtered_explanations.sort(key=lambda x: abs(x[1]), reverse=True)
        top_explanations = filtered_explanations[:10]
        
        # Konversi ke format yang mudah ditampilkan
        result = []
        for feature, weight, feature_type in top_explanations:
            result.append({
                'feature': feature,
                'weight': weight,
                'type': feature_type,
                'direction': 'positive' if weight > 0 else 'negative'
            })
        
        return {
            'prediction': prediction,
            'probabilities': probabilities,
            'explanations': result,
            'top_features': top_explanations
        }
        
    except Exception as e:
        st.error(f"Error in LIME explanation: {e}")
        return None

def get_simplified_word_importance(text, artifacts, model_type='sentiment', top_n=10):
    """
    Versi sederhana untuk mendapatkan kata penting dari model
    Hanya menampilkan kata-kata yang ADA dalam review text
    """
    try:
        # Preprocess text
        lemmatized_text = tokens_to_string(text).lower()
        words_in_review = set(lemmatized_text.split())
        
        if model_type == 'sentiment':
            model = artifacts['model']
            tfidf_vectorizer = artifacts['tfidf_vectorizer']
            feature_names = artifacts['feature_names']
        else:
            model = artifacts['model']
            tfidf_vectorizer = artifacts['tfidf_vectorizer']
            feature_names = artifacts['feature_names']
        
        # Dapatkan feature importance dari XGBoost
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type='gain')
        
        # Mapping dari feature index ke feature name
        word_importance = []
        
        # Iterasi melalui semua feature names
        for idx, feature_name in enumerate(feature_names):
            # Skip numerical features
            if feature_name in artifacts.get('numerical_features', []):
                continue
                
            # Cek apakah feature ini ada dalam importance_dict
            fname = f"f{idx}"
            importance_score = importance_dict.get(fname, 0)
            
            # Cek apakah kata ini ada dalam review
            if feature_name.lower() in words_in_review and importance_score > 0:
                word_importance.append({
                    'word': feature_name,
                    'importance': importance_score,
                    'direction': 'positive'  # Default, bisa disesuaikan
                })
        
        # Urutkan dan ambil top N
        word_importance.sort(key=lambda x: x['importance'], reverse=True)
        return word_importance[:top_n]
        
    except Exception as e:
        st.error(f"Error in simplified word importance: {e}")
        return []

def get_daily_spike_category(score):
    """Kategori untuk Daily Spike Score"""
    if score <= 0.5:
        return "Low", "low-badge"
    elif score <= 1.5:
        return "Moderate", "moderate-badge"
    else:
        return "High", "high-badge"

def get_hourly_spike_category(score):
    """Kategori untuk Hourly Spike Score"""
    if score == 0:
        return "Low", "low-badge"
    elif score <= 2:
        return "Moderate", "moderate-badge"
    elif score <= 4:
        return "High", "high-badge"
    else:
        return "Critical", "critical-badge"

def get_similarity_category(score):
    """Kategori untuk Max Similarity Score"""
    if score < 0.25:
        return "Low", "low-badge"
    elif score <= 0.40:
        return "Medium", "moderate-badge"
    elif score <= 0.80:
        return "High", "high-badge"
    else:
        return "Critical", "critical-badge"

def main():
    st.markdown("""
    <style>
    .main-header {
        background: rgba(255, 255, 255, 0.08);          /* Transparan tanpa gradient */
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border-radius: 18px;
        padding: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.25);
        box-shadow:
            0 4px 12px rgba(0, 0, 0, 0.25),
            0 0 25px rgba(140, 80, 255, 0.28);          /* Glow halus ungu-biru */
    }

    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 0 0 12px rgba(255,255,255,0.35);
    }

    .main-header p {
        color: rgba(235, 235, 255, 0.92);
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 300;
    }
    </style>

    <div class="main-header">
        <h1>Review Analysis Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models
    with st.spinner("🔄 Loading data and models..."):
        df = load_data()
        sentiment_artifacts = load_sentiment_model()
        fakereal_artifacts = load_fakereal_model()
    
    if df is None:
        st.error("❌ Failed to load dataset. Please check if 'Dummy_yelp_dataset_with_text.csv' exists.")
        return
    
    if sentiment_artifacts is None:
        st.error("❌ Failed to load sentiment model. Please check if 'Sentiment_Analysis_XGBoost.pkl' exists.")
        return
    
    if fakereal_artifacts is None:
        st.error("❌ Failed to load fake/real model. Please check if 'FakeReal_XGBoost.pkl' exists.")
        return
    
    # Sidebar untuk selection
    st.sidebar.header("🔍 Review Selection")
    total_reviews = len(df)
    
    review_idx = st.sidebar.number_input(
        "Select Review Index", 
        min_value=0, 
        max_value=total_reviews-1, 
        value=0,
        help=f"Enter a number between 0 and {total_reviews-1}"
    )
    
    # Quick stats di sidebar
    st.sidebar.markdown("---")
    st.sidebar.metric("📈 Total Reviews", total_reviews)
    st.sidebar.metric("✅ Real Reviews", len(df[df['is_fake'] == 0]))
    st.sidebar.metric("🚫 Fake Reviews", len(df[df['is_fake'] == 1]))
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Get selected review
    selected_review = df.iloc[review_idx]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### **📖 Review Text:**")
        st.text_area("", 
                   value=selected_review['text'], 
                   height=150, 
                   key=f"review_text_{review_idx}",
                   label_visibility="collapsed")
    
    with col2:
        st.markdown("### **📋 Review Information**")
        st.markdown(f"**ID:** `{selected_review['review_id'] if 'review_id' in selected_review else 'N/A'}`")
        
        stars = int(selected_review['stars_review'])
        # Display stars dengan visual yang menarik
        star_rating = "⭐" * stars + "☆" * (5 - stars)
        st.markdown(f"**Rating:** {star_rating} ({stars}/5)")
        
        st.markdown(f"**📅 Date:** {selected_review['date'].split()[0] if 'date' in selected_review else 'N/A'}")
        st.markdown(f"**🏢 Business:** {selected_review['business_name'] if 'business_name' in selected_review else 'N/A'}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # SENTIMENT ANALYSIS SECTION
    with col1:
        st.markdown("## 💭 Sentiment Analysis")
        
        try:
            # Prepare features
            sentiment_features = prepare_sentiment_features(
                selected_review['lemmatized_ml'],
                selected_review['stars_review'],
                sentiment_artifacts
            )
            
            # Prediction
            sentiment_proba = sentiment_artifacts['model'].predict_proba(sentiment_features)[0]
            sentiment_pred = sentiment_artifacts['model'].predict(sentiment_features)[0]
            
            sentiment_labels = sentiment_artifacts['class_names']
            predicted_label = sentiment_labels[sentiment_pred]
            
            # Prediction box dengan styling yang konsisten
            sentiment_color = {
                'negative': '#ff6b6b',
                'neutral': '#FFD700', 
                'positive': '#00D4AA'
            }
            
            sentiment_class = {
                'negative': 'negative-tag',
                'neutral': 'neutral-tag', 
                'positive': 'positive-tag'
            }
            
            st.markdown(f"""
            <div class="prediction-box" 
                style="border-left: 5px solid {sentiment_color[predicted_label]};">
                <h5>
                    🎯 Prediction:
                    <span style="color:{sentiment_color[predicted_label]};">
                        {predicted_label.upper()}
                    </span>
                </h5>
            </div>
            """, unsafe_allow_html=True)


            st.markdown('</div>', unsafe_allow_html=True)
            
            # Probabilities
            st.markdown("**📊 Confidence Scores:**")
            for i, label in enumerate(sentiment_labels):
                prob = sentiment_proba[i]
                bar_color = sentiment_color[label]
                st.markdown(f'''
                <div style="background: #2d2d2d; border-radius: 10px; padding: 0.8rem; margin: 0.5rem 0;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>{label.capitalize()}</span>
                        <span style="font-weight: bold;">{prob:.3f}</span>
                    </div>
                    <div style="background: #444; border-radius: 5px; height: 8px; margin-top: 5px;">
                        <div style="background: {bar_color}; width: {prob*100}%; height: 100%; border-radius: 5px;"></div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            # LIME Explanation
            st.markdown("**🔍 Key Words in This Review:**")
            
            # Dapatkan LIME explanation untuk kata-kata dalam review ini
            lime_result = get_lime_explanation_for_text(
                selected_review['lemmatized_ml'],
                selected_review['stars_review'],
                sentiment_artifacts,
                'sentiment'
            )
            
            if lime_result and lime_result['explanations']:
                # Filter hanya kata-kata (bukan numerical features)
                text_explanations = [exp for exp in lime_result['explanations'] 
                                   if exp['type'] in ['text', 'text_ngram']]
                
                if text_explanations:
                    for exp in text_explanations[:7]:  # Tampilkan 7 teratas
                        tag_class = "positive-tag" if exp['direction'] == 'positive' else "negative-tag"
                        influence = "SUPPORTS" if exp['direction'] == 'positive' else "OPPOSES"
                        
                        st.markdown(f'''
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span class="feature-tag {tag_class}" style="margin-right: 10px;">
                                • {exp['feature']}
                            </span>
                            <span style="font-size: 0.8rem; opacity: 0.8;">
                                ({influence} {predicted_label})
                            </span>
                            <span style="margin-left: auto; font-weight: bold;">
                                {exp['weight']:.4f}
                            </span>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    # Fallback ke simplified version
                    word_importance = get_simplified_word_importance(
                        selected_review['lemmatized_ml'],
                        sentiment_artifacts,
                        'sentiment',
                        7
                    )
                    
                    if word_importance:
                        for word_info in word_importance:
                            st.markdown(f'''
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <span class="feature-tag positive-tag" style="margin-right: 10px;">
                                    • {word_info['word']}
                                </span>
                                <span style="margin-left: auto; font-weight: bold;">
                                    {word_info['importance']:.4f}
                                </span>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ No significant keywords found in this review.")
            else:
                st.info("ℹ️ No LIME explanation available for this review.")
                
        except Exception as e:
            st.error(f"❌ Error in sentiment analysis: {e}")
    
    # FAKE/REAL ANALYSIS SECTION
    with col2:
        st.markdown("## 🔍 Authenticity Analysis")
        
        try:
            # Prepare features
            fakereal_features = prepare_fakereal_features(
                selected_review['lemmatized_ml'],
                selected_review['stars_review'],
                selected_review['daily_spike_score'],
                selected_review['hourly_spike_score'],
                selected_review['max_similarity_score'],
                fakereal_artifacts
            )
            
            # Prediction
            fakereal_proba = fakereal_artifacts['model'].predict_proba(fakereal_features)[0]
            fakereal_pred = fakereal_artifacts['model'].predict(fakereal_features)[0]
            
            predicted_authenticity = "FAKE" if fakereal_pred == 1 else "REAL"
            authenticity_color = "#ff6b6b" if fakereal_pred == 1 else "#00D4AA"
            authenticity_icon = "❌" if fakereal_pred == 1 else "✅"
            
            st.markdown(f"""
            <div class="prediction-box" 
                style="border-left: 5px solid {authenticity_color};">
                <h5>
                    {authenticity_icon} Prediction: 
                    <span style="color:{authenticity_color};">
                        {predicted_authenticity}
                    </span>
                </h5>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            
            # Probabilities Section
            st.markdown("**📊 Confidence Scores:**")
            
            # Real probability
            st.markdown(f'''
            <div class="probability-bar">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Real</span>
                    <span style="font-weight: bold;">{fakereal_proba[0]:.3f}</span>
                </div>
                <div style="background: #444; border-radius: 5px; height: 8px; margin-top: 5px;">
                    <div style="background: #00D4AA; width: {fakereal_proba[0]*100}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Fake probability
            st.markdown(f'''
            <div class="probability-bar">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>Fake</span>
                    <span style="font-weight: bold;">{fakereal_proba[1]:.3f}</span>
                </div>
                <div style="background: #444; border-radius: 5px; height: 8px; margin-top: 5px;">
                    <div style="background: #ff6b6b; width: {fakereal_proba[1]*100}%; height: 100%; border-radius: 5px;"></div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Behavioral Metrics Section
            st.markdown("**📈 Behavioral Metrics:**")
            
            # Daily Spike Score dengan kategori
            daily_score = selected_review["daily_spike_score"]
            daily_cat, daily_class = get_daily_spike_category(daily_score)
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>📅 Daily Spike</span>
                    <span style="font-weight: bold;">{daily_score:.3f}</span>
                </div>
                <span class="score-badge {daily_class}">{daily_cat}</span>
            </div>
            ''', unsafe_allow_html=True)
            
            # Hourly Spike Score dengan kategori
            hourly_score = selected_review["hourly_spike_score"]
            hourly_cat, hourly_class = get_hourly_spike_category(hourly_score)
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>⏰ Hourly Spike</span>
                    <span style="font-weight: bold;">{hourly_score:.3f}</span>
                </div>
                <span class="score-badge {hourly_class}">{hourly_cat}</span>
            </div>
            ''', unsafe_allow_html=True)
            
            # Similarity Score dengan kategori
            similarity_score = selected_review["max_similarity_score"]
            similarity_cat, similarity_class = get_similarity_category(similarity_score)
            st.markdown(f'''
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>🔍 Similarity</span>
                    <span style="font-weight: bold;">{similarity_score:.3f}</span>
                </div>
                <span class="score-badge {similarity_class}">{similarity_cat}</span>
            </div>
            ''', unsafe_allow_html=True)
            
            # LIME Explanation Section
            st.markdown("**🔍 Key Influencing Words:**")
            
            # Dapatkan LIME explanation
            lime_result = get_lime_explanation_for_text(
                selected_review['lemmatized_ml'],
                selected_review['stars_review'],
                fakereal_artifacts,
                'fakereal'
            )
            
            if lime_result and lime_result['explanations']:
                # Filter hanya kata-kata
                text_explanations = [exp for exp in lime_result['explanations'] 
                                   if exp['type'] in ['text', 'text_ngram']]
                
                if text_explanations:
                    for exp in text_explanations[:7]:
                        if fakereal_pred == 1:  # Fake
                            tag_class = "negative-tag" if exp['direction'] == 'positive' else "positive-tag"
                            influence = "SUPPORTS Fake" if exp['direction'] == 'positive' else "SUPPORTS Real"
                        else:  # Real
                            tag_class = "positive-tag" if exp['direction'] == 'positive' else "negative-tag"
                            influence = "SUPPORTS Real" if exp['direction'] == 'positive' else "SUPPORTS Fake"
                        
                        st.markdown(f'''
                        <div style="display: flex; align-items: center; margin-bottom: 8px;">
                            <span class="feature-tag {tag_class}" style="margin-right: 10px;">
                                • {exp['feature']}
                            </span>
                            <span style="font-size: 0.8rem; opacity: 0.8;">
                                ({influence})
                            </span>
                            <span style="margin-left: auto; font-weight: bold;">
                                {exp['weight']:.4f}
                            </span>
                        </div>
                        ''', unsafe_allow_html=True)
                else:
                    # Fallback
                    word_importance = get_simplified_word_importance(
                        selected_review['lemmatized_ml'],
                        fakereal_artifacts,
                        'fakereal',
                        7
                    )
                    
                    if word_importance:
                        for word_info in word_importance:
                            st.markdown(f'''
                            <div style="display: flex; align-items: center; margin-bottom: 8px;">
                                <span class="feature-tag positive-tag" style="margin-right: 10px;">
                                    • {word_info['word']}
                                </span>
                                <span style="margin-left: auto; font-weight: bold;">
                                    {word_info['importance']:.4f}
                                </span>
                            </div>
                            ''', unsafe_allow_html=True)
                    else:
                        st.info("ℹ️ No significant keywords found in this review.")
            else:
                st.info("ℹ️ No LIME explanation available for this review.")
                
        except Exception as e:
            st.error(f"❌ Error in authenticity analysis: {e}")

if __name__ == "__main__":
    main()