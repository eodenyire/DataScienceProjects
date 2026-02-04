"""
Streamlit Web Application for Plant Disease Prediction

This application provides an interactive web interface for predicting plant diseases
from leaf images using deep learning.

Author: Emmanuel Odenyire
Email: eodenyire@gmail.com
"""

import streamlit as st
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from plant_disease_predictor import PlantDiseasePredictor
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Plant Disease Prediction",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .healthy-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .disease-box {
        background-color: #FFEBEE;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #F44336;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_predictor():
    """Load or create the plant disease predictor model."""
    predictor = PlantDiseasePredictor()
    
    # Check if pre-trained model exists
    model_path = "plant_disease_model.keras"
    if not os.path.exists(model_path):
        model_path = "plant_disease_model.h5"  # Fallback to old format
    
    if os.path.exists(model_path):
        try:
            predictor.load_model(model_path)
            st.success("✅ Pre-trained model loaded successfully!")
        except Exception as e:
            st.warning(f"Could not load pre-trained model: {e}")
            st.info("Building a new model for demonstration...")
            predictor.build_model()
    else:
        st.info("Building model for demonstration (not trained on actual data)...")
        predictor.build_model()
        st.warning("⚠️ Note: This is a demo model. For production use, train the model with actual plant disease dataset.")
    
    return predictor


def display_image_with_preview(image_file):
    """Display uploaded image with a preview."""
    image = Image.open(image_file)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    return image


def format_disease_name(disease_name):
    """Format disease name for better display."""
    # Replace underscores with spaces
    formatted = disease_name.replace('___', ' - ').replace('_', ' ')
    return formatted


def main():
    """Main application function."""
    
    # Header
    st.markdown('<p class="main-header">🌿 Plant Disease Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Disease Detection for Healthier Crops</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("About the App")
        st.info("""
        This application uses deep learning and computer vision to identify plant diseases from leaf images.
        
        **Features:**
        - 🎯 High accuracy predictions
        - 🌱 38 disease classes supported
        - 📊 Confidence scores
        - 💊 Treatment recommendations
        
        **Developer:** Emmanuel Odenyire
        """)
        
        st.header("How to Use")
        st.markdown("""
        1. Upload an image of a plant leaf
        2. Click 'Predict Disease'
        3. View results and recommendations
        """)
        
        st.header("Supported Plants")
        st.markdown("""
        - 🍎 Apple
        - 🌽 Corn (Maize)
        - 🍇 Grape
        - 🍑 Peach
        - 🌶️ Pepper
        - 🥔 Potato
        - 🍓 Strawberry
        - 🍅 Tomato
        - And more...
        """)
    
    # Load model
    try:
        predictor = load_predictor()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()
    
    # Main content
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf for disease detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("📸 Uploaded Image")
        image = display_image_with_preview(uploaded_file)
        
        # Save temporarily for prediction
        temp_path = "temp_upload.jpg"
        image.save(temp_path)
        
        # Prediction button
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_button = st.button("🔍 Predict Disease", key="predict")
        
        if predict_button:
            with st.spinner("🔬 Analyzing image..."):
                try:
                    # Make prediction
                    results = predictor.predict(temp_path, top_k=3)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Prediction Results")
                    
                    # Top prediction
                    top_pred = results['top_prediction']
                    disease_name = format_disease_name(top_pred['disease'])
                    confidence = top_pred['confidence']
                    
                    # Determine if healthy or diseased
                    is_healthy = 'healthy' in top_pred['disease'].lower()
                    
                    if is_healthy:
                        st.markdown(f"""
                        <div class="healthy-box">
                            <h2>✅ Plant Status: HEALTHY</h2>
                            <h3>{disease_name}</h3>
                            <p style="font-size: 1.5rem;"><strong>Confidence: {confidence:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="disease-box">
                            <h2>⚠️ Disease Detected</h2>
                            <h3>{disease_name}</h3>
                            <p style="font-size: 1.5rem;"><strong>Confidence: {confidence:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Disease information
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📋 Description")
                        st.markdown(f"""
                        <div class="info-box">
                            {top_pred['description']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("### 💊 Recommended Treatment")
                        st.markdown(f"""
                        <div class="info-box">
                            {top_pred['treatment']}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Top 3 predictions
                    st.markdown("---")
                    st.subheader("🎯 Top 3 Predictions")
                    
                    for i, pred in enumerate(results['predictions'], 1):
                        with st.expander(f"#{i} - {format_disease_name(pred['disease'])} ({pred['confidence']:.2f}%)"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Description:**")
                                st.write(pred['description'])
                            with col2:
                                st.markdown("**Treatment:**")
                                st.write(pred['treatment'])
                    
                    # Confidence chart
                    st.markdown("---")
                    st.subheader("📈 Confidence Scores")
                    
                    # Create chart data
                    chart_data = {
                        'Disease': [format_disease_name(p['disease'])[:30] for p in results['predictions']],
                        'Confidence': [p['confidence'] for p in results['predictions']]
                    }
                    
                    st.bar_chart(data=chart_data, x='Disease', y='Confidence')
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
                    st.exception(e)
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
    
    else:
        # Instructions when no image uploaded
        st.info("👆 Please upload an image of a plant leaf to begin analysis")
        
        # Example section
        st.markdown("---")
        st.subheader("📷 Tips for Best Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **✅ Good Image Quality**
            - High resolution
            - Clear and focused
            - Good lighting
            - Single leaf visible
            """)
        
        with col2:
            st.markdown("""
            **❌ Avoid**
            - Blurry images
            - Dark/poor lighting
            - Multiple leaves overlapping
            - Heavily filtered images
            """)
        
        with col3:
            st.markdown("""
            **🎯 Best Practices**
            - Capture entire leaf
            - Plain background
            - Natural light
            - Show disease symptoms clearly
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>Developed by <strong>Emmanuel Odenyire</strong></p>
        <p>Part of the Data Science Projects Series</p>
        <p>🌿 Made with ❤️ for sustainable agriculture</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
