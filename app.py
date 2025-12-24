import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Vehicle Classification Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling modern
st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Card styling */
    .stCard {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 15px 30px;
        border-radius: 10px;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Title */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        text-align: center;
        padding: 20px 0;
    }
    
    /* Confidence badge */
    .confidence-high {
        background-color: #48bb78;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .confidence-medium {
        background-color: #ed8936;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    .confidence-low {
        background-color: #f56565;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- EMOJI UNTUK SETIAP KELAS ---
CLASS_EMOJI = {
    "Big Truck": "🚛",
    "City Car": "🚗",
    "Multi Purpose Vehicle": "🚐",
    "Sedan": "🚙",
    "Sport Utility Vehicle": "🚙",
    "Truck": "🚚",
    "Van": "🚐"
}

# --- LIST KELAS (LABELS) ---
CLASS_NAMES = [
    "Big Truck", 
    "City Car", 
    "Multi Purpose Vehicle", 
    "Sedan", 
    "Sport Utility Vehicle", 
    "Truck", 
    "Van"
]

# --- INFORMASI MODEL ---
MODEL_INFO = {
    "ResNet50 (Finetuned)": {
        "accuracy": "92%",
        "parameters": "25M",
        "architecture": "ResNet50V2",
        "description": "🏆 Best Model - Deep residual network dengan fine-tuning",
        "speed": "~120ms/image"
    },
    "MobileNetV2 (Finetuned)": {
        "accuracy": "82%",
        "parameters": "3.5M",
        "architecture": "MobileNetV2",
        "description": "⚡ Fastest Model - Optimized untuk mobile deployment",
        "speed": "~80ms/image"
    },
    "EfficientNet B0 (Finetuned)": {
        "accuracy": "80%",
        "parameters": "5M",
        "architecture": "EfficientNetB0",
        "description": "⚖️ Balanced Model - Good accuracy dengan efficiency",
        "speed": "~100ms/image"
    },
    "Base Model Final": {
        "accuracy": "84%",
        "parameters": "5.8M",
        "architecture": "Custom CNN",
        "description": "🔧 Custom Model - Built from scratch, trained on 128x128",
        "speed": "~50ms/image"
    }
}

# --- DAFTAR MODEL ---
MODEL_PATH = "model/"
AVAILABLE_MODELS = {
    "ResNet50 (Finetuned)": os.path.join(MODEL_PATH, "model_resnet50_finetuned.keras"),
    "MobileNetV2 (Finetuned)": os.path.join(MODEL_PATH, "model_mobilenetv2_finetuned.keras"),
    "EfficientNet B0 (Finetuned)": os.path.join(MODEL_PATH, "model_effb0_finetuned.keras"),
    "Base Model Final": os.path.join(MODEL_PATH, "model_base_final.keras"),
}

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_model(model_path):
    """Memuat model menggunakan cache agar tidak berat saat reload."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None

# --- FUNGSI PREDIKSI ---
def predict_image(model, image, model_name):
    """Melakukan preprocessing dan prediksi."""
    # Base model menggunakan 128x128, yang lain 224x224
    if "Base Model" in model_name:
        target_size = (128, 128)
    else:
        target_size = (224, 224)
    
    img = image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 
    
    predictions = model.predict(img_array, verbose=0)
    return predictions[0]

# --- FUNGSI CONFIDENCE BADGE ---
def get_confidence_badge(confidence):
    """Return HTML badge berdasarkan confidence level."""
    if confidence >= 70:
        return f'<span class="confidence-high">🎯 High Confidence: {confidence:.1f}%</span>'
    elif confidence >= 40:
        return f'<span class="confidence-medium">⚠️ Medium Confidence: {confidence:.1f}%</span>'
    else:
        return f'<span class="confidence-low">❓ Low Confidence: {confidence:.1f}%</span>'

# --- HEADER ---
st.markdown("""
    <h1>🚗 Vehicle Classification Dashboard</h1>
    <p style='text-align: center; font-size: 1.2rem; color: #4a5568; margin-bottom: 30px;'>
        Powered by Deep Learning & Transfer Learning | 7 Vehicle Categories | 92% Accuracy
    </p>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## Model Selection")
    
    selected_model_name = st.selectbox(
        "Choose Model:",
        list(AVAILABLE_MODELS.keys()),
        index=0  # Default ke ResNet50
    )
    
    # Tampilkan info model
    if selected_model_name in MODEL_INFO:
        info = MODEL_INFO[selected_model_name]
        st.markdown("---")
        st.markdown("### Model Information")
        st.markdown(f"**Architecture:** {info['architecture']}")
        st.markdown(f"**Test Accuracy:** {info['accuracy']}")
        st.markdown(f"**Parameters:** {info['parameters']}")
        st.markdown(f"**Inference Speed:** {info['speed']}")
        
        st.markdown("---")
        st.info(info['description'])
    
    st.markdown("---")
    st.markdown("### Categories")
    for class_name in CLASS_NAMES:
        st.markdown(f"{CLASS_EMOJI[class_name]} {class_name}")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
        This system uses state-of-the-art deep learning models 
        to classify vehicle images into 7 categories with high accuracy.
        
        **Training Dataset:** 15,645 images  
        **Test Accuracy:** Up to 92%  
        **Framework:** TensorFlow 2.x
    """)

# Load model yang dipilih
model_path = AVAILABLE_MODELS[selected_model_name]
if os.path.exists(model_path):
    with st.spinner('🔄 Loading model...'):
        model = load_model(model_path)
    if model is not None:
        st.sidebar.success(f"✅ Model loaded successfully!")
else:
    st.sidebar.error(f"❌ Model file not found: {model_path}")
    model = None

# --- MAIN CONTENT ---
tab1, tab2, tab3 = st.tabs(["Prediction", "Model Comparison", "How It Works"])

with tab1:
    # Section 1: Upload Images
    st.markdown("### Upload Images")
    uploaded_files = st.file_uploader(
        "Choose vehicle image(s) (JPG/PNG)...", 
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload clear images of vehicles for classification. You can upload multiple images at once."
    )
    
    # Display uploaded images
    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} image(s) uploaded**")
        
        # Show thumbnails in columns
        num_cols = min(len(uploaded_files), 4)  # Max 4 columns
        cols = st.columns(num_cols)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            col_idx = idx % num_cols
            with cols[col_idx]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Image {idx+1}", use_container_width=True)
                st.caption(f"Size: {image.size[0]}x{image.size[1]} | {uploaded_file.type}")
    
    # Section 2: Analyze Button
    st.markdown("---")
    if uploaded_files and model is not None:
        if st.button("Analyze Images", use_container_width=True, type="primary"):
            st.markdown("---")
            
            # Process each image
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f'Processing image {idx+1}/{len(uploaded_files)}...'):
                    try:
                        image = Image.open(uploaded_file)
                        
                        # Prediksi
                        predictions = predict_image(model, image, selected_model_name)
                        
                        # Hasil utama
                        predicted_class_idx = np.argmax(predictions)
                        predicted_class = CLASS_NAMES[predicted_class_idx]
                        confidence = predictions[predicted_class_idx] * 100
                        
                        # Container untuk setiap hasil
                        st.markdown(f"### Results for Image {idx+1}: {uploaded_file.name}")
                        
                        # Show image and prediction result in columns
                        col_img, col_result = st.columns([1, 2])
                        
                        with col_img:
                            st.image(image, use_container_width=True)
                        
                        with col_result:
                            # Tampilkan hasil utama dengan styling
                            st.markdown(f"""
                                <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border-radius: 15px; margin: 10px 0;'>
                                    <h2 style='margin: 0; color: #667eea;'>{CLASS_EMOJI[predicted_class]} {predicted_class}</h2>
                                    <p style='font-size: 1.2rem; margin: 10px 0;'>{get_confidence_badge(confidence)}</p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            # Metrics dalam 3 kolom
                            col_m1, col_m2, col_m3 = st.columns(3)
                            with col_m1:
                                st.metric("Prediction", predicted_class)
                            with col_m2:
                                st.metric("Confidence", f"{confidence:.2f}%")
                            with col_m3:
                                st.metric("Model", selected_model_name.split(" ")[0])
                        
                        # Buat dataframe untuk visualisasi
                        df_predictions = pd.DataFrame({
                            'Category': CLASS_NAMES,
                            'Confidence (%)': predictions * 100,
                            'Emoji': [CLASS_EMOJI[name] for name in CLASS_NAMES]
                        })
                        df_predictions = df_predictions.sort_values('Confidence (%)', ascending=True)
                        
                        # Section 3: Confidence Distribution & Detailed Table (2 columns)
                        col_chart, col_table = st.columns([1.5, 1])
                        
                        with col_chart:
                            st.markdown("#### Confidence Distribution")
                            
                            # Horizontal bar chart dengan Plotly
                            fig = go.Figure()
                            
                            colors = ['#667eea' if cat == predicted_class else '#cbd5e0' 
                                     for cat in df_predictions['Category']]
                            
                            fig.add_trace(go.Bar(
                                y=[f"{emoji} {cat}" for emoji, cat in 
                                   zip(df_predictions['Emoji'], df_predictions['Category'])],
                                x=df_predictions['Confidence (%)'],
                                orientation='h',
                                marker=dict(
                                    color=colors,
                                    line=dict(color='#667eea', width=2)
                                ),
                                text=[f"{val:.2f}%" for val in df_predictions['Confidence (%)']],
                                textposition='outside',
                                hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
                            ))
                            
                            fig.update_layout(
                                title={
                                    'text': "Confidence Score for All Categories",
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'font': {'size': 16, 'color': '#2d3748'}
                                },
                                xaxis_title="Confidence (%)",
                                yaxis_title="Vehicle Category",
                                height=350,
                                showlegend=False,
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(size=11),
                                margin=dict(l=10, r=10, t=50, b=10)
                            )
                            
                            fig.update_xaxes(
                                showgrid=True, 
                                gridwidth=1, 
                                gridcolor='#e2e8f0',
                                range=[0, 100]
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, key=f"chart_{idx}")
                        
                        with col_table:
                            st.markdown("#### Detailed Confidence Table")
                            df_display = df_predictions.sort_values('Confidence (%)', ascending=False)
                            df_display['Rank'] = range(1, len(df_display) + 1)
                            df_display['Category'] = df_display['Emoji'] + ' ' + df_display['Category']
                            df_display = df_display[['Rank', 'Category', 'Confidence (%)']]
                            df_display['Confidence (%)'] = df_display['Confidence (%)'].apply(lambda x: f"{x:.2f}%")
                            
                            st.dataframe(
                                df_display,
                                use_container_width=True,
                                hide_index=True,
                                height=350,
                                key=f"table_{idx}",
                                column_config={
                                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                                    "Category": st.column_config.TextColumn("Vehicle", width="medium"),
                                    "Confidence (%)": st.column_config.TextColumn("Conf.", width="small")
                                }
                            )
                        
                        # Separator between results
                        if idx < len(uploaded_files) - 1:
                            st.markdown("---")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing image {idx+1}: {e}")
                        st.exception(e)
    
    elif not uploaded_files:
        st.info("👆 Please upload one or more images to get started!")
    elif model is None:
        st.warning("⚠️ Model not loaded. Please check the model file.")

with tab2:
    st.markdown("### Model Performance Comparison")
    
    # Data perbandingan model
    comparison_data = {
        'Model': ['ResNet50V2', 'Base CNN', 'MobileNetV2', 'EfficientNetB0'],
        'Accuracy (%)': [92, 84, 82, 80],
        'Parameters (M)': [25, 5.8, 3.5, 5],
        'Speed (ms)': [120, 50, 80, 100],
        'F1-Score': [0.94, 0.88, 0.86, 0.85]
    }
    df_comparison = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy comparison
        fig_acc = px.bar(
            df_comparison, 
            x='Model', 
            y='Accuracy (%)',
            title='Test Accuracy Comparison',
            color='Accuracy (%)',
            color_continuous_scale='Blues',
            text='Accuracy (%)'
        )
        fig_acc.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig_acc.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        # Speed comparison
        fig_speed = px.bar(
            df_comparison, 
            x='Model', 
            y='Speed (ms)',
            title='Inference Speed Comparison (Lower is Better)',
            color='Speed (ms)',
            color_continuous_scale='Reds_r',
            text='Speed (ms)'
        )
        fig_speed.update_traces(texttemplate='%{text:.0f}ms', textposition='outside')
        fig_speed.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_speed, use_container_width=True)
    
    # Tabel lengkap
    st.markdown("#### Complete Performance Metrics")
    st.dataframe(
        df_comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy (%)": st.column_config.ProgressColumn(
                "Accuracy",
                format="%.0f%%",
                min_value=0,
                max_value=100,
            ),
        }
    )

with tab3:
    st.markdown("### How It Works")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Classification Process
        
        1. **Image Upload**   
           User uploads a vehicle image (JPG/PNG format)
        
        2. **Preprocessing**   
           - Resize to 224x224 (or 128x128 for Base Model)
           - Normalize pixel values (0-1 range)
           - Convert to numpy array
        
        3. **Model Inference**   
           - Deep neural network processes the image
           - Extracts features using convolutional layers
           - Outputs probability distribution for 7 classes
        
        4. **Results Display**   
           - Shows predicted category
           - Displays confidence scores
           - Visualizes all class probabilities
        """)
    
    with col2:
        st.markdown("""
        #### Model Architectures
        
        **ResNet50V2**   
        - 50 layers with residual connections
        - 25M parameters
        - Best accuracy: 92%
        
        **MobileNetV2**   
        - Lightweight depthwise separable convolutions
        - 3.5M parameters
        - Fastest inference: 80ms
        
        **EfficientNetB0**  
        - Compound scaling method
        - 5M parameters
        - Balanced performance
        
        **Custom CNN**  
        - Built from scratch
        - Trained on 128x128 resolution
        - Good baseline: 84%
        """)
    
    st.markdown("---")
    st.markdown("""
    #### About the Project
    
    This Vehicle Classification System was developed as part of **Machine Learning Practicum Final Project (UAP)**.
    The system compares traditional CNN (built from scratch) against modern Transfer Learning approaches 
    (MobileNetV2, ResNet50V2, EfficientNetB0) on a low-resolution vehicle dataset.
    
    **Key Achievements:**
    - Achieved 92% test accuracy with ResNet50 fine-tuning
    - Trained on 15,645 images across 7 vehicle categories
    - Optimized for both accuracy and inference speed
    - Successfully handled class imbalance using weighted loss
    
    **Technologies Used:**
    - TensorFlow 2.x & Keras
    - Streamlit for web interface
    - Plotly for interactive visualizations
    - Transfer Learning with ImageNet pre-trained weights
    """)

# --- FOOTER ---
st.markdown("---")
st.markdown("""
    
""", unsafe_allow_html=True)