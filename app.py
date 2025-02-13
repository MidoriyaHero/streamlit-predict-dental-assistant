%%writefile app.py
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from google import genai
from streamlit.components.v1 import html
import os

API = os.getenv('API_KEY')
# Page configuration
st.set_page_config(
    page_title="Medical Image Analysis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar for configuration
with st.sidebar:
    st.title("📋 Configuration")
    input_method = st.radio(
        "Select input method:",
        ("Upload Image", "Capture from Camera"),
        help="Choose how you want to input the image for analysis"
    )
    
    st.markdown("---")
    st.markdown("### Model Configuration")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        help="Adjust the confidence threshold for disease detection"
    )

# Main content
st.title("🏥 Medical Image Analysis System")
st.markdown("Upload or capture an image for automated disease detection and medical advice.")

# Initialize the image input based on selected method
if input_method == "Upload Image":
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image for analysis"
    )
else:
    uploaded_file = st.camera_input("Capture an image")

if uploaded_file is not None:
    # Process the image
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Create three columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📸 Input Image")
        st.image(image, use_container_width =True)

    # Load YOLO model and process image
    try:
        model = YOLO("/content/best.pt")
        results = model(image_cv)
        
        annotated_frame = results[0].plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        
        with col2:
            st.markdown("### 🔍 Analysis Result")
            st.image(annotated_frame_rgb, use_container_width =True)

        # Display detected diseases
        st.markdown("---")
        st.markdown("### 📊 Detection Results")
        
        detected_classes = []
        if results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()
            class_names = results[0].names
            
            # Create a container for results
            with st.container():
                for idx, prob in enumerate(probs):
                    if prob > confidence_threshold:
                        detected_classes.append(class_names[idx])
                        st.progress(float(prob))
                        st.write(f"**{class_names[idx]}**: {prob*100:.2f}%")

        # Chatbot interface
        if detected_classes:
            st.markdown("---")
            st.markdown("### 👨‍⚕️ Medical Assistant Advices 💬 ")
            
            # Create a chat-like interface
            chat_container = st.container()
            with chat_container:
                # Initialize Gemini
                client = genai.Client(api_key=API)
                
                disease_list = ", ".join(detected_classes)
                prompt = f"""Based on the image analysis, potential conditions detected: {disease_list}

                              Please provide:
                              1. Brief explanation of these conditions
                              2. Recommended next steps
                              3. General precautions
                              Please keep the response concise and clear."""
                with st.spinner("Getting medical advice..."):
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents=[prompt]
                        )
                        if response:
                            st.markdown(response.text)
                        else:
                            st.error("Unable to generate medical advice at the moment.")
                    except Exception as e:
                        st.error(f"Error generating medical advice: {str(e)}")

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.info("Please make sure the model file 'best.pt' is available in the correct location.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>⚠️ This is an automated analysis system. Always consult with a healthcare professional for medical advice.</p>
    </div>
""", unsafe_allow_html=True)