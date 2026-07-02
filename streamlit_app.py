import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import time
import threading

try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# Set page configuration with wide layout and custom title
st.set_page_config(
    page_title="Facial Emotion Detection System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and clean UI
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgb(239, 246, 255) 0%, rgb(255, 255, 255) 90%);
    }
    h1 {
        color: #1e3a8a;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    h3 {
        color: #2563eb;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e5e7eb;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2563eb;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #4b5563;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
""", unsafe_allow_html=True)

# Load the YOLO model using resource caching for speed
@st.cache_resource
def load_model():
    try:
        model = YOLO("best.pt")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load model
model = load_model()

# WebRTC Video Processor for real-time browser streaming
if WEBRTC_AVAILABLE:
    class EmotionDetectionProcessor(VideoProcessorBase):
        def __init__(self):
            self.conf_threshold = 0.34
            self.lock = threading.Lock()
            self.latest_detections = []
            self.latest_inference_time = 0.0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            if model is None:
                return frame
                
            with self.lock:
                conf_thresh = self.conf_threshold
                
            start_time = time.time()
            try:
                results = model.predict(
                    source=img,
                    conf=conf_thresh,
                    imgsz=640,
                    verbose=False
                )
                inference_time = (time.time() - start_time) * 1000  # in ms
                
                detections = []
                annotated_bgr = img
                
                if results and len(results) > 0:
                    result = results[0]
                    annotated_bgr = result.plot()
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
                        confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, "cpu") else result.boxes.conf
                        classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, "cpu") else result.boxes.cls
                        
                        for box, conf, cls in zip(xyxy, confs, classes):
                            label = model.names.get(int(cls), str(int(cls)))
                            detections.append({
                                "Emotion": label.capitalize(),
                                "Confidence": float(conf)
                            })
                
                with self.lock:
                    self.latest_detections = detections
                    self.latest_inference_time = inference_time
                    
                return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")
            except Exception as e:
                return frame
else:
    class EmotionDetectionProcessor:
        pass


# Title and introduction
st.title("😊 Facial Emotion Detection System")
st.markdown("An AI-powered application that detects human faces and classifies their facial emotions in real-time.")

# Sidebar Configuration
st.sidebar.title("⚙️ Control Panel")

# Confidence threshold slider
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.34,
    step=0.01,
    help="Minimum confidence score required to display a detection."
)

# Input method selection
input_method = st.sidebar.radio(
    "Select Input Source",
    ("Upload Image", "Live Webcam (Browser via WebRTC)", "Webcam Snapshot (Fallback)")
)

# Display class labels list if model is loaded successfully
if model:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detectable Emotions")
    # Clean and present labels
    emotions = list(model.names.values())
    for emotion in emotions:
        st.sidebar.markdown(f"- **{emotion.capitalize()}**")

# Application Logic
if model is None:
    st.error("Could not initialize the detection model. Make sure `best.pt` is in the project root directory.")
else:
    if input_method == "Live Webcam (Browser via WebRTC)":
        st.subheader("🎥 Live Web Browser Webcam Detection")
        st.markdown(
            "This mode captures live frames from your browser camera using WebRTC and performs real-time emotion detection. "
            "Please allow camera permissions in your browser when prompted, and click **Start**."
        )
        
        if not WEBRTC_AVAILABLE:
            st.warning("⚠️ `streamlit-webrtc` is not installed or supported in this environment. Falling back to Snapshot mode.")
            st.info("Please select **Webcam Snapshot (Fallback)** in the sidebar Control Panel to use your webcam.")
        else:
            # Webrtc streamer component
            ctx = webrtc_streamer(
                key="emotion-webrtc",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]
                },
                video_processor_factory=EmotionDetectionProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
            
            # Placeholders for statistics below the stream
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.info("💡 Real-time annotated video stream is displayed above. Emotion bounding boxes and confidence levels are plotted directly on the frames.")
                
            with col2:
                st.subheader("Live Stats")
                metric_col1, metric_col2 = st.columns(2)
                faces_placeholder = metric_col1.empty()
                fps_placeholder = metric_col2.empty()
                
                info_placeholder = st.empty()
                table_placeholder = st.empty()
                chart_placeholder = st.empty()

            if ctx.video_processor:
                # Update confidence threshold dynamically in the video processor
                ctx.video_processor.conf_threshold = conf_threshold
                
                # Retrieve latest detections and stats safely
                with ctx.video_processor.lock:
                    detections = ctx.video_processor.latest_detections
                    inference_time = ctx.video_processor.latest_inference_time
                
                # Draw stats
                faces_placeholder.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{len(detections)}</div>
                        <div class="metric-label">Faces</div>
                    </div>
                """, unsafe_allow_html=True)
                
                fps_val = 1000 / inference_time if inference_time > 0 else 0
                fps_placeholder.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{fps_val:.1f}</div>
                        <div class="metric-label">FPS (Inf)</div>
                    </div>
                """, unsafe_allow_html=True)
                
                if len(detections) > 0:
                    df = pd.DataFrame(detections)
                    primary_idx = df['Confidence'].idxmax()
                    primary_emotion = df.loc[primary_idx]['Emotion']
                    primary_conf = df.loc[primary_idx]['Confidence']
                    
                    info_placeholder.info(f"**Primary Emotion:** {primary_emotion} ({primary_conf*100:.1f}%)")
                    
                    display_df = df.copy()
                    display_df['Confidence'] = display_df['Confidence'].map(lambda x: f"{x*100:.1f}%")
                    table_placeholder.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    chart_data = df.groupby('Emotion')['Confidence'].mean().reset_index()
                    chart_placeholder.bar_chart(
                        data=chart_data,
                        x='Emotion',
                        y='Confidence',
                        color='#2563eb',
                        use_container_width=True
                    )
                else:
                    info_placeholder.warning("No emotions detected above threshold.")
                    table_placeholder.empty()
                    chart_placeholder.empty()
                
                st.caption("⚡ Note: Stats refresh whenever the Streamlit page is rerun or interacted with. The video stream itself is fully real-time.")
            
    else:
        # Static Image / Snapshot Detection Logic
        image_file = None
        
        if input_method == "Upload Image":
            image_file = st.file_uploader(
                "Choose an image...", 
                type=["jpg", "jpeg", "png", "bmp", "webp"]
            )
        else:
            image_file = st.camera_input("Take a photo")

        if image_file is not None:
            # Load the image using PIL and convert to NumPy array (RGB)
            image = Image.open(image_file)
            img_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV / YOLO processing
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Ultralytics YOLO commonly expects images in BGR format when you pass NumPy arrays that come from OpenCV.
            
            # Inference progress bar
            with st.spinner("Processing image... Running emotion detection model..."):
                start_time = time.time()
                results = model.predict(
                    source=img_bgr,
                    conf=conf_threshold,
                    imgsz=640,
                    verbose=False
                )
                inference_time = (time.time() - start_time) * 1000  # in ms
                
            if results and len(results) > 0:
                result = results[0]
                
                # Draw detections
                annotated_bgr = result.plot()
                # Convert BGR back to RGB for streamlit display
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                
                # Extract detection lists
                detections = []
                if result.boxes is not None and len(result.boxes) > 0:
                    xyxy = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, "cpu") else result.boxes.xyxy
                    confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes.conf, "cpu") else result.boxes.conf
                    classes = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, "cpu") else result.boxes.cls
                    
                    for box, conf, cls in zip(xyxy, confs, classes):
                        label = model.names.get(int(cls), str(int(cls)))
                        detections.append({
                            "Emotion": label.capitalize(),
                            "Confidence": float(conf),
                            "Box": [float(x) for x in box]
                        })
                
                # Layout: Display images and metrics
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Detection Result")
                    st.image(
                        annotated_rgb, 
                        caption="Processed Image with Detections", 
                        use_container_width=True
                    )
                    
                with col2:
                    st.subheader("Analytics & Stats")
                    
                    # Show key metrics
                    m_col1, m_col2 = st.columns(2)
                    with m_col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{len(detections)}</div>
                                <div class="metric-label">Faces Detected</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m_col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-value">{inference_time:.1f}ms</div>
                                <div class="metric-label">Inference Time</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    if len(detections) > 0:
                        df = pd.DataFrame(detections)
                        
                        # Primary Emotion calculation
                        primary_emotion = df.loc[df['Confidence'].idxmax()]['Emotion']
                        st.info(f"**Primary Emotion Detected:** {primary_emotion} "
                                f"({df['Confidence'].max()*100:.1f}% confidence)")
                        
                        st.markdown("#### Detailed Classifications")
                        # Display the detection details table
                        display_df = df[['Emotion', 'Confidence']].copy()
                        display_df['Confidence'] = display_df['Confidence'].map(lambda x: f"{x*100:.1f}%")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Emotion confidence distribution plot
                        st.markdown("#### Confidence Scores Chart")
                        chart_data = df.groupby('Emotion')['Confidence'].mean().reset_index()
                        st.bar_chart(
                            data=chart_data, 
                            x='Emotion', 
                            y='Confidence', 
                            color='#2563eb',
                            use_container_width=True
                        )
                    else:
                        st.warning("No faces or emotions were detected above the chosen confidence threshold. Try lowering the threshold in the sidebar.")
            else:
                st.warning("Could not run detection on the image. Please try a different image format.")
        else:
            # Prompt user to load image
            st.info("👈 Please upload an image or capture from your webcam using the sidebar options to begin detection.")
            
            # Display sample UI cards explaining features
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("""
                    <div class="metric-card" style="min-height: 150px;">
                        <div style="font-size: 2.5rem;">📁</div>
                        <div style="font-weight:600; margin-top: 10px;">Multiple Formats Supported</div>
                        <div style="font-size: 0.85rem; color: #6b7280; margin-top: 5px;">Upload PNG, JPG, JPEG, BMP or WebP files instantly.</div>
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                    <div class="metric-card" style="min-height: 150px;">
                        <div style="font-size: 2.5rem;">📸</div>
                        <div style="font-weight:600; margin-top: 10px;">Webcam Support</div>
                        <div style="font-size: 0.85rem; color: #6b7280; margin-top: 5px;">Capture live images directly using your system camera.</div>
                    </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown("""
                    <div class="metric-card" style="min-height: 150px;">
                        <div style="font-size: 2.5rem;">⚡</div>
                        <div style="font-weight:600; margin-top: 10px;">Cached Inference</div>
                        <div style="font-size: 0.85rem; color: #6b7280; margin-top: 5px;">Model weights are cached locally to provide ultra-fast response times.</div>
                    </div>
                """, unsafe_allow_html=True)

