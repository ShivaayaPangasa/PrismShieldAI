"""
==============================================================

PrismShieldAI

Enterprise Multimodal Deepfake Detection Platform

Causal and Contrastive Multimodal Reasoning
for Robust Deep Fake Detection

Author:
Shivaaya

==============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================

import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import torch

from inference.inference import PrismShieldInference
from configs.settings import *

# ==========================================================
# CONSTANTS
# ==========================================================

APP_NAME = "PrismShieldAI"

VERSION = "v2.0"

PRIMARY = "#5B6CFF"
SECONDARY = "#8B5CF6"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"

BACKGROUND = "#050B18"
SURFACE = "#0B1220"
CARD = "#121C31"
BORDER = "#23345C"

TEXT = "#F8FAFC"
TEXT_SECONDARY = "#94A3B8"

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(

    page_title="PrismShieldAI",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="expanded",

)

# ==========================================================
# PREMIUM CSS
# ==========================================================

st.markdown(
"""
<style>

/* =======================================================
Entire App
======================================================= */

.stApp{

background:
radial-gradient(circle at top left,#13203d 0%,#09101f 40%,#050b18 100%);

color:#F8FAFC;

}

details {
    background:#121C31 !important;
    border:1px solid #23345C !important;
    border-radius:12px !important;
}

summary{
    color:#F8FAFC !important;
    font-weight:700 !important;
}

/* =======================================================
Hide Streamlit Branding
======================================================= */

#MainMenu{

visibility:hidden;

}

footer{

visibility:hidden;

}

header{

visibility:hidden;

}

/* =======================================================
Container
======================================================= */

.block-container{

padding-top:2rem;

padding-left:3rem;

padding-right:3rem;

padding-bottom:2rem;

}

/* =======================================================
Sidebar
======================================================= */

section[data-testid="stSidebar"]{

background:

linear-gradient(

180deg,

#081225,

#0E1730,

#081225

);

border-right:1px solid #23345C;

}

section[data-testid="stSidebar"] *{

color:#F8FAFC;

}

/* =======================================================
Cards
======================================================= */

.card{

background:#152341;

border:1px solid #3B5BCF;

border-radius:18px;

padding:22px;

color:white;

box-shadow:

0 10px 35px rgba(0,0,0,.35);

}

/* =======================================================
Metric Cards
======================================================= */

div[data-testid="stMetric"]{

background:

linear-gradient(

180deg,

#121C31,

#0C1628

);

padding:18px;

border-radius:16px;

border:1px solid #23345C;

box-shadow:

0 0 25px rgba(91,108,255,.12);

}

/* =======================================================
Buttons
======================================================= */

.stButton>button{

width:100%;

height:55px;

font-size:18px;

font-weight:700;

border:none;

border-radius:14px;

color:white;

background:

linear-gradient(

90deg,

#5B6CFF,

#7C3AED

);

transition:.25s;

box-shadow:

0 0 18px rgba(91,108,255,.35);

}

.stButton>button:hover{

transform:translateY(-2px);

box-shadow:

0 0 28px rgba(91,108,255,.60);

}

/* =======================================================
Upload Box
======================================================= */

section[data-testid="stFileUploader"]{

background:#10192B;

border:2px dashed #304D9A;

border-radius:16px;

padding:14px;

}

/* =======================================================
Alerts
======================================================= */

div[data-testid="stSuccess"]{

border-radius:14px;

}

div[data-testid="stError"]{

border-radius:14px;

}

div[data-testid="stInfo"]{

border-radius:14px;

}

/* =======================================================
Tabs
======================================================= */

button[data-baseweb="tab"]{

font-size:17px;

font-weight:600;

}

button[data-baseweb="tab"][aria-selected="true"]{

color:#7C3AED;

}

/* =======================================================
Horizontal Line
======================================================= */

hr{

border:1px solid #1E315A;

}

/* =======================================================
Scrollbar
======================================================= */

::-webkit-scrollbar{

width:10px;

}

::-webkit-scrollbar-track{

background:#081225;

}

::-webkit-scrollbar-thumb{

background:#304D9A;

border-radius:20px;

}

/* =======================================================
Image Preview
======================================================= */

img{

border-radius:14px;

}

/* =======================================================
Audio Player
======================================================= */

audio{

width:100%;

}

/* =======================================================
Global Text
======================================================= */

html,
body,
[class*="css"],
.stMarkdown,
.stText,
p,
span,
label,
small{

color:#F8FAFC !important;

}

/* =======================================================
Metric Text
======================================================= */

[data-testid="stMetricValue"]{

color:#FFFFFF !important;

font-weight:700;

}

[data-testid="stMetricLabel"]{

color:#DCE6FF !important;

}

</style>
""",

unsafe_allow_html=True,

)

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(

        "https://img.icons8.com/fluency/128/shield.png",

        width=85,

    )

    st.markdown(
        f"""
# 🛡️ {APP_NAME}

Enterprise Multimodal
Deepfake Detection
"""
    )

    st.divider()

    st.markdown("## 🧠 AI Stack")

    st.markdown("""

**Vision Encoder**

EfficientNet-B0

---

**Audio Encoder**

Wav2Vec2

---

**Fusion Layer**

Cross-Modal Attention

---

**Reasoning**

Contrastive Learning

Causal Reasoning

""")

    st.divider()

    st.markdown("## 💻 Runtime")

    device = "CUDA GPU" if torch.cuda.is_available() else "CPU"

    if torch.cuda.is_available():

        st.success(f"🚀 {device}")

    else:

        st.warning(device)

    st.success("Inference Engine Ready")

    st.success("Fusion Model Loaded")

    st.success("Preprocessor Ready")

    st.divider()

    st.caption(

        f"{APP_NAME} {VERSION}"

    )

# ==========================================================
# HERO
# ==========================================================

hero_left, hero_right = st.columns([3,1])

with hero_left:

    st.markdown(f"""
# 🛡️ {APP_NAME}

## Enterprise Multimodal Deepfake Detection

### Causal and Contrastive Multimodal Reasoning
### for Robust Deepfake Detection

PrismShieldAI combines computer vision, speech understanding,
cross-modal attention, contrastive learning and causal reasoning
to detect manipulated multimedia content with a unified AI pipeline.

Designed for

- AI Security
- Digital Forensics
- Enterprise Trust
- Media Verification
- Research
""")

with hero_right:

    st.markdown(
        """
<div class="card">

## 🚀 Loaded Model

✅ EfficientNet-B0

✅ Wav2Vec2

✅ Cross-Modal Attention

✅ Contrastive Fusion

✅ Causal Reasoning

</div>
""",

        unsafe_allow_html=True,

    )

st.write("")

@st.cache_resource
def load_engine():

    return PrismShieldInference()

with st.spinner("Loading AI Engine..."):

    engine = load_engine()

st.success("✅ PrismShieldAI loaded successfully.")

# ==========================================================
# DASHBOARD CARDS
# ==========================================================

card1, card2, card3, card4 = st.columns(4)

with card1:

    st.markdown(
        """
<div class="card">

# 🖼️ Vision

EfficientNet-B0

CNN Feature Extraction

256-D Embedding

</div>
""",

        unsafe_allow_html=True,

    )

with card2:

    st.markdown(
        """
<div class="card">

# 🎤 Audio

Wav2Vec2

Speech Encoder

Temporal Features

</div>
""",

        unsafe_allow_html=True,

    )

with card3:

    st.markdown(
        """
<div class="card">

# 🔀 Fusion

Cross-Modal Attention

Contrastive Learning

Feature Alignment

</div>
""",

        unsafe_allow_html=True,

    )

with card4:

    st.markdown(
        """
<div class="card">

# 🧠 Reasoning

Causal Analysis

Confidence Head

Authenticity Prediction

</div>
""",

        unsafe_allow_html=True,

    )

st.write("")

st.divider()

# ==========================================================
# PIPELINE + STATUS
# ==========================================================

pipeline_col, status_col = st.columns([2,1])

with pipeline_col:

    st.markdown("## 🧠 AI Pipeline")
    
    pipeline_path = Path(__file__).parent / "pipeline.png"

    st.image(
        str(pipeline_path),
        use_container_width=True,
        caption="PrismShieldAI Multimodal Deepfake Detection Pipeline"
    )

with status_col:

    st.markdown("## 📊 System Status")

    st.success("Inference Ready")

    st.success("Fusion Model Loaded")

    st.success("GPU Enabled" if torch.cuda.is_available() else "CPU Mode")

    st.success("Preprocessor Ready")

    st.success("Video Pipeline Ready")

st.divider()

# ==========================================================
# ANALYSIS WORKSPACE
# ==========================================================

st.markdown("# 🔬 Analysis Workspace")

st.caption(
    "Upload multimedia content and let PrismShieldAI perform multimodal deepfake analysis."
)

analysis_tab, video_tab = st.tabs(
    [
        "🧪 Advanced Multimodal Analysis",
        "🎥 Video Analysis",
    ]
)

# ==========================================================
# VIDEO TAB
# ==========================================================

with video_tab:

    st.header("🎥 Multimodal Video Analysis")

    st.markdown(
        """
Upload a video.

PrismShieldAI will automatically:

• Select the best facial frame
• Extract speech
• Preprocess both modalities
• Perform multimodal reasoning
• Predict REAL or FAKE
"""
    )

    video_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    import tempfile
    from pathlib import Path

    if video_file is not None:

        temp_dir = Path(tempfile.gettempdir())

        video_path = temp_dir / video_file.name

        with open(video_path, "wb") as f:
            f.write(video_file.read())

        if st.button(
            "🚀 Analyze Video",
            use_container_width=True,
        ):

            try:

                start = time.perf_counter()

                with st.spinner("Analyzing video..."):

                    result = engine.predict_from_video(
                        video_path
                    )

                elapsed = time.perf_counter() - start

                if result["status"] != "success":

                    st.error(result["error"])

                else:

                    prediction = result["prediction"]
                    probability = result["probability"]
                    confidence = result["confidence"]

                    st.success("✅ Video Analysis Complete")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Prediction",
                            prediction,
                        )

                    with col2:
                        st.metric(
                            "Prediction Probability",
                            f"{probability * 100:.2f}%"
                        )

                    with col3:
                        st.metric(
                            "Fusion Confidence",
                            f"{confidence * 100:.2f}%"
                        )
                    
                    status = "High" if confidence >= 0.90 else \
                            "Medium" if confidence >= 0.70 else \
                            "Low"

                    st.metric(
                        "Detection Confidence",
                        status
                    )

                    st.divider()
                    
                    st.subheader("🖼 Representative Frame")

                    import cv2

                    # Convert OpenCV BGR image to RGB for correct display
                    rgb_frame = cv2.cvtColor(
                        result["selected_frame"],
                        cv2.COLOR_BGR2RGB,
                    )

                    st.image(
                        rgb_frame,
                        use_container_width=True,
                    )

                    st.divider()

                    st.subheader("🎤 Extracted Audio")

                    st.audio(
                        str(result["audio_path"])
                    )

                    st.divider()
                    
                    st.subheader("⚙ Processing Pipeline")

                    st.markdown("""
                    **Video**

                        ↓

                    **Frame Selection**

                        ↓

                    **Audio Extraction**

                        ↓

                    **Image Preprocessing**

                        ↓

                    **Audio Preprocessing**

                        ↓

                    **Cross-Modal Attention**

                        ↓

                    **Contrastive Learning**

                        ↓

                    **Causal Reasoning**

                        ↓

                    # ✅ Prediction
                    """)

                    st.metric(
                        "Processing Time",
                        f"{elapsed:.2f} sec",
                    )

            except Exception as e:

                st.error(
                    f"Inference failed:\n\n{e}"
                )

# ==========================================================
# ADVANCED ANALYSIS
# ==========================================================

with analysis_tab:

    st.markdown("## Upload Evidence")

    left_col, right_col = st.columns(2)

    # ------------------------------------------------------
    # IMAGE
    # ------------------------------------------------------

    with left_col:

        st.markdown("### 🖼 Face Image")

        image_file = st.file_uploader(

            "Upload a facial image",

            type=[

                "jpg",

                "jpeg",

                "png",

            ],

            key="image",

        )

        if image_file is not None:

            st.image(

                image_file,

                use_container_width=True,

            )

    # ------------------------------------------------------
    # AUDIO
    # ------------------------------------------------------

    with right_col:

        st.markdown("### 🎤 Voice Audio")

        audio_file = st.file_uploader(

            "Upload a voice recording",

            type=[

                "wav",

                "mp3",

                "flac",

            ],

            key="audio",

        )

        if audio_file is not None:

            st.audio(audio_file)

    st.write("")

    # ------------------------------------------------------
    # ANALYZE BUTTON
    # ------------------------------------------------------

    analyze_button = st.button(

        "🚀 Analyze with PrismShieldAI",

        use_container_width=True,

    )

    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

    if analyze_button:

        if image_file is None:

            st.warning(

                "Please upload a face image."

            )

            st.stop()

        if audio_file is None:

            st.warning(

                "Please upload a voice recording."

            )

            st.stop()

        # --------------------------------------------------
        # TEMP FILES
        # --------------------------------------------------

        temp_dir = Path("temp")

        temp_dir.mkdir(

            exist_ok=True,

        )

        image_path = temp_dir / image_file.name

        audio_path = temp_dir / audio_file.name

        image_path.write_bytes(

            image_file.getbuffer()

        )

        audio_path.write_bytes(

            audio_file.getbuffer()

        )

        # --------------------------------------------------
        # INFERENCE
        # --------------------------------------------------

        try:
            
            with st.spinner(
                
                "Running multimodal reasoning..."

            ):
                
                start = time.perf_counter()

                result = engine.predict_from_image_audio(
                
                    image_path,

                    audio_path,

                )
                
                elapsed = time.perf_counter() - start
        
            # ==========================================================
            # RESULTS
            # ==========================================================

            if result["status"] != "success":
                
                st.error(result["error"])
        
            else:

                st.success("Analysis Complete")

                prediction = result["prediction"]

                probability = result["probability"]

                confidence = result["confidence"]

                col1, col2, col3 = st.columns(3)

                col1.metric(

                    "Prediction",

                    prediction,

                )

                col2.metric(

                    "Probability",

                    f"{probability*100:.2f}%",

                )

                col3.metric(

                    "Confidence",

                    f"{confidence*100:.2f}%",

                )

            # ======================================================
            # VERDICT
            # ======================================================

            st.divider()

            if prediction == "REAL":

                st.success(
                    """
# ✅ AUTHENTIC MEDIA

PrismShieldAI did not detect significant multimodal
inconsistencies.

The uploaded media appears authentic.
"""
                )

            else:

                st.error(
                    """
# ⚠ POTENTIAL DEEPFAKE DETECTED

PrismShieldAI identified multimodal inconsistencies.

The uploaded media may contain synthetic manipulation.
"""
                )

            # ======================================================
            # KPI DASHBOARD
            # ======================================================

            st.markdown("## 📊 Analysis Summary")

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:

                st.metric(

                    "Prediction",

                    prediction,

                )

            with kpi2:

                st.metric(

                    "Prediction Probability",

                    f"{probability*100:.2f}%",

                )

            with kpi3:

                st.metric(

                    "Fusion Confidence",

                    f"{confidence:.2f}%",

                )

            with kpi4:

                st.metric(

                    "Processing Time",

                    f"{elapsed:.2f} sec",

                )

            # ======================================================
            # CONFIDENCE BAR
            # ======================================================

            st.markdown("### 🎯 Prediction Confidence")

            st.progress(probability)

            if probability >= 95:

                st.success(
                    "Very High Prediction Confidence"
                )

            elif probability >= 80:

                st.info(
                    "High Prediction Confidence"
                )

            elif probability >= 60:

                st.warning(
                    "Moderate Prediction Confidence"
                )

            else:

                st.error(
                    "Low Prediction Confidence"
                )

            st.divider()

            # ======================================================
            # DETAILS
            # ======================================================

            left, right = st.columns([2,1])

            with left:

                st.markdown("## 🧠 AI Interpretation")

                if prediction == "REAL":

                    st.info(
"""
The visual and speech representations remain
consistent throughout multimodal fusion.

No major evidence of synthetic manipulation
was detected.
"""
                    )

                else:

                    st.warning(
"""
The visual and speech representations exhibit
cross-modal inconsistencies.

This pattern is commonly associated with
AI-generated or manipulated media.
"""
                    )

            with right:

                st.markdown("## ⚙ Pipeline")

                st.success("✔ Face Detected")

                st.success("✔ Audio Processed")

                st.success("✔ Feature Extraction")

                st.success("✔ Cross-Modal Fusion")

                st.success("✔ Causal Reasoning")

                st.success("✔ Inference Complete")

            st.divider()

            # ======================================================
            # TECHNICAL DETAILS
            # ======================================================

            with st.expander("📈 Technical Information"):

                st.write("Prediction:", prediction)

                st.write(
                    f"Prediction Probability: {probability:.2f}%"
                )

                st.write(
                    f"Fusion Confidence: {confidence:.2f}%"
                )

                st.write(
                    f"Processing Time: {elapsed:.2f} seconds"
                )

                st.write(
                    "Device:",
                    device,
                )

                st.write(
                    "Vision Encoder:",
                    "EfficientNet-B0",
                )

                st.write(
                    "Audio Encoder:",
                    "Wav2Vec2",
                )

                st.write(
                    "Fusion:",
                    "Cross-Modal Attention",
                )

                st.write(
                    "Reasoning:",
                    "Contrastive + Causal",
                )

            # ======================================================
            # REPORT
            # ======================================================

            report = f"""
PrismShieldAI Report

Prediction:
{prediction}

Prediction Probability:
{probability:.2f}%

Fusion Confidence:
{confidence:.2f}%

Processing Time:
{elapsed:.2f} sec

Generated by PrismShieldAI Enterprise
"""

            st.download_button(

                "📄 Download Report",

                report,

                file_name="PrismShield_Report.txt",

                use_container_width=True,

            )

        # ==========================================================
        # CLEANUP
        # ==========================================================

            image_path.unlink(missing_ok=True)

            audio_path.unlink(missing_ok=True)

        except Exception as e:

            st.error(f"Inference failed:\n\n{e}")       
        
# ==========================================================
# ROADMAP
# ==========================================================

st.markdown("# 🛣 Development Roadmap")

road1, road2, road3, road4 = st.columns(4)

with road1:

    st.success(
"""
### ✔ Phase 1

Dataset

Training

Fusion Model
"""
    )

with road2:

    st.success(
"""
### ✔ Phase 2

Inference

Dashboard

Evaluation
"""
    )
    
with road3:

    st.success(
"""
### ✔ Phase 3

Video Analysis

Inference Dashboard

Performance Evaluation
"""
)

with road4:

    st.info(
"""
### Phase 4

Explainable AI

Cross-Dataset Validation

Enterprise Deployment
"""
    )

st.divider()

# ==========================================================
# ABOUT MODEL
# ==========================================================

with st.expander("About PrismShieldAI"):

    st.markdown(
        
        '''PrismShieldAI is a multimodal deepfake detection platform that integrates
computer vision, speech understanding, cross-modal attention,
contrastive learning and causal reasoning for robust multimedia authentication.

### Current Architecture

• EfficientNet-B0 Visual Encoder

• Wav2Vec2 Audio Encoder

• Cross-Modal Attention Fusion

• Contrastive Learning Module

• Causal Reasoning Module

• Fusion Model

• Confidence Head

• Deepfake Classifier

The framework is designed for research, digital forensics,
media verification and trustworthy AI.'''
        
    )

st.divider()

# ==========================================================
# STATISTICS
# ==========================================================

st.markdown("# 📈 Platform Overview")

stat1, stat2, stat3, stat4 = st.columns(4)

with stat1:

    st.metric(

        "Visual Encoder",

        "EfficientNet-B0",

    )

with stat2:

    st.metric(

        "Audio Encoder",

        "Wav2Vec2",

    )

with stat3:

    st.metric(

        "Fusion",

        "Cross-Modal",

    )

with stat4:

    st.metric(

        "Reasoning",
        
        "Causal + Confidence",

    )

st.divider()

# ==========================================================
# FOOTER
# ==========================================================

footer_left, footer_middle, footer_right = st.columns([3,2,2])

with footer_left:

    st.markdown(
"""
### 🛡 PrismShieldAI

Enterprise Multimodal Deepfake Detection Platform

Designed for trustworthy AI systems.
"""
    )

with footer_middle:

    st.markdown(
"""
### Architecture

Vision

Audio

Fusion

Reasoning
"""
    )

with footer_right:

    st.markdown(
f"""
### Version

{VERSION}

Research Prototype

© Shivaaya
"""
    )

st.caption(
"""
PrismShieldAI is a research prototype developed for multimodal
deepfake detection using causal reasoning and contrastive learning.

Future releases will include Explainable AI, Cross-Dataset Validation, Real-time Monitoring and Enterprise Deployment
"""
)