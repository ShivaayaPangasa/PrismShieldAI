"""
==============================================================
PrismShieldAI

Causal and Contrastive Multimodal Reasoning
for Robust Deep Fake Detection

Streamlit Application

Author:
Shivaaya

==============================================================
"""

# ==========================================================
# IMPORTS
# ==========================================================
import time

from pathlib import Path

import streamlit as st
import torch

from inference.inference import PrismShieldInference
from configs.settings import *

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(

    page_title="PrismShieldAI",

    page_icon="🛡️",

    layout="wide",

    initial_sidebar_state="expanded",

)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown(
    """<style>

.stApp{

background:#08111f;

color:white;

}

section[data-testid="stSidebar"]{

background:#0d1630;

}

div[data-testid="stMetric"]{

background:#121d3b;

padding:18px;

border-radius:14px;

border:1px solid #304d9a;

box-shadow:0 0 18px rgba(88,101,242,0.15);

}

.stButton>button{

background:linear-gradient(
90deg,
#5865F2,
#7C3AED
);

color:white;

font-weight:bold;

border:none;

border-radius:12px;

height:50px;

}

.stButton>button:hover{

background:linear-gradient(
90deg,
#6875FF,
#8B5CF6
);

}

hr{

border-color:#24345f;

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
        "https://img.icons8.com/fluency/96/shield.png",
        width=90,
    )

    st.title("PrismShieldAI")

    st.caption(
        "Multimodal Deepfake Detection"
    )

    st.divider()
    
    st.subheader("Architecture")

    st.markdown("""

    🖼 **Vision Encoder**

    EfficientNet-B0

    🎤 **Audio Encoder**

    Wav2Vec2

    🔀 **Fusion**

    Cross-Modal Attention

    🧠 **Reasoning**

    Contrastive + Causal

    """)

    st.divider()

    st.subheader("Status")

    st.success("Inference Engine Loaded")

    device = "CUDA GPU" if torch.cuda.is_available() else "CPU"

    st.info(f"Running on: {device}")

    st.divider()

    st.subheader("Version")

    st.write("PrismShieldAI v2.0")

# ==========================================================
# HEADER
# ==========================================================

st.title("🛡️ PrismShieldAI")

st.subheader(

    "Causal and Contrastive Multimodal Reasoning for Robust Deep Fake Detection"

)

st.write(

    """
Analyze multimedia content using a multimodal AI pipeline that combines:

- 🎥 Visual Analysis
- 🎤 Audio Analysis
- 🔀 Cross-Modal Attention
- 🧠 Contrastive Learning
- 🔬 Causal Reasoning

The system is designed for research, demonstrations, and real-world deepfake detection workflows.
"""
)

st.divider()

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_engine():

    return PrismShieldInference()


with st.spinner("Loading PrismShieldAI..."):

    engine = load_engine()

st.success("✅ PrismShieldAI loaded successfully.")

st.info(

    """
### AI Pipeline

Image
→ EfficientNet-B0

Audio
→ Wav2Vec2

↓

Cross-Modal Attention

↓

Contrastive Fusion

↓

Causal Reasoning

↓

Deepfake Classification
"""

)
# ==========================================================
# ANALYSIS TABS
# ==========================================================

video_tab, advanced_tab = st.tabs(

    [

        "🎥 Video Analysis",

        "🧪 Advanced Analysis",

    ]

)

with video_tab:

    st.header("🎥 Video Analysis")

    st.info(

        "Video pipeline will be connected next."

    )

with advanced_tab:

    st.header("🧪 Advanced Analysis")

    st.caption(

        "Analyze a matching face image and voice audio using the trained Fusion Model."

    )

    col1, col2 = st.columns(2)

    with col1:

        image_file = st.file_uploader(

            "Upload Face Image",

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

    with col2:

        audio_file = st.file_uploader(

            "Upload Voice Audio",

            type=[

                "wav",

                "mp3",

                "flac",

            ],

            key="audio",

        )
        
        if audio_file is not None:
            
            st.audio(audio_file)
            
        analyze_button = st.button(
            
            "🚀 Analyze",
                
            use_container_width=True,
        )

    st.divider()
    
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

        # ---------------------------------------------
        # Save uploaded files temporarily
        # ---------------------------------------------

        temp_dir = Path("temp")

        temp_dir.mkdir(

            exist_ok=True

        )

        image_path = temp_dir / image_file.name

        audio_path = temp_dir / audio_file.name

        image_path.write_bytes(

            image_file.getbuffer()

        )

        audio_path.write_bytes(

            audio_file.getbuffer()

        )
        
        # ---------------------------------------------
        # Run AI
        # ---------------------------------------------

        try:

            with st.spinner(

                "Running PrismShieldAI..."

            ):

                start = time.perf_counter()

                result = engine.predict_from_image_audio(

                    image_path,

                    audio_path,

                )

                elapsed = time.perf_counter() - start

            if result["status"] != "success":

                st.error(

                    result["error"]

                )

            else:

                col1, col2, col3, col4 = st.columns(4)

                with col1:

                    st.metric(

                        "Prediction",

                        result["prediction"],

                    )
                
                with col2:
                    st.metric(
                        
                        "Prediction Probability",
                        
                        f"{result['probability']*100:.2f}%",

                    )

                with col3:

                    st.metric(

                        "Fusion Confidence",

                        f"{result['confidence']*100:.2f}%",

                    )

                with col4:

                    st.metric(

                        "Processing Time",

                        f"{elapsed:.2f}s",

                    )

                st.divider()
                
                if result["prediction"] == "REAL":
                    
                    st.success(
                        "✅ Authentic media detected.\n\n"
                        "No significant multimodal inconsistencies were found."
                    )

                else:

                    st.error(
                        "⚠ Potential deepfake detected.\n\n"
                        "Visual and audio features indicate possible synthetic manipulation."
                    )

        except Exception as e:

            st.error(

                f"Inference failed: {e}"

            )

        finally:

            image_path.unlink(missing_ok=True)

            audio_path.unlink(missing_ok=True)
            
# ==========================================================
# FOOTER
# ==========================================================

st.divider()

st.caption(
    """
PrismShieldAI v2.0

Research Prototype

Causal and Contrastive Multimodal Reasoning for Robust Deep Fake Detection
"""
)