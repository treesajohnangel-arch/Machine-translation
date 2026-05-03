"""
app.py — Streamlit UI for Manglish → English translation.
Downloads the fine-tuned T5 model from Google Drive on first run.
"""

import os
import time
import torch
import gdown
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ──────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Manglish → English Translator",
    page_icon="🌴",
    layout="centered",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_PATH    = "./manglish_t5_model"
GDRIVE_FOLDER = "1N4045lmwqHBFjjUoUqQRZdoyFlePMC00"
PREFIX        = "translate manglish to english: "
MAX_LENGTH    = 128
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

EXAMPLES = [
    "njan nale varaam",
    "ivide casual aayi wait cheyyu",
    "aval vegam kazhichu",
    "ningal eppo varum?",
    "ente phone edukku",
    "nee ippo try cheyyu",
    "avide poyi nokkiyal manasilakum",
    "oru coffee kudikkanam",
]

# ──────────────────────────────────────────────
# Model loading (cached across sessions)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    # Check for model.safetensors specifically (not just the folder)
    model_file = os.path.join(MODEL_PATH, "model.safetensors")
    if not os.path.exists(model_file):
        st.info("⬇️ Downloading model from Google Drive (first run only, ~1 min)...")
        gdown.download_folder(
            id=GDRIVE_FOLDER,
            output=MODEL_PATH,
            quiet=False,
            use_cookies=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.eval()
    model.to(DEVICE)
    return tokenizer, model


def translate(text: str, tokenizer, model) -> str:
    inp = tokenizer(
        PREFIX + text.strip(),
        return_tensors="pt",
        max_length=MAX_LENGTH,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inp,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)


# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    .header-box {
        background: linear-gradient(135deg, #1a6e2e 0%, #2d9e4a 50%, #38b56a 100%);
        border-radius: 16px;
        padding: 2rem 1.5rem 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .header-box h1 { color: white; font-size: 2rem; margin: 0; }
    .header-box p  { color: #d0f0dc; margin: .4rem 0 0; }

    .result-card {
        background: #f0fdf4;
        border-left: 5px solid #22c55e;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: .8rem;
    }
    .result-card .label { font-size: .75rem; color: #666; font-weight: 600;
                          text-transform: uppercase; letter-spacing: .05em; }
    .result-card .translation { font-size: 1.3rem; color: #14532d;
                                 font-weight: 700; margin-top: .3rem; }

    div[data-testid="column"] button {
        border-radius: 20px !important;
        font-size: .82rem !important;
    }

    .footer { text-align:center; color:#999; font-size:.78rem; margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown(
    """
    <div class="header-box">
        <h1>🌴 Manglish → English</h1>
        <p>Translate Malayalam-English (Manglish) sentences instantly using a fine-tuned T5 model</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    num_beams = st.slider("Beam width", 1, 8, 4,
                          help="Higher = better quality, slower speed")
    st.divider()
    st.markdown(
        "**About Manglish**\n\n"
        "Manglish is an informal mix of Malayalam and English widely used in Kerala. "
        "This model was fine-tuned on a curated dataset of 5,000+ Manglish sentences "
        "covering categories like *casual chat*, *food*, *travel*, *workplace*, and more."
    )
    st.divider()
    st.markdown("Built with 🤗 Transformers + Streamlit")

# ──────────────────────────────────────────────
# Load model
# ──────────────────────────────────────────────
with st.spinner("Loading model … (first launch downloads ~240MB from Google Drive)"):
    try:
        tokenizer, model_obj = load_model()
        model_ready = True
        st.success(f"✅ Model loaded — running on **{DEVICE.upper()}**", icon="🤖")
    except Exception as e:
        model_ready = False
        st.error(f"❌ Could not load model:\n\n```\n{e}\n```")

# ──────────────────────────────────────────────
# Input area
# ──────────────────────────────────────────────
st.subheader("📝 Enter a Manglish sentence")

st.caption("Try an example:")
cols = st.columns(4)
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

for i, ex in enumerate(EXAMPLES):
    with cols[i % 4]:
        if st.button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.input_text = ex

user_input = st.text_area(
    label="Manglish text",
    value=st.session_state.input_text,
    height=120,
    placeholder="e.g.  njan nale varaam   /   oru coffee kudikkanam",
    label_visibility="collapsed",
)

translate_btn = st.button(
    "🔄 Translate",
    type="primary",
    disabled=not model_ready,
    use_container_width=True,
)

# ──────────────────────────────────────────────
# Translation
# ──────────────────────────────────────────────
if translate_btn and user_input.strip():
    with st.spinner("Translating …"):
        t0 = time.time()
        result = translate(user_input.strip(), tokenizer, model_obj)
        elapsed = time.time() - t0

    st.markdown(
        f"""
        <div class="result-card">
            <div class="label">English Translation</div>
            <div class="translation">{result}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"⏱ Translated in {elapsed:.2f}s using beam width {num_beams}")

elif translate_btn and not user_input.strip():
    st.warning("Please enter a Manglish sentence first.")

# ──────────────────────────────────────────────
# Batch translation
# ──────────────────────────────────────────────
if model_ready:
    with st.expander("📋 Batch translate multiple sentences"):
        batch_input = st.text_area(
            "One Manglish sentence per line",
            height=150,
            placeholder="njan nale varaam\naval vegam kazhichu\nente phone edukku",
        )
        if st.button("Translate All", key="batch_btn"):
            sentences = [s.strip() for s in batch_input.splitlines() if s.strip()]
            if sentences:
                results = []
                progress = st.progress(0)
                for idx, sent in enumerate(sentences):
                    eng = translate(sent, tokenizer, model_obj)
                    results.append({"Manglish": sent, "English": eng})
                    progress.progress((idx + 1) / len(sentences))

                import pandas as pd
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Download CSV", csv,
                                   "translations.csv", "text/csv")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    '<div class="footer">Model: T5-small fine-tuned on 5,000+ Manglish sentences &nbsp;|&nbsp; '
    'Categories: casual_chat · food · travel · workplace · family · more</div>',
    unsafe_allow_html=True,
)
