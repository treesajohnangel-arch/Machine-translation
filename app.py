"""
app.py  —  Manglish → English Translator (Streamlit App)
=========================================================
Model is downloaded automatically from Google Drive on first run.
 
Run:
    streamlit run app.py
"""
 
import os
import zipfile
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
 
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — paste your Google Drive file ID below
# ═══════════════════════════════════════════════════════════════════════════════
 
# How to get FILE_ID:
#   1. In Google Drive, right-click your best_model.zip → Share → Anyone with link
#   2. The share link looks like:
#      https://drive.google.com/file/d/1ABCxyz.../view?usp=sharing
#                                       ^^^^^^^^^^^ this part is the FILE_ID
#   3. Paste it below:
 
GDRIVE_FILE_ID    = "1rgGvHB1Z4WM1lxeFKtu4bXXaXZYF5q_k"
 
MODEL_PATH        = "outputs/best_model"
MAX_INPUT_LENGTH  = 128
MAX_TARGET_LENGTH = 128
INSTRUCTION_PREFIX = 'Translate the following Manglish text to English: '
 
EXAMPLE_SENTENCES = [
    "nee ippo try cheyyu",
    "ivide casual aayi wait cheyyu",
    "adipoli saadhanam aanu idu",
    "enthaa pakshe ningal paranjathu",
    "njan oru minute wait cheyyam",
    "super idea aanu",
    "sherikkum? enik vishwasikkan pattunilla",
    "nee enthina ivideyke vannathu",
]
 
# ═══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD MODEL FROM GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════════════════
 
def download_from_gdrive(file_id: str, dest_path: str):
    """Download a file from Google Drive using gdown."""
    import subprocess
    import sys
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
 
 
def prepare_model():
    """Download and unzip model from Google Drive if not already present."""
    if Path(MODEL_PATH).exists():
        return   # already downloaded, skip
 
    st.info("Downloading model from Google Drive (first run only, ~500MB)...")
    zip_path = "best_model.zip"
 
    # Download
    download_from_gdrive(GDRIVE_FILE_ID, zip_path)
 
    # Unzip
    st.info("Extracting model files...")
    os.makedirs("outputs", exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(".")
 
    # Clean up zip
    os.remove(zip_path)
    st.success("Model ready!")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING  (cached so it only loads once per session)
# ═══════════════════════════════════════════════════════════════════════════════
 
@st.cache_resource(show_spinner=False)
def load_model():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.config.tie_word_embeddings = False
    model.to(device)
    model.eval()
    return tokenizer, model, device
 
 
def translate(text: str, tokenizer, model, device: str) -> str:
    prompt = f'{INSTRUCTION_PREFIX}"{text}"'
    inputs = tokenizer(
        prompt,
        return_tensors = "pt",
        max_length     = MAX_INPUT_LENGTH,
        truncation     = True,
    ).to(device)
 
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length           = MAX_TARGET_LENGTH,
            num_beams            = 4,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  UI
# ═══════════════════════════════════════════════════════════════════════════════
 
def main():
    st.set_page_config(
        page_title = "Manglish → English Translator",
        page_icon  = "🌴",
        layout     = "centered",
    )
 
    st.title("🌴 Manglish → English Translator")
    st.markdown(
        "Translate **Malayalam–English code-mixed (Manglish)** sentences "
        "into natural English using a fine-tuned mT5 model."
    )
    st.divider()
 
    # ── Download model if needed ──────────────────────────────────────────────
    if GDRIVE_FILE_ID == "PASTE_YOUR_FILE_ID_HERE":
        st.error(
            "Please open app.py and set your Google Drive file ID.\n\n"
            "GDRIVE_FILE_ID = 'PASTE_YOUR_FILE_ID_HERE'  ← replace this"
        )
        st.stop()
 
    prepare_model()
 
    # ── Load model ────────────────────────────────────────────────────────────
    with st.spinner("Loading model..."):
        tokenizer, model, device = load_model()
 
    st.success(f"✅ Model loaded | Running on: **{device.upper()}**")
    st.divider()
 
    # ── Input ─────────────────────────────────────────────────────────────────
    st.subheader("✍️ Enter Manglish Text")
 
    selected_example = st.selectbox(
        "Or pick an example:",
        ["(type your own below)"] + EXAMPLE_SENTENCES,
    )
 
    default_text = "" if selected_example == "(type your own below)" else selected_example
    user_input   = st.text_area(
        "Manglish sentence:",
        value       = default_text,
        height      = 100,
        placeholder = "e.g. nee ippo try cheyyu",
    )
 
    translate_btn = st.button("🔁 Translate", type="primary", use_container_width=True)
 
    # ── Output ────────────────────────────────────────────────────────────────
    if translate_btn:
        if not user_input.strip():
            st.warning("Please enter a Manglish sentence first.")
        else:
            with st.spinner("Translating..."):
                result = translate(user_input.strip(), tokenizer, model, device)
 
            st.divider()
            st.subheader("📝 Translation Result")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Manglish (Input)**")
                st.info(user_input.strip())
            with col2:
                st.markdown("**English (Output)**")
                st.success(result)
 
    st.divider()
 
    # ── Batch translation ─────────────────────────────────────────────────────
    with st.expander("📋 Batch Translate (multiple sentences)"):
        st.markdown("Enter one Manglish sentence per line:")
        batch_input = st.text_area(
            "Sentences:", height=150,
            placeholder="nee ippo try cheyyu\nsuper idea aanu\n..."
        )
        batch_btn = st.button("🔁 Translate All", key="batch")
 
        if batch_btn and batch_input.strip():
            sentences = [s.strip() for s in batch_input.strip().splitlines() if s.strip()]
            progress  = st.progress(0)
            for i, sentence in enumerate(sentences):
                translation = translate(sentence, tokenizer, model, device)
                col1, col2  = st.columns(2)
                with col1:
                    st.info(sentence)
                with col2:
                    st.success(translation)
                progress.progress((i + 1) / len(sentences))
 
    st.divider()
    st.markdown(
        "<center><small>Built with mT5 + HuggingFace Transformers · "
        "Fine-tuned on Manglish dataset</small></center>",
        unsafe_allow_html=True,
    )
 
 
if __name__ == "__main__":
    main()
