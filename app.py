"""
app.py  —  Manglish → English Translator (Streamlit App)
"""
 
import os
import zipfile
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path
 
# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
 
GDRIVE_FILE_ID = "1rgGvHB1Z4WM1lxeFKtu4bXXaXZYF5q_k"

BASE_DIR   = Path(__file__).parent
TOKENIZER_PATH  = str(BASE_DIR / "outputs")
MODEL_PATH = str(BASE_DIR / "outputs"/ "best-model")  # hyphen, not underscore

MAX_INPUT_LENGTH   = 128
MAX_TARGET_LENGTH  = 128
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
    import subprocess, sys
    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest_path, quiet=False)
 
 
def prepare_model():
    if Path(MODEL_PATH).exists():
        return
 
    st.info("Downloading model from Google Drive (first run only, ~500MB)...")
    zip_path = str(BASE_DIR / "best_model.zip")
    download_from_gdrive(GDRIVE_FILE_ID, zip_path)
 
    st.info("Extracting model files...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(str(BASE_DIR))
 
    if os.path.exists(zip_path):
        os.remove(zip_path)
    st.success("Model ready!")
 
 
# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════
 
@st.cache_resource(show_spinner=False)
def load_model():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, local_files_only=True)
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
 
    if GDRIVE_FILE_ID == "PASTE_YOUR_FILE_ID_HERE":
        st.error("Please set your Google Drive file ID in app.py")
        st.stop()
 
    prepare_model()

    # TEMPORARY DEBUG
    import os
    best_model_dir = BASE_DIR / "outputs" / "best-model"
    if best_model_dir.exists():
        st.write("best-model/ contents:", os.listdir(str(best_model_dir)))
        for item in os.listdir(str(best_model_dir)):
            subpath = best_model_dir / item
            if subpath.is_dir():
                st.write(f"  {item}/ contents:", os.listdir(str(subpath)))
    st.stop()
 
    with st.spinner("Loading model..."):
        tokenizer, model, device = load_model()
 
    st.success(f"✅ Model loaded | Running on: **{device.upper()}**")
    st.divider()
 
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
