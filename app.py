"""
app.py  —  Manglish → English Translator (Streamlit App)
=========================================================
Run locally:
    streamlit run app.py

Make sure the model folder is in the same directory:
    outputs/best_model/
        config.json
        tokenizer.model
        pytorch_model.bin  (or model.safetensors)
        tokenizer_config.json
        special_tokens_map.json
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_PATH        = "outputs/best_model"   # folder with your saved model
MAX_INPUT_LENGTH  = 128
MAX_TARGET_LENGTH = 128
INSTRUCTION_PREFIX = 'Translate the following Manglish text to English: '

EMOTION_EMOJI = {
    "happy"    : "😊",
    "sad"      : "😢",
    "angry"    : "😠",
    "neutral"  : "😐",
    "casual"   : "😎",
    "surprised": "😲",
    "anxious"  : "😰",
    "exhausted": "😴",
}

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
#  MODEL LOADING  (cached so it only loads once)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model():
    """Load tokenizer and model from the saved checkpoint — cached after first load."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.config.tie_word_embeddings = False
    model.to(device)
    model.eval()

    return tokenizer, model, device


def translate(text: str, tokenizer, model, device: str) -> str:
    """Translate a Manglish sentence to English."""
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

    # ── Header ────────────────────────────────────────────────────────────────
    st.title("🌴 Manglish → English Translator")
    st.markdown(
        "Translate **Malayalam–English code-mixed (Manglish)** sentences "
        "into natural English using a fine-tuned mT5 model."
    )
    st.divider()

    # ── Load model ────────────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        st.error(
            f"❌ Model folder not found: `{MODEL_PATH}`\n\n"
            "Please make sure you have downloaded the trained model and placed it at:\n"
            f"`{MODEL_PATH}/`\n\n"
            "It should contain: `config.json`, `tokenizer.model`, `pytorch_model.bin`"
        )
        st.stop()

    with st.spinner("Loading model... (this takes ~30 seconds on first run)"):
        tokenizer, model, device = load_model()

    st.success(f"✅ Model loaded | Running on: **{device.upper()}**")
    st.divider()

    # ── Input area ────────────────────────────────────────────────────────────
    st.subheader("✍️ Enter Manglish Text")

    # Example sentence selector
    selected_example = st.selectbox(
        "Or pick an example:",
        ["(type your own below)"] + EXAMPLE_SENTENCES,
    )

    # Text input — prefill with selected example
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
        batch_input = st.text_area("Sentences:", height=150,
                                   placeholder="nee ippo try cheyyu\nsuper idea aanu\n...")
        batch_btn   = st.button("🔁 Translate All", key="batch")

        if batch_btn and batch_input.strip():
            sentences = [s.strip() for s in batch_input.strip().splitlines() if s.strip()]
            results   = []
            progress  = st.progress(0)

            for i, sentence in enumerate(sentences):
                translation = translate(sentence, tokenizer, model, device)
                results.append({"Manglish": sentence, "English": translation})
                progress.progress((i + 1) / len(sentences))

            st.markdown("**Results:**")
            for r in results:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(r["Manglish"])
                with col2:
                    st.success(r["English"])

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<center><small>Built with mT5 + HuggingFace Transformers · "
        "Fine-tuned on Manglish dataset</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
