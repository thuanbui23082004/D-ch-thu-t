import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Mã ngôn ngữ NLLB (3 ngôn ngữ chính)
lang_codes = {
    "Tiếng Việt": "vie_Latn",
    "Tiếng Anh": "eng_Latn",
    "Tiếng Pháp": "fra_Latn"
}

def translate(text, src_lang, tgt_lang, tokenizer, model):
    tokenizer.src_lang = lang_codes[src_lang]
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    bos_token_id = tokenizer.convert_tokens_to_ids(lang_codes[tgt_lang])
    generated = model.generate(**encoded, forced_bos_token_id=bos_token_id)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

# Streamlit UI
st.title("🌐 Dịch Anh - Việt - Pháp bằng NLLB")

with st.expander("Chọn ngôn ngữ dịch"):
    src = st.selectbox("Ngôn ngữ nguồn:", list(lang_codes.keys()))
    tgt = st.selectbox("Ngôn ngữ đích:", [l for l in lang_codes.keys() if l != src])

tokenizer, model = load_model()

uploaded_file = st.file_uploader("Hoặc tải file .txt lên để dịch", type=["txt"])
translated_text = None

if uploaded_file is not None:
    raw_text = uploaded_file.read().decode("utf-8")
    st.markdown(f"**Nội dung file đã tải lên ({len(raw_text)} ký tự):**")
    st.write(raw_text)
    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col2:
        if st.button("✨ Dịch file", key="btn_translate_file"):
            with st.spinner("Đang dịch file..."):
                translated_text = translate(raw_text, src, tgt, tokenizer, model)
                st.success("✅ Dịch file thành công!")

else:
    text = st.text_area(f"Nhập văn bản tiếng {src}:", height=150)
    col1, col2, col3 = st.columns([1.3, 1, 1])
    with col2:
        if st.button("✨ Dịch văn bản", key="btn_translate_text"):
            if not text.strip():
                st.warning("⚠️ Vui lòng nhập văn bản để dịch.")
            else:
                with st.spinner("Đang dịch văn bản..."):
                    translated_text = translate(text, src, tgt, tokenizer, model)
                    st.success("✅ Dịch văn bản thành công!")

if translated_text:
    st.markdown(f"### Kết quả dịch sang tiếng {tgt}:")
    st.text_area("Kết quả dịch", value=translated_text, height=150)
    st.download_button(
        label="⬇️ Tải kết quả dịch về",
        data=translated_text,
        file_name=f"translated_{src}_to_{tgt}.txt",
        mime="text/plain"
    )
