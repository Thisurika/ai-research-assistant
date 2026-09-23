import streamlit as st


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Assistant")

st.write(
    "Upload research papers and ask questions using "
    "Retrieval-Augmented Generation."
)

uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)

if uploaded_file is not None:
    st.success(f"Successfully uploaded: {uploaded_file.name}")
    st.write(f"File size: {uploaded_file.size} bytes")
else:
    st.info("Please upload a PDF research paper.")