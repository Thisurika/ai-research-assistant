import streamlit as st

from src.pdf_loader import extract_text_from_pdf


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 AI Research Assistant")

st.write(
    "Upload a research paper to extract and examine its text."
)

uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)

if uploaded_file is not None:
    try:
        pdf_bytes = uploaded_file.getvalue()

        pages = extract_text_from_pdf(pdf_bytes)

        total_words = sum(
            len(page["text"].split())
            for page in pages
        )

        st.success(
            f"Successfully processed: {uploaded_file.name}"
        )

        column1, column2, column3 = st.columns(3)

        column1.metric(
            "Pages",
            len(pages)
        )

        column2.metric(
            "Total words",
            total_words
        )

        column3.metric(
            "File size",
            f"{uploaded_file.size / 1024:.2f} KB"
        )

        st.subheader("Extracted text")

        for page in pages:
            page_number = page["page_number"]
            page_text = page["text"]

            with st.expander(f"Page {page_number}"):
                if page_text:
                    st.write(page_text)
                else:
                    st.warning(
                        "No selectable text was found on this page."
                    )

    except Exception as error:
        st.error(f"Could not process the PDF: {error}")

else:
    st.info("Please upload a PDF research paper.")