import streamlit as st

from src.pdf_loader import extract_text_from_pdf
from src.tokenizer import TextTokenizer


# Create the tokenizer
tokenizer = TextTokenizer()


# Configure the Streamlit page
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


st.title("🔬 AI Research Assistant")

st.write(
    "Upload a research paper to extract, clean and tokenize its text."
)


# PDF upload component
uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)


if uploaded_file is not None:
    try:
        # Read the uploaded PDF as binary data
        pdf_bytes = uploaded_file.getvalue()

        # Extract and clean text page by page
        pages = extract_text_from_pdf(pdf_bytes)

        # Calculate total word count
        total_words = sum(
            len(page["text"].split())
            for page in pages
        )

        # Calculate total token count
        total_tokens = sum(
            tokenizer.count_tokens(page["text"])
            for page in pages
        )

        st.success(
            f"Successfully processed: {uploaded_file.name}"
        )

        # Display document statistics
        column1, column2, column3, column4 = st.columns(4)

        column1.metric(
            "Pages",
            len(pages)
        )

        column2.metric(
            "Total words",
            total_words
        )

        column3.metric(
            "Total tokens",
            total_tokens
        )

        column4.metric(
            "File size",
            f"{uploaded_file.size / 1024:.2f} KB"
        )

        st.subheader("Extracted text")

        # Display the text and token count for every page
        for page in pages:
            page_number = page["page_number"]
            page_text = page["text"]

            page_token_count = tokenizer.count_tokens(
                page_text
            )

            with st.expander(
                f"Page {page_number} — "
                f"{page_token_count} tokens"
            ):
                if page_text:
                    st.write(page_text)

                    st.caption(
                        f"Word count: "
                        f"{len(page_text.split())} | "
                        f"Token count: {page_token_count}"
                    )
                else:
                    st.warning(
                        "No selectable text was found "
                        "on this page."
                    )

    except ValueError as error:
        st.error(str(error))

    except Exception as error:
        st.error(
            f"Could not process the PDF: {error}"
        )

else:
    st.info(
        "Please upload a PDF research paper."
    )