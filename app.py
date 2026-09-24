import streamlit as st

from src.chunker import create_chunks
from src.pdf_loader import extract_text_from_pdf
from src.tokenizer import TextTokenizer


tokenizer = TextTokenizer()


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


st.title("🔬 AI Research Assistant")

st.write(
    "Upload a research paper to extract, clean, tokenize "
    "and divide it into chunks."
)


# Chunk settings
with st.sidebar:
    st.header("Chunk settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        help="Maximum number of tokens in each chunk."
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=250,
        value=80,
        step=10,
        help="Number of repeated tokens between chunks."
    )


uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)


if uploaded_file is not None:
    try:
        if chunk_overlap >= chunk_size:
            st.error(
                "Chunk overlap must be smaller "
                "than chunk size."
            )
            st.stop()

        pdf_bytes = uploaded_file.getvalue()

        pages = extract_text_from_pdf(pdf_bytes)

        chunks = create_chunks(
            pages=pages,
            source_name=uploaded_file.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        total_words = sum(
            len(page["text"].split())
            for page in pages
        )

        total_tokens = sum(
            tokenizer.count_tokens(page["text"])
            for page in pages
        )

        st.success(
            f"Successfully processed: {uploaded_file.name}"
        )

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
            "Chunks",
            len(chunks)
        )

        page_tab, chunk_tab = st.tabs(
            ["Extracted pages", "Generated chunks"]
        )

        with page_tab:
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
                    else:
                        st.warning(
                            "No selectable text was found "
                            "on this page."
                        )

        with chunk_tab:
            st.caption(
                f"Chunk size: {chunk_size} tokens | "
                f"Overlap: {chunk_overlap} tokens"
            )

            for chunk in chunks:
                with st.expander(
                    f"Chunk {chunk['chunk_id']} — "
                    f"Page {chunk['page_number']} — "
                    f"{chunk['token_count']} tokens"
                ):
                    st.write(chunk["text"])

                    st.json(
                        {
                            "chunk_id": chunk["chunk_id"],
                            "source": chunk["source"],
                            "page_number": chunk["page_number"],
                            "token_count": chunk["token_count"]
                        }
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