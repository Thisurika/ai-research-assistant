import streamlit as st

from src.chunker import create_chunks
from src.embedding_model import EmbeddingModel
from src.pdf_loader import extract_text_from_pdf
from src.tokenizer import TextTokenizer
from src.vector_store import VectorStore


tokenizer = TextTokenizer()


@st.cache_resource
def load_embedding_model() -> EmbeddingModel:
    """
    Load the embedding model once and reuse it.
    """

    return EmbeddingModel()


@st.cache_resource
def load_vector_store() -> VectorStore:
    """
    Create one persistent ChromaDB connection.
    """

    return VectorStore()


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


st.title("🔬 AI Research Assistant")

st.write(
    "Upload a research paper to extract, clean, tokenize, "
    "chunk, embed and store its content in ChromaDB."
)


# Chunk configuration
with st.sidebar:
    st.header("Chunk Settings")

    chunk_size = st.slider(
        "Chunk size",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        help="Maximum number of tokens in one chunk."
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=250,
        value=80,
        step=10,
        help=(
            "Number of tokens repeated between "
            "neighbouring chunks."
        )
    )

    st.caption(
        "Recommended starting values: "
        "400 tokens with an 80-token overlap."
    )


uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)


if uploaded_file is not None:
    try:
        # Validate chunk settings
        if chunk_overlap >= chunk_size:
            st.error(
                "Chunk overlap must be smaller than chunk size."
            )
            st.stop()

        # Convert the uploaded PDF into bytes
        pdf_bytes = uploaded_file.getvalue()

        # Extract and clean text page by page
        pages = extract_text_from_pdf(pdf_bytes)

        # Divide the pages into token-based chunks
        chunks = create_chunks(
            pages=pages,
            source_name=uploaded_file.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        # Calculate statistics
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

        # Display statistics
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
            "Generated chunks",
            len(chunks)
        )

        # Application tabs
        page_tab, chunk_tab, database_tab = st.tabs(
            [
                "Extracted Pages",
                "Generated Chunks",
                "Vector Database"
            ]
        )

        # Page display
        with page_tab:
            st.subheader("Extracted and cleaned text")

            for page in pages:
                page_number = page["page_number"]
                page_text = page["text"]

                page_token_count = (
                    tokenizer.count_tokens(page_text)
                )

                with st.expander(
                    f"Page {page_number} — "
                    f"{page_token_count} tokens"
                ):
                    if page_text:
                        st.write(page_text)

                        st.caption(
                            f"Words: {len(page_text.split())} | "
                            f"Tokens: {page_token_count}"
                        )
                    else:
                        st.warning(
                            "No selectable text was found "
                            "on this page."
                        )

        # Chunk display
        with chunk_tab:
            st.subheader("Token-based chunks")

            st.caption(
                f"Chunk size: {chunk_size} tokens | "
                f"Overlap: {chunk_overlap} tokens"
            )

            if not chunks:
                st.warning(
                    "No chunks were generated."
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
                            "page_number": (
                                chunk["page_number"]
                            ),
                            "token_count": (
                                chunk["token_count"]
                            )
                        }
                    )

        # Embedding and ChromaDB storage
        with database_tab:
            st.subheader(
                "Embeddings and ChromaDB storage"
            )

            st.write(
                "Generate a semantic embedding for every "
                "chunk and store the vectors, text and "
                "metadata in ChromaDB."
            )

            vector_store = load_vector_store()

            stored_count_placeholder = st.empty()

            stored_count_placeholder.metric(
                "Chunks currently stored",
                vector_store.count()
            )

            if not chunks:
                st.warning(
                    "No chunks are available for embedding."
                )

            elif st.button(
                "Generate and Store Embeddings",
                type="primary"
            ):
                try:
                    with st.spinner(
                        "Loading the embedding model, "
                        "generating vectors and saving "
                        "them to ChromaDB..."
                    ):
                        embedding_model = (
                            load_embedding_model()
                        )

                        chunk_embeddings = (
                            embedding_model.embed_chunks(
                                chunks
                            )
                        )

                        stored_chunks = (
                            vector_store.add_chunks(
                                chunks=chunks,
                                embeddings=chunk_embeddings
                            )
                        )

                    st.success(
                        f"{stored_chunks} chunks were "
                        "stored successfully."
                    )

                    # Update the displayed database count
                    stored_count_placeholder.metric(
                        "Chunks currently stored",
                        vector_store.count()
                    )

                    result_column1, result_column2 = (
                        st.columns(2)
                    )

                    result_column1.metric(
                        "Generated embeddings",
                        chunk_embeddings.shape[0]
                    )

                    result_column2.metric(
                        "Embedding dimensions",
                        chunk_embeddings.shape[1]
                    )

                    st.write(
                        "First 10 values from the first "
                        "chunk embedding:"
                    )

                    embedding_preview = (
                        chunk_embeddings[0][:10].tolist()
                    )

                    st.code(
                        str(embedding_preview),
                        language="python"
                    )

                    st.info(
                        "Every chunk now has a semantic "
                        "vector, original text and metadata "
                        "stored in the vector database."
                    )

                except Exception as database_error:
                    st.error(
                        "Could not generate or store "
                        f"embeddings: {database_error}"
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