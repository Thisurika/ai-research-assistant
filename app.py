import streamlit as st
from dotenv import load_dotenv

from src.chunker import create_chunks
from src.embedding_model import EmbeddingModel
from src.llm_service import GeminiService
from src.pdf_loader import extract_text_from_pdf
from src.retriever import SemanticRetriever
from src.tokenizer import TextTokenizer
from src.vector_store import VectorStore


# Load environment variables from .env
load_dotenv()


tokenizer = TextTokenizer()


@st.cache_resource
def load_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource
def load_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource
def load_llm_service() -> GeminiService:
    return GeminiService()


st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide"
)


st.title("🔬 AI Research Assistant")

st.write(
    "Upload a research paper, create a searchable knowledge "
    "base and ask questions using RAG."
)


# Connect to ChromaDB
vector_store = load_vector_store()


# Sidebar settings
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
            "Number of repeated tokens between "
            "neighbouring chunks."
        )
    )

    st.caption(
        "Recommended: 400 tokens with an "
        "80-token overlap."
    )

    st.divider()

    st.subheader("Database Information")

    st.metric(
        "Stored chunks",
        vector_store.count()
    )


# PDF uploader
uploaded_file = st.file_uploader(
    "Upload a research paper",
    type=["pdf"]
)


if uploaded_file is not None:
    try:
        # Validate settings
        if chunk_overlap >= chunk_size:
            st.error(
                "Chunk overlap must be smaller "
                "than chunk size."
            )
            st.stop()

        # Read PDF bytes
        pdf_bytes = uploaded_file.getvalue()

        # Extract and clean PDF text
        pages = extract_text_from_pdf(pdf_bytes)

        # Create token-based chunks
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

        # Document statistics
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
        (
            page_tab,
            chunk_tab,
            database_tab,
            search_tab,
            rag_tab
        ) = st.tabs(
            [
                "Extracted Pages",
                "Generated Chunks",
                "Vector Database",
                "Semantic Search",
                "Ask AI"
            ]
        )

        # --------------------------------------------------
        # Extracted pages
        # --------------------------------------------------
        with page_tab:
            st.subheader(
                "Extracted and Cleaned Text"
            )

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
                            f"Words: "
                            f"{len(page_text.split())} | "
                            f"Tokens: {page_token_count}"
                        )
                    else:
                        st.warning(
                            "No selectable text was found "
                            "on this page."
                        )

        # --------------------------------------------------
        # Generated chunks
        # --------------------------------------------------
        with chunk_tab:
            st.subheader("Token-Based Chunks")

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

        # --------------------------------------------------
        # Vector database
        # --------------------------------------------------
        with database_tab:
            st.subheader(
                "Embeddings and ChromaDB Storage"
            )

            st.write(
                "Generate semantic embeddings and store "
                "the vectors, text and metadata in ChromaDB."
            )

            stored_count_placeholder = st.empty()

            stored_count_placeholder.metric(
                "Chunks currently stored",
                vector_store.count()
            )

            if not chunks:
                st.warning(
                    "No chunks are available."
                )

            elif st.button(
                "Generate and Store Embeddings",
                type="primary",
                key="store_embeddings"
            ):
                try:
                    with st.spinner(
                        "Generating embeddings and storing "
                        "them in ChromaDB..."
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
                        "embedding:"
                    )

                    st.code(
                        str(
                            chunk_embeddings[0][
                                :10
                            ].tolist()
                        ),
                        language="python"
                    )

                except Exception as database_error:
                    st.error(
                        "Could not store embeddings: "
                        f"{database_error}"
                    )

        # --------------------------------------------------
        # Semantic search
        # --------------------------------------------------
        with search_tab:
            st.subheader("Semantic Search")

            st.write(
                "Search for document passages based "
                "on meaning."
            )

            stored_count = vector_store.count()

            st.metric(
                "Available searchable chunks",
                stored_count
            )

            if stored_count == 0:
                st.warning(
                    "Store document embeddings before "
                    "using semantic search."
                )

            else:
                search_question = st.text_input(
                    "Enter your search question",
                    placeholder=(
                        "What dataset was used?"
                    ),
                    key="semantic_question"
                )

                search_result_count = st.slider(
                    "Number of search results",
                    min_value=1,
                    max_value=min(10, stored_count),
                    value=min(3, stored_count),
                    key="semantic_result_count"
                )

                if st.button(
                    "Search Documents",
                    type="primary",
                    key="semantic_search"
                ):
                    try:
                        with st.spinner(
                            "Searching for relevant chunks..."
                        ):
                            embedding_model = (
                                load_embedding_model()
                            )

                            retriever = SemanticRetriever(
                                embedding_model=(
                                    embedding_model
                                ),
                                vector_store=vector_store
                            )

                            retrieved_chunks = (
                                retriever.retrieve(
                                    question=search_question,
                                    number_of_results=(
                                        search_result_count
                                    )
                                )
                            )

                        st.success(
                            f"Found "
                            f"{len(retrieved_chunks)} "
                            "relevant chunks."
                        )

                        for result in retrieved_chunks:
                            st.markdown(
                                f"### Result "
                                f"{result['rank']}"
                            )

                            source_column, page_column = (
                                st.columns(2)
                            )

                            source_column.write(
                                f"Source: "
                                f"{result['source']}"
                            )

                            page_column.write(
                                f"Page: "
                                f"{result['page_number']}"
                            )

                            st.write(result["text"])

                            st.caption(
                                f"Chunk ID: "
                                f"{result['chunk_id']} | "
                                f"Tokens: "
                                f"{result['token_count']} | "
                                f"Distance: "
                                f"{result['distance']:.4f}"
                            )

                            st.divider()

                    except ValueError as search_error:
                        st.error(str(search_error))

                    except Exception as search_error:
                        st.error(
                            f"Semantic search failed: "
                            f"{search_error}"
                        )

        # --------------------------------------------------
        # RAG answer generation
        # --------------------------------------------------
        with rag_tab:
            st.subheader(
                "Ask the AI Research Assistant"
            )

            st.write(
                "The system retrieves relevant evidence "
                "from ChromaDB and sends that evidence "
                "to Gemini to generate an answer."
            )

            stored_count = vector_store.count()

            if stored_count == 0:
                st.warning(
                    "The vector database is empty. "
                    "Generate and store embeddings first."
                )

            else:
                rag_question = st.text_area(
                    "Ask a question",
                    placeholder=(
                        "What methodology was used "
                        "in this research?"
                    ),
                    key="rag_question"
                )

                context_chunk_count = st.slider(
                    "Number of context chunks",
                    min_value=1,
                    max_value=min(10, stored_count),
                    value=min(4, stored_count),
                    key="rag_context_count"
                )

                if st.button(
                    "Generate Answer",
                    type="primary",
                    key="generate_answer"
                ):
                    try:
                        with st.spinner(
                            "Retrieving evidence and "
                            "generating an answer..."
                        ):
                            # Load the embedding model
                            embedding_model = (
                                load_embedding_model()
                            )

                            # Retrieve related chunks
                            retriever = SemanticRetriever(
                                embedding_model=(
                                    embedding_model
                                ),
                                vector_store=vector_store
                            )

                            retrieved_chunks = (
                                retriever.retrieve(
                                    question=rag_question,
                                    number_of_results=(
                                        context_chunk_count
                                    )
                                )
                            )

                            # Load Gemini
                            llm_service = (
                                load_llm_service()
                            )

                            # Generate the grounded answer
                            answer = (
                                llm_service.generate_answer(
                                    question=rag_question,
                                    retrieved_chunks=(
                                        retrieved_chunks
                                    )
                                )
                            )

                        st.subheader("Answer")

                        st.write(answer)

                        st.subheader(
                            "Retrieved Evidence"
                        )

                        for result in retrieved_chunks:
                            with st.expander(
                                f"Source: "
                                f"{result['source']} — "
                                f"Page "
                                f"{result['page_number']}"
                            ):
                                st.write(
                                    result["text"]
                                )

                                st.caption(
                                    f"Retrieval rank: "
                                    f"{result['rank']} | "
                                    f"Chunk ID: "
                                    f"{result['chunk_id']} | "
                                    f"Distance: "
                                    f"{result['distance']:.4f}"
                                )

                    except ValueError as rag_error:
                        st.error(str(rag_error))

                    except Exception as rag_error:
                        st.error(
                            "Could not generate the answer: "
                            f"{rag_error}"
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

    stored_count = vector_store.count()

    if stored_count > 0:
        st.caption(
            f"ChromaDB currently contains "
            f"{stored_count} searchable chunks."
        )