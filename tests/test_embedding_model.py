from src.embedding_model import EmbeddingModel


def test_embedding_shape():
    model = EmbeddingModel()

    texts = [
        "Machine learning",
        "Artificial intelligence"
    ]

    embeddings = model.embed_texts(texts)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 384


def test_one_embedding_per_chunk():
    model = EmbeddingModel()

    chunks = [
        {
            "text": "First research chunk."
        },
        {
            "text": "Second research chunk."
        }
    ]

    embeddings = model.embed_chunks(chunks)

    assert len(embeddings) == len(chunks)