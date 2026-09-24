from src.tokenizer import TextTokenizer


def create_chunks(
    pages: list[dict],
    source_name: str,
    chunk_size: int = 400,
    chunk_overlap: int = 80
) -> list[dict]:
    """
    Divide PDF pages into overlapping token-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError(
            "Chunk size must be greater than zero."
        )

    if chunk_overlap < 0:
        raise ValueError(
            "Chunk overlap cannot be negative."
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be smaller than chunk size."
        )

    tokenizer = TextTokenizer()
    chunks = []

    chunk_id = 1
    step_size = chunk_size - chunk_overlap

    for page in pages:
        page_number = page["page_number"]
        page_text = page["text"]

        tokens = tokenizer.encode(page_text)

        for start_position in range(
            0,
            len(tokens),
            step_size
        ):
            chunk_tokens = tokens[
                start_position:start_position + chunk_size
            ]

            if not chunk_tokens:
                continue

            chunk_text = tokenizer.decode(
                chunk_tokens
            ).strip()

            if not chunk_text:
                continue

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source_name,
                    "page_number": page_number,
                    "text": chunk_text,
                    "token_count": len(chunk_tokens)
                }
            )

            chunk_id += 1

            if start_position + chunk_size >= len(tokens):
                break

    return chunks