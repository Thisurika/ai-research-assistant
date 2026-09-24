from src.tokenizer import TextTokenizer


def test_token_count_is_greater_than_zero():
    tokenizer = TextTokenizer()

    count = tokenizer.count_tokens(
        "Retrieval-Augmented Generation"
    )

    assert count > 0


def test_encode_returns_token_ids():
    tokenizer = TextTokenizer()

    tokens = tokenizer.encode(
        "Artificial intelligence"
    )

    assert isinstance(tokens, list)
    assert all(isinstance(token, int) for token in tokens)


def test_decode_reconstructs_original_text():
    tokenizer = TextTokenizer()

    original_text = "Machine learning is useful."
    tokens = tokenizer.encode(original_text)
    reconstructed_text = tokenizer.decode(tokens)

    assert reconstructed_text == original_text


def test_empty_text_has_zero_tokens():
    tokenizer = TextTokenizer()

    assert tokenizer.count_tokens("") == 0