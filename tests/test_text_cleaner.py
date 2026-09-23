from src.text_cleaner import clean_text


def test_removes_extra_spaces():
    raw_text = "Artificial     intelligence"
    result = clean_text(raw_text)

    assert result == "Artificial intelligence"


def test_joins_broken_words():
    raw_text = "machine-\nlearning"
    result = clean_text(raw_text)

    assert result == "machinelearning"


def test_removes_null_characters():
    raw_text = "Artificial\x00intelligence"
    result = clean_text(raw_text)

    assert result == "Artificial intelligence"


def test_removes_extra_blank_lines():
    raw_text = "First paragraph.\n\n\n\nSecond paragraph."
    result = clean_text(raw_text)

    assert result == "First paragraph.\n\nSecond paragraph."

