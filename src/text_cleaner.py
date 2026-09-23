import re


def clean_text(text: str) -> str:
    """
    Clean text extracted from a PDF.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned text.
    """

    # Remove null characters
    text = text.replace("\x00", " ")

    # Join words broken by a hyphen and newline
    # Example: "machine-\nlearning" becomes "machinelearning"
    text = re.sub(r"-\s*\n\s*", "", text)

    # Replace single line breaks with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Replace repeated spaces and tabs with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce three or more newlines to two
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()