import pymupdf

from src.text_cleaner import clean_text


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract and clean text from a PDF page by page.
    """

    document = pymupdf.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    try:
        for page_index in range(len(document)):
            page = document.load_page(page_index)

            raw_text = page.get_text("text")
            cleaned_text = clean_text(raw_text)

            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": cleaned_text
                }
            )

    finally:
        document.close()

    has_text = any(page["text"] for page in pages)

    if not has_text:
        raise ValueError(
            "No selectable text was found. "
            "This PDF may require OCR."
        )

    return pages