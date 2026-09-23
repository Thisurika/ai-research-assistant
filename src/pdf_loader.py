import fitz


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from a PDF page by page.

    Args:
        pdf_bytes: PDF file content as bytes.

    Returns:
        A list containing page numbers and extracted text.
    """

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_index in range(len(document)):
        page = document.load_page(page_index)
        text = page.get_text("text")

        pages.append(
            {
                "page_number": page_index + 1,
                "text": text.strip()
            }
        )

    document.close()

    return pages