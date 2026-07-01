"""
pdf_utils.py — Helper functions for PDF processing
Mainly for locating text on a PDF page and generating highlighting annotations
for streamlit-pdf-viewer.
"""

import logging
import fitz

logger = logging.getLogger(__name__)


def get_pdf_annotations(file_path: str, page_num: int, search_text: str) -> list[dict]:
    """
    Search for text on a specific PDF page and return bounding box annotations.

    Args:
        file_path: Path to the PDF file.
        page_num: 1-indexed page number.
        search_text: The text string to search for and highlight.

    Returns:
        A list of dictionary annotations compatible with streamlit-pdf-viewer.
    """
    annotations = []

    try:
        pdf = fitz.open(file_path)
        page_index = page_num - 1

        if page_index < 0 or page_index >= len(pdf):
            pdf.close()
            return annotations

        page = pdf[page_index]

        search_lines = [line.strip() for line in search_text.split('\n') if line.strip()]

        if not search_lines:
             search_lines = [search_text]

        seen_rects = set()

        for line in search_lines:
            rects = page.search_for(line)

            for rect in rects:
                rect_key = (round(rect.x0), round(rect.y0), round(rect.width), round(rect.height))
                if rect_key not in seen_rects:
                    seen_rects.add(rect_key)

                    annotations.append({
                        "page": page_num,
                        "x": rect.x0,
                        "y": rect.y0,
                        "width": rect.width,
                        "height": rect.height,
                        "color": "red"
                    })

        pdf.close()
    except Exception as e:
        logger.error("Error getting PDF annotations: %s", e)

    return annotations
