"""
formula_extractor.py — Regex-based extraction of LaTeX and ASCII math formulas.
Returns formula text + surrounding context for semantic search.
"""

import re
import logging

logger = logging.getLogger(__name__)

# Patterns for formula detection
FORMULA_PATTERNS = [
    # LaTeX display math: $$...$$
    re.compile(r'\$\$(.+?)\$\$', re.DOTALL),
    # LaTeX inline math: $...$
    re.compile(r'(?<!\$)\$([^\$]+?)\$(?!\$)'),
    # LaTeX commands: \frac{}{}, \int, \sum, etc.
    re.compile(r'\\(?:frac|int|sum|prod|lim|sqrt|partial|nabla|infty)\{[^}]*\}(?:\{[^}]*\})?'),
    # ASCII math: E = mc^2, F = ma, PV = nRT style
    re.compile(r'[A-Z][a-z]?\s*=\s*[A-Za-z0-9\^\*\+\-\/\(\)\s]{2,}'),
    # Equation-like patterns with common operators
    re.compile(r'(?:^|\s)([A-Za-z_]+\s*[=<>≥≤]+\s*[A-Za-z0-9\^\*\+\-\/\(\)\s\.]+)', re.MULTILINE),
]


def extract_formulas(text: str, source: str = "", page: int = 0) -> list[dict]:
    """
    Extract formulas from text and return them with surrounding context.

    Args:
        text: The document text to search for formulas
        source: Source filename for metadata
        page: Page number for metadata

    Returns:
        List of dicts: {"formula": str, "context": str, "source": str, "page": int}
    """
    formulas = []
    seen = set()

    lines = text.split('\n')

    for pattern in FORMULA_PATTERNS:
        for match in pattern.finditer(text):
            formula = match.group().strip()

            # Skip very short matches (likely false positives)
            if len(formula) < 4:
                continue

            # Deduplicate
            if formula in seen:
                continue
            seen.add(formula)

            # Get surrounding context (2 lines before and after)
            match_pos = match.start()
            text_before = text[:match_pos]
            line_num = text_before.count('\n')

            start_line = max(0, line_num - 2)
            end_line = min(len(lines), line_num + 3)
            context = '\n'.join(lines[start_line:end_line])

            formulas.append({
                "formula": formula,
                "context": context,
                "source": source,
                "page": page,
            })

    logger.debug("Extracted %d formulas from %s page %d", len(formulas), source, page)
    return formulas


def is_formula_query(question: str) -> bool:
    """Detect if a question is asking about formulas or equations."""
    keywords = ["formula", "equation", "derive", "derivation", "proof",
                "calculate", "compute", "expression for", "mathematical"]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)
