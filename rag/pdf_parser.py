import os
import re
from langchain_community.document_loaders import PyPDFLoader
from text_cleaner import clean_text


def extract_metadata(path):
    filename = os.path.basename(path)
    print(filename)
    pattern = r"([A-Z0-9]+)_annual_reports_(\d{4})_(\d{4})\.pdf"

    match = re.match(pattern, filename)

    if not match:
        raise ValueError(f"Invalid filename format: {filename}")

    symbol = match.group(1)
    from_year = match.group(2)
    to_year = match.group(3)

    return symbol, from_year, to_year


def load_pdf(path):

    symbol, from_year, to_year = extract_metadata(path)

    loader = PyPDFLoader(path)
    pages = loader.load()

    text = ""

    for page in pages:
        text += page.page_content

    cleaned = clean_text(text)

    return {
        "symbol": symbol,
        "from_year": from_year,
        "to_year": to_year,
        "text": cleaned
    }