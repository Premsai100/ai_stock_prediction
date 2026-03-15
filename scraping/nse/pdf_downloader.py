import requests
import os
import json
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
     ),
    "Referer": "https://www.nseindia.com/",
    "Accept":  "application/pdf,*/*;q=0.9",
    "Accept-Language":"en-US,en;q=0.9",
    "DNT": "1",
    "Connection":  "keep-alive",
}

def json_to_pdf(data, symbol):
    for i, d in enumerate(data):
        url = d.get("pdf_url")
        from_year = d.get("from_year")
        to_year = d.get("to_year")
        if not url:
            continue
        path = f"stock-ai-system/data/raw/annual_reports/pdfs/{symbol}_{from_year}_{to_year}.pdf"

        response = requests.get(url, headers=HEADERS, stream=True)
        if response.status_code != 200:
            print(f"Failed: {url}")
            continue

        with open(path, "wb") as writer:
            for chunk in response.iter_content(8192):
                writer.write(chunk)

    print(f"Downloaded {path}")

def listing_json():
    files = os.listdir("stock-ai-system/data/raw/annual_reports")
    for file in files:
         symbol = file.replace(".json", "")
         with open(f"stock-ai-system/data/raw/annual_reports/{file}","r",encoding="UTF-8") as reader:
            data = json.load(reader)
            json_to_pdf(data,symbol)

if __name__=="__main__":
     listing_json()