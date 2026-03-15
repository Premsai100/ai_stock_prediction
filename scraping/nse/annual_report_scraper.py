
from playwright.sync_api import sync_playwright
import json

def scrape_annual_reports(symbol) -> list[dict]:
    url = (
        f"https://www.nseindia.com/companies-listing/"
        f"corporate-filings-annual-reports?symbol={symbol}&tabIndex=equity"
    )
    results = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False,channel='chrome')
        context = browser.new_context(
        user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
             ),
             extra_http_headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.nseindia.com/",
            },
        )
        page = context.new_page()

        page.goto(url, wait_until="networkidle", timeout=60_000)
        wrapper_id = "AREquityWrapper"
        try:
            page.wait_for_selector(f"#{wrapper_id}:not(.hide)", timeout=15_000)
        except Exception:
             page.wait_for_timeout(5_000)
        try:
            page.wait_for_selector("#CFannualreportEquityTable tbody tr", timeout=15_000)
        except Exception:
            print("[!] Table rows not found — the page may require manual CAPTCHA or JS execution.")
            browser.close()
            return results
        rows = page.query_selector_all("#CFannualreportEquityTable tbody tr")
        print(f"[*] Found {len(rows)} row(s) in the table.")
        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) < 4:
                continue

            company  = cells[0].inner_text().strip()
            from_yr = cells[1].inner_text().strip()
            to_yr  = cells[2].inner_text().strip()
            anchor = cells[3].query_selector("a")
            pdf_url = anchor.get_attribute("href") if anchor else None

            sub_type = cells[4].inner_text().strip() if len(cells) > 4 else ""

            results.append({
                "company": company,
                 "from_year": from_yr,
                 "to_year": to_yr,
                 "pdf_url": pdf_url,
                "submission_type": sub_type,
            })

        browser.close()
    return results




def save_results(data: list[dict], symbol: str) -> None:
    if not data:
        print("[!] No data to save.")
        return

    json_path = f"stock-ai-system/data/raw/annual_reports/{symbol}_annual_reports.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[+] Saved JSON  → {json_path}")


if __name__ == "__main__":
    with open("stock-ai-system/scraping/nse/symbols.json","r") as reader:
            data = json.load(reader)

    for stocks in data.get('nse_top_stocks'):
        symbol = stocks.get("symbol")
        reports = scrape_annual_reports(symbol)
        if reports:
            save_results(reports, symbol)
        else:
             print("[!] No reports scraped. Possible causes:")