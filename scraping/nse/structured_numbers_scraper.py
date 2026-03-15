import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import json

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def fetch_screener(symbol: str) -> BeautifulSoup | None:
    url = f"https://www.screener.in/company/{symbol.upper()}/consolidated/"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"  [ERROR] {symbol} — HTTP {response.status_code}")
            return None
        return BeautifulSoup(response.text, "html.parser")
    except Exception as e:
        print(f"  [ERROR] {symbol} — {e}")
        return None



def extract_ratios(soup: BeautifulSoup) -> dict:
    ratios = {}
    try:
        for li in soup.select("#top-ratios li"):
            name = li.select_one(".name")
            value = li.select_one(".value")
            if name and value:
                ratios[name.get_text(strip=True)] = value.get_text(strip=True)
    except Exception as e:
        print(f"  [WARN] Ratios extraction failed: {e}")
    return ratios



def extract_table(soup: BeautifulSoup, section_id: str, max_rows: int = 8) -> dict:
    result = {}
    try:
        section = soup.find("section", {"id": section_id})
        if not section:
            return result

        table = section.find("table")
        if not table:
            return result

       
        headers = [th.get_text(strip=True) for th in table.select("thead th")][1:]  
        headers = headers[-max_rows:] 
        for row in table.select("tbody tr"):
            cols = row.find_all("td")
            if not cols:
                continue
            row_name = cols[0].get_text(strip=True)
            values = [c.get_text(strip=True) for c in cols[1:]]
            values = values[-max_rows:]  
            if row_name and any(v.strip() for v in values):
                result[row_name] = dict(zip(headers, values))

    except Exception as e:
        print(f"  [WARN] Table {section_id} extraction failed: {e}")
    return result



def extract_shareholding(soup: BeautifulSoup) -> dict:
    shareholding = {}
    try:
        section = soup.find("section", {"id": "shareholding"})
        if not section:
            return shareholding

        table = section.find("table")
        if not table:
            return shareholding

        headers = [th.get_text(strip=True) for th in table.select("thead th")][1:]

        for row in table.select("tbody tr"):
            cols = row.find_all("td")
            if not cols:
                continue
            name = cols[0].get_text(strip=True)
            values = [c.get_text(strip=True) for c in cols[1:]]
            if name and values:
                shareholding[name] = values[-1] 

    except Exception as e:
        print(f"  [WARN] Shareholding extraction failed: {e}")
    return shareholding


def extract_peers(soup: BeautifulSoup) -> list[dict]:
    peers = []
    try:
        section = soup.find("section", {"id": "peers"})
        if not section:
            return peers

        table = section.find("table")
        if not table:
            return peers

        headers = [th.get_text(strip=True) for th in table.select("thead th")]

        for row in table.select("tbody tr"):
            cols = row.find_all("td")
            if not cols:
                continue
            peer = {headers[i]: cols[i].get_text(strip=True) for i in range(min(len(headers), len(cols)))}
            if peer:
                peers.append(peer)

    except Exception as e:
        print(f"  [WARN] Peers extraction failed: {e}")
    return peers[:6] 


def scrape_screener(symbol: str) -> dict:
    print(f"  Scraping Screener for: {symbol.upper()}")

    soup = fetch_screener(symbol)
    if not soup:
        return {}

    data = {
        "stock":symbol,
        "ratios":        extract_ratios(soup),
        "quarters":      extract_table(soup, "quarters", max_rows=6),
        "profit_loss":   extract_table(soup, "profit-loss", max_rows=5),
        "balance_sheet": extract_table(soup, "balance-sheet", max_rows=5),
        "cash_flow":     extract_table(soup, "cash-flow", max_rows=5),
        "shareholding":  extract_shareholding(soup),
        "peers":         extract_peers(soup),
    }

    return data


