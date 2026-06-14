"""
Scrape Zerodha Varsity finance content and save raw text chunks to data/raw/.

Run from repo root:
    python foundry/desi-finance-advisor/phase1_data_prep/collect_data.py
"""

import json
import os
import time

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
BASE_URL = "https://zerodha.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

# Varsity modules most relevant to personal finance
MODULE_SLUGS = [
    "personal-finance",
    "mutual-funds",
    "markets-and-taxation",
    "introduction-to-stock-markets",
]

# Known chapter URLs as fallback if module discovery fails
FALLBACK_CHAPTERS = [
    "/varsity/chapter/personal-finance-ladder/",
    "/varsity/chapter/the-5-point-checklist/",
    "/varsity/chapter/insurance-basics/",
    "/varsity/chapter/understanding-sips/",
    "/varsity/chapter/what-are-mutual-funds/",
    "/varsity/chapter/types-of-mutual-fund-schemes/",
    "/varsity/chapter/expense-ratio-and-other-charges/",
    "/varsity/chapter/direct-vs-regular-mutual-funds/",
    "/varsity/chapter/all-about-epf/",
    "/varsity/chapter/public-provident-fund/",
    "/varsity/chapter/national-pension-system/",
    "/varsity/chapter/income-tax-for-individuals/",
    "/varsity/chapter/capital-gains-tax/",
    "/varsity/chapter/tax-on-mf-returns/",
    "/varsity/chapter/emi-and-home-loan-basics/",
    "/varsity/chapter/how-to-open-a-demat-account/",
    "/varsity/chapter/understanding-index-funds/",
    "/varsity/chapter/sovereign-gold-bonds/",
    "/varsity/chapter/real-estate-as-an-investment/",
    "/varsity/chapter/old-vs-new-tax-regime/",
]


def get_chapter_links(module_slug: str) -> list[str]:
    url = f"{BASE_URL}/varsity/module/{module_slug}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/varsity/chapter/" in href:
                full = BASE_URL + href if not href.startswith("http") else href
                if full not in links:
                    links.append(full)
        print(f"  {module_slug}: {len(links)} chapters found")
        return links
    except Exception as e:
        print(f"  Warning: could not fetch module '{module_slug}': {e}")
        return []


def scrape_chapter(url: str) -> dict | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")

        title_tag = soup.find("h1") or soup.find("h2")
        title = title_tag.get_text(strip=True) if title_tag else url.rstrip("/").split("/")[-1]

        content = (
            soup.find("div", class_="chapter-content")
            or soup.find("div", class_="post-content")
            or soup.find("article")
            or soup.find("main")
        )
        if not content:
            return None

        paragraphs = []
        for tag in content.find_all(["p", "li", "h2", "h3", "td"]):
            text = tag.get_text(separator=" ", strip=True)
            if len(text) > 50:
                paragraphs.append(text)

        return {"url": url, "title": title, "paragraphs": paragraphs} if paragraphs else None
    except Exception as e:
        print(f"  Failed {url}: {e}")
        return None


def chunk_paragraphs(record: dict, chunk_words: int = 350) -> list[dict]:
    chunks, words = [], []
    for para in record["paragraphs"]:
        words.extend(para.split())
        if len(words) >= chunk_words:
            chunks.append({"url": record["url"], "title": record["title"],
                           "text": " ".join(words[:chunk_words])})
            words = words[chunk_words // 2:]  # 50% overlap
    if len(words) > 40:
        chunks.append({"url": record["url"], "title": record["title"], "text": " ".join(words)})
    return chunks


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    chapter_urls: set[str] = set()

    print("Discovering chapters from module pages...")
    for slug in MODULE_SLUGS:
        links = get_chapter_links(slug)
        chapter_urls.update(links)
        time.sleep(1)

    for path in FALLBACK_CHAPTERS:
        chapter_urls.add(BASE_URL + path)

    chapter_list = sorted(chapter_urls)
    print(f"\nTotal unique chapters: {len(chapter_list)}\n")

    all_chunks = []
    for i, url in enumerate(chapter_list, 1):
        slug = url.rstrip("/").split("/")[-1]
        print(f"  [{i:3d}/{len(chapter_list)}] {slug}")
        record = scrape_chapter(url)
        if record:
            all_chunks.extend(chunk_paragraphs(record))
        time.sleep(0.6)

    out = os.path.join(DATA_DIR, "varsity_chunks.jsonl")
    with open(out, "w") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"\n{'='*55}")
    print(f"Saved {len(all_chunks)} chunks → {out}")
    print("Next: python foundry/desi-finance-advisor/phase1_data_prep/clean_and_format.py")


if __name__ == "__main__":
    main()
