import asyncio
import hashlib
import json
import os
import re
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from rich.console import Console
from html_checker import url_presente, carica_html_da_url
from database import get_connection
from fastapi import HTTPException

def estrai_fifa_chirurgico(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # Se il titolo è vuoto o il body contiene solo il div root di caricamento, è un errore
    testo_grezzo = soup.get_text(strip=True)
    if not testo_grezzo or "page-loading-indicator" in html_content:
        raise ValueError("LOADING_PAGE_DETECTED")

    # CONTROLLO ANTI-CONSENT ---
    # Se la pagina contiene questi elementi è un cookie wall
    testo_pagina = soup.get_text().lower()
    if any(keyword in testo_pagina for keyword in ["onetrust-consent-sdk", "your privacy", "we care about your privacy", "cookie settings"]):
        raise ValueError("COOKIE_WALL_DETECTED")

    # TITOLO ---
    tag_titolo = (
        soup.find('h1', class_=re.compile(r'fusion-title-heading')) or 
        soup.find('h1', class_=re.compile(r'simple-hero-module_title')) or
        soup.find('h1') or
        soup.find('title')
    )
    titolo = tag_titolo.get_text(strip=True) if tag_titolo else "Title not found"

    # PULIZIA PREVENTIVA ---
    for spazzatura in soup.find_all(['header', 'footer', 'nav', 'script', 'style', 'button', 'iframe', 'noscript']):
        spazzatura.decompose()
    
    # Toglie widget calendario e selettori date che sporcano l'output
    for widget in soup.find_all('div', class_=re.compile(r'calendar|date-selector|filter-menu')):
        widget.decompose()

    for controlli_video in soup.find_all('div', class_=re.compile(r'theo|vjs-|video-js|player')):
        controlli_video.decompose()

    # IDENTIFICAZIONE CONTENITORE ---
    articolo_completo = (
        soup.find('div', class_=re.compile(r'rich-text-module_richTextContainer')) or
        soup.find('article', class_=re.compile(r'article-section_body')) or
        soup.find('div', class_=re.compile(r'Post_post|content-block|text-module|standings-module')) or 
        soup.find('main')
    )

    if not articolo_completo:
        articolo_completo = soup.find('div', id='main-content') or soup.find('body')

    # ESTRAZIONE TESTO---
    testo_markdown_raw = ""
    tags_da_estrarre = ['p', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'a', 'div', 'span']
    blacklist_testo = ["quality automatic", "background opacity", "character edge style", "selected descriptions off", "powered by theoplayer"]

    for tag in articolo_completo.find_all(tags_da_estrarre):
        if tag.name in ['div', 'span'] and tag.find(['p', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'a']):
            continue

        testo = tag.get_text(separator=" ", strip=True)
        testo = " ".join(testo.split())
        
        soglia = 1 
        if not testo or len(testo) < soglia or testo.lower() == titolo.lower():
            continue
            
        if any(bad_word in testo.lower() for bad_word in blacklist_testo):
            continue

        # --- LOGICA MARKDOWN ---
        if tag.name == 'h2':
            testo_markdown_raw += f"\n\n## {testo}\n\n"
        elif tag.name in ['h3', 'h4', 'h5', 'h6'] or 'competitionName' in str(tag.get('class', '')):
            testo_markdown_raw += f"\n\n### {testo}\n\n"
        elif tag.name == 'li':
            testo_markdown_raw += f"\n* {testo}"
        elif 'standings' in str(tag.get('class', '')) or 'group' in testo.lower():
            testo_markdown_raw += f"\n\n#### {testo}\n"
        elif tag.name == 'p':
            testo_markdown_raw += f"\n\n{testo}\n\n"
        else:
            # Per dati sparsi o span (es. nomi squadre o punteggi)
            testo_markdown_raw += f"{testo} "

    # MICRO-PULIZIA FINALE ---
    testo_quasi_finito = re.sub(r'\n{3,}', '\n\n', testo_markdown_raw).strip()
    paragrafi = testo_quasi_finito.split('\n\n')
    visti = set()
    risultato_unico = []
    
    for p in paragrafi:
        norm = re.sub(r'\W+', '', p).lower()
        if norm and norm not in visti:
            if norm in ["england", "uominidonne", "uomini", "donne"]:
                continue
            risultato_unico.append(p.strip())
            visti.add(norm)
    
    return titolo, "\n\n".join(risultato_unico)


async def parse_fifa(url: str, html_text: str = None):
    # Se ha già l'HTML, bypass
    if html_text:
        try:
            titolo_finale, testo_pulito = estrai_fifa_chirurgico(html_text)
            parsed_uri = urlparse(url)
            return { 
                "url": url, 
                "domain": parsed_uri.netloc, 
                "title": titolo_finale, 
                "html_text": html_text, 
                "parsed_text": testo_pulito 
            }
        except ValueError as e:
            if str(e) == "COOKIE_WALL_DETECTED":
                raise HTTPException(status_code=403, detail="WALL RILEVATO su FIFA. Usa l'HTML salvato nel DB.")
            raise e

    # Se non ha l'HTML, usa Crawl4AI in modalità Live
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, magic=True)
        
        if not result.success:
            raise HTTPException(status_code=500, detail="Impossibile scaricare la pagina.")
        
        try:
            titolo_finale, testo_pulito = estrai_fifa_chirurgico(result.html)
        except ValueError as e:
            if str(e) == "COOKIE_WALL_DETECTED":
                raise HTTPException(status_code=403, detail="WALL RILEVATO su FIFA. Carica l'html manualmente in gs_builder!")
            raise e
            
        parsed_uri = urlparse(url)
        return {
            "url": url,
            "domain": parsed_uri.netloc,
            "title": titolo_finale,
            "html_text": result.html,
            "parsed_text": testo_pulito
        }