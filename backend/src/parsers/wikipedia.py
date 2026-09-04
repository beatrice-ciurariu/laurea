import asyncio
import json
import re
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from rich.console import Console
from database import get_connection
from fastapi import HTTPException

def estrai_wiki_chirurgico(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # TITOLO ---
    tag_titolo = soup.find('h1', id='firstHeading')
    titolo = tag_titolo.get_text(strip=True) if tag_titolo else "Title not found"
    
    # SELEZIONE CONTENITORI
    content_div = soup.select_one('#mw-content-text > .mw-parser-output')
    if not content_div:
        content_div = soup.find('div', class_='mw-parser-output')
    
    if not content_div:
        return titolo, "Article content not found."

    # PULIZIA PREVENTIVA ---
    classi_spazzatura = [
        'mw-editsection', 'hatnote', 'navbox', 'infobox', 'toc', 
        'reference', 'noprint', 'catlinks', 'metadata', 'shortdescription',
        'ambox', 'mw-indicator', 'sidebar', 'vertical-navbox'
    ]
    
    for classe in classi_spazzatura:
        for elemento in content_div.find_all(class_=classe):
            elemento.decompose()
            
    for tag in content_div.find_all(['style', 'figure', 'sup', 'script']):
        tag.decompose()

    # FASE DI RICOSTRUZIONE ---
    testo_risultante = "" 
    sezioni_di_stop = ['see also', 'notes', 'references', 'further reading', 'external links', 'bibliography']
    
    for tag in content_div.find_all(['h2', 'h3', 'p', 'li', 'dt', 'dd', 'table']):
        if tag.decomposed: 
            continue

        # GESTIONE WIKITABLE (MARKDOWN COMPLIANT)
        if tag.name == 'table':
            if 'wikitable' in tag.get('class', []):
                righe_tabella = []
                # Estraiamo tutte le righe (tr)
                raw_rows = tag.find_all('tr')
                for i, riga in enumerate(raw_rows):
                    celle = [c.get_text(separator=" ", strip=True) for c in riga.find_all(['th', 'td'])]
                    celle_pulite = [" ".join(c.split()) for c in celle if c]
                    
                    if celle_pulite:
                        righe_tabella.append("| " + " | ".join(celle_pulite) + " |")
                        # Markdown richiede la riga di separazione | --- | dopo l'header (prima riga)
                        if i == 0:
                            separatore = "|" + " --- |" * len(celle_pulite)
                            righe_tabella.append(separatore)
                
                if righe_tabella:
                    testo_risultante += "\n" + "\n".join(righe_tabella) + "\n\n"
            continue 

        # PROCESSAMENTO STANDARD (TITOLI, LISTE, PARAGRAFI)
        testo = tag.get_text(separator=" ", strip=True)
        testo = " ".join(testo.split())
        
        if not testo: 
            continue 
        
        if tag.name == 'h2':
            titolo_h2 = testo.lower().strip()
            if any(s in titolo_h2 for s in sezioni_di_stop):
                break 
            # Cancelletti per il Markdown h2
            testo_risultante += f"\n## {testo}\n" 
        elif tag.name == 'h3':
            # Cancelletti per il Markdown h3
            testo_risultante += f"\n### {testo}\n"
        elif tag.name in ['li', 'dt']:
            # Trasforma li e dt in liste puntate Markdown
            testo_risultante += f"* {testo}\n"
        elif tag.name in ['p', 'dd']:
            testo_risultante += f"{testo}\n\n"
            
    # PULIZIA POST-ESTRAZIONE ---
    testo_finale = re.sub(r'\s+([,.!?;:])', r'\1', testo_risultante)
    testo_finale = re.sub(r'\(\s+', '(', testo_finale)
    testo_finale = re.sub(r'\s+\)', ')', testo_finale)
    testo_finale = re.sub(r'/\s+', '/', testo_finale)
    testo_finale = re.sub(r'\s+/', '/', testo_finale)
    testo_finale = re.sub(r'([A-Za-z]+)\s+:\s+\[', r'\1:[', testo_finale)

    return titolo, testo_finale.strip()

async def parse_wikipedia(url: str, html_text: str = None):
    # SE ABBIAMO GIA' L'HTML, BYPASS
    if html_text:
        titolo_finale, testo_pulito = estrai_wiki_chirurgico(html_text)
        parsed_uri = urlparse(url)
        return {
            "url": url,
            "domain": parsed_uri.netloc,
            "title": titolo_finale,
            "html_text": html_text,
            "parsed_text": testo_pulito
        }

    # ALTRIMENTI ACCENDIAMO CRAWL4AI SOLO PER SCARICARE DAL WEB (Modalità Live)
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        if not result.success:
            raise Exception("Failed to fetch page")
        titolo_finale, testo_pulito = estrai_wiki_chirurgico(result.html)
        parsed_uri = urlparse(url)
        return {
            "url": url,
            "domain": parsed_uri.netloc,
            "title": titolo_finale,
            "html_text": result.html,
            "parsed_text": testo_pulito
        }
