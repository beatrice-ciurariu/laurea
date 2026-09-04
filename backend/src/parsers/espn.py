import asyncio
import json
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from rich.console import Console
from crawl4ai import AsyncWebCrawler
from html_checker import url_presente, carica_html_da_url
from database import get_connection
from fastapi import HTTPException

def estrai_espn_chirurgico(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # CONTROLLO ANTI-BOT / AWS WAF ---
    if "awswaf.com" in html_content or "challenge.js" in html_content or "challenge-container" in html_content:
        raise ValueError("AWS_WAF_DETECTED")

    # TITOLO ---
    tag_h1 = soup.find('h1')
    if tag_h1:
        titolo = tag_h1.get_text(strip=True)
    else:
        titolo = soup.title.string.replace("- ESPN", "").strip() if soup.title else "ESPN Page"
    
    # PULIZIA PREVENTIVA ---
    for s in soup(['script', 'style', 'nav', 'header', 'footer', 'button']):
        s.decompose()

    contenuto_finale = []

    # LOGICA TABELLE  ---
    tabelle = soup.find_all('table', class_=re.compile(r'tablehead|Table', re.I))
    
    if tabelle:
        for tabella in tabelle:
            righe = tabella.find_all('tr')
            for i, riga in enumerate(righe):
                if 'stathead' in riga.get('class', []):

                    contenuto_finale.append(f"## {riga.get_text(strip=True).upper()}")
                    continue

                colonne = riga.find_all(['td', 'th'])
                testi = [" ".join(c.get_text(" ", strip=True).split()) for c in colonne]
                
                if len(testi) >= 2:
                    # COSTRUZIONE RIGA MARKDOWN
                    riga_md = "| " + " | ".join(testi) + " |"
                    
                    if 'colhead' in riga.get('class', []) or riga.find('th'):
                        header = "| " + " | ".join(testi).upper().replace("RUNNER-UP", "RUNNER UP") + " |"
                        contenuto_finale.append(header)

                        separatore = "|" + " --- |" * len(testi)
                        contenuto_finale.append(separatore)
                    else:
                        contenuto_finale.append(riga_md)
        
        if contenuto_finale:
            return titolo, "\n".join(contenuto_finale).strip()

    # LOGICA ARTICOLI / STORY ---
    for junk in soup(['aside', 'figure']):
        junk.decompose()
    
    for link_junk in soup.find_all(['p', 'div'], string=re.compile(r'Last updated:|full class of|order by rank|jump to the', re.I)):
        link_junk.decompose()

    corpo = soup.find('div', class_=re.compile(r'article-body', re.I))
    if corpo:
        for elemento in corpo.find_all(['p', 'h2', 'blockquote']):
            testo = elemento.get_text(" ", strip=True)
            testo = " ".join(testo.split())
            
            if any(x in testo for x in ["A post shared by", "pic.twitter.com", "View this post", "Instagram"]):
                continue
            
            # COSTRUZIONE MARKDOWN
            if elemento.name == 'h2':
                contenuto_finale.append(f"## {testo}")
            elif elemento.name == 'blockquote':
                contenuto_finale.append(f"> {testo}")
            elif "Ranking:" in testo and "Committed to:" in testo:
                # Formattazione liste dati recruit
                testo_md = testo.replace("Committed to:", "\n* **Committed to:**")
                testo_md = testo_md.replace("Commitment date:", "\n* **Commitment date:**")
                contenuto_finale.append(testo_md)
            elif testo and len(testo) > 3:
                contenuto_finale.append(testo)

    # FALLBACK BIOGRAFIE ---
    if not contenuto_finale:
        sezioni = soup.find_all(['p', 'div'], class_=re.compile(r'article-body|PlayerHeader__Bio'))
        for s in sezioni:
            testo = " ".join(s.get_text(" ", strip=True).split())
            if len(testo) > 30: 
                contenuto_finale.append(testo)

    return titolo, "\n\n".join(contenuto_finale).strip()


async def parse_espn(url: str, html_text: str = None):
    # Bypass se abbiamo l'HTML
    if html_text:
        try:
            titolo_finale, testo_pulito = estrai_espn_chirurgico(html_text)
            parsed_uri = urlparse(url)
            return { "url": url, "domain": parsed_uri.netloc, "title": titolo_finale, "html_text": html_text, "parsed_text": testo_pulito }
        except ValueError as e:
            if str(e) == "AWS_WAF_DETECTED":
                raise HTTPException(status_code=403, detail="WAF RILEVATO su ESPN. Usa l'HTML salvato nel DB.")
            raise e

    # Live con Crawl4AI (WAF richiede obbligatoriamente JS attivo)
    async with AsyncWebCrawler() as crawler:
        # Magic dovrebbe superare il wall
        result = await crawler.arun(url=url, magic=True)
        
        if not result.success:
            raise HTTPException(status_code=500, detail="Impossibile scaricare la pagina.")
        
        try:
            titolo_finale, testo_pulito = estrai_espn_chirurgico(result.html)
        except ValueError as e:
            if str(e) == "AWS_WAF_DETECTED":
                raise HTTPException(status_code=403, detail="WAF RILEVATO su ESPN. . Carica l'html manualmente in gs_builder!")
            raise e
            
        parsed_uri = urlparse(url)
        return {
            "url": url,
            "domain": parsed_uri.netloc,
            "title": titolo_finale,
            "html_text": result.html,
            "parsed_text": testo_pulito
        }