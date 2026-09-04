import asyncio
import json
import re
from urllib.parse import urlparse
from crawl4ai import AsyncWebCrawler
from bs4 import BeautifulSoup
from rich.console import Console
from html_checker import url_presente, carica_html_da_url
from database import get_connection
from fastapi import HTTPException

def estrai_yahoo_chirurgico(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')

    # CONTROLLO COOKIES WALL ---
    html_lower = html_content.lower()
    
    # Ricerca dell'ID/classi che Yahoo usa per il blocco
    if soup.find(id="consent-page") or soup.find(class_=re.compile(r'consent-form|consent-container')):
        raise ValueError("COOKIE_WALL_DETECTED")
        
    # Controllo esplicito su dove puntano i form
    for form in soup.find_all('form'):
        action = form.get('action', '').lower()
        if 'consent.yahoo.com' in action or 'guce.yahoo.com' in action:
            raise ValueError("COOKIE_WALL_DETECTED")

    # Testo univoco del banner (ITA/ENG)
    if ("famiglia di brand yahoo" in html_lower or "yahoo family of brands" in html_lower) and ("accetta" in html_lower or "accept" in html_lower or "agree" in html_lower):
        raise ValueError("COOKIE_WALL_DETECTED")
        
    # Fallback sul tag title
    title_str = str(soup.title).lower() if soup.title else ""
    if "guce.yahoo.com" in title_str or "yahoo privacy" in title_str or "yahoo consent" in title_str:
        raise ValueError("COOKIE_WALL_DETECTED")
        
    # TITOLO ---
    tag_titolo = (
        soup.find('h1', data_testid="headline") or 
        soup.find('h1', class_=re.compile(r'headline|caas-header|yf-')) or
        soup.find('h2', class_=re.compile(r'header yf-1kayatz')) or
        soup.find('h1') or
        soup.find('title')
    )
    
    titolo = "Title not found"
    if tag_titolo:
        raw_title = tag_titolo.get_text(strip=True)
        titolo = raw_title.split(" - Yahoo")[0].split(" | Yahoo")[0]
    

    # ESTRAZIONE HTML GREZZO ---
    soup_html = BeautifulSoup(html_content, 'html.parser')
    for s in soup_html.find_all(['script', 'style', 'iframe', 'noscript']):
        s.decompose()
    html_text_grezzo = str(soup_html)

    # PULIZIA PREVENTIVA ---
    soup_clean = BeautifulSoup(html_content, 'html.parser')
    for spazzatura in soup_clean.find_all(['header', 'footer', 'nav', 'script', 'style', 'button', 'iframe', 'noscript', 'form', 'aside']):
        spazzatura.decompose()

    # LOGICA ARTICOLO  ---
    articolo_completo = soup_clean.find('div', class_=re.compile(r'article-body|caas-body|body-wrap')) or soup_clean.find('article')
    testo_markdown = ""
    if articolo_completo:
        break_words = ["read next", "building wealth across", "see also", "rad intel", "mode mobile"]
        for tag in articolo_completo.find_all(['p', 'h2', 'h3', 'h4', 'li']):
            testo = " ".join(tag.get_text(separator=" ", strip=True).split())
            if any(bw in testo.lower() for bw in break_words): break
            if len(testo) < 10 or any(bad in testo.lower() for bad in ["read more", "subscribe"]): continue
            
            if tag.name in ['h2', 'h3', 'h4']: 
                testo_markdown += f"\n\n## {testo}\n\n"
            elif tag.name == 'li': 
                testo_markdown += f"\n* {testo}"
            else: 
                testo_markdown += f"{testo}\n\n"

    parsed_final = testo_markdown.strip()

    # lOGICA MODELLI STOCK
    if len(parsed_final) < 500:
        target_ids = ['quote-summary-stats', 'quote-statistics', 'valuation-measures', 'financial-highlights', 'balance-sheet', 'cash-flow']
        ha_dati_stock = any(soup.find(attrs={'data-testid': sid}) for sid in target_ids)

        if ha_dati_stock:
            # Il titolo fa parte del contenuto informativo e garantisce anche
            # un'intestazione Markdown coerente per le pagine quotazione.
            dati_stock = [f"# {titolo}\n", "## News headlines\n"]
            intro = soup.find('div', class_=re.compile(r'yf-1028f67|quote-summary|news-content|description'))
            if intro:
                for p in intro.find_all('p'):
                    txt = p.get_text(strip=True)
                    if len(txt) > 80:
                        dati_stock.append(txt + "\n")

            performance = soup.find(attrs={'data-testid': 'performance-overview'})
            if performance:
                performance_text = " ".join(
                    performance.get_text(" ", strip=True).split()
                )
                if performance_text:
                    dati_stock.append(f"\n## Performance Overview\n{performance_text}")

            for sid in target_ids:
                section = soup.find(attrs={'data-testid': sid})
                if section:
                    h = section.find(['h3', 'h4'])
                    # Sottotitoli in Markdown
                    if h: dati_stock.append(f"\n## {h.get_text(strip=True)}\n")
                    for row in section.find_all(['li', 'tr']):
                        cells = row.find_all(['p', 'span', 'td'], recursive=False)
                        if len(cells) >= 2:
                            l, v = cells[0].get_text(strip=True), cells[1].get_text(strip=True)
                            if l and v and l.lower() not in v.lower():
                                # Formato identico al tuo originale per non perdere punti F1
                                dati_stock.append(f"* **{l}**: {v}")

            desc_section = soup.find('section', {'data-testid': 'description'}) or soup.find('div', class_=re.compile(r'quote-description|business-summary'))
            if desc_section:
                dati_stock.append(f"\n## {titolo} Overview")
                p_desc = desc_section.find('p') or desc_section
                d_txt = p_desc.get_text(strip=True).replace("Show More", "")
                dati_stock.append(d_txt)
            
            parsed_final = "\n\n".join(dati_stock).strip()

        # LOGICA TABELLA MERCATI 
        else:
            tabella = soup.find('table')
            if tabella:
                righe = []
                headers = [th.get_text(strip=True) for th in tabella.find_all('th') if th.get_text(strip=True)]
                if headers: 
                    righe.append("| " + " | ".join(headers) + " |")
                    righe.append("|" + " --- |" * len(headers))
                for tr in tabella.find_all('tr')[1:]:
                    cols = [ " ".join(td.get_text(separator=" ", strip=True).split()) for td in tr.find_all('td')]
                    if cols: righe.append("| " + " | ".join(cols) + " |")
                
                parsed_final = "\n".join(righe)

    return titolo, parsed_final

# LOGICA DI PROCESSO ---
async def parse_yahoo(url: str, html_text: str = None):
    # MODALITÀ LOCALE
    if html_text:
        try:
            titolo_finale, testo_pulito = estrai_yahoo_chirurgico(html_text)
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
                raise HTTPException(status_code=400, detail="L'HTML fornito è una pagina di blocco Privacy/Cookie.")
            raise e

    # MODALITÀ LIVE ---
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url, magic=True)
        
        if not result.success:
            raise HTTPException(status_code=500, detail="Impossibile scaricare la pagina.")
        
        try:
            titolo_finale, testo_pulito = estrai_yahoo_chirurgico(result.html)
        except ValueError as e:
            # Intercetta il muro di yahoo
            if str(e) == "COOKIE_WALL_DETECTED":
                # Controlliamo se l'URL è nel GS per dare il messaggio corretto
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("SELECT url FROM gold_standard WHERE url = ?", (url,))
                esiste = cur.fetchone()
                conn.close()

                if esiste:
                    raise HTTPException(
                        status_code=403, 
                        detail="WALL RILEVATO. URL presente nel Gold Standard: usa local=true per usare l'HTML salvato."
                    )
                else:
                    raise HTTPException(
                        status_code=403, 
                        detail="WALL RILEVATO. URL non in GS: inserisci l'HTML manualmente nel DB."
                    )
            raise e # Se è un altro errore, lo RILANCIA

        parsed_uri = urlparse(url)
        return {
            "url": url,
            "domain": parsed_uri.netloc,
            "title": titolo_finale,
            "html_text": result.html,
            "parsed_text": testo_pulito
        }
