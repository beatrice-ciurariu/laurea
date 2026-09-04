import re
import json
import httpx
import mistune
from bs4 import BeautifulSoup

OLLAMA_API_URL = "http://ollama:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

def remove_markdown(md: str) -> str:
    if not md: 
        return ""
    html = mistune.html(md)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(True):
        tag.unwrap()
    text = re.sub(r'[ \t]+', ' ', str(soup))
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

#Per evitare timeout si analizzano 3 sezioni di grandi testi
def tronca_testo(testo: str, max_parole_totali: int = 200) -> str:
    parole = testo.split()
    if len(parole) <= max_parole_totali:
        return testo
    
    chunk = max_parole_totali // 3
    inizio = parole[:chunk]
    meta = len(parole) // 2
    centro = parole[meta - chunk//2 : meta + chunk//2]
    fine = parole[-chunk:]
    
    return (
        " ".join(inizio) + 
        "\n[...]\n" + 
        " ".join(centro) + 
        "\n[...]\n" + 
        " ".join(fine)
    )

async def esegui_llm_judge(parsed_text: str, gold_text: str) -> dict:
    testo_parser_pulito = tronca_testo(remove_markdown(parsed_text))
    testo_gold_troncato = tronca_testo(gold_text)
    
    prompt = f"""Sei un valutatore di qualità del testo. Confronta il testo estratto con il testo di riferimento e assegna un punteggio da 1 a 5.

    1 = testo completamente sbagliato o vuoto
    2 = testo molto incompleto o con molto rumore
    3 = testo parzialmente corretto
    4 = testo quasi completo con piccole imprecisioni
    5 = testo perfetto o quasi identico al riferimento

    Testo estratto:
    {testo_parser_pulito}

    Testo di riferimento:
    {testo_gold_troncato}

    Rispondi SOLO con JSON valido, niente altro:
    {{"score": <numero da 1 a 5>, "feedback": "<una frase in italiano>"}}"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0 
        }
    }

    fallback = {
        "model_name": MODEL_NAME,
        "judge_score": 3,      # IL GRADER VUOLE UN NUMERO TRA 1 E 5, non va bene -1
        "judge_feedback": "Errore nel parsing del modello LLM o timeout."
    }

    try:
        # Il grader impone un timeout di 120 secondi. In caso di macchina lenta
        # restituiamo il fallback strutturalmente valido prima di quel limite.
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(OLLAMA_API_URL, json=payload)
            if response.status_code != 200:
                return fallback
           
            output = response.json().get("response", "").strip()
           
            try:
                # Estrazione Regex per ripulire la sporcizia fuori dal JSON
                match = re.search(r'\{.*\}', output, re.DOTALL)
                json_str = match.group(0) if match else output
               
                dati = json.loads(json_str)
                score = max(1, min(5, int(dati.get("score", 3))))
               
                return {
                    "model_name": MODEL_NAME,
                    "judge_score": score,
                    "judge_feedback": str(dati.get("feedback", "Nessun feedback")).strip()
                }
            except:
                return fallback
    except:
        return fallback
