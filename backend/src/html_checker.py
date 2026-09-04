import json
import os
from rich.console import Console

console = Console()

def normalizza_url(url: str) -> str:
    if not url:
        return url
    return url[:-1] if url[-1] == "/" else url

def url_presente(url: str) -> bool:
    url = normalizza_url(url)
    file_json = "html_list.json"
    base_path = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_path, file_json)
    
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        urls = data.get("urls", [])
        
        return url in urls
    
    except FileNotFoundError:
        raise Exception(f"File {file_json} non trovato")
    except json.JSONDecodeError:
        raise Exception(f"Errore nel parsing del file JSON")


def carica_html_da_url(url: str) -> str | None:
    url = normalizza_url(url)
    console.print("Caricamento da HTML locale", style="bold yellow")

    base_path = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(base_path, "html_list.json"), "r") as f:
        data = json.load(f)

    file_rel_path = data.get("urls", {}).get(url)
    if not file_rel_path:
        console.print("URL non presente in html_list.json", style="bold yellow")
        return None

    file_path = os.path.join(base_path, file_rel_path)

    if not os.path.exists(file_path):
        console.print(f"File HTML locale non trovato: {file_path}", style="bold yellow")
        return None

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()