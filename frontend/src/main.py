from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
import requests

app = FastAPI()
templates = Jinja2Templates(directory="templates")
BACKEND = "http://backend:8003"

def get_domains():
    try:
        return requests.get(f"{BACKEND}/domains", timeout=5).json().get("domains", [])
    except:
        return []

def get_status():
    try:
        return requests.get(f"{BACKEND}/status", timeout=5).json()
    except:
        return {"backend": "error", "database": "error", "ollama": "error"}

# -----------------------
# HOME
# -----------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="home.html", context={
        "domains": get_domains(),
        "status": get_status()
    })

# -----------------------
# PARSER & EVAL
# -----------------------
@app.get("/parser", response_class=HTMLResponse)
def parser_page(request: Request):
    domains = get_domains()
    gs_urls = []
    for d in domains:
        try:
            data = requests.get(f"{BACKEND}/gold_standard_urls", params={"domain": d}, timeout=5).json()
            gs_urls.extend(data.get("gold_standard_urls", []))
        except:
            continue
    return templates.TemplateResponse(request=request, name="parser.html", context={
        "gs_urls": gs_urls,
        "result": None,
        "error": None
    })

@app.post("/parser", response_class=HTMLResponse)
def parser_run(request: Request, url: str = Form(...), mode: str = Form("live")):
    domains = get_domains()
    gs_urls = []
    for d in domains:
        try:
            data = requests.get(f"{BACKEND}/gold_standard_urls", params={"domain": d}, timeout=5).json()
            gs_urls.extend(data.get("gold_standard_urls", []))
        except:
            continue

    parsed = gs = eval_result = error_msg = None

    try:
        local = mode == "local"
        resp = requests.post(f"{BACKEND}/parse", json={"url": url, "local": local}, timeout=60)
        if resp.status_code != 200:
            error_msg = resp.json().get("detail", "Errore durante il parsing")
        else:
            parsed = resp.json()
            try:
                gs_resp = requests.get(f"{BACKEND}/gold_standard", params={"url": url}, timeout=5)
                if gs_resp.status_code == 200:
                    gs = gs_resp.json()
                    ev = requests.post(f"{BACKEND}/evaluate", json={
                        "parsed_text": parsed["parsed_text"],
                        "gold_text": gs["gold_text"]
                    }, timeout=10).json()
                    judge = requests.post(f"{BACKEND}/evaluate_judge", json={
                        "parsed_text": parsed["parsed_text"],
                        "gold_text": gs["gold_text"]
                    }, timeout=600).json()
                    eval_result = {**ev, **judge}
            except:
                pass
    except:
        error_msg = "Impossibile contattare il backend"

    return templates.TemplateResponse(request=request, name="parser.html", context={
        "gs_urls": gs_urls,
        "result": {"parsed": parsed, "gs": gs, "eval": eval_result} if parsed else None,
        "error": error_msg
    })

# -----------------------
# GS BUILDER
# -----------------------
@app.get("/gs_builder", response_class=HTMLResponse)
def gs_builder_page(request: Request, domain: str = None):
    domains = get_domains()
    gs_urls = []

    if not domain and domains:
        domain = domains[0]
        
    if domain:
        try:
            data = requests.get(f"{BACKEND}/gold_standard_urls", params={"domain": domain}, timeout=5).json()
            gs_urls = data.get("gold_standard_urls", [])
        except:
            pass
    return templates.TemplateResponse(request=request, name="gs_builder.html", context={
        "domains": domains,
        "selected_domain": domain,
        "gs_urls": gs_urls,
        "fetched_html": None,
        "fetched_url": None,
        "error": None,
        "success": None
    })

@app.post("/gs_builder/fetch", response_class=HTMLResponse)
def gs_fetch(request: Request, domain: str = Form(...), url: str = Form(...)):
    domains = get_domains()
    
    # Auto-selezione del dominio
    detected_domain = next((d for d in domains if d in url), None)
    if detected_domain and detected_domain != domain:
        domain = detected_domain 

    gs_urls = []
    warning_msg = None
    try:
        data = requests.get(f"{BACKEND}/gold_standard_urls", params={"domain": domain}, timeout=5).json()
        gs_urls = data.get("gold_standard_urls", [])
    except:
        pass

    # NUOVO: Impostiamo solo un warning, senza bloccare l'esecuzione
    if url in gs_urls:
        warning_msg = "Nota: Questo URL è già presente nel Gold Standard. Se salvi il testo in basso, lo sovrascriverai."

    fetched_html = error_msg = None
    try:
        resp = requests.post(f"{BACKEND}/parse", json={"url": url, "local": False}, timeout=60)
        if resp.status_code == 200:
            fetched_html = resp.json().get("html_text", "")
            # Salva la web_resource nel DB
            requests.post(f"{BACKEND}/add_web_resource", json={"url": url, "html_text": fetched_html}, timeout=60)
        else:
            error_msg = resp.json().get("detail", "Errore nel fetch")
    except:
        error_msg = "Errore nel contattare il backend"

    return templates.TemplateResponse(request=request, name="gs_builder.html", context={
        "domains": domains,
        "selected_domain": domain,
        "gs_urls": gs_urls,
        "fetched_html": fetched_html,
        "fetched_url": url,
        "error": error_msg,
        "success": None,
        "warning": warning_msg 
    })

@app.post("/gs_builder/save", response_class=HTMLResponse)
def gs_save(request: Request, domain: str = Form(...), url: str = Form(...), gold_text: str = Form(...)):
    domains = get_domains()
    gs_urls = []
    success_msg = error_msg = None
    try:
        data = requests.get(f"{BACKEND}/gold_standard_urls", params={"domain": domain}, timeout=5).json()
        gs_urls = data.get("gold_standard_urls", [])
    except:
        pass

    try:
        resp = requests.post(f"{BACKEND}/add_gold_standard", json={"url": url, "gold_text": gold_text}, timeout=10)
        if resp.json().get("status") == "ok":
            success_msg = f"Gold Standard salvato per {url}"
            # Ricarica lista
            try:
                data = requests.get(f"{BACKEND}/gold_standard_urls", params={"domain": domain}, timeout=5).json()
                gs_urls = data.get("gold_standard_urls", [])
            except:
                pass
        else:
            error_msg = resp.json().get("detail", "Errore nel salvataggio")
    except:
        error_msg = "Errore nel contattare il backend"

    return templates.TemplateResponse(request=request, name="gs_builder.html", context={
        "domains": domains,
        "selected_domain": domain,
        "gs_urls": gs_urls,
        "fetched_html": None,
        "fetched_url": url,
        "error": error_msg,
        "success": success_msg
    })

@app.post("/gs_builder/delete", response_class=HTMLResponse)
def gs_delete(request: Request, domain: str = Form(...), url: str = Form(...)):
    try:
        requests.request("DELETE", f"{BACKEND}/web_resource", json={"url": url}, timeout=10)
    except:
        pass
    return RedirectResponse(url=f"/gs_builder?domain={domain}", status_code=303)

# Definiamo la struttura del JSON in arrivo
class ManualSavePayload(BaseModel):
    domain: str
    url: str
    html_text: str
    gold_text: str

@app.post("/gs_builder/save_manual_json")
def gs_save_manual_json(payload: ManualSavePayload):
    try:
        # Salva l'HTML grezzo in web_resources passando i dati dal payload
        resp_wr = requests.post(f"{BACKEND}/add_web_resource", json={"url": payload.url, "html_text": payload.html_text}, timeout=60)
        
        if resp_wr.status_code == 200 and resp_wr.json().get("status") == "ok":
            # Se l'HTML è salvato, salva il Gold Standard
            resp_gs = requests.post(f"{BACKEND}/add_gold_standard", json={"url": payload.url, "gold_text": payload.gold_text}, timeout=10)
            
            if resp_gs.status_code == 200 and resp_gs.json().get("status") == "ok":
                return {"success": True, "message": f"Bypass riuscito! HTML e Gold Standard salvati."}
            else:
                return {"success": False, "error": resp_gs.json().get("detail", "Errore nel salvataggio del Gold Standard")}
        else:
            return {"success": False, "error": resp_wr.json().get("detail", "Errore nel salvataggio dell'HTML")}
    except Exception as e:
        return {"success": False, "error": f"Errore nel contattare il backend: {str(e)}"}

# -----------------------
# STATS
# -----------------------
@app.get("/stats", response_class=HTMLResponse)
def stats_page(request: Request):
    stats = error = None
    try:
        stats = requests.get(f"{BACKEND}/db_stats", timeout=10).json()
    except:
        error = "Impossibile caricare le statistiche"
    return templates.TemplateResponse(request=request, name="stats.html", context={
        "stats": stats,
        "error": error
    })

@app.post("/stats/recalc", response_class=HTMLResponse)
def stats_recalc(request: Request):
    error = None
    try:
        requests.post(f"{BACKEND}/recalc_all", timeout=600)
    except Exception as e:
        error = f"Errore: {str(e)}"
    
    stats = None
    try:
        stats = requests.get(f"{BACKEND}/db_stats", timeout=10).json()
    except:
        error = "Impossibile caricare le statistiche"
    
    return templates.TemplateResponse(request=request, name="stats.html", context={
        "stats": stats,
        "error": error,
        "recalc_errors": []
    })