from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime, timezone
import hashlib
import math
import re
import requests
import uuid

app = FastAPI()
templates = Jinja2Templates(directory="templates")
BACKEND = "http://backend:8003"
DEFAULT_JUDGE_MODEL = "llama3.2:3b"
JUDGE_MODELS = ("llama3.2:3b", "gemma3:4b")


class EvaluationLabRequest(BaseModel):
    domain: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    language: Literal["en", "it"] = "en"
    model_name: Literal["llama3.2:3b", "gemma3:4b"] = DEFAULT_JUDGE_MODEL
    timeout_seconds: int = Field(default=240, ge=30, le=600)
    max_html_chars: int = Field(default=4000, ge=500, le=12_000)
    max_parsed_chars: int = Field(default=4000, ge=500, le=12_000)


def _backend_error(response):
    try:
        body = response.json()
        if isinstance(body, dict):
            return str(body.get("detail") or body.get("error") or body)
    except (ValueError, TypeError):
        pass
    return f"Backend HTTP {response.status_code}"


def _stage_error(message, status="error"):
    return {"status": status, "result": None, "error": message}


def _text_metadata(value):
    encoded = value.encode("utf-8")
    return {"chars": len(value), "sha256": hashlib.sha256(encoded).hexdigest()}


def _filename_slug(value):
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _valid_export_judge(stage):
    if not isinstance(stage, dict) or stage.get("status") != "success":
        return False
    result = stage.get("result")
    if not isinstance(result, dict):
        return False
    dimensions = result.get("dimensions")
    if not isinstance(dimensions, dict):
        return False
    dimension_values = [dimensions.get(key) for key in ("completeness", "fidelity", "noise", "readability")]
    values = [*dimension_values, result.get("overall_score")]
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in values
    ):
        return False
    return all(1 <= value <= 5 for value in values)


def _export_judge_differences(no_gs, with_gs):
    if not (_valid_export_judge(no_gs) and _valid_export_judge(with_gs)):
        return None
    left = no_gs["result"]
    right = with_gs["result"]
    differences = {
        key: abs(left["dimensions"][key] - right["dimensions"][key])
        for key in ("completeness", "fidelity", "noise", "readability")
    }
    differences["overall_score"] = abs(left["overall_score"] - right["overall_score"])
    return differences


def _judge_prompt_metadata(stage):
    configuration = stage.get("configuration") if isinstance(stage, dict) else None
    configuration = configuration if isinstance(configuration, dict) else {}
    data_hashes = configuration.get("data_hashes")
    data_hashes = data_hashes if isinstance(data_hashes, dict) else {}
    return {
        "prompt_version": configuration.get("prompt_version"),
        "prompt_template_sha256": configuration.get("prompt_template_sha256"),
        "prompt_sha256": configuration.get("prompt_sha256") or data_hashes.get("prompt_sha256"),
        "response_schema_sha256": configuration.get("response_schema_sha256"),
        "configuration_sha256": configuration.get("configuration_sha256"),
    }


def _build_experiment_export(experiment, document, requested_model, parameters, evaluations):
    no_gs = evaluations.get("judge_no_gs")
    with_gs = evaluations.get("judge_with_gs")
    no_gs_configuration = no_gs.get("configuration") if isinstance(no_gs, dict) else None
    with_gs_configuration = with_gs.get("configuration") if isinstance(with_gs, dict) else None
    no_gs_configuration = no_gs_configuration if isinstance(no_gs_configuration, dict) else {}
    with_gs_configuration = with_gs_configuration if isinstance(with_gs_configuration, dict) else {}
    export = {
        "schema_version": "minerva-evaluation-export-v1",
        "experiment": {
            **experiment,
            "document": document,
            "models": {
                "requested": requested_model,
                "effective": {
                    "judge_no_gs": no_gs_configuration.get("response_model"),
                    "judge_with_gs": with_gs_configuration.get("response_model"),
                },
            },
            "parameters": parameters,
            "judge_prompt_metadata": {
                "judge_no_gs": _judge_prompt_metadata(no_gs),
                "judge_with_gs": _judge_prompt_metadata(with_gs),
            },
        },
        
        "evaluations": evaluations,
    }
    differences = _export_judge_differences(no_gs, with_gs)
    if differences is not None:
        export["judge_absolute_differences"] = differences
    return export


def _post_evaluation(path, payload, timeout):
    try:
        response = requests.post(f"{BACKEND}{path}", json=payload, timeout=timeout)
    except requests.Timeout:
        return _stage_error(f"Timeout dopo {timeout} secondi.", "timeout")
    except requests.RequestException as exc:
        return _stage_error(f"Errore di comunicazione con il backend: {exc}")

    try:
        result = response.json()
    except (ValueError, TypeError):
        result = None
    if response.status_code != 200:
        return _stage_error(_backend_error(response), "timeout" if response.status_code == 504 else "error")
    if result is None:
        return _stage_error("Il backend ha restituito una risposta JSON non valida.", "invalid_response")
    return {"status": "success", "result": result, "error": None}


def _post_no_gs_judge(payload, timeout):
    try:
        response = requests.post(
            f"{BACKEND}/evaluate_judge_no_gs", json=payload, timeout=timeout
        )
        body = response.json()
    except requests.Timeout:
        stage = _stage_error(f"Timeout frontend dopo {timeout} secondi.", "timeout")
        stage["backend_completed"] = False
        return stage
    except requests.RequestException as exc:
        stage = _stage_error(f"Errore di comunicazione con il backend: {exc}")
        stage["backend_completed"] = False
        return stage
    except (ValueError, TypeError):
        stage = _stage_error("Il backend ha restituito una risposta JSON non valida.", "invalid_response")
        stage["backend_completed"] = True
        return stage

    if not isinstance(body, dict):
        stage = _stage_error("La risposta del Judge non è un oggetto JSON.", "invalid_response")
        stage["backend_completed"] = True
        return stage
    status = body.get("status")
    if status not in {"success", "timeout", "request_error", "model_error", "invalid_response", "invalid_input"}:
        stage = _stage_error(_backend_error(response), "error")
        stage["backend_completed"] = True
        return stage
    if status == "success" and response.status_code != 200:
        stage = _stage_error(f"Risposta incoerente dal backend: HTTP {response.status_code} con status success.", "invalid_response")
        stage["backend_completed"] = True
        return stage
    return {
        "status": status,
        "result": body.get("result") if status == "success" else None,
        "error": body.get("error"),
        "configuration": body.get("configuration"),
        "backend_completed": True,
    }


def _post_gs_judge(payload, timeout):
    try:
        response = requests.post(
            f"{BACKEND}/evaluate_judge_with_gs_advanced", json=payload, timeout=timeout
        )
        body = response.json()
    except requests.Timeout:
        return _stage_error(f"Timeout dopo {timeout} secondi.", "timeout")
    except requests.RequestException as exc:
        return _stage_error(f"Errore di comunicazione con il backend: {exc}")
    except (ValueError, TypeError):
        return _stage_error("Il backend ha restituito una risposta JSON non valida.", "invalid_response")

    if not isinstance(body, dict):
        return _stage_error("La risposta del Judge non è un oggetto JSON.", "invalid_response")
    status = body.get("status")
    if status not in {"success", "timeout", "request_error", "model_error", "invalid_response", "invalid_input"}:
        return _stage_error(_backend_error(response), "error")
    if status == "success" and response.status_code != 200:
        return _stage_error(
            f"Risposta incoerente dal backend: HTTP {response.status_code} con status success.",
            "invalid_response",
        )
    return {
        "status": status,
        "result": body.get("result") if status == "success" else None,
        "error": body.get("error"),
        "configuration": body.get("configuration"),
    }

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
# EVALUATION LAB
# -----------------------
@app.get("/evaluation-lab", response_class=HTMLResponse)
def evaluation_lab_page(request: Request, domain: str = ""):
    domains = get_domains()
    selected_domain = domain if domain in domains else (domains[0] if domains else "")
    urls = []
    error = None
    if selected_domain:
        try:
            response = requests.get(
                f"{BACKEND}/gold_standard_urls",
                params={"domain": selected_domain},
                timeout=10,
            )
            if response.status_code == 200:
                urls = response.json().get("gold_standard_urls", [])
            else:
                error = _backend_error(response)
        except requests.Timeout:
            error = "Timeout durante il caricamento degli URL Gold Standard."
        except requests.RequestException as exc:
            error = f"Impossibile caricare gli URL: {exc}"
    return templates.TemplateResponse(request=request, name="evaluation_lab.html", context={
        "domains": domains,
        "selected_domain": selected_domain,
        "urls": urls,
        "judge_models": JUDGE_MODELS,
        "default_judge_model": DEFAULT_JUDGE_MODEL,
        "error": error,
    })


@app.post("/evaluation-lab/run")
def evaluation_lab_run(payload: EvaluationLabRequest):
    experiment_id = uuid.uuid4().hex
    experiment_started = datetime.now(timezone.utc)
    experiment_timestamp = experiment_started.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    try:
        domain_response = requests.get(f"{BACKEND}/domains", timeout=10)
        if domain_response.status_code != 200:
            return {"status": "error", "error": _backend_error(domain_response)}
        domains = domain_response.json().get("domains", [])
        if payload.domain not in domains:
            return {"status": "error", "error": "Dominio non disponibile."}

        urls_response = requests.get(
            f"{BACKEND}/gold_standard_urls",
            params={"domain": payload.domain},
            timeout=10,
        )
        if urls_response.status_code != 200:
            return {"status": "error", "error": _backend_error(urls_response)}
        urls = urls_response.json().get("gold_standard_urls", [])
        if payload.url not in urls:
            return {"status": "error", "error": "URL non disponibile per il dominio selezionato."}

        source_response = requests.get(
            f"{BACKEND}/gold_standard", params={"url": payload.url}, timeout=15
        )
        if source_response.status_code != 200:
            return {"status": "error", "error": _backend_error(source_response)}
        source = source_response.json()
        if source.get("url") != payload.url or source.get("domain") != payload.domain:
            return {"status": "error", "error": "Il documento restituito non corrisponde alla selezione."}
        html_text = source.get("html_text")
        gold_text = source.get("gold_text")
        if not isinstance(html_text, str) or not html_text.strip():
            return {"status": "error", "error": "HTML locale vuoto o non disponibile."}
        if not isinstance(gold_text, str) or not gold_text.strip():
            return {"status": "error", "error": "Gold Standard vuoto o non disponibile."}
    except requests.Timeout:
        return {"status": "error", "error": "Timeout durante il recupero del documento."}
    except (requests.RequestException, ValueError, TypeError) as exc:
        return {"status": "error", "error": f"Errore durante il recupero del documento: {exc}"}

    try:
        parse_response = requests.post(
            f"{BACKEND}/parse",
            json={"url": payload.url, "local": True},
            timeout=90,
        )
        if parse_response.status_code != 200:
            return {"status": "error", "error": f"Parsing locale non riuscito: {_backend_error(parse_response)}"}
        parsed = parse_response.json()
    except requests.Timeout:
        return {"status": "error", "error": "Timeout durante il parsing locale."}
    except (requests.RequestException, ValueError, TypeError) as exc:
        return {"status": "error", "error": f"Errore durante il parsing locale: {exc}"}

    if not isinstance(parsed, dict):
        return {"status": "error", "error": "Il parser ha restituito una risposta non valida."}
    parsed_text = parsed.get("parsed_text")
    if parsed.get("url") != payload.url or parsed.get("html_text") != html_text:
        return {"status": "error", "error": "Il parser ha restituito un documento o HTML diverso da quello selezionato."}
    if not isinstance(parsed_text, str):
        return {"status": "error", "error": "Il parser non ha restituito parsed_text."}

    original = _post_evaluation("/evaluate", {
        "parsed_text": parsed_text, "gold_text": gold_text
    }, timeout=30)
    if original["status"] == "timeout":
        advanced = _stage_error("Non avviato perché la richiesta precedente è scaduta.", "not_run")
        judge_response = _stage_error("Non avviato perché una richiesta precedente è scaduta.", "not_run")
        judge_with_gs_response = _stage_error("Non avviato perché una richiesta precedente è scaduta.", "not_run")
    else:
        advanced = _post_evaluation("/evaluate_advanced", {
            "parsed_text": parsed_text,
            "gold_text": gold_text,
            "language": payload.language,
        }, timeout=300)
        if advanced["status"] == "timeout":
            judge_response = _stage_error(
                "Non avviato perché la richiesta delle metriche avanzate è scaduta; il backend potrebbe ancora elaborarla.",
                "not_run",
            )
            judge_with_gs_response = _stage_error(
                "Non avviato perché una richiesta precedente potrebbe essere ancora in elaborazione.",
                "not_run",
            )
        else:
            judge_response = _post_no_gs_judge({
                "html_text": html_text,
                "parsed_text": parsed_text,
                "model_name": payload.model_name,
                "timeout_seconds": payload.timeout_seconds,
                "max_html_chars": payload.max_html_chars,
                "max_parsed_chars": payload.max_parsed_chars,
            }, timeout=payload.timeout_seconds + 30)

            if judge_response.get("backend_completed") is False:
                judge_with_gs_response = _stage_error(
                    "Non avviato: il timeout di comunicazione non conferma che il Judge senza GS abbia terminato.",
                    "not_run",
                )
            else:
                judge_with_gs_response = _post_gs_judge({
                    "parsed_text": parsed_text,
                    "gold_text": gold_text,
                    "model_name": payload.model_name,
                    "timeout_seconds": payload.timeout_seconds,
                    "max_parsed_chars": payload.max_parsed_chars,
                    "max_gold_chars": payload.max_html_chars,
                }, timeout=payload.timeout_seconds + 30)

    document = {"url": payload.url, "domain": payload.domain}
    evaluations = {
        "original": original,
        "advanced": advanced,
        "judge_no_gs": judge_response,
        "judge_with_gs": judge_with_gs_response,
    }
    experiment = {
        "id": experiment_id,
        "timestamp_utc": experiment_timestamp,
        "html_source": {
            "kind": "local_database_html",
            "is_local": True,
            **_text_metadata(html_text),
        },
        "gold_standard": _text_metadata(gold_text),
        "parsed_text": _text_metadata(parsed_text),
    }
    parameters = {
        "language": payload.language,
        "requested_model": payload.model_name,
        "judge_timeout_seconds": payload.timeout_seconds,
        "max_html_chars": payload.max_html_chars,
        "max_parsed_chars": payload.max_parsed_chars,
        "max_gold_chars": payload.max_html_chars,
        "original_metrics_http_timeout_seconds": 30,
        "advanced_metrics_http_timeout_seconds": 300,
        "judge_http_timeout_seconds": payload.timeout_seconds + 30,
        "judge_num_ctx": {
            "judge_no_gs": (judge_response.get("configuration") or {}).get("num_ctx"),
            "judge_with_gs": (judge_with_gs_response.get("configuration") or {}).get("num_ctx"),
        },
    }
    export = _build_experiment_export(experiment, document, payload.model_name, parameters, evaluations)
    timestamp_for_filename = experiment_started.strftime("%Y%m%dT%H%M%S%fZ")
    export_filename = (
        f"minerva-evaluation-{_filename_slug(payload.domain)}-"
        f"{_filename_slug(payload.model_name)}-{timestamp_for_filename}-{experiment_id[:8]}.json"
    )
    return {
        "status": "success",
        "document": document,
        "model_name": payload.model_name,
        "experiment": experiment,
        "export": export,
        "export_filename": export_filename,
        "parser": {"status": "success", "parsed_text": parsed_text},
        "evaluations": evaluations,
    }

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
