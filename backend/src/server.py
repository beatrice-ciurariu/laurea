import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Optional
from contextlib import asynccontextmanager
from parser_router import get_parser
from evaluation import evaluate
import json
import os
import traceback
import httpx
from database import get_connection
from judge import MODEL_NAME, esegui_llm_judge
from advanced_metrics import (
    MetricDependencyError,
    MetricInputSizeError,
    bertscore_eval,
    meteor_eval,
)
from advanced_judge import (
    DEFAULT_MAX_HTML_CHARS,
    DEFAULT_MAX_PARSED_CHARS,
    DEFAULT_TIMEOUT_SECONDS,
    evaluate_without_gold,
)
from advanced_judge_with_gs import (
    DEFAULT_MAX_GOLD_CHARS,
    DEFAULT_MAX_PARSED_CHARS as DEFAULT_GS_JUDGE_PARSED_CHARS,
    DEFAULT_MODEL_NAME as DEFAULT_GS_JUDGE_MODEL,
    DEFAULT_TIMEOUT_SECONDS as DEFAULT_GS_JUDGE_TIMEOUT,
    evaluate_with_gold,
)


# impedisce al server di aprire troppi browser simultaneamente se il grader spamma le richieste
browser_lock = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app):
    print("=== LIFESPAN START ===")
    try:
        from init_db import carica_gs
        carica_gs() # Sincrono e veloce per non bloccare la porta
    except Exception as e:
        print(f"ERRORE: {e}")
        traceback.print_exc()
    print("=== LIFESPAN END ===")
    yield

app = FastAPI(lifespan=lifespan)
BASE_DIR = os.path.dirname(__file__)
GS_DIR = "/app/gs_data"

# --- MODELLI PYDANTIC ---
class ParseRequest(BaseModel):
    url: str
    local: Optional[bool] = False

class EvaluateRequest(BaseModel):
    parsed_text: str
    gold_text: str

class AdvancedEvaluateRequest(BaseModel):
    parsed_text: str = Field(..., min_length=1, max_length=2_000_000)
    gold_text: str = Field(..., min_length=1, max_length=2_000_000)
    language: Literal["en", "it"] = "en"

class NoGSJudgeRequest(BaseModel):
    html_text: str = Field(..., min_length=1, max_length=5_000_000)
    parsed_text: str = Field(..., max_length=2_000_000)
    model_name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=30, le=600)
    max_html_chars: int = Field(default=DEFAULT_MAX_HTML_CHARS, ge=500, le=12_000)
    max_parsed_chars: int = Field(default=DEFAULT_MAX_PARSED_CHARS, ge=500, le=12_000)

    class Config:
        extra = "forbid"

class AdvancedGSJudgeRequest(BaseModel):
    parsed_text: str = Field(..., max_length=2_000_000)
    gold_text: str = Field(..., min_length=1, max_length=2_000_000)
    model_name: Optional[str] = Field(default=None, min_length=1, max_length=128)
    timeout_seconds: float = Field(default=DEFAULT_GS_JUDGE_TIMEOUT, ge=30, le=600)
    max_parsed_chars: int = Field(default=DEFAULT_GS_JUDGE_PARSED_CHARS, ge=500, le=12_000)
    max_gold_chars: int = Field(default=DEFAULT_MAX_GOLD_CHARS, ge=500, le=12_000)

    class Config:
        extra = "forbid"

class AddWebResourceRequest(BaseModel):
    url: str
    html_text: str

class AddGoldStandardRequest(BaseModel):
    url: str
    gold_text: str

class DeleteRequest(BaseModel):
    url: str

# -----------------------
# GET /status
# -----------------------
@app.get("/status")
def system_status():
    # DB
    try:
        conn = get_connection(retries=1, delay=1)
        conn.close()
        db_status = "ok"
    except:
        db_status = "error"

    # Ollama
    try:
        import httpx
        r = httpx.get("http://ollama:11434/api/tags", timeout=3)
        models = r.json().get("models", []) if r.status_code == 200 else []
        model_ready = any(
            model.get("name", "").split(":latest")[0] == MODEL_NAME
            for model in models
        )
        ollama_status = "ok" if model_ready else "error"
    except:
        ollama_status = "error"

    return {"backend": "ok", "database": db_status, "ollama": ollama_status}

# -----------------------
# GET /domains
# -----------------------
@app.get("/domains")
def get_domains():
    path = os.path.join(BASE_DIR, "domains.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------
# POST /parse
# -----------------------
@app.post("/parse") 
async def parse_document(req: ParseRequest): 
    url = req.url 
    html_locale = None 
    
    if req.local: 
        conn = get_connection() 
        cur = conn.cursor() 
        cur.execute("SELECT html_text FROM web_resources WHERE url = ?", (url,)) 
        row = cur.fetchone() 
        cur.close() 
        conn.close() 
        if not row: 
            raise HTTPException(status_code=404, detail="URL non presente nel DB locale") 
        html_locale = row[0] 
    try: 
        parser = get_parser(url) 
    except Exception: 
        raise HTTPException(status_code=400, detail="Dominio non supportato") 
    
    try: 
        result = await parser(url, html_text=html_locale) 
    except Exception as e: 
        if "COOKIE_WALL_DETECTED" in str(e): 
            raise HTTPException( status_code=422, detail="Impossibile superare il Cookie Wall di questo dominio. Carica manualmente l'HTML nel database tramite il builder del Gold Standard e abilita la modalità Locale." ) 
        raise HTTPException(status_code=400, detail=f"URL irraggiungibile o non scansionabile, carica l'HTML manualmente tramite il builder del Gold Standard e abilita la modalità Locale: {str(e)}") 
    
    return result

# -----------------------
# GET /gold_standard
# -----------------------
@app.get("/gold_standard")
def get_gs(url: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT wr.url, wr.domain, wr.title, wr.html_text, gs.gold_text
        FROM web_resources wr
        JOIN gold_standard gs ON wr.url = gs.url
        WHERE wr.url = ?
    """, (url,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        raise HTTPException(status_code=404, detail="URL non presente nel Gold Standard")

    return {
        "url": row[0],
        "domain": row[1],
        "title": row[2],
        "html_text": row[3],
        "gold_text": row[4]
    }

# -----------------------
# GET /gold_standard_urls
# -----------------------
@app.get("/gold_standard_urls")
def get_gs_urls(domain: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT gs.url FROM gold_standard gs
        JOIN web_resources wr ON gs.url = wr.url
        WHERE wr.domain = ?
    """, (domain,))
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Dominio non supportato o senza GS")

    return {"gold_standard_urls": [r[0] for r in rows]}

# -----------------------
# POST /evaluate
# -----------------------
@app.post("/evaluate")
def eval_api(data: EvaluateRequest):
    return evaluate(data.parsed_text, data.gold_text)

@app.post("/recalc_all")
async def recalc_all():
    try:
        path = os.path.join(BASE_DIR, "domains.json")
        with open(path, "r") as f:
            domains = json.load(f).get("domains", [])
        for domain in domains:
            try:
                await full_eval(domain)
                print(f"✓ Ricalcolato {domain}")
            except Exception as e:
                print(f"✗ {domain}: {e}")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# -----------------------
# POST /evaluate_judge
# -----------------------
@app.post("/evaluate_judge")
async def evaluate_judge_api(data: EvaluateRequest):
    if not data.parsed_text or not data.gold_text:
        raise HTTPException(status_code=400, detail="I testi sono vuoti")
    return await esegui_llm_judge(data.parsed_text, data.gold_text)

# -----------------------
# POST /evaluate_advanced
# -----------------------
@app.post("/evaluate_advanced")
async def evaluate_advanced_api(data: AdvancedEvaluateRequest):
    try:
        
        meteor = await asyncio.to_thread(
            meteor_eval, data.parsed_text, data.gold_text, data.language
        )
        bertscore = await asyncio.to_thread(
            bertscore_eval, data.parsed_text, data.gold_text
        )
    except MetricDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MetricInputSizeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {**meteor, **bertscore}

# -----------------------
# POST /evaluate_judge_no_gs
# -----------------------
@app.post("/evaluate_judge_no_gs")
async def evaluate_judge_no_gs_api(data: NoGSJudgeRequest):
    result = await evaluate_without_gold(
        data.html_text,
        data.parsed_text,
        **({"model_name": data.model_name} if data.model_name else {}),
        timeout_seconds=data.timeout_seconds,
        max_html_chars=data.max_html_chars,
        max_parsed_chars=data.max_parsed_chars,
    )
    status_codes = {
        "success": 200,
        "invalid_input": 422,
        "timeout": 504,
        "request_error": 503,
        "model_error": 502,
        "invalid_response": 502,
    }
    return JSONResponse(
        status_code=status_codes.get(result.get("status"), 502),
        content=result,
    )

# -----------------------
# POST /evaluate_judge_with_gs_advanced
# -----------------------
@app.post("/evaluate_judge_with_gs_advanced")
async def evaluate_judge_with_gs_advanced_api(data: AdvancedGSJudgeRequest):
    result = await evaluate_with_gold(
        data.parsed_text,
        data.gold_text,
        model_name=(data.model_name or DEFAULT_GS_JUDGE_MODEL),
        timeout_seconds=data.timeout_seconds,
        max_parsed_chars=data.max_parsed_chars,
        max_gold_chars=data.max_gold_chars,
    )
    status_codes = {
        "success": 200,
        "invalid_input": 422,
        "timeout": 504,
        "request_error": 503,
        "model_error": 502,
        "invalid_response": 502,
    }
    return JSONResponse(
        status_code=status_codes.get(result.get("status"), 502),
        content=result,
    )

# -----------------------
# GET /full_gs_eval
# -----------------------
@app.get("/full_gs_eval")
async def full_eval(domain: str):
    # Recupero dati iniziali e giudizi già calcolati. Ogni entry viene giudicata
    # una sola volta e poi riutilizzata: il grader richiama questo endpoint più
    # volte e non deve provocare decine di inferenze identiche.
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT wr.url, wr.html_text, gs.gold_text FROM web_resources wr JOIN gold_standard gs ON wr.url = gs.url WHERE wr.domain = ?", (domain,))
        rows = cur.fetchall()
        cur.execute("""
            SELECT lj.url, lj.judge_score
            FROM llm_judgments lj
            JOIN web_resources wr ON lj.url = wr.url
            WHERE wr.domain = ?
        """, (domain,))
        cached_judges = {url: score for url, score in cur.fetchall()}
    finally:
        cur.close()
        conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Dominio non supportato o senza GS")

    eval_results = []
    judge_scores = []

    for url, html_text, gold_text in rows:
        try:
            # Parsing
            async with browser_lock:
                parser = get_parser(url)
                parsed = await parser(url, html_text=html_text)

            # Metriche classiche (F1)
            ev = evaluate(parsed["parsed_text"], gold_text)
            tl = ev["token_level_eval"]
            eval_results.append(tl)

            judge = None
            judge_score = cached_judges.get(url)
            if judge_score is None:
                try:
                    # Dieci entry per dominio: un limite più stretto evita che
                    # una CPU lenta faccia scadere il timeout aggregato.
                    judge = await asyncio.wait_for(
                        esegui_llm_judge(parsed["parsed_text"], gold_text),
                        timeout=20.0,
                    )
                except asyncio.TimeoutError:
                    judge = {
                        "model_name": MODEL_NAME,
                        "judge_score": 3,
                        "judge_feedback": "Timeout durante la valutazione LLM.",
                    }
                judge_score = judge.get("judge_score", 3)
            judge_scores.append(float(judge_score))

            # Si mantiene una sola valutazione aggiornata per URL e salva il
            # giudizio solo quando non era già in cache.
            conn2 = get_connection()
            try:
                cur2 = conn2.cursor()
                cur2.execute("DELETE FROM evaluations WHERE url = ?", (url,))
                cur2.execute("INSERT INTO evaluations (url, precision_score, recall_score, f1_score) VALUES (?, ?, ?, ?)", 
                             (url, tl["precision"], tl["recall"], tl["f1"]))
                if judge is not None:
                    cur2.execute("DELETE FROM llm_judgments WHERE url = ?", (url,))
                    cur2.execute(
                        "INSERT INTO llm_judgments (url, model_name, judge_score, judge_feedback) VALUES (?, ?, ?, ?)",
                        (
                            url,
                            judge.get("model_name", MODEL_NAME),
                            judge_score,
                            judge.get("judge_feedback", "Nessun feedback"),
                        ),
                    )
            finally:
                cur2.close()
                conn2.close()

        except Exception as e:
            print(f"Errore su {url}: {e}")
            continue

    if not eval_results:
        raise HTTPException(status_code=422, detail="Nessuna entry del Gold Standard valutabile")

    return {
        "token_level_eval": {
            "precision": sum(r["precision"] for r in eval_results) / len(eval_results),
            "recall": sum(r["recall"] for r in eval_results) / len(eval_results),
            "f1": sum(r["f1"] for r in eval_results) / len(eval_results),
        },
        "judge_score": sum(judge_scores) / len(judge_scores) if judge_scores else 3.0
    }

# -----------------------
# POST /add_web_resource
# -----------------------
@app.post("/add_web_resource")
async def add_web_resource(req: AddWebResourceRequest):
    try:
        # Estrae il dominio senza librerie esterne
        domain = req.url.split("/")[2] if "//" in req.url else req.url
       
        conn = get_connection()
        cur = conn.cursor()
        # Upsert senza DELETE implicito: REPLACE romperebbe le FK quando la
        # risorsa possiede già un Gold Standard.
        cur.execute("""
            INSERT INTO web_resources (url, domain, title, html_text)
            VALUES (?, ?, ?, ?)
            ON DUPLICATE KEY UPDATE
                domain = VALUES(domain),
                html_text = VALUES(html_text),
                title = IF(title IS NULL OR title = '', VALUES(title), title)
        """, (req.url, domain, "Titolo Risorsa", req.html_text))
        conn.commit()
        cur.close()
        conn.close()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# -----------------------
# POST /add_gold_standard
# -----------------------
@app.post("/add_gold_standard")
def add_gold_standard(req: AddGoldStandardRequest):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT url FROM web_resources WHERE url = ?", (req.url,))
        if not cur.fetchone():
            return {"status": "error", "detail": "URL non presente in web_resources"}

        cur.execute("""
            INSERT INTO gold_standard (url, gold_text)
            VALUES (?, ?)
            ON DUPLICATE KEY UPDATE gold_text = VALUES(gold_text)
        """, (req.url, req.gold_text))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        conn.close()  # si chiude sempre

# -----------------------
# DELETE /web_resource
# -----------------------
@app.delete("/web_resource")
def delete_web_resource(req: DeleteRequest):
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Le DELETE esplicite mantengono compatibilità con volumi creati prima
        # dell'introduzione di ON DELETE CASCADE.
        cur.execute("DELETE FROM evaluations WHERE url = ?", (req.url,))
        cur.execute("DELETE FROM llm_judgments WHERE url = ?", (req.url,))
        cur.execute("DELETE FROM parsed_documents WHERE url = ?", (req.url,))
        cur.execute("DELETE FROM gold_standard WHERE url = ?", (req.url,))
        cur.execute("DELETE FROM web_resources WHERE url = ?", (req.url,))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        conn.close()

# -----------------------
# DELETE /gold_standard
# -----------------------
@app.delete("/gold_standard")
def delete_gold_standard(req: DeleteRequest):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT url FROM gold_standard WHERE url = ?", (req.url,))
        if not cur.fetchone():
            return {"status": "error", "detail": "URL non presente"}
        
        cur.execute("DELETE FROM gold_standard WHERE url = ?", (req.url,))
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
    finally:
        conn.close()

# -----------------------
# GET /db_stats
# -----------------------
@app.get("/db_stats")
def db_stats():
    # Domini ufficiali
    path = os.path.join(BASE_DIR, "domains.json")
    with open(path, "r", encoding="utf-8") as f:
        valid_domains = json.load(f).get("domains", [])
       
    conn = get_connection()
    cur = conn.cursor()

    # Inizializzazione pulita
    stats = {
        "web_resources": {d: 0 for d in valid_domains},
        "gold_standard": {d: 0 for d in valid_domains},
        "avg_eval": {d: {"token_level_eval": {"precision": 0.0, "recall": 0.0, "f1": 0.0}} for d in valid_domains},
        "avg_eval_judge": {d: {"judge_score": -1} for d in valid_domains}
    }

    def get_base_domain(d_str):
        for vd in valid_domains:
            if vd in d_str: return vd
        return d_str

    cur.execute("SELECT domain, COUNT(*) FROM web_resources GROUP BY domain")
    for row in cur.fetchall():
        bd = get_base_domain(row[0])
        if bd in stats["web_resources"]: stats["web_resources"][bd] += row[1]

    cur.execute("""
        SELECT wr.domain, COUNT(*) FROM gold_standard gs
        JOIN web_resources wr ON gs.url = wr.url
        GROUP BY wr.domain
    """)
    for row in cur.fetchall():
        bd = get_base_domain(row[0])
        if bd in stats["gold_standard"]: stats["gold_standard"][bd] += row[1]

    cur.execute("""
        SELECT wr.domain, AVG(e.precision_score), AVG(e.recall_score), AVG(e.f1_score)
        FROM evaluations e
        JOIN web_resources wr ON e.url = wr.url
        GROUP BY wr.domain
    """)
    for row in cur.fetchall():
        bd = get_base_domain(row[0])
        if bd in stats["avg_eval"]:
            stats["avg_eval"][bd]["token_level_eval"] = {
                "precision": row[1] or 0.0,
                "recall": row[2] or 0.0,
                "f1": row[3] or 0.0
            }

    cur.execute("""
        SELECT wr.domain, AVG(lj.judge_score)
        FROM llm_judgments lj
        JOIN web_resources wr ON lj.url = wr.url
        GROUP BY wr.domain
    """)
    for row in cur.fetchall():
        bd = get_base_domain(row[0])
        if bd in stats["avg_eval_judge"]:
            stats["avg_eval_judge"][bd]["judge_score"] = row[1] or -1

    cur.close()
    conn.close()
    return stats

# -----------------------
# GET /db_schema
# -----------------------
@app.get("/db_schema")
def db_schema():
    return {
        "web_resources": {
            "url": "varchar(2048), PK",
            "domain": "varchar(255)",
            "title": "varchar(2048)",
            "html_text": "longtext",
            "created_at": "datetime"
        },
        "gold_standard": {
            "url": "varchar(2048), PK, FK(web_resources.url)",
            "gold_text": "longtext",
            "created_at": "datetime"
        },
        "parsed_documents": {
            "id": "int, PK, AUTO_INCREMENT",
            "url": "varchar(2048), FK(web_resources.url)",
            "parsed_text": "longtext",
            "created_at": "datetime"
        },
        "evaluations": {
            "id": "int, PK, AUTO_INCREMENT",
            "url": "varchar(2048), FK(web_resources.url)",
            "precision_score": "float",
            "recall_score": "float",
            "f1_score": "float",
            "created_at": "datetime"
        },
        "llm_judgments": {
            "id": "int, PK, AUTO_INCREMENT",
            "url": "varchar(2048), FK(web_resources.url)",
            "model_name": "varchar(100)",
            "judge_score": "int",
            "judge_feedback": "text",
            "created_at": "datetime"
        }
    }

# -----------------------
# GET /full_gold_standard
# -----------------------
@app.get("/full_gold_standard")
def full_gs(domain: str):
    parts = domain.split(".")
    filename = parts[-2] + "_gs.json"
    path = os.path.join(GS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="GS not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
