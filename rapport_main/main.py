import os, time, asyncio, json, logging, re
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv(dotenv_path="rapport_main/.env")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("moviroo")
DEV_MODE = os.getenv("DEV_MODE", "false").lower() == "true"

# ── CONFIG ────────────────────────────────────────────────────────────────────

GEMINI_KEY   = os.environ["GEMINI_API_KEY"]
BASE_URL     = "https://generativelanguage.googleapis.com/v1beta/models"

MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-001",
]

MOBILITY_API = os.getenv("MOBILITY_API", "http://localhost:3000")
# FIX: ML Service runs on port 8005
ML_API       = os.getenv("ML_API", "http://127.0.0.1:8005")

CACHE_TTL = 60

# ── PYDANTIC MODELS ───────────────────────────────────────────────────────────

class Issue(BaseModel):
    level: Literal["critical", "warning", "info"]
    text:  str

class Prediction(BaseModel):
    text: str
    bold: str | None = None

class Recommendation(BaseModel):
    num:   str
    text:  str
    color: str

class HealthItem(BaseModel):
    dot:  str
    text: str

class Report(BaseModel):
    summary:         str
    alertLevel:      Literal["critical", "warning", "normal"]
    issues:          list[Issue]
    predictions:     list[Prediction]
    recommendations: list[Recommendation]
    health:          list[HealthItem]
    generatedAt:     str  = ""
    fromCache:       bool = False
    model:           str  = ""

FALLBACK = Report(
    summary="Gemini failed — fallback response used.",
    alertLevel="warning",
    issues=[Issue(level="warning", text="AI failure — check /debug endpoint")],
    predictions=[],
    recommendations=[Recommendation(num="01", text="Check /debug for details", color="#FFB74D")],
    health=[
        HealthItem(dot="#FFB74D", text="AI degraded"),
        HealthItem(dot="#4CAF50", text="ML API OK"),
        HealthItem(dot="#4CAF50", text="Platform OK"),
        HealthItem(dot="#FFB74D", text="Gemini unstable"),
    ],
)

# ── STATE ─────────────────────────────────────────────────────────────────────

_cache: dict[str, tuple[float, Report]] = {}
_debug: dict = {}

# ── APP ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Moviroo Intelligence Proxy", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── CACHE ─────────────────────────────────────────────────────────────────────

def cache_get(key):
    e = _cache.get(key)
    if not e: return None
    ts, data = e
    if time.time() - ts > CACHE_TTL:
        del _cache[key]; return None
    return data

def cache_set(key, val):
    _cache[key] = (time.time(), val)

# ── PLATFORM DATA ─────────────────────────────────────────────────────────────

async def safe_get(client, url):
    try:
        r = await client.get(url, timeout=6)
        if r.is_success:
            return r.json()
        log.warning("API %s → %s", url, r.status_code)
        return None
    except Exception as e:
        log.warning("API %s → unreachable: %s", url, e)
        return None

async def safe_post(client, url, body=None):
    try:
        r = await client.post(url, json=body or {}, timeout=6)
        if r.is_success:
            return r.json()
        log.warning("API POST %s → %s", url, r.status_code)
        return None
    except Exception as e:
        log.warning("API POST %s → unreachable: %s", url, e)
        return None

async def fetch_platform_data():
    """
    FIX: Appelle les vrais endpoints du ML Service (port 8005).
    Anciens endpoints (port 8002, supprimés) :
      - /anomalies/churn?min_risk=0&limit=20       → remplacé par /predict/anomalies?hours=24
      - /model-registry/drift/report               → supprimé (n'existe pas sur 8005)
      - /demand-forecast/surge                     → remplacé par /intelligence/zones

    Nouveaux endpoints (port 8005) :
      - GET  /dashboard/kpis
      - GET  /predict/demand?hours=24
      - GET  /predict/revenue?forecast_days=7
      - GET  /predict/anomalies?hours=24
      - GET  /intelligence/zones
    """
    async with httpx.AsyncClient() as c:
        res = await asyncio.gather(
            # Support tickets depuis Mobility API
            safe_get(c, f"{MOBILITY_API}/api/support/tickets/stats"),
            # ML Service — port 8005
            safe_get(c, f"{ML_API}/dashboard/kpis"),
            safe_get(c, f"{ML_API}/predict/demand?hours=24"),
            safe_get(c, f"{ML_API}/predict/revenue?forecast_days=7"),
            safe_get(c, f"{ML_API}/predict/anomalies?hours=24"),
            safe_get(c, f"{ML_API}/intelligence/zones"),
        )

    keys = ["support", "kpis", "demand", "revenue", "anomalies", "zones"]
    data = dict(zip(keys, res))
    available = [k for k, v in data.items() if v]
    log.info("Platform APIs available: %s", available)
    _debug["platform_available"] = available
    _debug["platform_data"]      = {k: v for k, v in data.items() if v}
    return data

# ── PROMPT ────────────────────────────────────────────────────────────────────

PROMPT = """You are an AI analyst for Moviroo, a Tunisian ride-hailing platform.
Generate a complete operational intelligence report from the data below.

PLATFORM DATA:
{data}

DATA GUIDE (use these fields to fill the report with real numbers):
- kpis: total_rides, completed_rides, cancelled_rides, cancellation_rate_pct, avg_fare_tnd,
        revenue_today_tnd, revenue_total_tnd, total_drivers, drivers_online, drivers_offline,
        avg_driver_rating, avg_wait_minutes, completion_rate_pct
- demand: predictions[].demand, summary.busiest_hour, summary.max_demand, summary.avg_demand
- revenue: predictions[].predicted_revenue, total_predicted, trend_slope
- anomalies: anomalies[].type, anomalies[].severity, anomalies[].impact,
             total, critical_count, anomaly_rate
- zones: zones[].zone, zones[].coverage_status, zones[].avg_wait_min,
         zones[].demand_coverage_ratio, under_served[]

Return ONLY a JSON object. No markdown, no explanation, no text outside the JSON.

Required format:
{{
  "summary": "2-3 sentences about platform status with real numbers from the data",
  "alertLevel": "warning",
  "issues": [
    {{"level": "critical", "text": "specific issue with real value from data"}},
    {{"level": "warning",  "text": "another issue with real value"}},
    {{"level": "info",     "text": "informational note"}}
  ],
  "predictions": [
    {{"text": "Peak demand of 37 rides expected at 14h today.", "bold": "37 rides"}},
    {{"text": "Revenue forecast for the week: 10,448 TND total.", "bold": "10,448 TND"}},
    {{"text": "Sousse zone severely under-served — 200 demand coverage ratio.", "bold": "Sousse"}}
  ],
  "recommendations": [
    {{"num": "01", "text": "Freeze driver b2a70fd7 — 19 unresolved anomalies, 9.5% anomaly rate.", "color": "#E57373"}},
    {{"num": "02", "text": "Deploy more drivers in Sousse — 55 min avg wait, coverage ratio 200.", "color": "#E57373"}},
    {{"num": "03", "text": "Apply surge pricing at 14h — peak demand forecast 37 rides.", "color": "#A855F7"}},
    {{"num": "04", "text": "Investigate cancellation rate of 17.72% — above acceptable threshold.", "color": "#A855F7"}}
  ],
  "health": [
    {{"dot": "#4CAF50", "text": "ML models: demand, revenue, anomaly, zones active"}},
    {{"dot": "#FFB74D", "text": "Completion rate 91.44% — 8.56% drop risk"}},
    {{"dot": "#E57373", "text": "Critical: 19 fraud anomalies unresolved on driver b2a70fd7"}},
    {{"dot": "#FFB74D", "text": "Avg wait 67.1 min — above target"}}
  ]
}}

Rules:
- alertLevel: one of critical, warning, normal
- issues level: one of critical, warning, info — 3 to 5 items
- health: exactly 4 items
- recommendations: 4 to 6 items, num from 01 to 06
- color: use #E57373 for urgent items, #A855F7 for normal, #FFB74D for caution
- bold: must be a short phrase that appears verbatim in the text field
- Replace ALL example values with REAL values from the DATA above
- If a data section is null/missing, skip it gracefully — do not invent values
"""

# ── JSON PARSING ──────────────────────────────────────────────────────────────

def extract_json(raw: str) -> str:
    clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    if clean.startswith("{"):
        return clean
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return match.group(0) if match else clean

def coerce(p: dict) -> dict:
    if p.get("alertLevel") not in {"critical", "warning", "normal"}:
        p["alertLevel"] = "warning"
    for i in p.get("issues", []):
        if i.get("level") not in {"critical", "warning", "info"}:
            i["level"] = "info"
    for r in p.get("recommendations", []):
        if r.get("color") not in {"#E57373", "#A855F7", "#FFB74D"}:
            r["color"] = "#A855F7"
    h = p.get("health", [])
    while len(h) < 4:
        h.append({"dot": "#4CAF50", "text": "System OK"})
    p["health"] = h[:4]
    return p

def try_parse(raw: str, model: str) -> Report | None:
    _debug["last_raw"]   = raw
    _debug["last_model"] = model
    log.info("RAW from %s (%d chars):\n%s", model, len(raw), raw[:1500])

    if len(raw) < 100:
        _debug["last_error"] = f"Response too short ({len(raw)} chars) — likely truncated"
        log.warning("Response too short: %d chars", len(raw))
        return None

    clean = extract_json(raw)
    _debug["last_clean"] = clean[:800]
    try:
        parsed  = json.loads(clean)
        coerced = coerce(parsed)
        return Report(**coerced,
                      generatedAt=time.strftime("%d %b %Y, %H:%M"),
                      model=model)
    except json.JSONDecodeError as e:
        _debug["last_error"] = f"JSONDecodeError: {e} | raw_len={len(raw)}"
        log.warning("JSON error (%d chars): %s\nSnippet: %s", len(raw), e, clean[:300])
        return None
    except ValidationError as e:
        _debug["last_error"] = f"ValidationError: {str(e)[:400]}"
        log.warning("Pydantic error: %s", e)
        return None

# ── GEMINI CALL ───────────────────────────────────────────────────────────────

async def call_one_model(model: str, data: dict) -> Report | None:
    url  = f"{BASE_URL}/{model}:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": PROMPT.format(data=json.dumps(data, indent=2))}]}],
        "generationConfig": {
            "temperature":      0.15,
            "maxOutputTokens":  8192,
            "responseMimeType": "application/json",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=45) as c:
            r = await c.post(url, json=body)
    except Exception as e:
        log.warning("Network error for %s: %s", model, e)
        return None

    log.info("Model %s → HTTP %s", model, r.status_code)
    _debug.setdefault("model_attempts", {})[model] = r.status_code

    if r.status_code == 429:
        log.warning("Model %s → 429 quota exceeded", model)
        return None

    if not r.is_success:
        log.warning("Model %s → error %s: %s", model, r.status_code, r.text[:200])
        return None

    rj         = r.json()
    candidates = rj.get("candidates", [])

    if not candidates:
        log.warning("Model %s → no candidates. Response: %s", model, rj)
        _debug["last_error"] = f"{model}: no candidates"
        return None

    finish = candidates[0].get("finishReason", "STOP")
    log.info("Model %s → finishReason: %s", model, finish)

    usage = rj.get("usageMetadata", {})
    log.info("Token usage: %s", usage)
    _debug["token_usage"] = usage

    if finish in ("SAFETY", "RECITATION", "OTHER"):
        log.warning("Model %s → blocked (%s)", model, finish)
        _debug["last_error"] = f"{model}: blocked by Gemini ({finish})"
        return None

    raw = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

    if not raw:
        log.warning("Model %s → empty text. finishReason=%s", model, finish)
        _debug["last_error"] = f"{model}: empty text, finishReason={finish}"
        return None

    return try_parse(raw, model)


async def call_gemini(data: dict) -> Report:
    if DEV_MODE:
        log.warning("⚡ DEV MODE ACTIVE — skipping Gemini")
        return FALLBACK.model_copy(update={
            "summary": "DEV MODE REPORT — no Gemini call",
            "generatedAt": time.strftime("%d %b %Y, %H:%M"),
            "model": "dev-mock"
        })

    _debug["model_attempts"] = {}

    for model in MODELS:
        log.info("Trying model: %s", model)
        result = await call_one_model(model, data)
        if result:
            log.info("✅ Success with: %s", model)
            return result
        await asyncio.sleep(1)

    log.error("❌ All models failed")
    return FALLBACK

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":       "ok",
        "cache":        len(_cache),
        "models":       MODELS,
        "mobility_api": MOBILITY_API,
        "ml_api":       ML_API,  # affiche la valeur réelle utilisée
        "ml_endpoints": [
            f"{ML_API}/dashboard/kpis",
            f"{ML_API}/predict/demand?hours=24",
            f"{ML_API}/predict/revenue?forecast_days=7",
            f"{ML_API}/predict/anomalies?hours=24",
            f"{ML_API}/intelligence/zones",
        ],
    }

@app.post("/report", response_model=Report)
async def report(request: Request, force: bool = False):
    if not force:
        cached = cache_get("report")
        if cached:
            log.info("Cache HIT")
            return cached
    data   = await fetch_platform_data()
    result = await call_gemini(data)
    cache_set("report", result)
    return result

@app.delete("/report/cache")
def clear_cache():
    _cache.clear()
    return {"cleared": True}

@app.get("/debug")
def debug():
    return _debug