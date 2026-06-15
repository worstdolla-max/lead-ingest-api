import asyncio
import json
import os
import time
from typing import List, Literal, Optional

import anthropic
import resend
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, EmailStr
from supabase import create_client, Client

from datetime import datetime, timedelta, timezone
from datetime import time as dt_time


from fastapi.responses import HTMLResponse as FastAPIHTMLResponse

import urllib.parse

# ---------- App FastAPI ----------

app = FastAPI(title="Immo AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key"],
)

# ---------- Route formulaire HTML ----------


@app.get("/", response_class=HTMLResponse)
async def serve_form():
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


# ---------- Config Resend ----------

RESEND_API_KEY = os.environ.get("RESEND_API_KEY")
print(">>> RESEND_API_KEY existe en prod ?", bool(RESEND_API_KEY))
if RESEND_API_KEY:
    print(">>> RESEND_API_KEY commence par :", RESEND_API_KEY[:5])
    resend.api_key = RESEND_API_KEY
else:
    print(">>> RESEND_API_KEY manquante !")

# ---------- Config Supabase (lazy — server starts even if creds are missing/invalid) ----------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

supabase: Optional[Client] = None
_supabase_error: Optional[str] = None

if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    except Exception as e:
        _supabase_error = str(e)
else:
    _supabase_error = "SUPABASE_URL or SUPABASE_SERVICE_KEY not set"


def get_supabase() -> Client:
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail=f"Supabase unavailable: {_supabase_error}",
        )
    return supabase

def generate_time_slots(tz_name: str = "Europe/Paris", days_ahead: int = 3) -> list:
    import zoneinfo
    from datetime import date as dt_date
    tz = zoneinfo.ZoneInfo(tz_name)
    now = datetime.now(tz)
    slots = []
    for i in range(days_ahead):
        target_day = now.date() + timedelta(days=i + 1)
        slot_time = datetime.combine(target_day, dt_time(hour=11, minute=0), tz)
        if slot_time <= now:
            slot_time = datetime.combine(target_day, dt_time(hour=15, minute=0), tz)
        slots.append(slot_time)
    return slots



# ---------- Config Anthropic ----------

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

_CLAUDE_SYSTEM = (
    "Tu es un assistant IA pour une agence immobilière française. "
    "Analyse le message d'un prospect et retourne un JSON avec exactement ces champs : "
    "summary (résumé en 2 phrases), "
    "score (A/B/C où A=très qualifié), "
    "budget (montant en euros ou null), "
    "timeline (délai exprimé en mois ou null), "
    "property_type (type de bien ou null), "
    "location (ville ou null), "
    "email_reply_suggestion (suggestion de réponse email courte et professionnelle en français), "
    "tags (liste de mots-clés pertinents)."
)

_FALLBACK_INSIGHTS = {
    "summary": "Résumé non disponible (analyse IA désactivée).",
    "score": "B",
    "budget": None,
    "timeline": None,
    "property_type": None,
    "location": None,
    "email_reply_suggestion": (
        "Bonjour, merci pour votre message. Nous revenons vers vous rapidement."
    ),
    "tags": [],
}


def analyse_lead_with_claude(message: str) -> dict:
    if not ANTHROPIC_API_KEY:
        return _FALLBACK_INSIGHTS.copy()

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=512,
            system=_CLAUDE_SYSTEM,
            messages=[{"role": "user", "content": message}],
        )
        raw = response.content[0].text.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        insights = json.loads(raw)
        return insights
    except Exception:
        return _FALLBACK_INSIGHTS.copy()


# ---------- API Key auth ----------

_API_KEY = os.environ.get("API_KEY")


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    if not _API_KEY:
        raise HTTPException(
            status_code=500, detail="API_KEY not configured on the server."
        )
    if x_api_key != _API_KEY:
        raise HTTPException(
            status_code=401, detail="Unauthorized: missing or invalid X-API-Key."
        )


# ---------- Logging Middleware ----------


async def _write_api_log(
    method: str, path: str, status_code: int, duration_ms: int
) -> None:
    if supabase is None:
        return
    try:
        await asyncio.to_thread(
            lambda: supabase.table("api_logs")
            .insert(
                {
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                }
            )
            .execute()
        )
    except Exception:
        pass


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.monotonic()
    response = await call_next(request)
    duration_ms = int((time.monotonic() - start) * 1000)
    asyncio.create_task(
        _write_api_log(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
    )
    return response


# ---------- Modèles Pydantic ----------


class LeadIngest(BaseModel):
    agency_id: str
    crm_lead_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    source: Optional[str] = None
    message: str
    property_reference: Optional[str] = None


class LeadOut(BaseModel):
    lead_id: str
    agency_id: str
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    source: Optional[str]
    message: str
    created_at: str
    manual_score: Optional[str]
    score: Optional[str]
    summary: Optional[str]


class LeadPatchRequest(BaseModel):
    manual_score: Optional[Literal["A", "B", "C"]] = None
    status: Optional[Literal["new", "contacted", "qualified", "won", "lost"]] = None
    notes: Optional[str] = None


class LeadPatchResponse(BaseModel):
    lead_id: str
    agency_id: str
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    source: Optional[str]
    message: str
    manual_score: Optional[str]
    status: Optional[str]
    notes: Optional[str]
    created_at: str


class LeadDetail(BaseModel):
    lead_id: str
    agency_id: str
    crm_lead_id: Optional[str]
    name: Optional[str]
    email: Optional[str]
    phone: Optional[str]
    source: Optional[str]
    message: str
    property_reference: Optional[str]
    created_at: str
    summary: Optional[str]
    manual_score: Optional[str]
    score: Optional[str]
    budget: Optional[str]
    timeline: Optional[str]
    property_type: Optional[str]
    location: Optional[str]
    email_reply_suggestion: Optional[str]
    tags: Optional[List[str]]


class AppointmentConfirmQuery(BaseModel):
    lead_id: str
    slot_iso: str  # datetime ISO8601


# ---------- Routes ----------


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/confirm-appointment")
def confirm_appointment(lead_id: str, slot_iso: str):
    import zoneinfo
    db = get_supabase()

    # 1) Récupérer le lead
    try:
        lead_resp = (
            db.table("leads")
            .select("id, agency_id, name, email, phone")
            .eq("id", lead_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        return HTMLResponse(
            content=f"<h2>Erreur</h2><p>Impossible de récupérer le lead: {e}</p>",
            status_code=500,
        )

    if not lead_resp.data:
        return HTMLResponse(
            content="<h2>Lead introuvable</h2><p>Le lien est invalide ou expiré.</p>",
            status_code=404,
        )

    lead = lead_resp.data[0]

    # 2) Parser le slot — on gère le + mal encodé en espace
    try:
        slot_iso_clean = slot_iso.replace(" ", "+")
        dt_slot = datetime.fromisoformat(slot_iso_clean)
    except Exception as ex:
        return HTMLResponse(
            content=f"<h2>Créneau invalide</h2><p>Valeur reçue : {slot_iso!r} — Erreur : {ex}</p>",
            status_code=400,
        )

    # 3) Insérer le rendez-vous
    try:
        db.table("appointments").insert(
            {
                "lead_id": lead["id"],
                "agency_id": lead["agency_id"],
                "prospect_name": lead.get("name"),
                "prospect_email": lead.get("email"),
                "prospect_phone": lead.get("phone"),
                "status": "confirmed",
                "start_at": dt_slot.isoformat(),
            }
        ).execute()
    except Exception as e:
        return HTMLResponse(
            content=f"<h2>Erreur</h2><p>Impossible d'enregistrer le rendez-vous : {e}</p>",
            status_code=500,
        )

    # 4) Page de confirmation
    tz_paris = zoneinfo.ZoneInfo("Europe/Paris")
    human_date = dt_slot.astimezone(tz_paris).strftime("%A %d %B %Y à %Hh%M")
    prospect_name = lead.get("name") or "le prospect"

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <title>Rendez-vous confirmé — LeadFlow</title>
        <style>
          body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: #05060a;
            color: #f5f3ee;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            margin: 0;
          }}
          .card {{
            background: #11131b;
            padding: 32px 36px;
            border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.1);
            max-width: 440px;
            width: 90%;
            text-align: center;
          }}
          .icon {{ font-size: 48px; margin-bottom: 16px; }}
          h2 {{ margin: 0 0 12px; font-size: 22px; }}
          .date {{
            background: rgba(255,255,255,0.06);
            padding: 12px 16px;
            border-radius: 10px;
            font-size: 17px;
            font-weight: 600;
            margin: 16px 0;
          }}
          p {{ font-size: 14px; color: #9ca3af; line-height: 1.6; }}
        </style>
      </head>
      <body>
        <div class="card">
          <div class="icon">✅</div>
          <h2>Rendez-vous confirmé</h2>
          <p>avec <strong style="color:#f5f3ee;">{prospect_name}</strong></p>
          <div class="date">{human_date}</div>
          <p>Ce rendez-vous a été enregistré dans votre espace LeadFlow.</p>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html, status_code=200)

@app.post("/api/ingest-lead", dependencies=[Depends(verify_api_key)])
def ingest_lead(payload: LeadIngest):
    print(">>> /api/ingest-lead appelé (prod)")

    db = get_supabase()
    try:
        lead_insert = (
            db.table("leads")
            .insert(
                {
                    "agency_id": payload.agency_id,
                    "crm_lead_id": payload.crm_lead_id,
                    "name": payload.name,
                    "email": payload.email,
                    "phone": payload.phone,
                    "source": payload.source,
                    "message": payload.message,
                    "property_reference": payload.property_reference,
                }
            )
            .execute()
        )
    except Exception as e:
        print(">>> Erreur insertion lead:", repr(e))
        raise HTTPException(status_code=500, detail=f"Erreur insertion lead: {e}")

    if not lead_insert.data:
        print(">>> Insertion lead échouée (pas de data)")
        raise HTTPException(status_code=500, detail="Insertion lead échouée.")

    lead_row = lead_insert.data[0]
    lead_id = lead_row["id"]

    insights = analyse_lead_with_claude(payload.message)

    summary = insights.get("summary")
    budget = insights.get("budget")
    timeline = insights.get("timeline")
    property_type = insights.get("property_type")
    location = insights.get("location")
    score = insights.get("score", "B")
    email_reply_suggestion = insights.get("email_reply_suggestion")
    tags = insights.get("tags", [])

    try:
        db.table("lead_insights").insert(
            {
                "lead_id": lead_id,
                "summary": summary,
                "budget": budget,
                "timeline": timeline,
                "property_type": property_type,
                "location": location,
                "score": score,
                "email_reply_suggestion": email_reply_suggestion,
                "tags": tags,
            }
        ).execute()
    except Exception as e:
        print(">>> Erreur insertion insights:", repr(e))
        raise HTTPException(status_code=500, detail=f"Erreur insertion insights: {e}")

    # ---------- Email Resend ----------
    # ----------- Partie email Resend + créneaux IA ----------    # ----------- Partie email Resend + créneaux RDV -----------

    print(">>> RESEND_API_KEY truthy dans la route ?", bool(RESEND_API_KEY))

    try:
        agency_row = (
            db.table("agencies")
            .select("contact_email, name")
            .eq("id", payload.agency_id)
            .limit(1)
            .execute()
        )
        print(">>> agency_row.data:", agency_row.data)

        if agency_row.data:
            agency_contact_email = agency_row.data[0].get("contact_email")
            agency_name = agency_row.data[0].get("name") or "Agence"

            # ---- Génération des 3 créneaux ----
            slots = generate_time_slots(tz_name="Europe/Paris", days_ahead=3)

            base_url = os.environ.get(
                "PUBLIC_BASE_URL",
                "https://lead-ingest-api-production.up.railway.app"
            )

            slot_links_html = ""
            for dt_slot in slots:
                import urllib.parse
                iso = dt_slot.isoformat()
                confirm_url = (
                    f"{base_url}/confirm-appointment"
                    f"?lead_id={lead_id}"
                    f"&slot_iso={urllib.parse.quote(iso)}"
                )
                label = dt_slot.strftime("%A %d %B %Y à %Hh%M")
                slot_links_html += (
                    f'<p style="margin:8px 0;">'
                    f'<a href="{confirm_url}" style="color:#1d4ed8;font-weight:600;text-decoration:none;">'
                    f'✅ Confirmer : {label}'
                    f'</a></p>'
                )

            if RESEND_API_KEY and agency_contact_email:
                print(">>> Envoi email via Resend...")
                try:
                    html_body = f"""
                    <div style="font-family:system-ui,sans-serif;max-width:600px;margin:0 auto;padding:24px;">
                      <h3 style="margin-top:0;">Nouveau lead reçu</h3>
                      <p><strong>Agence :</strong> {agency_name}</p>
                      <p><strong>Nom :</strong> {payload.name or ''}</p>
                      <p><strong>Email :</strong> {payload.email or ''}</p>
                      <p><strong>Téléphone :</strong> {payload.phone or ''}</p>
                      <p><strong>Source :</strong> {payload.source or ''}</p>
                      <p><strong>Message :</strong> {payload.message}</p>
                      <p><strong>Score IA :</strong> {score}</p>
                      <hr style="border:none;border-top:1px solid #e5e7eb;margin:16px 0;"/>
                      <h4 style="margin-bottom:8px;">Résumé IA</h4>
                      <p style="color:#374151;">{summary or ''}</p>
                      <hr style="border:none;border-top:1px solid #e5e7eb;margin:16px 0;"/>
                      <h4 style="margin-bottom:8px;">📅 Proposer un rendez-vous</h4>
                      <p style="color:#6b7280;margin-bottom:12px;">
                        Cliquez sur un créneau ci-dessous pour confirmer le rendez-vous automatiquement :
                      </p>
                      {slot_links_html}
                      <p style="font-size:12px;color:#9ca3af;margin-top:20px;">
                        Ce rendez-vous sera enregistré dans votre espace LeadFlow dès confirmation.
                      </p>
                    </div>
                    """
                    email_resp = resend.Emails.send(
                        {
                            "from": "LeadFlow <onboarding@resend.dev>",
                            "to": [agency_contact_email],
                            "subject": f"Nouveau lead : {payload.name or 'Nouveau contact'} (Score {score})",
                            "html": html_body,
                        }
                    )
                    print(">>> Resend OK:", email_resp)
                except Exception as e:
                    print(">>> Resend ERROR:", repr(e))
            else:
                print(">>> Pas d'envoi email: RESEND_API_KEY ou agency_contact_email manquant")
        else:
            print(">>> Aucune agence trouvée pour id:", payload.agency_id)
    except Exception as e:
        print(">>> Erreur récupération agence / envoi email:", repr(e))

    return {
        "success": True,
        "lead_id": lead_id,
        "score": score,
        "summary": summary,
    }


@app.get(
    "/api/leads", response_model=List[LeadOut], dependencies=[Depends(verify_api_key)]
)
def get_leads(agency_id: str):
    db = get_supabase()
    try:
        resp = (
            db.table("leads")
            .select(
                "id, agency_id, crm_lead_id, name, email, phone, source, message, created_at, manual_score, "
                "lead_insights(score, summary, budget, timeline, property_type, "
                "location, email_reply_suggestion, tags)"
            )
            .eq("agency_id", agency_id)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture leads: {e}")

    leads = []
    for row in resp.data:
        insights = row.get("lead_insights") or []
        score = insights[0]["score"] if insights else None
        summary = insights[0]["summary"] if insights else None

        leads.append(
            LeadOut(
                lead_id=row["id"],
                agency_id=row["agency_id"],
                name=row.get("name"),
                email=row.get("email"),
                phone=row.get("phone"),
                source=row.get("source"),
                message=row.get("message"),
                created_at=row["created_at"],
                manual_score=row.get("manual_score"),
                score=score,
                summary=summary,
            )
        )

    return leads


@app.get(
    "/api/leads/{lead_id}",
    response_model=LeadDetail,
    dependencies=[Depends(verify_api_key)],
)
def get_lead(lead_id: str):
    db = get_supabase()
    try:
        resp = (
            db.table("leads")
            .select(
                "id, agency_id, crm_lead_id, name, email, phone, source, message, "
                "property_reference, created_at, manual_score, "
                "lead_insights(summary, score, budget, timeline, property_type, "
                "location, email_reply_suggestion, tags)"
            )
            .eq("id", lead_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture lead: {e}")

    if not resp.data:
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} introuvable.")

    row = resp.data[0]
    insights = (row.get("lead_insights") or [{}])[0]

    return LeadDetail(
        lead_id=row["id"],
        agency_id=row["agency_id"],
        crm_lead_id=row.get("crm_lead_id"),
        name=row.get("name"),
        email=row.get("email"),
        phone=row.get("phone"),
        source=row.get("source"),
        message=row.get("message"),
        property_reference=row.get("property_reference"),
        created_at=row["created_at"],
        summary=insights.get("summary"),
        manual_score=row.get("manual_score"),
        score=insights.get("score"),
        budget=insights.get("budget"),
        timeline=insights.get("timeline"),
        property_type=insights.get("property_type"),
        location=insights.get("location"),
        email_reply_suggestion=insights.get("email_reply_suggestion"),
        tags=insights.get("tags"),
    )


@app.patch(
    "/api/leads/{lead_id}",
    response_model=LeadPatchResponse,
    dependencies=[Depends(verify_api_key)],
)
def patch_lead(lead_id: str, body: LeadPatchRequest):
    db = get_supabase()
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(
            status_code=422, detail="Aucun champ à mettre à jour fourni."
        )

    try:
        resp = db.table("leads").update(updates).eq("id", lead_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur mise à jour: {e}")

    if not resp.data:
        raise HTTPException(status_code=404, detail=f"Lead {lead_id} introuvable.")

    row = resp.data[0]
    return LeadPatchResponse(
        lead_id=row["id"],
        agency_id=row["agency_id"],
        name=row.get("name"),
        email=row.get("email"),
        phone=row.get("phone"),
        source=row.get("source"),
        message=row.get("message"),
        manual_score=row.get("manual_score"),
        status=row.get("status"),
        notes=row.get("notes"),
        created_at=row["created_at"],
    )


@app.get("/api/logs", dependencies=[Depends(verify_api_key)])
def get_logs(limit: int = 100):
    db = get_supabase()
    try:
        resp = (
            db.table("api_logs")
            .select("id, created_at, method, path, status_code, duration_ms")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture logs: {e}")
    return resp.data

# ─── Route : créneaux disponibles d'une agence ───────────────────────────────
@app.get("/api/availability")
def get_availability(agency_id: str):
    """Retourne les créneaux non encore réservés d'une agence."""
    db = get_supabase()
    try:
        resp = (
            db.table("agency_availability")
            .select("id, start_at, duration_minutes")
            .eq("agency_id", agency_id)
            .eq("is_booked", False)
            .gte("start_at", datetime.now(timezone.utc).isoformat())
            .order("start_at")
            .execute()
        )
        return {"slots": resp.data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Route : réservation d'un créneau par le prospect ────────────────────────
class BookAppointmentPayload(BaseModel):
    agency_id: str
    slot_id: str          # UUID du créneau dans agency_availability
    prospect_name: str
    prospect_email: str
    prospect_phone: str = ""
    prospect_message: str = ""


@app.post("/api/book-appointment")
def book_appointment(payload: BookAppointmentPayload):
    db = get_supabase()

    # 1) Récupérer le créneau et vérifier qu'il est encore libre
    try:
        slot_resp = (
            db.table("agency_availability")
            .select("id, start_at, is_booked, agency_id")
            .eq("id", payload.slot_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture créneau : {e}")

    if not slot_resp.data:
        raise HTTPException(status_code=404, detail="Créneau introuvable.")

    slot = slot_resp.data[0]

    if slot["is_booked"]:
        raise HTTPException(status_code=409, detail="Ce créneau est déjà réservé.")

    # 2) Créer un lead minimal
    try:
        lead_resp = (
            db.table("leads")
            .insert(
                {
                    "agency_id": payload.agency_id,
                    "name": payload.prospect_name,
                    "email": payload.prospect_email,
                    "phone": payload.prospect_phone,
                    "message": payload.prospect_message or "Réservation via calendrier",
                    "source": "calendrier",
                }
            )
            .execute()
        )
        lead_id = lead_resp.data[0]["id"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur création lead : {e}")

    # Score IA sur le message du prospect (non bloquant)
    try:
        insights = analyse_lead_with_claude(
            payload.prospect_message or "Réservation via calendrier"
        )
        db.table("lead_insights").insert(
            {
                "lead_id": lead_id,
                "summary": insights.get("summary"),
                "score": insights.get("score", "B"),
                "budget": insights.get("budget"),
                "timeline": insights.get("timeline"),
                "property_type": insights.get("property_type"),
                "location": insights.get("location"),
                "email_reply_suggestion": insights.get("email_reply_suggestion"),
                "tags": insights.get("tags", []),
            }
        ).execute()
    except Exception:
        pass
    # 3) Créer le rendez-vous dans appointments
    try:
        db.table("appointments").insert({
            "lead_id": lead_id,
            "agency_id": payload.agency_id,
            "prospect_name": payload.prospect_name,
            "prospect_email": payload.prospect_email,
            "prospect_phone": payload.prospect_phone,
            "status": "confirmed",
            "start_at": slot["start_at"],
        }).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur création RDV : {e}")

    # 4) Marquer le créneau comme réservé
    try:
        db.table("agency_availability").update({"is_booked": True}).eq("id", payload.slot_id).execute()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur mise à jour créneau : {e}")

    # 5) Email à l'agence
    try:
        agency_resp = (
            db.table("agencies")
            .select("contact_email, name")
            .eq("id", payload.agency_id)
            .limit(1)
            .execute()
        )
        if agency_resp.data and RESEND_API_KEY:
            agency_email = agency_resp.data[0].get("contact_email")
            agency_name = agency_resp.data[0].get("name") or "Agence"
            dt_slot = datetime.fromisoformat(slot["start_at"])
            import zoneinfo
            human_date = dt_slot.astimezone(zoneinfo.ZoneInfo("Europe/Paris")).strftime("%A %d %B %Y à %Hh%M")
            resend.Emails.send({
                "from": "LeadFlow <onboarding@resend.dev>",
                "to": [agency_email],
                "subject": f"Nouveau RDV : {payload.prospect_name} — {human_date}",
                "html": f"""
                <div style="font-family:system-ui,sans-serif;max-width:600px;padding:24px;">
                  <h3>Rendez-vous confirmé via LeadFlow</h3>
                  <p><strong>Prospect :</strong> {payload.prospect_name}</p>
                  <p><strong>Email :</strong> {payload.prospect_email}</p>
                  <p><strong>Téléphone :</strong> {payload.prospect_phone or 'Non renseigné'}</p>
                  <p><strong>Message :</strong> {payload.prospect_message or '—'}</p>
                  <p><strong>Date du RDV :</strong> {human_date}</p>
                </div>
                """,
            })
    except Exception as e:
        print(">>> Email agence ERROR:", repr(e))

    # 6) Email de confirmation au prospect
    try:
        if RESEND_API_KEY and payload.prospect_email:
            dt_slot = datetime.fromisoformat(slot["start_at"])
            import zoneinfo
            human_date = dt_slot.astimezone(zoneinfo.ZoneInfo("Europe/Paris")).strftime("%A %d %B %Y à %Hh%M")
            resend.Emails.send({
                "from": "LeadFlow <onboarding@resend.dev>",
                "to": [payload.prospect_email],
                "subject": f"Votre rendez-vous est confirmé — {human_date}",
                "html": f"""
                <div style="font-family:system-ui,sans-serif;max-width:600px;padding:24px;">
                  <h3>Votre rendez-vous est confirmé ✅</h3>
                  <p>Bonjour {payload.prospect_name},</p>
                  <p>Votre rendez-vous avec notre agence est bien enregistré pour :</p>
                  <p style="font-size:18px;font-weight:bold;">{human_date}</p>
                  <p>À très bientôt.</p>
                </div>
                """,
            })
    except Exception as e:
        print(">>> Email prospect ERROR:", repr(e))

    return {"success": True, "message": "Rendez-vous confirmé."}
