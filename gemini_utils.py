from __future__ import annotations

import json
import os
import re
import urllib.parse
from typing import Any, Optional

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
_client = genai.Client(api_key=API_KEY) if (genai and API_KEY) else None

PLATFORM_URLS = {
    "amazon": "https://www.amazon.in/s?k={q}",
    "flipkart": "https://www.flipkart.com/search?q={q}",
    "ikea": "https://www.ikea.com/in/en/search/?q={q}",
    "myntra": "https://www.myntra.com/{q}",
    "ajio": "https://www.ajio.com/search/?text={q}",
    "meesho": "https://www.meesho.com/search?q={q}",
    "bigbasket": "https://www.bigbasket.com/ps/?q={q}",
    "swiggy": "https://www.swiggy.com/search?query={q}",
    "zomato": "https://www.zomato.com/search?q={q}",
    "bookmyshow": "https://in.bookmyshow.com/explore/search?text={q}",
    "oyorooms": "https://www.oyorooms.com/search/?location={q}",
    "booking": "https://www.booking.com/searchresults.html?ss={q}",
    "makemytrip": "https://www.makemytrip.com/hotels/hotel-listing/?city={q}",
    "nobroker": "https://www.nobroker.in/property/rent/{q}",
    "bluestone": "https://www.bluestone.com/search.html?query={q}",
    "tanishq": "https://www.tanishq.co.in/search?q={q}",
    "caratlane": "https://www.caratlane.com/catalogsearch/result/?q={q}",
    "melorra": "https://www.melorra.com/jewellery/?q={q}",
    "google": "https://www.google.com/search?q={q}",
}

def ai_enabled() -> bool:
    return _client is not None

def extract_json_from_response(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("Empty AI response")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for i, char in enumerate(cleaned):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(cleaned[i:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    raise ValueError("Could not find a valid JSON object in AI response")

def shopping_links(search_terms: str, platforms: list[str]) -> dict[str, str]:
    q = urllib.parse.quote_plus(search_terms.strip())
    return {p: PLATFORM_URLS[p].format(q=q) for p in platforms if p in PLATFORM_URLS}

def _num(v: Any, default: float = 0.0) -> float:
    try:
        return max(float(v), 0.0)
    except (TypeError, ValueError):
        return default

def _normalize_item(item: dict[str, Any], platforms: list[str]) -> dict[str, Any]:
    name = str(item.get("name") or item.get("item") or "Recommended item").strip()
    search = str(item.get("search_terms") or name).strip()
    qty = max(int(_num(item.get("quantity"), 1)), 1)
    unit = _num(item.get("unit_price"), _num(item.get("estimated_price"), 0))
    total = _num(item.get("estimated_price"), unit * qty)
    if total <= 0 and unit > 0:
        total = unit * qty
    if unit <= 0 and qty:
        unit = total / qty
    return {
        "name": name,
        "description": str(item.get("description") or "Budget-friendly option selected for your requirements."),
        "quantity": qty,
        "unit_price": round(unit, 2),
        "estimated_price": round(total, 2),
        "search_terms": search,
        "shopping_links": shopping_links(search, platforms),
    }

def _finalize_budget(result: dict[str, Any], requested_budget: float,
                     default_platforms: list[str],
                     category_platforms: Optional[dict[str, list[str]]] = None) -> dict[str, Any]:
    budget = round(float(requested_budget), 2)
    breakdown = result.get("budget_breakdown")
    if not isinstance(breakdown, list):
        breakdown = []
    normalized = []
    spent = 0.0
    for raw_cat in breakdown:
        if not isinstance(raw_cat, dict):
            continue
        category = str(raw_cat.get("category") or "Miscellaneous").strip()
        platforms = default_platforms
        if category_platforms:
            low = category.lower()
            for needle, mapped in category_platforms.items():
                if needle in low:
                    platforms = mapped
                    break
        items = []
        for raw_item in raw_cat.get("items", []):
            if not isinstance(raw_item, dict):
                continue
            item = _normalize_item(raw_item, platforms)
            remaining = budget - spent
            if remaining <= 0:
                break
            if item["estimated_price"] > remaining:
                item["estimated_price"] = round(remaining, 2)
                item["unit_price"] = round(remaining / item["quantity"], 2)
            if item["estimated_price"] > 0:
                spent += item["estimated_price"]
                items.append(item)
        if items:
            total = round(sum(i["estimated_price"] for i in items), 2)
            normalized.append({"category": category, "allocated_budget": total, "items": items})
    result["total_budget"] = budget
    result["budget_breakdown"] = normalized
    result["remaining_budget"] = round(max(budget - spent, 0), 2)
    result["calculation_table"] = [{
        "category": cat["category"],
        "items_count": len(cat["items"]),
        "total_cost": round(sum(i["estimated_price"] for i in cat["items"]), 2),
        "percentage_of_budget": round(
            sum(i["estimated_price"] for i in cat["items"]) / budget * 100 if budget else 0, 2
        ),
    } for cat in normalized]
    if not isinstance(result.get("additional_suggestions"), list):
        result["additional_suggestions"] = []
    result["ai_mode"] = "gemini" if ai_enabled() else "demo"
    return result

def _gemini_json(prompt: str, image_path: str | None = None) -> dict[str, Any]:
    if not _client:
        raise RuntimeError("Gemini is not configured")

    if image_path:
        with Image.open(image_path) as image:
            image.load()
            contents = [prompt, image.copy()]
    else:
        contents = prompt

    config = types.GenerateContentConfig(
        temperature=0.4,
        response_mime_type="application/json",
    )

    try:
        response = _client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=config,
        )

        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")

        return extract_json_from_response(response.text)

    except Exception as e:
        raise RuntimeError(f"Gemini API Error: {e}")
def get_home_recommendations(data: dict[str, Any]) -> dict[str, Any]:
    budget = float(data["total_budget"])
    prompt = f"""
You are PocketSmart AI, an India-focused budget planner.
Create a realistic home-interior purchase plan in INR that never exceeds ₹{budget:.2f}.
Requirements:
- Lights: {data.get('num_lights',0)}
- Ceiling fans: {data.get('num_fans',0)}
- Furniture pieces: {data.get('num_furniture',0)}
- Dining tables: {data.get('num_dining_tables',0)}
- Living room: {bool(data.get('has_living_room'))}
- Kitchen: {bool(data.get('has_kitchen'))}
- Bedroom: {bool(data.get('has_bedroom'))}
- Additional requirements: {data.get('additional_requirements') or 'None'}
Return only JSON with total_budget, budget_breakdown (category + items), remaining_budget,
and additional_suggestions. Each item must include name, description, quantity, unit_price,
estimated_price, search_terms. Sum of estimated_price must be <= total_budget.
"""
    try:
        raw = _gemini_json(prompt) if ai_enabled() else _home_fallback(data)
    except Exception:
        raw = _home_fallback(data)
    return _finalize_budget(raw, budget, ["amazon", "flipkart", "ikea"])
def _home_fallback(data: dict[str, Any]) -> dict[str, Any]:
    budget = float(data["total_budget"])

    items = []

    if data.get("num_lights", 0):
        items.append({
            "category": "Lighting",
            "items": [{
                "name": "LED Lights",
                "description": "Energy efficient LED lights",
                "quantity": data["num_lights"],
                "unit_price": 500,
                "estimated_price": data["num_lights"] * 500,
                "search_terms": "LED lights"
            }]
        })

    if data.get("num_fans", 0):
        items.append({
            "category": "Fans",
            "items": [{
                "name": "Ceiling Fan",
                "description": "Standard ceiling fan",
                "quantity": data["num_fans"],
                "unit_price": 2500,
                "estimated_price": data["num_fans"] * 2500,
                "search_terms": "Ceiling fan"
            }]
        })

    if data.get("num_furniture", 0):
        items.append({
            "category": "Furniture",
            "items": [{
                "name": "Furniture",
                "description": "Basic furniture",
                "quantity": data["num_furniture"],
                "unit_price": 6000,
                "estimated_price": data["num_furniture"] * 6000,
                "search_terms": "Furniture"
            }]
        })

    if data.get("num_dining_tables", 0):
        items.append({
            "category": "Dining",
            "items": [{
                "name": "Dining Table",
                "description": "Wooden dining table",
                "quantity": data["num_dining_tables"],
                "unit_price": 12000,
                "estimated_price": data["num_dining_tables"] * 12000,
                "search_terms": "Dining table"
            }]
        })

    return {
        "total_budget": budget,
        "budget_breakdown": items,
        "remaining_budget": 0,
        "additional_suggestions": [
            "Compare prices before purchasing.",
            "Look for festive discounts."
        ]
    }

def _party_fallback(data: dict[str, Any]) -> dict[str, Any]:
    budget = float(data["total_budget"])
    weights = []
    if str(data.get("venue_type","")).lower() not in {"home","own home","house"}:
        weights.append(("Venue", .20))
    if data.get("needs_catering"):
        weights.append(("Catering", .38))
    if data.get("needs_decoration"):
        weights.append(("Decoration", .16))
    if data.get("needs_entertainment"):
        weights.append(("Entertainment", .16))
    weights.append(("Contingency", .10))
    total_weight = sum(w for _, w in weights)
    cats = []
    for category, weight in weights:
        amount = budget * weight / total_weight
        search = f"{data.get('party_type','party')} {category.lower()} {data.get('num_guests',1)} guests"
        cats.append({"category": category, "items": [{
            "name": f"{category} plan",
            "description": f"Suggested {category.lower()} allocation for the event.",
            "quantity": 1, "unit_price": round(amount,2),
            "estimated_price": round(amount,2), "search_terms": search,
        }]})
    return {"total_budget": budget, "budget_breakdown": cats, "additional_suggestions": [
        "Confirm vendor quotations before paying advances.",
        "Keep contingency available until the event is complete.",
        "For small events, home-made décor and buffet service can reduce costs.",
    ]}

def get_party_recommendations(data: dict[str, Any]) -> dict[str, Any]:
    budget = float(data["total_budget"])
    prompt = f"""
You are PocketSmart AI, an Indian event-budget assistant. Build an event plan <= ₹{budget:.2f}.
Type: {data.get('party_type')}; Guests: {data.get('num_guests')};
Venue: {data.get('venue_type') or 'Not specified'};
Catering: {bool(data.get('needs_catering'))}; Decoration: {bool(data.get('needs_decoration'))};
Entertainment: {bool(data.get('needs_entertainment'))};
Extra: {data.get('additional_requirements') or 'None'}.
Return only JSON with total_budget, budget_breakdown, remaining_budget, additional_suggestions.
Each item needs name, description, quantity, unit_price, estimated_price, search_terms.
Use Venue/Catering/Decoration/Entertainment/Contingency only as appropriate.
"""
    try:
        raw = _gemini_json(prompt) if ai_enabled() else _party_fallback(data)
    except Exception:
        raw = _party_fallback(data)
    mapping = {
        "venue": ["google","booking","makemytrip","oyorooms","nobroker"],
        "catering": ["swiggy","zomato","google"],
        "food": ["swiggy","zomato","bigbasket"],
        "decoration": ["amazon","flipkart","meesho"],
        "entertainment": ["bookmyshow","google","amazon"],
        "contingency": ["amazon","flipkart","google"],
    }
    return _finalize_budget(raw, budget, ["google","amazon","flipkart"], mapping)

def _jewelry_fallback(data: dict[str, Any], image_path: str | None) -> dict[str, Any]:
    budget = float(data["total_budget"])
    recs = []
    for name, share in [("Pendant necklace",.36),("Earrings",.26),("Bracelet",.20)]:
        amount = round(budget*share,2)
        recs.append({
            "name": name,
            "description": f"Versatile option suitable for a {data.get('occasion','special')} occasion.",
            "quantity": 1, "unit_price": amount, "estimated_price": amount,
            "search_terms": f"{data.get('preferences') or 'elegant'} {name} India",
        })
    result = {"total_budget": budget, "jewelry_recommendations": recs, "styling_tips": [
        "Choose one statement piece and keep other accessories subtle.",
        "Match metal tones across your accessories.",
        "Verify size, return policy, hallmark/certification, and material details before buying.",
    ]}
    if image_path:
        result["outfit_analysis"] = {
            "colors": ["Image supplied"],
            "style": "Demo mode cannot reliably infer style",
            "formality": "Configure Gemini for visual outfit analysis",
        }
    return result

def get_jewelry_recommendations(data: dict[str, Any], image_path: str | None = None) -> dict[str, Any]:
    budget = float(data["total_budget"])
    image_instruction = ("An outfit image is included. Analyze dominant colors, style and formality."
                         if image_path else "No outfit image is included.")
    prompt = f"""
You are PocketSmart AI, an India-focused jewelry assistant.
Budget: ₹{budget:.2f}; Occasion: {data.get('occasion')};
Preferences: {data.get('preferences') or 'Not specified'}.
{image_instruction}
Return only JSON with optional outfit_analysis, total_budget, jewelry_recommendations,
remaining_budget, styling_tips. Each jewelry item needs name, description, quantity,
unit_price, estimated_price, search_terms. Keep total estimated_price <= total_budget.
"""
    try:
        raw = _gemini_json(prompt, image_path) if ai_enabled() else _jewelry_fallback(data, image_path)
    except Exception:
        raw = _jewelry_fallback(data, image_path)
    recs = raw.get("jewelry_recommendations")
    if not isinstance(recs, list):
        recs = []
    spent = 0.0
    out = []
    platforms = ["amazon","flipkart","bluestone","tanishq","caratlane","melorra","meesho"]
    for raw_item in recs:
        if not isinstance(raw_item, dict):
            continue
        item = _normalize_item(raw_item, platforms)
        remaining = budget - spent
        if remaining <= 0:
            break
        if item["estimated_price"] > remaining:
            item["estimated_price"] = round(remaining,2)
            item["unit_price"] = round(remaining/item["quantity"],2)
        if item["estimated_price"] > 0:
            spent += item["estimated_price"]
            out.append(item)
    result = {
        "total_budget": round(budget,2),
        "jewelry_recommendations": out,
        "remaining_budget": round(max(budget-spent,0),2),
        "styling_tips": raw.get("styling_tips") if isinstance(raw.get("styling_tips"),list) else [],
        "calculation_table": [{
            "category":"Jewelry","items_count":len(out),"total_cost":round(spent,2),
            "percentage_of_budget":round(spent/budget*100 if budget else 0,2),
        }],
        "ai_mode":"gemini" if ai_enabled() else "demo",
    }
    if image_path and isinstance(raw.get("outfit_analysis"), dict):
        result["outfit_analysis"] = raw["outfit_analysis"]
    return result
