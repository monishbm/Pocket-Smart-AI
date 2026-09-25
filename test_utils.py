from gemini_utils import extract_json_from_response, get_home_recommendations, shopping_links

def test_extract_json_plain():
    assert extract_json_from_response('{"a":1}') == {"a":1}

def test_extract_json_fenced():
    assert extract_json_from_response('```json\n{"a":2}\n```') == {"a":2}

def test_shopping_link_encoding():
    assert "warm+white+LED+bulb" in shopping_links("warm white LED bulb",["amazon"])["amazon"]

def test_home_fallback_stays_in_budget(monkeypatch):
    import gemini_utils
    monkeypatch.setattr(gemini_utils,"_client",None)
    result = get_home_recommendations({
        "total_budget":5000,"num_lights":5,"num_fans":2,"num_furniture":2,
        "num_dining_tables":1,"has_living_room":True,"has_kitchen":False,
        "has_bedroom":True,"additional_requirements":""
    })
    spent=sum(i["estimated_price"] for c in result["budget_breakdown"] for i in c["items"])
    assert spent <= 5000.01
    assert result["remaining_budget"] >= 0
