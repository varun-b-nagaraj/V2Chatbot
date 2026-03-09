from starlette.testclient import TestClient

from api.index import app
from shopping_assistant import app as assistant_module


client = TestClient(app)


def test_options_preflight_returns_cors_headers() -> None:
    response = client.options(
        "/chat",
        headers={
            "Origin": "https://rrhscoop.roundrockisd.org",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 204
    assert response.headers["Access-Control-Allow-Origin"] == "*"


def test_health_route_with_mocked_catalog(monkeypatch) -> None:
    async def fake_get_catalog_for_query(query: str, limit: int = 24):
        return {"products": [], "total": 0, "last_updated": 0}

    monkeypatch.setattr(assistant_module.mcp_client, "get_catalog_for_query", fake_get_catalog_for_query)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_chat_route_with_mocked_dependencies(monkeypatch) -> None:
    catalog = {
        "products": [
            {
                "id": 101,
                "name": "Protein Shake",
                "url": "https://example.com/protein-shake",
                "variants": [
                    {
                        "variantKey": "101:1",
                        "combinationId": 1,
                        "label": "Chocolate",
                        "price": 4.99,
                        "effectiveSku": "SHAKE-CHOC",
                        "in_stock": True,
                        "options": [{"name": "Flavor", "value": "Chocolate"}],
                    }
                ],
            }
        ],
        "total": 1,
        "last_updated": 0,
    }

    async def fake_tool_roundtrip(request):
        return "[V:101:1] Protein Shake - Chocolate - $4.99\nGreat post-workout pick.", [], catalog

    monkeypatch.setattr(assistant_module, "_run_llm_tool_roundtrip", fake_tool_roundtrip)

    headers = {}
    if assistant_module.API_KEY:
        headers["Authorization"] = f"Bearer {assistant_module.API_KEY}"

    response = client.post(
        "/chat",
        json={"message": "What protein drinks do you have?"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["validated"] is True
    assert body["products"][0]["variantKey"] == "101:1"
    assert "[V:" not in body["message"]


def test_chat_route_supports_remove_from_cart(monkeypatch) -> None:
    catalog = {
        "products": [
            {
                "id": 101,
                "name": "Protein Shake",
                "url": "https://example.com/protein-shake",
                "variants": [
                    {
                        "variantKey": "101:1",
                        "combinationId": 1,
                        "label": "Chocolate",
                        "price": 4.99,
                        "effectiveSku": "SHAKE-CHOC",
                        "in_stock": True,
                        "options": [{"name": "Flavor", "value": "Chocolate"}],
                    }
                ],
            }
        ],
        "total": 1,
        "last_updated": 0,
    }

    async def fake_tool_roundtrip(request):
        return "[V:101:1] Protein Shake - Chocolate - $4.99", [], catalog

    monkeypatch.setattr(assistant_module, "_run_llm_tool_roundtrip", fake_tool_roundtrip)

    headers = {}
    if assistant_module.API_KEY:
        headers["Authorization"] = f"Bearer {assistant_module.API_KEY}"

    response = client.post(
        "/chat",
        json={"message": "Remove the protein shake from my cart"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Removed from your cart."
    assert body["cart_actions"][0]["quantity"] == -1


def test_chat_route_supports_pending_remove_selection(monkeypatch) -> None:
    catalog = {"products": [], "total": 0, "last_updated": 0}

    async def fake_get_catalog_for_query(query: str, limit: int = 24):
        return catalog

    monkeypatch.setattr(assistant_module.mcp_client, "get_catalog_for_query", fake_get_catalog_for_query)

    headers = {}
    if assistant_module.API_KEY:
        headers["Authorization"] = f"Bearer {assistant_module.API_KEY}"

    response = client.post(
        "/chat",
        json={
            "message": "the first one",
            "pending": {
                "type": "choose_for_cart",
                "quantity": -1,
                "options": [
                    {
                        "id": 101,
                        "name": "Protein Shake",
                        "combinationId": 1,
                        "variantKey": "101:1",
                        "variantLabel": "Chocolate",
                        "price": 4.99,
                        "sku": "SHAKE-CHOC",
                        "url": "https://example.com/protein-shake",
                        "selectedOptions": [{"name": "Flavor", "value": "Chocolate"}]
                    },
                    {
                        "id": 101,
                        "name": "Protein Shake",
                        "combinationId": 2,
                        "variantKey": "101:2",
                        "variantLabel": "Vanilla",
                        "price": 4.99,
                        "sku": "SHAKE-VAN",
                        "url": "https://example.com/protein-shake",
                        "selectedOptions": [{"name": "Flavor", "value": "Vanilla"}]
                    }
                ]
            }
        },
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "Removed from your cart."
    assert body["cart_actions"][0]["quantity"] == -1
    assert body["cart_actions"][0]["combinationId"] == 1


def test_chat_route_returns_graceful_response_when_catalog_is_down(monkeypatch) -> None:
    async def fake_tool_roundtrip(request):
        raise assistant_module.CatalogUnavailableError("down")

    monkeypatch.setattr(assistant_module, "_run_llm_tool_roundtrip", fake_tool_roundtrip)

    headers = {}
    if assistant_module.API_KEY:
        headers["Authorization"] = f"Bearer {assistant_module.API_KEY}"

    response = client.post(
        "/chat",
        json={"message": "show me chips"},
        headers=headers,
    )
    assert response.status_code == 503


def test_chat_route_handles_mcp_session_open_failure(monkeypatch) -> None:
    async def fake_tool_roundtrip(request):
        raise RuntimeError("session open failed")

    monkeypatch.setattr(assistant_module, "_run_llm_tool_roundtrip", fake_tool_roundtrip)

    headers = {}
    if assistant_module.API_KEY:
        headers["Authorization"] = f"Bearer {assistant_module.API_KEY}"

    response = client.post(
        "/chat",
        json={"message": "show me chips"},
        headers=headers,
    )
    assert response.status_code == 502


def test_chat_tools_executes_llm_requested_tool(monkeypatch) -> None:
    calls = {"count": 0}

    class DummyProtocolClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

    async def fake_call_ollama_message(messages, tools=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "catalog_search",
                                "arguments": {"keyword": "chips", "limit": 3},
                            }
                        }
                    ],
                }
            }
        return {
            "message": {
                "role": "assistant",
                "content": "Here are a few good chip options right now.",
            }
        }

    async def fake_tool_call(tool_name: str, payload: dict, **kwargs):
        assert tool_name == "catalog_search"
        assert payload["keyword"] == "chips"
        return {"items": [{"id": 1, "name": "Sea Salt Chips"}], "total": 1, "count": 1}

    async def fake_get_tool_schemas(force_refresh: bool = False):
        return [{"type": "function", "function": {"name": "catalog_search", "parameters": {"type": "object"}}}]

    monkeypatch.setattr(assistant_module, "call_ollama_message", fake_call_ollama_message)
    monkeypatch.setattr(assistant_module.mcp_client, "_call_tool_with_retry", fake_tool_call)
    monkeypatch.setattr(assistant_module.mcp_client, "get_tool_schemas", fake_get_tool_schemas)
    monkeypatch.setattr(assistant_module.mcp_client, "_new_protocol_client", lambda: DummyProtocolClient())

    headers = {}
    if assistant_module.API_KEY:
        headers["Authorization"] = f"Bearer {assistant_module.API_KEY}"

    response = client.post(
        "/chat-tools",
        json={"message": "show me chips"},
        headers=headers,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["tool_calls"][0]["name"] == "catalog_search"
    assert body["tool_calls"][0]["ok"] is True
    assert body["message"] == "Here are a few good chip options right now."
