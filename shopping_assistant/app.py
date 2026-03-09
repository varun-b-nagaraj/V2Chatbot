"""Vercel-friendly shopping assistant API backed by Ollama and an MCP tools server."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from dotenv import load_dotenv
from fastmcp import Client as FastMCPClient
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from e_series_mcp.server import mcp as local_mcp_server

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "8000"))
MCP_URL = os.getenv("MCP_URL", "").strip()
USE_LOCAL_MCP = os.getenv("USE_LOCAL_MCP", "true").strip().lower() in {"1", "true", "yes", "on"}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-v3.1:671b-cloud").strip()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
OLLAMA_TIMEOUT_MS = int(os.getenv("OLLAMA_TIMEOUT_MS", "120000"))
OLLAMA_FIRST_TOKEN_TIMEOUT_MS = int(os.getenv("OLLAMA_FIRST_TOKEN_TIMEOUT_MS", "20000"))
API_KEY = os.getenv("API_KEY", "").strip()
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.15"))
CATALOG_CACHE_TTL_SEC = int(os.getenv("CATALOG_CACHE_TTL_SEC", "300"))
CHAT_CATALOG_LIMIT = int(os.getenv("CHAT_CATALOG_LIMIT", "24"))

SYSTEM_PROMPT = """
You are a concise, high-conversion shopping assistant for an online store.

Goals:
- Help the shopper find the best match fast.
- Reduce friction and get to checkout with as little back-and-forth as possible.
- Increase conversion by being specific, confident, and useful without sounding pushy.

Rules:
- Only use products and options from the live catalog provided.
- Never invent names, prices, stock, or product details.
- Keep answers short and commercially useful.
- Default to 2 or 3 strong recommendations, not long lists.
- Explain each recommendation with one concrete reason tied to the shopper's request.
- When the shopper wants to add to cart and there is one exact match, confirm it directly.
- When several in-stock options match, ask one short clarifying question.
- Prefer decisive guidance over vague summaries.
- Never mention internal tags or internal data structures to the shopper.

Style:
- Warm, direct, fast.
- Short paragraphs or short lists.
- Helpful salesperson, not hype.
"""

TOOL_SYSTEM_PROMPT = """
You are a store assistant with live tools.

Rules:
- Use tools whenever the user asks for product, stock, options, policies, or store info.
- Never invent product data.
- Prefer `catalog_search` first for shopping requests.
- Use `product_get` for a specific product.
- Use `product_variants_get` when the user needs options, sizes, or flavors.
- Use `store_info_get` and `faq_search` for policies and store operations.
- After you receive tool results, answer clearly and briefly.
- If a tool fails, explain that live inventory is temporarily unavailable instead of making anything up.
- If you mention a specific sellable option, include exactly one internal tag in this format: [V:productId:combinationId]
- Use those tags only for real in-stock options returned by the tools.
- If a product has multiple options and the shopper wants to add or remove it, ask one short clarification question.
- Keep responses short and specific.
"""

TOOL_LOOP_PROMPT = """
You are operating in an agent loop with live store tools.

You must respond with JSON only, no markdown.

Valid shapes:
{"action":"tool","name":"catalog_search","arguments":{"keyword":"chips","limit":8}}
{"action":"answer","message":"Here are a few chip options..."}

Rules:
- Pick exactly one action each turn.
- Use tools whenever product, stock, options, FAQ, or store-info facts are needed.
- Do not invent product data.
- Use one tool at a time.
- After tool results are provided, either call another tool or answer.
- If you answer with specific sellable options, include [V:productId:combinationId] tags.
- Use only IDs and facts present in tool results.
- Keep answers concise.
"""

SELECTION_SYSTEM_PROMPT = """
Return only valid JSON in the shape {"indexes":[0,1]}.
Use 0-based indexes.
If the user did not clearly pick an option, return {"indexes":[]}.
"""

VARIANT_TAG_RE = re.compile(r"\[V:(\d+:\d+)\]")
ORDINAL_WORDS = {
    "first": 0,
    "second": 1,
    "third": 2,
    "fourth": 3,
    "fifth": 4,
    "sixth": 5,
}
SEARCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "best",
    "buy",
    "cart",
    "drink",
    "for",
    "get",
    "good",
    "have",
    "i",
    "in",
    "is",
    "item",
    "items",
    "like",
    "me",
    "my",
    "need",
    "of",
    "on",
    "or",
    "recommend",
    "show",
    "something",
    "some",
    "snack",
    "that",
    "the",
    "these",
    "those",
    "to",
    "want",
    "what",
    "with",
}


class CatalogUnavailableError(RuntimeError):
    pass


class Message(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=5000)


class ProductLink(BaseModel):
    id: int
    name: str
    combinationId: int
    variantKey: str
    variantLabel: str
    price: float
    sku: str | None = None
    url: str | None = None
    selectedOptions: list[dict[str, Any]] = Field(default_factory=list)


class PendingChoice(BaseModel):
    type: str = Field(..., pattern="^choose_for_cart$")
    options: list[ProductLink] = Field(default_factory=list)
    quantity: int = Field(1, ge=-20, le=20)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    history: list[Message] | None = Field(None, max_length=20)
    pending: PendingChoice | None = None
    stream: bool | None = False


class CartAction(BaseModel):
    type: str
    productId: int
    combinationId: int
    quantity: int
    sku: str | None = None
    options: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    message: str
    model: str
    in_stock_products: int
    validated: bool = True
    products: list[ProductLink] = Field(default_factory=list)
    cart_actions: list[CartAction] = Field(default_factory=list)
    pending: PendingChoice | None = None


class ToolCallTrace(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ToolChatResponse(BaseModel):
    message: str
    model: str
    tool_calls: list[ToolCallTrace] = Field(default_factory=list)
    rounds: int


def _normalize_ollama_base_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return "http://localhost:11434"
    if "ollama.com" in url and url.startswith("http://"):
        url = "https://" + url.removeprefix("http://")
    return url.rstrip("/")


def _ollama_chat_url() -> str:
    base = _normalize_ollama_base_url(OLLAMA_BASE_URL)
    if base.endswith("/api"):
        return f"{base}/chat"
    return f"{base}/api/chat"


def _make_variant_key(product_id: Any, combination_id: Any) -> str:
    return f"{int(product_id)}:{int(combination_id)}"


def _strip_variant_tags(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"^\s*(?:[-•]\s*)?\[V:\d+:\d+\]\s*", "", text, flags=re.MULTILINE)
    cleaned = VARIANT_TAG_RE.sub("", cleaned)
    return cleaned.strip()


def _extract_variant_keys_in_order(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for match in VARIANT_TAG_RE.finditer(text or ""):
        key = match.group(1)
        lowered = key.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered.append(key)
    return ordered


def _looks_like_recommendation(text: str) -> bool:
    return bool(re.search(r"\$\s*\d", text or "")) and bool(re.search(r"(^|\n)\s*(?:[-•]|\d+\.)", text or ""))


def _parse_quantity(text: str) -> int:
    match = re.search(r"\b(\d{1,2})\s*(?:x|items?)?\b", (text or "").lower())
    if not match:
        return 1
    return max(1, min(int(match.group(1)), 20))


def _should_add_to_cart(text: str) -> bool:
    value = (text or "").lower()
    return bool(re.search(r"\b(add|put|throw)\b", value)) and ("cart" in value or "bag" in value or value.startswith("add "))


def _should_remove_from_cart(text: str) -> bool:
    value = (text or "").lower()
    return bool(
        re.search(r"\b(remove|delete|take)\b", value)
        and ("cart" in value or "bag" in value or "out" in value)
    )


def _cart_quantity_delta(text: str) -> int:
    quantity = _parse_quantity(text)
    if _should_remove_from_cart(text):
        return -quantity
    return quantity


def _is_cart_update_request(text: str) -> bool:
    return _should_add_to_cart(text) or _should_remove_from_cart(text)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _normalize_selected_options(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        return [{"name": str(k), "value": str(v)} for k, v in raw.items() if k and v]
    if not isinstance(raw, list):
        return []
    selected: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("optionName") or item.get("option_name")
        value = item.get("value") or item.get("text") or item.get("valueName") or item.get("title")
        if name and value:
            selected.append({"name": str(name), "value": str(value)})
    return selected


def _build_product_link(product: dict[str, Any], variant: dict[str, Any]) -> ProductLink | None:
    variant_key = str(variant.get("variantKey") or "").strip()
    if not variant_key:
        return None
    return ProductLink(
        id=int(product.get("id") or variant_key.split(":")[0]),
        name=product.get("name", "Unknown"),
        combinationId=int(variant.get("combinationId") or variant_key.split(":")[1]),
        variantKey=variant_key,
        variantLabel=variant.get("label") or "Default",
        price=float(variant.get("price", 0) or 0),
        sku=(variant.get("effectiveSku") or "").strip() or None,
        url=product.get("url") or None,
        selectedOptions=_normalize_selected_options(variant.get("options")),
    )


def _safe_fallback(catalog: dict[str, Any]) -> tuple[str, list[ProductLink]]:
    products = catalog.get("products", [])
    links: list[ProductLink] = []
    for product in products:
        for variant in product.get("variants") or []:
            if variant.get("in_stock") is False:
                continue
            link = _build_product_link(product, variant)
            if link:
                links.append(link)
            if len(links) == 3:
                break
        if len(links) == 3:
            break
    if not links:
        return "I’m having trouble reaching live inventory right now. Please try again in a moment.", []
    lines = ["A few strong options right now:"]
    for link in links:
        lines.append(f"- {link.name} - {link.variantLabel} - ${link.price:.2f}")
    lines.append("Tell me what you want and I’ll narrow it down fast.")
    return "\n".join(lines), links


def _inventory_unavailable_message(user_text: str) -> str:
    if _is_cart_update_request(user_text):
        return "I’m having trouble reaching live inventory right now, so I can’t safely update your cart yet. Try again in a moment."
    return "I’m having trouble reaching live inventory right now. Try again in a moment and I’ll pull live options for you."


def _format_catalog_for_prompt(catalog: dict[str, Any]) -> str:
    products = catalog.get("products", [])
    if not products:
        return "CATALOG: []\nALLOWED_VARIANTS: []"

    rows: list[dict[str, Any]] = []
    for product in products:
        for variant in product.get("variants") or []:
            if variant.get("in_stock") is False or not variant.get("variantKey"):
                continue
            rows.append(
                {
                    "variantKey": variant.get("variantKey"),
                    "name": product.get("name", "Unknown"),
                    "label": variant.get("label") or "Default",
                    "price": float(variant.get("price", 0) or 0),
                    "url": product.get("url") or "",
                }
            )
    preview = json.dumps(rows[:80], ensure_ascii=True)
    return f"ALLOWED_VARIANTS: {preview}"


class MCPClient:
    def __init__(self) -> None:
        self.catalog_cache: dict[str, Any] | None = None
        self.cache_timestamp = 0.0
        self.cache_ttl = CATALOG_CACHE_TTL_SEC
        self.search_cache: dict[str, tuple[float, dict[str, Any]]] = {}
        self.tool_schema_cache: list[dict[str, Any]] | None = None
        self.tool_schema_timestamp = 0.0
        self.timeout_sec = float(os.getenv("MCP_TIMEOUT_SEC", "30"))
        self.max_retries = int(os.getenv("MCP_MAX_RETRIES", "3"))
        self.retry_base_sec = float(os.getenv("MCP_RETRY_BASE_SEC", "0.4"))
        self.lock = asyncio.Lock()
        self.search_lock = asyncio.Lock()
        self.tool_lock = asyncio.Lock()

    def _cache_valid(self, timestamp: float) -> bool:
        return (time.time() - timestamp) < self.cache_ttl

    def _get_search_cache(self, key: str) -> dict[str, Any] | None:
        cached = self.search_cache.get(key)
        if not cached:
            return None
        timestamp, value = cached
        if self._cache_valid(timestamp):
            return value
        self.search_cache.pop(key, None)
        return None

    @staticmethod
    def _keyword_from_query(query: str | None) -> str:
        tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
        filtered = [token for token in tokens if len(token) > 1 and token not in SEARCH_STOPWORDS]
        return " ".join(filtered[:6]).strip()

    def _new_protocol_client(self) -> FastMCPClient:
        if USE_LOCAL_MCP:
            return FastMCPClient(local_mcp_server, timeout=self.timeout_sec, name="v2-chatbot")
        if not MCP_URL:
            raise RuntimeError("MCP_URL is not configured and USE_LOCAL_MCP is disabled")
        return FastMCPClient(MCP_URL, timeout=self.timeout_sec, name="v2-chatbot")

    @staticmethod
    def _parse_protocol_tool_result(result: Any) -> dict[str, Any]:
        data = getattr(result, "data", None)
        if isinstance(data, dict):
            return data
        if data is not None:
            return {"result": data}

        structured = getattr(result, "structured_content", None)
        if isinstance(structured, dict):
            return structured
        if structured is not None:
            return {"result": structured}

        content = getattr(result, "content", None) or []
        text_parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                text_parts.append(text)
        if len(text_parts) == 1:
            try:
                parsed = json.loads(text_parts[0])
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                return parsed
            if parsed is not None:
                return {"result": parsed}
        if text_parts:
            return {"content": text_parts}
        return {}

    @staticmethod
    def _tool_to_ollama_schema(tool: Any) -> dict[str, Any]:
        parameters = getattr(tool, "inputSchema", None)
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            parameters = {"type": "object", "properties": {}}
        return {
            "type": "function",
            "function": {
                "name": getattr(tool, "name", ""),
                "description": getattr(tool, "description", "") or "",
                "parameters": parameters,
            },
        }

    async def get_tool_schemas(self, force_refresh: bool = False) -> list[dict[str, Any]]:
        if not force_refresh and self.tool_schema_cache and self._cache_valid(self.tool_schema_timestamp):
            return self.tool_schema_cache

        async with self.tool_lock:
            if not force_refresh and self.tool_schema_cache and self._cache_valid(self.tool_schema_timestamp):
                return self.tool_schema_cache

            try:
                async with self._new_protocol_client() as client:
                    tools = await client.list_tools()
            except Exception as exc:
                raise CatalogUnavailableError("Tool schema discovery unavailable") from exc
            schemas = [self._tool_to_ollama_schema(tool) for tool in tools if getattr(tool, "name", None)]
            self.tool_schema_cache = schemas
            self.tool_schema_timestamp = time.time()
            return schemas

    async def _call_tool_with_retry(
        self,
        tool_name: str,
        payload: dict[str, Any],
        *,
        client: FastMCPClient | None = None,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                if client is not None:
                    result = await client.call_tool(tool_name, payload or {}, raise_on_error=True)
                else:
                    async with self._new_protocol_client() as fresh_client:
                        result = await fresh_client.call_tool(tool_name, payload or {}, raise_on_error=True)
                if getattr(result, "is_error", False):
                    raise RuntimeError(f"MCP tool returned error: {tool_name}")
                return self._parse_protocol_tool_result(result)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "MCP %s failed attempt %s/%s: %s: %s",
                    tool_name,
                    attempt,
                    self.max_retries,
                    type(exc).__name__,
                    exc,
                )
                if attempt < self.max_retries:
                    delay = self.retry_base_sec * (2 ** (attempt - 1)) + random.uniform(0, self.retry_base_sec)
                    await asyncio.sleep(delay)
        raise RuntimeError(f"MCP {tool_name} failed") from last_error

    async def _fetch_variants_for_product(
        self,
        product_id: int,
        *,
        client: FastMCPClient | None = None,
    ) -> list[dict[str, Any]]:
        data = await self._call_tool_with_retry("product_variants_get", {"product_id": product_id}, client=client)
        variants = data.get("variants", [])
        if not isinstance(variants, list):
            return []
        normalized: list[dict[str, Any]] = []
        for variant in variants:
            if not isinstance(variant, dict):
                continue
            if variant.get("in_stock") is False:
                continue
            cid = variant.get("combinationId")
            if cid is None:
                continue
            pid = int(variant.get("productId") or product_id)
            normalized.append(
                {
                    "productId": pid,
                    "combinationId": int(cid),
                    "variantKey": variant.get("variantKey") or _make_variant_key(pid, cid),
                    "effectiveSku": variant.get("effectiveSku") or "",
                    "label": variant.get("label") or "Default",
                    "price": float(variant.get("price", 0) or 0),
                    "in_stock": True,
                    "options": _normalize_selected_options(variant.get("options")),
                }
            )
        return normalized

    async def get_catalog(self, force_refresh: bool = False) -> dict[str, Any]:
        now = time.time()
        if not force_refresh and self.catalog_cache and self._cache_valid(self.cache_timestamp):
            return self.catalog_cache

        async with self.lock:
            now = time.time()
            if not force_refresh and self.catalog_cache and self._cache_valid(self.cache_timestamp):
                return self.catalog_cache

            try:
                async with self._new_protocol_client() as client:
                    all_products: list[dict[str, Any]] = []
                    offset = 0
                    limit = 100

                    while True:
                        data = await self._call_tool_with_retry("catalog_search", {"limit": limit, "offset": offset}, client=client)
                        items = data.get("items", [])
                        if not isinstance(items, list):
                            break
                        enabled_batch = [item for item in items if isinstance(item, dict) and item.get("enabled", False)]
                        all_products.extend(enabled_batch)
                        if len(items) < limit or offset > 1000:
                            break
                        offset += limit

                    await self._enrich_products(all_products, client=client)

                catalog = {
                    "products": all_products,
                    "total": len(all_products),
                    "last_updated": now,
                }
                self.catalog_cache = catalog
                self.cache_timestamp = now
                return catalog
            except Exception as exc:
                if self.catalog_cache:
                    logger.warning("Returning stale full catalog after MCP failure: %s", exc)
                    return self.catalog_cache
                raise CatalogUnavailableError("Live catalog unavailable") from exc

    async def _enrich_products(
        self,
        products: list[dict[str, Any]],
        *,
        client: FastMCPClient | None = None,
    ) -> None:
        semaphore = asyncio.Semaphore(8)

        async def enrich(product: dict[str, Any]) -> None:
            product_id = int(product.get("id") or 0)
            if product_id <= 0:
                product["variants"] = []
                return
            async with semaphore:
                try:
                    product["variants"] = await self._fetch_variants_for_product(product_id, client=client)
                except Exception as exc:
                    logger.warning("Variant enrichment failed for %s: %s", product_id, exc)
                    product["variants"] = []

        await asyncio.gather(*(enrich(product) for product in products))

    async def get_catalog_for_query(self, query: str, limit: int = CHAT_CATALOG_LIMIT) -> dict[str, Any]:
        keyword = self._keyword_from_query(query)
        cache_key = f"{keyword}:{limit}"
        cached = self._get_search_cache(cache_key)
        if cached:
            return cached

        async with self.search_lock:
            cached = self._get_search_cache(cache_key)
            if cached:
                return cached

            search_attempts = []
            if keyword:
                search_attempts.append({"keyword": keyword, "limit": limit, "offset": 0})
            search_attempts.append({"limit": min(limit, 12), "offset": 0})

            last_exc: Exception | None = None
            try:
                async with self._new_protocol_client() as client:
                    for payload in search_attempts:
                        try:
                            data = await self._call_tool_with_retry("catalog_search", payload, client=client)
                            items = data.get("items", [])
                            if not isinstance(items, list):
                                items = []
                            products = [item for item in items if isinstance(item, dict) and item.get("enabled", False)]
                            await self._enrich_products(products, client=client)
                            catalog = {
                                "products": products,
                                "total": len(products),
                                "last_updated": time.time(),
                            }
                            self.search_cache[cache_key] = (time.time(), catalog)
                            return catalog
                        except Exception as exc:
                            last_exc = exc
                            logger.warning("Search catalog attempt failed for query '%s': %s", query, exc)
            except Exception as exc:
                last_exc = exc
                logger.warning("Failed to open MCP session for query '%s': %s", query, exc)

            if self.catalog_cache:
                logger.warning("Falling back to cached full catalog for query '%s'", query)
                return self.catalog_cache
            raise CatalogUnavailableError("Search catalog unavailable") from last_exc


mcp_client = MCPClient()
ollama_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global ollama_client
    ollama_client = httpx.AsyncClient(timeout=httpx.Timeout(OLLAMA_TIMEOUT_MS / 1000))
    yield
    if ollama_client is not None:
        await ollama_client.aclose()


app = FastAPI(
    title="FactMCP Shopping Assistant",
    version="1.0.0",
    description="Vercel-friendly chat endpoint that uses Ollama and an MCP inventory backend.",
    lifespan=lifespan,
)

def _apply_cors_headers(response: Response) -> Response:
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Max-Age"] = "86400"
    return response


@app.middleware("http")
async def allow_all_cors(request: Request, call_next):
    if request.method.upper() == "OPTIONS":
        return _apply_cors_headers(Response(status_code=204))

    response = await call_next(request)
    return _apply_cors_headers(response)


async def call_ollama(messages: list[dict[str, str]]) -> str:
    response = await call_ollama_message(messages)
    return response.get("message", {}).get("content", "")


async def call_ollama_message(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": MODEL_TEMPERATURE},
    }
    if tools:
        payload["tools"] = tools
    headers: dict[str, str] = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    try:
        response = await ollama_client.post(_ollama_chat_url(), json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {}
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Ollama timed out") from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Ollama returned %s: %s", exc.response.status_code, exc.response.text[:300])
        raise HTTPException(status_code=502, detail="Ollama request failed") from exc


async def call_ollama_stream(messages: list[dict[str, str]]):
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="LLM client not initialized")

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": MODEL_TEMPERATURE},
    }
    headers: dict[str, str] = {}
    if OLLAMA_API_KEY:
        headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

    timeout = httpx.Timeout(OLLAMA_TIMEOUT_MS / 1000, read=OLLAMA_FIRST_TOKEN_TIMEOUT_MS / 1000)
    async with httpx.AsyncClient(timeout=timeout) as stream_client:
        async with stream_client.stream("POST", _ollama_chat_url(), json=payload, headers=headers) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content")
                if content:
                    yield content
                if data.get("done"):
                    break


def _resolve_product_links(variant_keys: list[str], products: list[dict[str, Any]]) -> list[ProductLink]:
    indexed: dict[str, ProductLink] = {}
    for product in products:
        for variant in product.get("variants") or []:
            link = _build_product_link(product, variant)
            if link:
                indexed[link.variantKey.lower()] = link
    resolved: list[ProductLink] = []
    for key in variant_keys:
        link = indexed.get(key.lower())
        if link:
            resolved.append(link)
    return resolved


def _validate_variant_keys(text: str, products: list[dict[str, Any]]) -> tuple[bool, list[str], list[ProductLink]]:
    keys = _extract_variant_keys_in_order(text)
    links = _resolve_product_links(keys, products)
    linked_keys = {link.variantKey.lower() for link in links}
    invalid = [key for key in keys if key.lower() not in linked_keys]
    return len(invalid) == 0, invalid, links


def _build_cart_actions(request_text: str, product_links: list[ProductLink]) -> list[CartAction]:
    if not _is_cart_update_request(request_text) or not product_links:
        return []
    quantity = _cart_quantity_delta(request_text)
    chosen = _select_cart_candidates(request_text, product_links)
    if not chosen:
        return []
    return [
        CartAction(
            type="cart.add",
            productId=link.id,
            combinationId=link.combinationId,
            quantity=quantity,
            sku=link.sku,
            options=link.selectedOptions,
        )
        for link in chosen
    ]


def _build_pending_choice(request_text: str, product_links: list[ProductLink]) -> PendingChoice | None:
    if not _is_cart_update_request(request_text) or len(product_links) <= 1:
        return None
    return PendingChoice(type="choose_for_cart", options=product_links[:4], quantity=_cart_quantity_delta(request_text))


def _select_cart_candidates(request_text: str, product_links: list[ProductLink]) -> list[ProductLink]:
    if not product_links:
        return []
    if len(product_links) == 1:
        return [product_links[0]]

    lowered = request_text.lower()
    ordinal = None
    for word, idx in ORDINAL_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            ordinal = idx
            break
    if ordinal is not None and ordinal < len(product_links):
        return [product_links[ordinal]]

    chosen = [
        link
        for link in product_links
        if link.variantLabel.lower() in lowered or link.name.lower() in lowered
    ]
    return chosen if len(chosen) == 1 else []


def _build_cart_actions_from_pending(request_text: str, pending: PendingChoice | None) -> list[CartAction]:
    if not pending or not pending.options:
        return []
    chosen = _select_cart_candidates(request_text, pending.options)
    if not chosen:
        return []
    quantity = pending.quantity or 1
    return [
        CartAction(
            type="cart.add",
            productId=link.id,
            combinationId=link.combinationId,
            quantity=quantity,
            sku=link.sku,
            options=link.selectedOptions,
        )
        for link in chosen
    ]


def _sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _normalize_tool_arguments(raw_arguments: Any) -> dict[str, Any]:
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _format_tool_schemas_for_prompt(tool_schemas: list[dict[str, Any]]) -> str:
    compact: list[dict[str, Any]] = []
    for tool in tool_schemas:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        compact.append(
            {
                "name": function.get("name", ""),
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {}),
            }
        )
    return json.dumps(compact, ensure_ascii=True)


def _parse_tool_decision(text: str) -> dict[str, Any]:
    parsed = _extract_json_object(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Model did not return valid JSON tool decision")
    action = parsed.get("action")
    if action not in {"tool", "answer"}:
        raise RuntimeError("Model returned invalid action")
    if action == "tool":
        name = parsed.get("name")
        if not isinstance(name, str) or not name.strip():
            raise RuntimeError("Model returned tool action without name")
        parsed["arguments"] = _normalize_tool_arguments(parsed.get("arguments"))
    else:
        if not isinstance(parsed.get("message"), str):
            raise RuntimeError("Model returned answer action without message")
    return parsed


def _catalog_from_tool_traces(traces: list[ToolCallTrace]) -> dict[str, Any]:
    product_index: dict[int, dict[str, Any]] = {}

    def ensure_product(product_id: int, name: str | None = None) -> dict[str, Any]:
        existing = product_index.get(product_id)
        if existing is None:
            existing = {
                "id": product_id,
                "name": name or f"Product {product_id}",
                "url": "",
                "options": [],
                "variants": [],
            }
            product_index[product_id] = existing
        elif name and not existing.get("name"):
            existing["name"] = name
        return existing

    for trace in traces:
        if not trace.ok:
            continue
        result = trace.result or {}
        if trace.name == "catalog_search":
            items = result.get("items", [])
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    product_id = int(item.get("id") or 0)
                    if product_id <= 0:
                        continue
                    product = ensure_product(product_id, item.get("name"))
                    for key, value in item.items():
                        if key != "variants":
                            product[key] = value
                    product.setdefault("variants", [])
        elif trace.name == "product_get":
            product_data = result.get("product")
            if isinstance(product_data, dict):
                product_id = int(product_data.get("id") or 0)
                if product_id > 0:
                    product = ensure_product(product_id, product_data.get("name"))
                    product.update(product_data)
                    product.setdefault("variants", product_data.get("variants") or [])
        elif trace.name == "product_variants_get":
            product_id = int(result.get("product_id") or 0)
            if product_id > 0:
                product = ensure_product(product_id, result.get("product"))
                if result.get("product"):
                    product["name"] = result.get("product")
                if isinstance(result.get("options"), list):
                    product["options"] = result.get("options")
                if isinstance(result.get("variants"), list):
                    product["variants"] = result.get("variants")

    products = list(product_index.values())
    return {
        "products": products,
        "total": len(products),
    }


async def _run_llm_tool_roundtrip(
    request: ChatRequest,
) -> tuple[str, list[ToolCallTrace], dict[str, Any]]:
    tool_schemas = await mcp_client.get_tool_schemas()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": TOOL_SYSTEM_PROMPT},
        {"role": "system", "content": TOOL_LOOP_PROMPT},
        {"role": "system", "content": f"AVAILABLE_TOOLS: {_format_tool_schemas_for_prompt(tool_schemas)}"},
    ]
    for item in request.history or []:
        messages.append({"role": item.role, "content": item.content})
    messages.append({"role": "user", "content": request.message})

    traces: list[ToolCallTrace] = []
    max_rounds = 4

    async with mcp_client._new_protocol_client() as protocol_client:
        for round_num in range(1, max_rounds + 1):
            response = await call_ollama_message(messages)
            message = response.get("message", {})
            if not isinstance(message, dict):
                return "", traces, _catalog_from_tool_traces(traces)
            content = message.get("content", "") or ""
            decision = _parse_tool_decision(content)

            if decision["action"] == "answer":
                return decision["message"], traces, _catalog_from_tool_traces(traces)

            name = decision["name"]
            arguments = decision["arguments"]
            messages.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=True)})
            try:
                result = await mcp_client._call_tool_with_retry(name, arguments, client=protocol_client)
                trace = ToolCallTrace(name=name, arguments=arguments, ok=True, result=result)
                tool_payload = result
            except Exception as exc:
                trace = ToolCallTrace(
                    name=name,
                    arguments=arguments,
                    ok=False,
                    result={},
                    error=str(exc),
                )
                tool_payload = {"error": str(exc)}
            traces.append(trace)
            messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(
                        {
                            "tool_name": name,
                            "arguments": arguments,
                            "result": tool_payload,
                        },
                        ensure_ascii=True,
                    ),
                }
            )

    final_response = await call_ollama_message(
        messages
        + [
            {
                "role": "user",
                "content": 'Respond now with JSON only: {"action":"answer","message":"..."}',
            }
        ]
    )
    final_message = final_response.get("message", {}) if isinstance(final_response, dict) else {}
    final_content = final_message.get("content", "") if isinstance(final_message, dict) else ""
    try:
        final_decision = _parse_tool_decision(final_content)
        if final_decision["action"] == "answer":
            return final_decision["message"], traces, _catalog_from_tool_traces(traces)
    except Exception:
        pass
    return final_content, traces, _catalog_from_tool_traces(traces)


async def _run_llm_tool_chat(request: ChatRequest) -> ToolChatResponse:
    message, traces, _ = await _run_llm_tool_roundtrip(request)
    return ToolChatResponse(
        message=message,
        model=OLLAMA_MODEL,
        tool_calls=traces,
        rounds=min(4, max(1, len(traces) or 1)),
    )


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "ok", "service": "factmcp-shopping-assistant"}


@app.get("/health")
async def health() -> dict[str, Any]:
    try:
        catalog = await mcp_client.get_catalog_for_query("", limit=1)
        return {
            "status": "healthy",
            "model": OLLAMA_MODEL,
            "mcp_mode": "local" if USE_LOCAL_MCP else "remote",
            "mcp_url": None if USE_LOCAL_MCP else MCP_URL,
            "catalog_products": catalog.get("total", 0),
        }
    except CatalogUnavailableError:
        return {
            "status": "degraded",
            "model": OLLAMA_MODEL,
            "mcp_mode": "local" if USE_LOCAL_MCP else "remote",
            "mcp_url": None if USE_LOCAL_MCP else MCP_URL,
            "catalog_products": 0,
        }


@app.get("/catalog")
async def get_catalog(authorization: str | None = Header(None)) -> dict[str, Any]:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        return await mcp_client.get_catalog()
    except CatalogUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Catalog unavailable") from exc


@app.post("/clear-cache")
async def clear_cache(authorization: str | None = Header(None)) -> dict[str, str]:
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    try:
        await mcp_client.get_catalog(force_refresh=True)
    except CatalogUnavailableError as exc:
        raise HTTPException(status_code=503, detail="Catalog unavailable") from exc
    return {"status": "cache refreshed"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, authorization: str | None = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    if request.pending and request.pending.options:
        pending_actions = _build_cart_actions_from_pending(request.message, request.pending)
        if pending_actions:
            total_delta = sum(action.quantity for action in pending_actions)
            if total_delta == -1:
                message = "Removed from your cart."
            elif total_delta < 0:
                message = "Removed those from your cart."
            elif total_delta == 1:
                message = "Added to your cart."
            else:
                message = "Added those to your cart."
            return ChatResponse(
                message=message,
                model=OLLAMA_MODEL,
                in_stock_products=len(request.pending.options),
                validated=True,
                products=[],
                cart_actions=pending_actions,
                pending=None,
            )

    try:
        raw_text, _traces, catalog = await _run_llm_tool_roundtrip(request)
    except CatalogUnavailableError:
        raise HTTPException(status_code=503, detail="Inventory backend unavailable")
    except Exception as exc:
        logger.error("Tool chat failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail="Assistant backend error") from exc

    products = catalog.get("products", [])

    def finalize_response(raw_text: str) -> ChatResponse:
        valid, invalid_keys, product_links = _validate_variant_keys(raw_text, products)
        if not raw_text:
            return ChatResponse(
                message="I couldn't complete that request.",
                model=OLLAMA_MODEL,
                in_stock_products=catalog.get("total", 0),
                validated=False,
                products=[],
                cart_actions=[],
            )

        pending = None
        cart_actions: list[CartAction] = []
        if request.pending and request.pending.options:
            cart_actions = _build_cart_actions_from_pending(request.message, request.pending)
        if not cart_actions:
            cart_actions = _build_cart_actions(request.message, product_links)
        if not cart_actions:
            pending = _build_pending_choice(request.message, product_links)

        cleaned = _strip_variant_tags(raw_text)
        if cart_actions:
            total_delta = sum(action.quantity for action in cart_actions)
            if total_delta == -1:
                cleaned = "Removed from your cart."
            elif total_delta < 0:
                cleaned = "Removed those from your cart."
            elif total_delta == 1:
                cleaned = "Added to your cart."
            else:
                cleaned = "Added those to your cart."

        response_products = pending.options if pending else product_links
        if cart_actions:
            response_products = []

        return ChatResponse(
            message=cleaned,
            model=OLLAMA_MODEL,
            in_stock_products=catalog.get("total", 0),
            validated=valid,
            products=response_products,
            cart_actions=cart_actions,
            pending=pending,
        )

    if request.stream:
        final = finalize_response(raw_text)

        async def event_generator():
            yield _sse_event({"event": "final", **final.model_dump()})

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return finalize_response(raw_text)


@app.post("/chat-tools", response_model=ToolChatResponse)
async def chat_tools(request: ChatRequest, authorization: str | None = Header(None)):
    if API_KEY and authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    return await _run_llm_tool_chat(request)


@app.exception_handler(Exception)
async def handle_exception(request: Request, exc: Exception):
    logger.error("Unhandled error on %s: %s", request.url.path, exc, exc_info=True)
    return _apply_cors_headers(JSONResponse(status_code=500, content={"error": "Internal server error"}))
