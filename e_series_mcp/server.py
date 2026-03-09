"""FastMCP server for Lightspeed eCom / Ecwid (E-Series).

Designed for Vercel serverless deployment with a mountable ASGI app.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
PORT = int(os.getenv("PORT", "8000"))
ECWID_STORE_ID = os.getenv("ECWID_STORE_ID", "").strip()
ECWID_SECRET_TOKEN = os.getenv("ECWID_SECRET_TOKEN", "").strip()
DATA_DIR = (BASE_DIR / os.getenv("DATA_DIR", "data").strip()).resolve()
STORE_INFO_PATH = (BASE_DIR / os.getenv("STORE_INFO_PATH", "data/store_info.json").strip()).resolve()
FAQ_PATH = (BASE_DIR / os.getenv("FAQ_PATH", "data/faq.json").strip()).resolve()
CATALOG_CACHE_TTL_SEC = int(os.getenv("CATALOG_CACHE_TTL_SEC", "900"))
USER_AGENT = "Mozilla/5.0 (compatible; ESeriesMCP/1.0)"
ECWID_API_BASE = f"https://app.ecwid.com/api/v3/{ECWID_STORE_ID}" if ECWID_STORE_ID else ""

mcp = FastMCP(
    name="E-Series Inventory MCP",
    instructions=(
        "Use these tools to retrieve real E-Series inventory and store information. "
        "Do not invent product details."
    ),
)


class EcwidAPIError(Exception):
    pass


class ConfigurationError(Exception):
    pass


@dataclass
class CacheEntry:
    expires_at: float
    value: Any


_CACHE: dict[str, CacheEntry] = {}


def _cache_get(key: str) -> Any | None:
    entry = _CACHE.get(key)
    if entry and time.time() <= entry.expires_at:
        return entry.value
    _CACHE.pop(key, None)
    return None


def _cache_set(key: str, value: Any, ttl: int) -> None:
    _CACHE[key] = CacheEntry(time.time() + ttl, value)


def _load_json(path: Path, default: Any) -> Any:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return default


def _require_ecwid() -> None:
    if not ECWID_STORE_ID or not ECWID_SECRET_TOKEN:
        raise ConfigurationError("Missing ECWID_STORE_ID or ECWID_SECRET_TOKEN")


async def _ecwid_get(path: str, params: dict[str, Any] | None = None) -> Any:
    _require_ecwid()
    url = f"{ECWID_API_BASE}{path}"
    headers = {"Authorization": f"Bearer {ECWID_SECRET_TOKEN}"}
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers, params=params or {})
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as exc:
        raise EcwidAPIError(
            f"API error {exc.response.status_code}: {exc.response.text[:200]}"
        ) from exc
    except Exception as exc:
        raise EcwidAPIError(f"Request failed: {exc}") from exc


def _extract_options(item: dict[str, Any]) -> list[dict[str, Any]]:
    options = item.get("options")
    if not isinstance(options, list):
        return []

    normalized: list[dict[str, Any]] = []
    for option in options:
        if not isinstance(option, dict):
            continue
        name = (
            option.get("name")
            or option.get("title")
            or option.get("optionName")
            or option.get("option_name")
        )
        if not name:
            continue

        choices_raw = option.get("choices")
        choices: list[dict[str, str]] = []
        if isinstance(choices_raw, list):
            for choice in choices_raw:
                text: str | None = None
                if isinstance(choice, dict):
                    text = (
                        choice.get("text")
                        or choice.get("value")
                        or choice.get("title")
                        or choice.get("name")
                    )
                elif choice:
                    text = str(choice)
                if text:
                    choices.append({"text": str(text)})

        normalized.append({"name": str(name), "choices": choices})
    return normalized


def _normalize_product(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id"),
        "sku": item.get("sku", ""),
        "name": item.get("name", "Unnamed Product"),
        "price": float(item.get("price", 0) or 0),
        "enabled": bool(item.get("enabled", False)),
        "inStock": bool(item.get("inStock", False)),
        "url": item.get("url", ""),
        "thumbnailUrl": item.get("thumbnailUrl") or item.get("imageUrl", ""),
        "description": (item.get("description") or "")[:800],
        "options": _extract_options(item),
    }


def _variant_selected_options(variant: dict[str, Any]) -> list[dict[str, str]]:
    for key in ("options", "optionValues", "option_values", "optionsValues", "options_values"):
        options = variant.get(key)
        if not isinstance(options, list):
            continue
        selected: list[dict[str, str]] = []
        for option in options:
            if isinstance(option, dict):
                name = option.get("name") or option.get("optionName") or option.get("option_name")
                value = (
                    option.get("value")
                    or option.get("text")
                    or option.get("valueName")
                    or option.get("value_name")
                    or option.get("title")
                )
            else:
                name = None
                value = None
            if name and value:
                selected.append({"name": str(name), "value": str(value)})
        if selected:
            return selected
    return []


def _variant_option_values(variant: dict[str, Any]) -> list[str]:
    for key in ("options", "optionValues", "option_values", "optionsValues", "options_values"):
        options = variant.get(key)
        if not isinstance(options, list):
            continue
        parts: list[str] = []
        for option in options:
            if isinstance(option, dict):
                value = (
                    option.get("value")
                    or option.get("valueName")
                    or option.get("value_name")
                    or option.get("title")
                )
                if value:
                    parts.append(str(value))
                    continue
                name = option.get("name") or option.get("optionName") or option.get("option_name")
                if name:
                    parts.append(str(name))
            elif option:
                parts.append(str(option))
        if parts:
            return parts
    return []


def _variant_label(variant: dict[str, Any], product: dict[str, Any]) -> str:
    parts = _variant_option_values(variant)
    if parts:
        return " · ".join(parts)
    for key in ("name", "combinationName", "combination_name", "title"):
        value = variant.get(key)
        if value:
            return str(value).strip()
    return product.get("name", "Variant")


def _variant_id(variant: dict[str, Any]) -> Any | None:
    for key in ("combinationId", "id", "combination_id", "variantId", "variant_id"):
        if key in variant and variant.get(key) is not None:
            return variant.get(key)
    return None


def _variant_sku(variant: dict[str, Any]) -> str:
    for key in ("sku", "skuCode", "code", "sku_code"):
        value = variant.get(key)
        if value:
            return str(value).strip()
    return ""


def _effective_variant_sku(variant: dict[str, Any], product: dict[str, Any]) -> str:
    sku = _variant_sku(variant)
    if sku:
        return sku
    base_sku = product.get("sku")
    return str(base_sku).strip() if base_sku else ""


def _variant_in_stock(variant: dict[str, Any]) -> bool:
    if "inStock" in variant:
        return bool(variant.get("inStock"))
    if "instock" in variant:
        return bool(variant.get("instock"))
    quantity = variant.get("quantity")
    if quantity is not None:
        try:
            return float(quantity) > 0
        except (TypeError, ValueError):
            return bool(quantity)
    return True


def _variant_enabled(variant: dict[str, Any]) -> bool:
    enabled = variant.get("enabled")
    if enabled is None:
        return True
    return bool(enabled)


def _extract_variants(item: dict[str, Any], base: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    combinations = item.get("combinations") or item.get("variants")
    base_price = (base or {}).get("price")
    if base_price is None:
        base_price = float(item.get("price", 0) or 0)
    base_enabled = bool(item.get("enabled", False))
    base_in_stock = bool(item.get("inStock", False))
    product_id = item.get("id")
    variants: list[dict[str, Any]] = []

    if isinstance(combinations, list) and combinations:
        for variant in combinations:
            if not isinstance(variant, dict):
                continue
            if not _variant_enabled(variant) or not _variant_in_stock(variant):
                continue
            combination_id = _variant_id(variant)
            if combination_id is None:
                continue
            price = variant.get("price", base_price)
            try:
                price_val = float(price or 0)
            except (TypeError, ValueError):
                price_val = float(base_price or 0)
            variants.append(
                {
                    "productId": product_id,
                    "combinationId": combination_id,
                    "variantKey": f"{product_id}:{combination_id}",
                    "effectiveSku": _effective_variant_sku(variant, item),
                    "label": _variant_label(variant, item),
                    "price": price_val,
                    "in_stock": True,
                    "options": _variant_selected_options(variant),
                }
            )
        return variants

    if base_enabled and base_in_stock:
        variants.append(
            {
                "productId": product_id,
                "combinationId": 0,
                "variantKey": f"{product_id}:0",
                "effectiveSku": str(item.get("sku") or "").strip(),
                "label": item.get("name", "Default"),
                "price": float(base_price or 0),
                "in_stock": True,
                "options": [],
            }
        )
    return variants


def _extract_variants_from_combinations(
    combinations: Any,
    base: dict[str, Any] | None = None,
    product_id: Any | None = None,
) -> list[dict[str, Any]]:
    if isinstance(combinations, dict):
        if isinstance(combinations.get("items"), list):
            combinations = combinations["items"]
        elif isinstance(combinations.get("combinations"), list):
            combinations = combinations["combinations"]
        else:
            combinations = []
    if not isinstance(combinations, list):
        return []

    base_product = base or {}
    base_price = base_product.get("price") or 0.0
    pid = product_id if product_id is not None else base_product.get("id")
    try:
        pid_val = int(pid)
    except (TypeError, ValueError):
        pid_val = pid

    variants: list[dict[str, Any]] = []
    for variant in combinations:
        if not isinstance(variant, dict):
            continue
        if not _variant_enabled(variant) or not _variant_in_stock(variant):
            continue
        combination_id = _variant_id(variant)
        if combination_id is None:
            continue

        price = variant.get("price")
        if price is None:
            price = variant.get("defaultDisplayedPrice")
        if price is None:
            price = base_price
        try:
            price_val = float(price or 0)
        except (TypeError, ValueError):
            price_val = float(base_price or 0)

        variants.append(
            {
                "productId": pid_val,
                "combinationId": combination_id,
                "variantKey": f"{pid_val}:{combination_id}",
                "effectiveSku": _effective_variant_sku(variant, base_product),
                "label": _variant_label(variant, base_product),
                "price": price_val,
                "in_stock": True,
                "options": _variant_selected_options(variant),
            }
        )
    return variants


def _normalize_product_details(item: dict[str, Any]) -> dict[str, Any]:
    product = _normalize_product(item)
    product["options"] = _extract_options(item)
    product["variants"] = _extract_variants(item, base=product)
    return product


def _text_score(query: str, text: str) -> float:
    if not query or not text:
        return 0.0
    q = query.lower()
    t = text.lower()
    return sum(1.0 for token in re.findall(r"\w+", q) if token in t)


def _clean_text(html: str, max_chars: int = 15000) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text(separator="\n")[:max_chars].strip()


class CatalogSearchOutput(BaseModel):
    total: int
    count: int
    items: list[dict[str, Any]]
    error: str | None = None


class ProductGetOutput(BaseModel):
    product: dict[str, Any]


class ProductVariantsOutput(BaseModel):
    product_id: int
    product: str | None = None
    options: list[dict[str, Any]] = Field(default_factory=list)
    variants: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class FAQSearchOutput(BaseModel):
    matches: list[dict[str, Any]]


class WebFetchOutput(BaseModel):
    url: str
    status_code: int
    title: str | None
    text: str


class HealthOutput(BaseModel):
    status: str
    service: str


class CatalogSearchInput(BaseModel):
    keyword: str | None = Field(None, max_length=500)
    sku: str | None = Field(None, max_length=100)
    categoryId: int | None = Field(None, ge=0)
    priceFrom: float | None = Field(None, ge=0)
    priceTo: float | None = Field(None, ge=0)
    limit: int = Field(50, ge=1, le=100)
    offset: int = Field(0, ge=0)


class ProductGetInput(BaseModel):
    product_id: int = Field(..., ge=0)


class ProductVariantsInput(BaseModel):
    product_id: int = Field(..., ge=0)


class FAQSearchInput(BaseModel):
    query: str = Field(..., max_length=500)
    limit: int = Field(5, ge=1, le=20)


class WebFetchInput(BaseModel):
    url: str = Field(..., max_length=2000)
    max_chars: int = Field(15000, ge=100, le=50000)


@mcp.tool()
def health_check() -> HealthOutput:
    """Lightweight health check for MCP clients."""
    return HealthOutput(status="ok", service="e-series-mcp")


@mcp.tool()
async def catalog_search(
    keyword: str | None = None,
    sku: str | None = None,
    categoryId: int | None = None,
    priceFrom: float | None = None,
    priceTo: float | None = None,
    limit: int = 50,
    offset: int = 0,
) -> CatalogSearchOutput:
    """Search products in Ecwid catalog with filters."""
    try:
        cache_key = f"search:{keyword}:{sku}:{categoryId}:{priceFrom}:{priceTo}:{limit}:{offset}"
        cached = _cache_get(cache_key)
        if cached:
            return CatalogSearchOutput(**cached)

        params: dict[str, Any] = {"limit": min(limit, 100), "offset": offset}
        if keyword:
            params["keyword"] = keyword
        if sku:
            params["sku"] = sku
        if categoryId is not None:
            params["category"] = categoryId
        if priceFrom is not None:
            params["priceFrom"] = priceFrom
        if priceTo is not None:
            params["priceTo"] = priceTo

        data = await _ecwid_get("/products", params)
        items = [_normalize_product(item) for item in data.get("items", [])]
        result = {"total": data.get("total", 0), "count": len(items), "items": items}
        _cache_set(cache_key, result, CATALOG_CACHE_TTL_SEC)
        return CatalogSearchOutput(**result)
    except Exception as exc:
        logger.error("catalog_search error: %s", exc)
        return CatalogSearchOutput(total=0, count=0, items=[], error=str(exc))


@mcp.tool()
async def product_get(product_id: int) -> ProductGetOutput:
    """Get detailed product information by ID."""
    try:
        cache_key = f"product:{product_id}"
        cached = _cache_get(cache_key)
        if cached:
            return ProductGetOutput(product=cached)

        data = await _ecwid_get(f"/products/{product_id}")
        product = _normalize_product_details(data)
        _cache_set(cache_key, product, CATALOG_CACHE_TTL_SEC)
        return ProductGetOutput(product=product)
    except Exception as exc:
        logger.error("product_get error: %s", exc)
        return ProductGetOutput(product={"error": str(exc)})


@mcp.tool()
async def product_variants_get(product_id: int) -> ProductVariantsOutput:
    """Get sellable variants for a product."""
    try:
        cache_key = f"product_variants:{product_id}"
        cached = _cache_get(cache_key)
        if cached:
            return ProductVariantsOutput(**cached)

        data = await _ecwid_get(f"/products/{product_id}")
        options = _extract_options(data)

        combinations: Any = []
        try:
            combinations = await _ecwid_get(f"/products/{product_id}/combinations")
        except Exception as exc:
            logger.warning("Combinations endpoint failed for %s: %s", product_id, exc)
            combinations = data.get("combinations") or data.get("variants") or []

        variants = _extract_variants_from_combinations(combinations, base=data, product_id=product_id)
        if not variants:
            variants = _extract_variants(data)

        result = {
            "product_id": product_id,
            "product": data.get("name"),
            "options": options,
            "variants": variants,
        }
        _cache_set(cache_key, result, CATALOG_CACHE_TTL_SEC)
        return ProductVariantsOutput(**result)
    except Exception as exc:
        logger.error("product_variants_get error: %s", exc)
        return ProductVariantsOutput(product_id=product_id, error=str(exc))


@mcp.tool()
def store_info_get() -> dict[str, Any]:
    """Get store operational info (hours, policies)."""
    return {"store_info": _load_json(STORE_INFO_PATH, {})}


@mcp.tool()
def faq_search(query: str, limit: int = 5) -> FAQSearchOutput:
    """Search local FAQ entries."""
    try:
        faqs = _load_json(FAQ_PATH, [])
        scored = [(_text_score(query, f"{faq.get('q', '')} {faq.get('a', '')}"), faq) for faq in faqs]
        matches = [
            {"q": faq.get("q", ""), "a": faq.get("a", "")}
            for score, faq in sorted(scored, reverse=True)[:limit]
            if score > 0
        ]
        return FAQSearchOutput(matches=matches)
    except Exception as exc:
        return FAQSearchOutput(matches=[{"error": str(exc)}])


@mcp.tool()
async def web_fetch(url: str, max_chars: int = 15000) -> WebFetchOutput:
    """Fetch URL and extract visible text."""
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(url, headers={"User-Agent": USER_AGENT})
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("title")
            return WebFetchOutput(
                url=str(response.url),
                status_code=response.status_code,
                title=title.get_text(strip=True) if title else None,
                text=_clean_text(response.text, max_chars),
            )
    except Exception as exc:
        logger.error("web_fetch error: %s", exc)
        return WebFetchOutput(url=url, status_code=0, title=None, text=f"Error: {exc}")


@mcp.resource("store://info")
def store_info_resource() -> str:
    """Store information resource."""
    return json.dumps(_load_json(STORE_INFO_PATH, {}), indent=2)


@mcp.resource("store://faq")
def faq_resource() -> str:
    """FAQ resource."""
    return json.dumps(_load_json(FAQ_PATH, []), indent=2)


@mcp.prompt()
def storefront_assistant() -> str:
    """Storefront assistant prompt."""
    return """You are a storefront shopping assistant for a Lightspeed eCom E-Series store.

Rules:
- Always use catalog_search/product_get for product info - never invent details
- If a product has multiple in-stock variants, use product_variants_get and select or ask
- Keep recommendations to 3 items max with rationale
- Use store_info_get for hours/policies/pickup
- Use web_fetch for website content grounding
- Include actions array for cart operations: ADD_TO_CART, OPEN_CART, OPEN_PRODUCT
"""


_TOOL_HTTP_HANDLERS: dict[str, tuple[Any, type[BaseModel] | None]] = {
    "catalog_search": (catalog_search, CatalogSearchInput),
    "product_get": (product_get, ProductGetInput),
    "product_variants_get": (product_variants_get, ProductVariantsInput),
    "store_info_get": (store_info_get, None),
    "faq_search": (faq_search, FAQSearchInput),
    "web_fetch": (web_fetch, WebFetchInput),
}


def _serialize_result(result: Any) -> Any:
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result


@mcp.custom_route("/", methods=["GET"], include_in_schema=False)
async def root(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(_: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


@mcp.custom_route("/tools/{tool_name}", methods=["POST"], include_in_schema=False)
async def tool_call(request: Request) -> JSONResponse:
    tool_name = request.path_params.get("tool_name", "").strip()
    handler = _TOOL_HTTP_HANDLERS.get(tool_name)
    if not handler:
        return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)

    func, input_model = handler
    payload: dict[str, Any] = {}
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    if input_model is not None:
        try:
            payload = input_model(**(payload or {})).model_dump()
        except Exception as exc:
            return JSONResponse({"error": f"Invalid input: {exc}"}, status_code=400)

    try:
        if inspect.iscoroutinefunction(func):
            result = await func(**payload)
        else:
            result = func(**payload)
    except Exception as exc:
        logger.error("HTTP tool call %s failed: %s", tool_name, exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    return JSONResponse(_serialize_result(result))


@mcp.custom_route("/products/{product_id:int}/variants", methods=["GET"], include_in_schema=False)
async def product_variants_route(request: Request) -> JSONResponse:
    product_id = request.path_params["product_id"]
    try:
        result = await product_variants_get(product_id=product_id)
    except Exception as exc:
        logger.error("HTTP product variants failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(_serialize_result(result))


app = mcp.http_app(path="/mcp", json_response=True, stateless_http=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
