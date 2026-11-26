"""ADK demo agent, refactored from the DeepLearning.AI's
'Evaluating and Improving Agentic Applications' labs (Northwind Apparel Shop) and
instrumented with chain/llm/tool spans.

This is a reference example showing how an agent plugs into evalkit — copy
the pattern (`build_agent(tracer)`) for your own agent; nothing here is
required by the toolkit.

Instrumentation uses ADK's native agent callbacks, so the agent's own code
stays untouched: before/after_model -> llm spans, before/after_tool -> tool
spans, and the caller wraps runner.run_async in a chain span.
"""

from __future__ import annotations

import os
from typing import Any

from google.adk.agents import LlmAgent

from evalkit.instrumentation import llm_pacer, traced_callbacks
from evalkit.tracer import Tracer

# --- Shop data (tiny in-memory catalog, mirrors the lab's SQLite shop) ---

CATALOG = [
    {"id": "TEE-001", "name": "Classic Tee", "category": "tops", "price": 24.99,
     "colors": ["black", "heather gray", "navy", "white"], "sizes": ["XS", "S", "M", "L", "XL"],
     "care": "Machine wash cold, tumble dry low"},
    {"id": "TEE-002", "name": "Heavyweight Tee", "category": "tops", "price": 32.0,
     "colors": ["olive", "charcoal"], "sizes": ["S", "M", "L", "XL"],
     "care": "Machine wash cold"},
    {"id": "HOD-001", "name": "Pullover Hoodie", "category": "tops", "price": 58.0,
     "colors": ["black", "gray", "forest"], "sizes": ["S", "M", "L", "XL"],
     "care": "Machine wash cold, inside out"},
    {"id": "SWE-001", "name": "Merino Wool Sweater", "category": "tops", "price": 89.0,
     "colors": ["navy", "cream", "burgundy"], "sizes": ["S", "M", "L"],
     "care": "Machine washable on wool cycle"},
    {"id": "PAN-001", "name": "Chino Pants", "category": "bottoms", "price": 64.0,
     "colors": ["khaki", "navy", "black"], "sizes": ["28", "30", "32", "34", "36"],
     "care": "Machine wash cold"},
    {"id": "PAN-002", "name": "Cargo Pants", "category": "bottoms", "price": 72.0,
     "colors": ["olive", "black"], "sizes": ["28", "30", "32", "34"],
     "care": "Machine wash cold"},
    {"id": "SHT-001", "name": "Oxford Shirt", "category": "tops", "price": 54.0,
     "colors": ["white", "sky blue", "pink"], "sizes": ["XS", "S", "M", "L", "XL"],
     "care": "Machine wash warm, warm iron"},
    {"id": "BEA-001", "name": "Wool Beanie", "category": "accessories", "price": 24.0,
     "colors": ["black", "rust", "cream"], "sizes": ["One size"],
     "care": "Hand wash cold, lay flat to dry"},
    {"id": "SOX-001", "name": "Crew Socks (3-pack)", "category": "accessories", "price": 18.0,
     "colors": ["white", "black", "mixed"], "sizes": ["S/M", "L/XL"],
     "care": "Machine wash warm"},
    {"id": "JKT-001", "name": "Quilted Jacket", "category": "outerwear", "price": 129.0,
     "colors": ["black", "olive"], "sizes": ["S", "M", "L", "XL"],
     "care": "Machine wash cold, gentle cycle"},
]

INVENTORY = {
    ("TEE-001", "M"): 14, ("TEE-001", "L"): 9, ("TEE-001", "S"): 22,
    ("SWE-001", "M"): 6, ("SWE-001", "L"): 4,
    ("BEA-001", "ONE SIZE"): 31,
    ("PAN-001", "32"): 11, ("JKT-001", "L"): 3,
}

ORDERS = {
    "ORD-1001": {"status": "delivered", "items": ["TEE-001", "SOX-001"],
                 "tracking": ["Label created", "In transit", "Out for delivery", "Delivered"]},
    "ORD-1002": {"status": "in_transit", "items": ["SWE-001"],
                 "tracking": ["Label created", "In transit — arriving Thursday"]},
}

# --- Tools: plain functions; ADK builds schemas from docstrings + type hints ---

def search_catalog(query: str) -> list[dict[str, Any]]:
    """Search the clothing catalog by keyword. Matches product name, category, or color."""
    q = query.lower()
    return [
        {k: p[k] for k in ("id", "name", "category", "price", "colors", "sizes", "care")}
        for p in CATALOG
        if q == p["category"] or q in p["name"].lower() or any(q in c for c in p["colors"])
    ]


def check_inventory(product_id: str, size: str) -> dict[str, Any]:
    """Check stock for a product ID and size. Returns quantity and availability."""
    qty = INVENTORY.get((product_id.upper(), size.upper()), 0)
    return {"product_id": product_id.upper(), "size": size.upper(),
            "available": qty > 0, "quantity": qty}


def get_order_status(order_id: str) -> dict[str, Any]:
    """Look up an order by ID (e.g. ORD-1001). Returns status, items, and tracking."""
    order = ORDERS.get(order_id.strip().upper())
    return order if order else {"order_id": order_id, "found": False,
                                "reason": "order_not_found"}


def return_policy() -> str:
    """Return the store's return and exchange policy."""
    return ("Items may be returned within 30 days of delivery with tags attached. "
            "Exchanges for a different size or color are free; refunds go to the "
            "original payment method within 5 business days.")


TOOLS = [search_catalog, check_inventory, get_order_status, return_policy]

INSTRUCTION = """You are a friendly assistant for Northwind Apparel, a clothing store.
Help customers find products, check sizes and stock, look up orders and tracking,
and explain the return policy. Use the available tools for catalog, inventory,
order, and policy questions. Keep answers short and concrete."""


def build_agent(
    tracer: Tracer | None = None,
    *,
    model: str | None = None,
    tools: list | None = None,
    instruction: str | None = None,
) -> LlmAgent:
    """Build the shop agent. With `tracer`, spans go to our local JSON tracer;
    without it, only LLM pacing is applied (e.g. when Phoenix/OTel does tracing)."""
    callbacks = traced_callbacks(tracer) if tracer is not None else {
        "before_model_callback": llm_pacer(),
    }
    return LlmAgent(
        name="shop_assistant",
        model=model or os.getenv("AGENT_MODEL")
        or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        instruction=instruction or INSTRUCTION,
        tools=tools or TOOLS,
        **callbacks,
    )
