# src/core/prompts.py
from __future__ import annotations

from typing import Any

from src.core.menu import MenuCatalog


STATIC_SYSTEM_PROMPT_TEMPLATE = """You are a friendly, efficient, and accurate AI food-ordering assistant.
Your responses are simultaneously displayed in chat and spoken aloud via Text-to-Speech (TTS).

VOICE & STYLE RULES:
1. Keep every response concise (1 to 2 short sentences).
2. NEVER use Markdown formatting (no asterisks **, no bullet lists -, no numbered lists, no tables). Plain conversational sentences only.
3. Format prices clearly with a dollar sign and two decimals (e.g., $11.50).
4. Never calculate prices yourself. Always read the exact prices and totals from the LIVE CART CONTEXT or tool outputs.

MENU & CART TOOL RULES:
1. Use `add_item`, `edit_item`, `remove_item`, and `set_special_instructions` to manage the order.
2. REQUIRED OPTIONS WITHOUT DEFAULTS: Notice that `milkshake` requires a `flavor` (vanilla, chocolate, strawberry, or oreo) and has NO default. If a user orders a milkshake without specifying a flavor, do NOT guess a flavor—ask them which flavor they want before adding it (you may still add any other valid items they requested in the same turn).
3. INVALID OPTIONS OR EXTRAS: If a user asks for an option or extra that is not supported by that item (e.g., a veggie patty on a Spicy Jalapeño Burger, or pepperoni on a burger), politely explain that the option is unavailable for that item and mention which menu item supports it if applicable.
4. MULTIPLE ITEMS & PARTIAL EDITS:
   - Each cart line has a unique `line_id` (e.g., `line_1`).
   - If a line has `quantity > 1` (e.g., 2 Classic Burgers) and the user wants to modify only a subset (e.g., "put bacon on just one of them" or "remove one burger"), pass `target_quantity=1` to `edit_item` or `quantity=1` to `remove_item`.
   - If multiple different lines in the cart match the user's description (e.g., both a Classic Burger and a Spicy Burger are in the cart and the user says "make the burger large"), do NOT guess—ask the user to clarify which one they mean.
5. SPECIAL INSTRUCTIONS: If the user mentions dietary notes or preparation instructions (e.g., "no onions please", "extra napkins"), pass them via `special_instructions` on `add_item`/`edit_item` or call `set_special_instructions`.

CONFIRMATION & SUBMISSION PROTOCOL (CRITICAL):
1. After adding, editing, or removing items, briefly state what was added or updated and end with: "Anything else?" (Do NOT ask "Ready to submit?" on the same turn you modify the cart).
2. When the user indicates they are finished adding items (e.g., "That's it", "Nothing else", "Checkout", or an initial "Submit" before you have asked for confirmation):
   - Do NOT call `submit_order` yet if `AWAITING_FINAL_CONFIRMATION` is `False`.
   - Summarize the full order and exact total price from LIVE CART CONTEXT, and ask: "Ready to submit?" (Example: "Your order: Regular Classic Burger with cheese — $9.50. Ready to submit?").
3. Only call `submit_order` (and call it ONLY ONCE) when `AWAITING_FINAL_CONFIRMATION` is `True` and the user confirms (e.g., "Yes", "Submit", "Confirm", "Go ahead") WITHOUT requesting any new changes to the cart.
4. If the user says "Yes, and add a cola", you must call `add_item` for the cola, state the updated total, and ask for confirmation again—never call `submit_order` in the same turn the cart is modified.
5. HANDLING `submit_order` RESPONSES:
   - If `submit_order` returns `success: true`: Confirm the submission with the `order_id` and `estimated_time` (e.g., "Order submitted! Order #ORD-12345. Estimated time: 15-20 minutes.").
   - If `submit_order` fails because the kitchen is busy or a temporary server error occurred: Apologize briefly and ask if they would like you to try submitting again (e.g., "Sorry, the order couldn't be submitted — the kitchen is busy. Would you like me to try again?").
   - If `submit_order` fails because the order total exceeds the $50.00 limit or due to a menu validation refusal: Explain the exact reason and ask which item they would like to remove or modify.

{menu_section}
"""


def build_static_system_prompt(menu: MenuCatalog) -> str:
    """
    Builds the 100% static system prompt containing rules and the menu schema.
    Because this string never changes across turns, Ollama keeps it locked in its KV cache
    for near-zero Time-To-First-Token (TTFT).
    """
    return STATIC_SYSTEM_PROMPT_TEMPLATE.format(menu_section=menu.format_for_prompt()).strip()


def format_user_turn_with_context(
    user_message: str,
    cart_formatted: str,
    awaiting_confirmation: bool,
) -> str:
    """
    Attaches the live cart state and confirmation flag to the current user message
    so the static system prompt prefix remains 100% cache-valid in Ollama.
    """
    return (
        f"[LIVE CART CONTEXT]\n"
        f"{cart_formatted}\n"
        f"AWAITING_FINAL_CONFIRMATION: {awaiting_confirmation}\n"
        f"[/LIVE CART CONTEXT]\n\n"
        f"User: {user_message}"
    )


def build_ollama_tools(menu: MenuCatalog) -> list[dict[str, Any]]:
    """
    Dynamically constructs standard OpenAI/Ollama JSON-Schema tool definitions
    directly from `MenuCatalog`, enforcing strict `enum` constraints for every
    `item_id`, item option, and extra.
    """
    valid_item_ids = list(menu.by_id.keys())

    # 1. Build per-option enum schemas from the menu catalog
    option_choices_map: dict[str, list[str]] = {}
    option_items_map: dict[str, list[str]] = {}
    for item in menu.items:
        for opt_name, opt_def in item.options.items():
            option_items_map.setdefault(opt_name, []).append(item.id)
            for choice in opt_def.choices:
                if choice not in option_choices_map.setdefault(opt_name, []):
                    option_choices_map[opt_name].append(choice)

    option_properties: dict[str, Any] = {}
    for opt_name, choices in option_choices_map.items():
        per_item_details = "; ".join(
            f"{it.id}: {it.options[opt_name].choices} (default={it.options[opt_name].default})"
            for it in menu.items
            if opt_name in it.options
        )
        option_properties[opt_name] = {
            "type": "string",
            "enum": choices,
            "description": f"Valid only for {option_items_map[opt_name]}. Allowed values -> {per_item_details}",
        }

    options_schema = {
        "type": "object",
        "properties": option_properties,
        "description": "Item options (only specify options supported by the target item_id).",
    }

    # 2. Collect all valid extra IDs across the menu
    all_extra_ids: list[str] = []
    extra_details_by_item: list[str] = []
    for item in menu.items:
        if item.valid_extra_ids:
            extra_details_by_item.append(f"{item.id}: {item.valid_extra_ids}")
            for ext_id in item.valid_extra_ids:
                if ext_id not in all_extra_ids:
                    all_extra_ids.append(ext_id)

    extras_array_schema = {
        "type": "array",
        "items": {"type": "string", "enum": all_extra_ids},
        "description": f"List of extra IDs. Valid extras per item -> {'; '.join(extra_details_by_item)}",
    }

    return [
        {
            "type": "function",
            "function": {
                "name": "add_item",
                "description": (
                    "Add a food or drink item from the menu to the user's cart. "
                    "Identical items are automatically merged into a single cart line. "
                    "Note: 'milkshake' requires 'flavor' to be specified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "item_id": {
                            "type": "string",
                            "enum": valid_item_ids,
                            "description": "Menu item ID to add.",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Number of units to add (1 or more). Defaults to 1.",
                        },
                        "options": options_schema,
                        "extras": extras_array_schema,
                        "special_instructions": {
                            "type": "string",
                            "description": "Optional order-level dietary notes or special instructions.",
                        },
                    },
                    "required": ["item_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "edit_item",
                "description": (
                    "Modify an existing line item in the cart by its `line_id`. "
                    "Supports changing options, adding/removing specific extras, changing total quantity, "
                    "or splitting a multi-quantity line via `target_quantity`."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_id": {
                            "type": "string",
                            "description": "The cart line_id to modify (e.g., 'line_1').",
                        },
                        "target_quantity": {
                            "type": "integer",
                            "description": (
                                "How many units of this line to apply option/extra changes to. "
                                "Set to 1 if the line has quantity=2 and the user says 'add bacon to just one of them'. "
                                "Leave omitted to modify all units in the line."
                            ),
                        },
                        "new_quantity": {
                            "type": "integer",
                            "description": "New total quantity for this line (e.g., 'make that 3 burgers' -> 3).",
                        },
                        "options": options_schema,
                        "add_extras": extras_array_schema,
                        "remove_extras": {
                            "type": "array",
                            "items": {"type": "string", "enum": all_extra_ids},
                            "description": "Extra IDs to remove from this item without affecting its other extras.",
                        },
                        "special_instructions": {
                            "type": "string",
                            "description": "Optional order-level special instructions to save.",
                        },
                    },
                    "required": ["line_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "remove_item",
                "description": "Remove a line item (or decrement part of its quantity) from the cart.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_id": {
                            "type": "string",
                            "description": "The cart line_id to remove (e.g., 'line_1').",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Number of units to remove. Omit to remove the entire line.",
                        },
                    },
                    "required": ["line_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_special_instructions",
                "description": "Save or update order-level special instructions (e.g., 'No onions please').",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "special_instructions": {
                            "type": "string",
                            "description": "The special instructions for the order.",
                        },
                    },
                    "required": ["special_instructions"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_order",
                "description": (
                    "Submit the final confirmed order to the kitchen via MCP. "
                    "ONLY call this ONCE after you have already summarized the order and total price "
                    "and the user has explicitly confirmed they want to submit."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "special_instructions": {
                            "type": "string",
                            "description": "Optional final special instructions for the order.",
                        },
                    },
                },
            },
        },
    ]