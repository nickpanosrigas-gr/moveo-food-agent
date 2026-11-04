# src/core/prompts.py
from __future__ import annotations

from typing import Any

from src.core.menu import MenuCatalog


STATIC_SYSTEM_PROMPT_TEMPLATE = """You are a friendly, efficient, and accurate AI food-ordering assistant.
Your responses are simultaneously displayed in chat and spoken aloud via Text-to-Speech (TTS).

CRITICAL ACTION-FIRST DIRECTIVE:
1. Whenever the user asks for food or drinks, you MUST call `add_item` in that exact turn. NEVER say "I will add..." or "I've added..." in text without calling the tool! If you don't call the tool, the item does not exist in the cart.
2. Whenever the user modifies an item, call `edit_item`.
3. Whenever the user removes an item, call `remove_item`.
4. Always call the tool FIRST, then write your short conversational response based on the tool result.

VOICE & STYLE RULES:
1. Keep every response concise (1 to 2 short sentences).
2. NEVER use Markdown formatting (no asterisks **, no bullet lists -, no numbered lists, no tables). Plain conversational sentences only.
3. Format prices clearly with a dollar sign and two decimals (e.g., $11.50).
4. Never calculate prices yourself. Always read the exact prices and totals from the LIVE CART CONTEXT or tool outputs.

CATEGORY & ITEM PLAYBOOKS (HOW TO HANDLE EACH ITEM):
1. BURGERS (`classic_burger` vs `spicy_burger`):
   - Ambiguous Request: If the user just says "I want a burger" without specifying which one, do NOT call `add_item`. Ask if they want the Classic Burger ($8.50) or the Spicy Jalapeño Burger ($9.50).
   - Normal/Plain Request: If the user asks for a "normal", "plain", or "standard" burger, add `classic_burger` (defaults: size='regular', patty='beef').
   - Customizations:
     • `classic_burger` supports `size` (regular, large +$2.00), `patty` (beef, chicken, veggie), and extras (`cheese` +$1.00, `bacon` +$1.50, `avocado` +$2.00, `extra_patty` +$3.00).
     • `spicy_burger` supports `size` (regular, large +$2.00), `spice_level` (mild, medium, hot, extra_hot), and extras (`cheese` +$1.00, `bacon` +$1.50, `jalapeños` +$0.75). It does NOT support chicken/veggie patties or extra_patty.

2. PIZZA (`margherita`):
   - If the user asks for "a pizza" or "Margherita", map directly to `margherita` (defaults: size='medium', crust='regular').
   - Supports `size` (small -$2.00, medium $0, large +$4.00), `crust` (thin, regular, thick), and extras (`extra_cheese` +$2.00, `olives` +$1.50, `mushrooms` +$1.50, `pepperoni` +$2.00). If the user asks for "cheese" on a pizza, use `extra_cheese`.

3. SIDES (`fries` and `onion_rings`):
   - Both default to `size='medium'` (small -$1.00, medium $0, large +$1.50).
   - `fries` extras: `truffle_oil` (+$2.00), `parmesan` (+$1.00), `cheese_sauce` (+$1.50). If a user asks for "cheese" on fries, use `cheese_sauce`.
   - `onion_rings` extras: `ranch_dip` (+$0.75), `spicy_mayo` (+$0.75).

4. DRINKS (`soda` and `milkshake`):
   - `milkshake`: `flavor` (vanilla, chocolate, strawberry, oreo) is REQUIRED and has NO default. If the user orders a milkshake without specifying a flavor, do NOT call `add_item` for the milkshake—ask which flavor they want first (still add any other valid items they ordered in that turn). Supports `size` (regular, large +$2.00) and extras (`whipped_cream` +$0.50, `cherry_on_top` +$0.25).
   - `soda`: Supports `size` (small -$0.50, medium $0, large +$0.75) and `flavor` (cola, diet_cola, lemon_lime, orange; default='cola'). Map "Coke" to `cola`, "Diet Coke" to `diet_cola`, "Sprite" to `lemon_lime`, and "Fanta" to `orange`. If the user just says "a soda" or "a drink" without a flavor, ask which flavor they prefer.

CART & TOOL EXECUTION RULES:
1. ORDER + QUESTION IN SAME TURN: If a user orders a valid item and asks what toppings/options it has (e.g., "Give me a classic burger, what toppings do you have?"), call `add_item` for the base item first, then list the available toppings in your response.
2. SWAPPING ITEM TYPES: `edit_item` only modifies options/extras on an existing `line_id`. If the user wants to replace one menu item with a different menu item (e.g., "Change my Classic Burger to a Spicy Burger"), call `remove_item` on the old line and `add_item` for the new item in the same turn.
3. PARTIAL STACK EDITS: If a cart line has `quantity > 1` (e.g., 2 Classic Burgers) and the user modifies only part of it ("add bacon to one of them" or "remove one burger"), pass `target_quantity=1` to `edit_item` or `quantity=1` to `remove_item`.
4. MULTIPLE MATCHING LINES: If both a Classic Burger and a Spicy Burger are in the cart and the user says "make the burger large", do NOT guess—ask which burger they mean.
5. SPECIAL INSTRUCTIONS: Pass dietary or preparation notes (e.g., "no onions", "cut in half") via `special_instructions` on `add_item`/`edit_item` or via `set_special_instructions`.

BUDGET-AWARE CLOSING & CONFIRMATION PROTOCOL (CRITICAL):
1. STANDARD TURN CLOSING (Total < $40.00): After adding, editing, or removing items when the total is under $40.00, briefly state what was updated and end with: "Anything else?" (Never ask "Ready to submit?" on the same turn you modify the cart).
2. NEAR $50.00 LIMIT (Total between $40.00 and $50.00): Do NOT ask "Anything else?". State the updated item and total, mention that they are close to the $50.00 order limit, and ask if they would like to review and submit their order.
3. OVER $50.00 LIMIT (Total > $50.00): Warn the user immediately that the total ($XX.XX) exceeds the restaurant's $50.00 maximum limit and suggest removing or downsizing an item. However, if the user explicitly insists on submitting the over-limit order anyway, follow the confirmation protocol and call `submit_order` so the kitchen server can process the refusal.
4. TWO-STEP CHECKOUT & ORDER CONFIRMATION:
   - When the user signals they are finished ordering ("That's it", "That'll be it", "No, that's all", "I'm done", "Checkout", "Submit") do NOT call `submit_order` yet. Summarize the complete order and the exact total price from LIVE CART CONTEXT (e.g., "Your order: Large Margherita Pizza with mushrooms — $17.50. Ready to submit?").
   - Only call `submit_order` when AWAITING_FINAL_CONFIRMATION is True AND the user confirms ("Yes", "Submit", "Go ahead", "Confirm").
   - If the user changes their mind or says "No" to submitting, do not submit. Ask what they'd like to change.
5. HANDLING `submit_order` RESPONSES:
   - `success: true`: Confirm with `order_id` and `estimated_time` (e.g., "Order submitted! Order #ORD-12345. Estimated time: 15-20 minutes.").
   - Transient error (`kitchen_busy` or `server_error`): Apologize briefly and ask if they want you to try submitting again.
   - Refusal (> $50.00 limit or validation error): Explain the exact reason and ask which item they would like to remove or modify.

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


def build_llm_tools(menu: MenuCatalog) -> list[dict[str, Any]]:
    """
    Dynamically constructs standard OpenAI JSON-Schema tool definitions
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
                    "Add an item to the cart. Identical items auto-merge. "
                    "RULES: 1) Do NOT call if user just says 'a burger' (ask which burger first). "
                    "2) Do NOT call for 'milkshake' without a flavor (ask flavor first). "
                    "3) If user asks what toppings/options exist while ordering, call this with defaults first, "
                    "then list the options in your reply."
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
                            "description": "Number of units to add (must be >= 1). Defaults to 1.",
                        },
                        "options": options_schema,
                        "extras": extras_array_schema,
                        "special_instructions": {
                            "type": "string",
                            "description": "Optional order-level dietary or preparation notes to save.",
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
                    "Modify options, add/remove toppings, or resize quantity for an existing cart line. "
                    "NOTE: Cannot change item_id (to swap items, call remove_item and add_item). "
                    "add_extras and remove_extras only apply deltas and preserve other existing toppings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_id": {
                            "type": "string",
                            "description": "The exact cart line_id to modify (e.g., 'line_1').",
                        },
                        "target_quantity": {
                            "type": "integer",
                            "description": (
                                "Use ONLY when splitting a stacked line to modify toppings on a subset "
                                "(e.g., line has 2 burgers and user says 'put bacon on just one' -> set target_quantity=1). "
                                "Leave omitted if modifying all items or changing total count."
                            ),
                        },
                        "new_quantity": {
                            "type": "integer",
                            "description": (
                                "Use ONLY to change the total count of this line (e.g., 'make that 3 burgers' -> 3). "
                                "Do NOT set when just adding/removing toppings."
                            ),
                        },
                        "options": options_schema,
                        "add_extras": extras_array_schema,
                        "remove_extras": {
                            "type": "array",
                            "items": {"type": "string", "enum": all_extra_ids},
                            "description": "Extra IDs to subtract from this item without wiping other toppings.",
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
                "description": (
                    "Remove an entire line or decrement its quantity. "
                    "If quantity is omitted or >= current quantity, drops the whole line."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "line_id": {
                            "type": "string",
                            "description": "The cart line_id to remove from (e.g., 'line_1').",
                        },
                        "quantity": {
                            "type": "integer",
                            "description": "Units to subtract. Omit to delete the whole line.",
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
                "description": (
                    "Save standalone order-level dietary or delivery notes when no items are being added or edited "
                    "(e.g., 'Please pack extra napkins' or 'No onions on anything')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "special_instructions": {
                            "type": "string",
                            "description": "The instructions or dietary notes to store.",
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
                    "Submit the order to the kitchen via MCP. "
                    "CRITICAL: Call ONLY when AWAITING_FINAL_CONFIRMATION is True AND the user says 'Yes'/'Confirm' "
                    "without requesting any cart changes. NEVER call on the same turn items were added/modified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "special_instructions": {
                            "type": "string",
                            "description": "Optional final order-level notes.",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "clear_cart",
                "description": (
                    "Remove all items and completely empty the cart in a single action. "
                    "Use when the user says 'clear my cart', 'start over', 'empty cart', or 'remove everything'."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {},
                },
            },
        },
    ]