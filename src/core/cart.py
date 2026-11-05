from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from src.core.menu import MenuCatalog, MenuValidationError, ValidatedOrderItem

MAX_ORDER_TOTAL = Decimal("50.00")


def _safe_int(val: Any, default: int | None = None) -> int | None:
    if val is None or isinstance(val, bool):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


# =====================================================================
# 1. Line Item Model & Tool Input Schemas
# =====================================================================

class CartLineItem(BaseModel):
    """
    Represents a single distinct line in the order cart.
    Items with identical (item_id, options, extras) share a single CartLineItem.
    """
    model_config = ConfigDict(extra="forbid")

    line_id: str
    item_id: str
    name: str
    quantity: int = Field(ge=1)
    options: dict[str, str]
    extras: list[str]
    unit_price: float
    line_total: float

    @property
    def signature(self) -> tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]]:
        """
        Canonical identity key used to auto-merge identical items into the same line.
        Because `MenuCatalog.validate_and_resolve` already normalizes options and sorts
        extras in canonical menu order, two items with the same customizations always
        produce the exact same signature.
        """
        return (
            self.item_id,
            tuple(sorted(self.options.items())),
            tuple(self.extras),
        )

    def recalculate_total(self) -> None:
        """Recomputes `line_total` from `unit_price * quantity` using exact Decimal math."""
        total = (Decimal(str(self.unit_price)) * Decimal(self.quantity)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        self.line_total = float(total)

    def format_description(self) -> str:
        """Human-readable summary of the line item (e.g., '2x Large Classic Burger (beef) with cheese, bacon — $23.00')."""
        opt_parts = [v for k, v in self.options.items() if k != "size"]
        size_prefix = f"{self.options['size'].title()} " if "size" in self.options else ""
        opt_suffix = f" ({', '.join(opt_parts)})" if opt_parts else ""
        extra_suffix = f" with {', '.join(e.replace('_', ' ') for e in self.extras)}" if self.extras else ""
        qty_prefix = f"{self.quantity}x " if self.quantity > 1 else "1x "
        return (
            f"[{self.line_id}] {qty_prefix}{size_prefix}{self.name}"
            f"{opt_suffix}{extra_suffix} "
            f"(${self.unit_price:.2f} ea, line total: ${self.line_total:.2f})"
        )


class EditItemToolInput(BaseModel):
    """Schema for the `edit_item` tool using delta parameters and partial-stack splitting."""
    model_config = ConfigDict(extra="forbid")

    line_id: str = Field(
        ...,
        description="The unique line_id in the current cart to modify (e.g., 'line_1').",
    )
    target_quantity: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many units from this line to apply option/extra changes to. "
            "If the line has quantity=2 and the user says 'add bacon to just one of them', "
            "set target_quantity=1 to split the line. Leave None to modify the entire line."
        ),
    )
    new_quantity: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Set this ONLY if the user wants to change the total count of this line "
            "(e.g., 'make that 3 burgers' -> new_quantity=3). Leave None otherwise."
        ),
    )
    options: dict[str, str] | None = Field(
        default=None,
        description="Option updates to merge into existing options (e.g., {'size': 'large'}).",
    )
    add_extras: list[str] | None = Field(
        default=None,
        description="Extra IDs to ADD to the item's existing extras without wiping previous extras.",
    )
    remove_extras: list[str] | None = Field(
        default=None,
        description="Extra IDs to REMOVE from the item's existing extras.",
    )
    special_instructions: str | None = Field(
        default=None,
        description="Optional order-level special instructions or dietary notes to save.",
    )


class RemoveItemToolInput(BaseModel):
    """Schema for the `remove_item` tool supporting full-line or partial-quantity removal."""
    model_config = ConfigDict(extra="forbid")

    line_id: str = Field(
        ...,
        description="The unique line_id in the current cart to remove from (e.g., 'line_1').",
    )
    quantity: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of units to remove from this line. Leave None (or set equal to the line's "
            "current quantity) to remove the entire line from the cart."
        ),
    )


# =====================================================================
# 2. Stateful OrderCart Engine (Add, Edit, Remove, Split & Merge)
# =====================================================================

class OrderCart:
    """
    Deterministic per-session cart state engine.
    Handles:
      - Auto-merging identical items into a single line (`_consolidate_lines`).
      - Auto-splitting stacks when `edit_item` targets a subset (`target_quantity < line.quantity`).
      - Atomic validation rollback (invalid edits never corrupt existing cart state).
      - Persistent `special_instructions` aggregation across turns.
      - Turn-level mutation tracking (`modified_in_current_turn`) to prevent unconfirmed submissions.
      - Automatic cart reset when starting a new order after a successful submission.
    """

    def __init__(self, menu: MenuCatalog):
        self.menu = menu
        self.lines: list[CartLineItem] = []
        self.special_instructions: str | None = None
        self.is_submitted: bool = False
        self.last_submitted_order_id: str | None = None
        self.modified_in_current_turn: bool = False
        self._line_counter: int = 0

    # -----------------------------------------------------------------
    # Turn & Lifecycle Management
    # -----------------------------------------------------------------
    def start_turn(self) -> None:
        """
        Called at the start of every `agent.send()` turn.
        Resets the per-turn modification flag so we can detect the 'Post-Confirmation Cold Feet'
        edge case ('Yes, and also add a cola' -> must not submit on the same turn).
        """
        self.modified_in_current_turn = False

    def mark_submitted(self, order_id: str | None = None) -> None:
        """Marks the current cart as submitted after MCP `submit_order` succeeds."""
        self.is_submitted = True
        self.last_submitted_order_id = order_id

    def clear(self) -> None:
        """Resets the cart completely for a fresh order."""
        self.lines.clear()
        self.special_instructions = None
        self.is_submitted = False
        self.last_submitted_order_id = None
        self.modified_in_current_turn = False
        self._line_counter = 0

    def _ensure_active_cart(self) -> None:
        """
        Edge Case #6 (Post-Submission Second Order):
        If the previous order was already submitted and the user starts adding new items,
        automatically clear the old submitted cart so they begin a fresh order cleanly.
        """
        if self.is_submitted:
            self.clear()

    def _next_line_id(self) -> str:
        self._line_counter += 1
        return f"line_{self._line_counter}"

    def _find_line(self, line_id: str) -> CartLineItem | None:
        normalized = line_id.strip().lower()
        for line in self.lines:
            if line.line_id.lower() == normalized:
                return line
        return None

    # -----------------------------------------------------------------
    # Totals & $50 Limit Inspection
    # -----------------------------------------------------------------
    @property
    def total_price(self) -> float:
        total = sum((Decimal(str(line.line_total)) for line in self.lines), Decimal("0.00"))
        return float(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    @property
    def total_item_count(self) -> int:
        return sum(line.quantity for line in self.lines)

    @property
    def exceeds_limit(self) -> bool:
        return Decimal(str(self.total_price)) > MAX_ORDER_TOTAL

    def update_special_instructions(self, notes: str | None, replace: bool = False) -> None:
        """
        Edge Case #5 (Special Instructions Aggregation & Clearing):
        Persists special instructions across turns so early dietary notes ('no onions please')
        are deterministically included when `submit_order` is called later, while allowing
        explicit replacement or clearing when requested.
        """
        if notes is None:
            if replace:
                self.special_instructions = None
            return

        cleaned = notes.strip()
        if not cleaned or cleaned.lower() in ("none", "clear", "null", "n/a", "no special instructions"):
            if replace or cleaned.lower() in ("none", "clear", "null", "n/a", "no special instructions"):
                self.special_instructions = None
            return

        if replace or not self.special_instructions:
            self.special_instructions = cleaned
        elif cleaned.lower() not in self.special_instructions.lower():
            self.special_instructions = f"{self.special_instructions}; {cleaned}"

    # -----------------------------------------------------------------
    # Core Function 1: add_item
    # -----------------------------------------------------------------
    def add_item(
        self,
        item_id: str,
        quantity: int = 1,
        options: dict[str, Any] | None = None,
        extras: list[str] | None = None,
        special_instructions: str | None = None,
    ) -> dict[str, Any]:
        """
        Validates an item against `MenuCatalog`, resolves default options, and adds it to the cart.
        If an identical item (same item_id, resolved options, and canonical extras) already exists
        in the cart, increments that existing line's quantity instead of creating a duplicate line.
        """
        self._ensure_active_cart()

        try:
            validated: ValidatedOrderItem = self.menu.validate_and_resolve(
                item_id=item_id,
                quantity=quantity,
                options=options,
                extras=extras,
            )
        except MenuValidationError as exc:
            return {
                "success": False,
                "error": str(exc),
                "cart": self.get_state_snapshot(),
            }

        if special_instructions:
            self.update_special_instructions(special_instructions)

        new_line = CartLineItem(
            line_id=self._next_line_id(),
            item_id=validated.item_id,
            name=validated.name,
            quantity=validated.quantity,
            options=validated.options,
            extras=validated.extras,
            unit_price=validated.unit_price,
            line_total=validated.line_total,
        )

        # Auto-merge if an identical line already exists in the cart
        merged_target = self._merge_or_append(new_line)
        self.modified_in_current_turn = True

        return {
            "success": True,
            "action": "added_or_merged",
            "line": merged_target.model_dump(),
            "cart": self.get_state_snapshot(),
        }

    # -----------------------------------------------------------------
    # Core Function 2: edit_item (Delta Updates + Split & Merge)
    # -----------------------------------------------------------------
    def edit_item(
        self,
        line_id: str,
        target_quantity: int | None = None,
        new_quantity: int | None = None,
        options: dict[str, Any] | None = None,
        add_extras: list[str] | None = None,
        remove_extras: list[str] | None = None,
        special_instructions: str | None = None,
    ) -> dict[str, Any]:
        """
        Modifies an existing cart line using delta parameters (`options`, `add_extras`, `remove_extras`).
        Supports:
          1. Partial Stack Split: If `line.quantity == 2` and `target_quantity == 1` with option/extra
             changes, decrements the original line to 1 and splits the modified 1 into a new (or merged) line.
          2. Full Line Edit: Modifies the line in-place and auto-merges if it now matches another line.
          3. Quantity Resize: If `new_quantity` is provided, updates the line quantity (or removes if 0).
          4. Atomic Rollback: Validates new options/extras before mutating the cart.
        """
        if self.is_submitted:
            return {
                "success": False,
                "error": "The previous order has already been submitted. Please add items to start a new order.",
                "cart": self.get_state_snapshot(),
            }

        target_line = self._find_line(line_id)
        if target_line is None:
            available = [f"{l.line_id} ({l.quantity}x {l.name})" for l in self.lines]
            return {
                "success": False,
                "error": f"Line '{line_id}' not found in cart. Current lines: {available}",
                "cart": self.get_state_snapshot(),
            }

        # Handle `new_quantity == 0` as a removal
        if new_quantity is not None and new_quantity <= 0:
            return self.remove_item(line_id=line_id)

        if special_instructions:
            self.update_special_instructions(special_instructions)

        # Compute candidate options by merging delta options over the line's current options
        menu_item = self.menu.get_item(target_line.item_id)
        candidate_options = dict(target_line.options)
        if options:
            cleaned_opts = {k: str(v).lower().strip() for k, v in options.items() if v is not None}
            valid_keys_present = any(k in menu_item.options for k in cleaned_opts)
            for k, v in cleaned_opts.items():
                # Strip hallucinated cross-item default options (e.g., flavor='cola' when editing a pizza)
                if k not in menu_item.options and valid_keys_present and v in ("cola", "regular", "medium", "beef"):
                    continue
                candidate_options[k] = v

        # Compute candidate extras by applying `add_extras` and `remove_extras` deltas
        remove_set = {str(e).lower().strip() for e in (remove_extras or []) if e}
        candidate_extras = [e for e in target_line.extras if e not in remove_set]
        for extra_to_add in add_extras or []:
            cleaned_extra = str(extra_to_add).lower().strip()
            if cleaned_extra and cleaned_extra not in candidate_extras:
                candidate_extras.append(cleaned_extra)

        has_customization_change = bool(options or add_extras or remove_extras)

        # Determine if this is a partial stack split (e.g., 2 burgers in line_1, edit 1 of them)
        is_partial_split = (
            target_quantity is not None
            and 0 < target_quantity < target_line.quantity
            and has_customization_change
        )

        if is_partial_split:
            split_qty = target_quantity  # type: ignore[assignment]
            # Validate the split-off item BEFORE mutating `target_line` (atomic rollback guarantee)
            try:
                validated_split = self.menu.validate_and_resolve(
                    item_id=target_line.item_id,
                    quantity=split_qty,
                    options=candidate_options,
                    extras=candidate_extras,
                )
            except MenuValidationError as exc:
                return {
                    "success": False,
                    "error": str(exc),
                    "cart": self.get_state_snapshot(),
                }

            # Decrement original line and recalculate its total
            target_line.quantity -= split_qty
            target_line.recalculate_total()

            # Create the split line and merge it if an identical customization already exists
            split_line = CartLineItem(
                line_id=self._next_line_id(),
                item_id=validated_split.item_id,
                name=validated_split.name,
                quantity=validated_split.quantity,
                options=validated_split.options,
                extras=validated_split.extras,
                unit_price=validated_split.unit_price,
                line_total=validated_split.line_total,
            )
            result_line = self._merge_or_append(split_line)
            self.modified_in_current_turn = True

            return {
                "success": True,
                "action": "split_and_updated",
                "original_line": target_line.model_dump(),
                "updated_line": result_line.model_dump(),
                "cart": self.get_state_snapshot(),
            }

        # Standard full-line edit (mutates the entire stack or updates its total quantity)
        final_qty = new_quantity if new_quantity is not None else target_line.quantity
        try:
            validated_full = self.menu.validate_and_resolve(
                item_id=target_line.item_id,
                quantity=final_qty,
                options=candidate_options,
                extras=candidate_extras,
            )
        except MenuValidationError as exc:
            return {
                "success": False,
                "error": str(exc),
                "cart": self.get_state_snapshot(),
            }

        target_line.quantity = validated_full.quantity
        target_line.options = validated_full.options
        target_line.extras = validated_full.extras
        target_line.unit_price = validated_full.unit_price
        target_line.line_total = validated_full.line_total

        # If editing this line made it identical to another line in the cart, merge them!
        consolidated_line = self._consolidate_lines(preferred_line_id=target_line.line_id)
        self.modified_in_current_turn = True

        return {
            "success": True,
            "action": "updated",
            "line": (consolidated_line or target_line).model_dump(),
            "cart": self.get_state_snapshot(),
        }

    # -----------------------------------------------------------------
    # Core Function 3: remove_item (Full Line or Partial Quantity)
    # -----------------------------------------------------------------
    def remove_item(
        self,
        line_id: str,
        quantity: int | None = None,
    ) -> dict[str, Any]:
        """
        Removes an item line from the cart, or decrements its quantity if `quantity < line.quantity`.
        """
        if self.is_submitted:
            return {
                "success": False,
                "error": "The previous order has already been submitted.",
                "cart": self.get_state_snapshot(),
            }

        target_line = self._find_line(line_id)
        if target_line is None:
            available = [f"{l.line_id} ({l.quantity}x {l.name})" for l in self.lines]
            return {
                "success": False,
                "error": f"Line '{line_id}' not found in cart. Current lines: {available}",
                "cart": self.get_state_snapshot(),
            }

        if quantity is not None and quantity < 1:
            return {
                "success": False,
                "error": f"Removal quantity must be at least 1 (got {quantity}).",
                "cart": self.get_state_snapshot(),
            }

        # Partial quantity decrement (e.g., user has 2 burgers and says "remove one of the burgers")
        if quantity is not None and quantity < target_line.quantity:
            target_line.quantity -= quantity
            target_line.recalculate_total()
            self.modified_in_current_turn = True
            return {
                "success": True,
                "action": "decremented",
                "removed_quantity": quantity,
                "remaining_line": target_line.model_dump(),
                "cart": self.get_state_snapshot(),
            }

        # Full line removal
        self.lines = [l for l in self.lines if l.line_id != target_line.line_id]
        self.modified_in_current_turn = True
        return {
            "success": True,
            "action": "removed_line",
            "removed_line": target_line.model_dump(),
            "cart": self.get_state_snapshot(),
        }
    
    # -----------------------------------------------------------------
    # Core Function 4: clear_cart (Remove All Items From Cart)
    # -----------------------------------------------------------------
     
    def clear_cart(self) -> dict[str, Any]:
        """Tool implementation: empties all items from the cart."""
        self.clear()
        self.modified_in_current_turn = True
        return {
            "success": True,
            "action": "cleared_cart",
            "message": "Cart has been cleared.",
            "cart": self.get_state_snapshot(),
        }

    # -----------------------------------------------------------------
    # Auto-Merge Consolidation Helpers
    # -----------------------------------------------------------------
    def _merge_or_append(self, new_line: CartLineItem) -> CartLineItem:
        """
        Checks if a line with the exact same `(item_id, options, extras)` signature
        already exists. If so, adds `new_line.quantity` to it; otherwise appends `new_line`.
        """
        for existing in self.lines:
            if existing.signature == new_line.signature:
                existing.quantity += new_line.quantity
                existing.recalculate_total()
                return existing

        self.lines.append(new_line)
        return new_line

    def _consolidate_lines(self, preferred_line_id: str | None = None) -> CartLineItem | None:
        """
        Scans the cart after an `edit_item` call and merges any two lines that now share
        the exact same signature `(item_id, options, extras)`.
        """
        by_sig: dict[tuple[str, tuple[tuple[str, str], ...], tuple[str, ...]], CartLineItem] = {}
        consolidated: list[CartLineItem] = []
        matched_target: CartLineItem | None = None

        for line in self.lines:
            sig = line.signature
            if sig not in by_sig:
                by_sig[sig] = line
                consolidated.append(line)
            else:
                primary = by_sig[sig]
                primary.quantity += line.quantity
                primary.recalculate_total()
                if line.line_id == preferred_line_id:
                    matched_target = primary

        self.lines = consolidated
        if matched_target is not None:
            return matched_target
        return self._find_line(preferred_line_id) if preferred_line_id else None

    # -----------------------------------------------------------------
    # Tool Dispatcher, MCP Serialization & System Prompt Formatting
    # -----------------------------------------------------------------
    def execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
        """Unified dispatcher called by the agentic loop for local cart tools."""
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name == "add_item":
            raw_qty = tool_args.get("quantity", 1)
            qty = _safe_int(raw_qty, default=1 if raw_qty is None else -1)
            return self.add_item(
                item_id=str(tool_args.get("item_id") or ""),
                quantity=qty if qty is not None else 1,
                options=tool_args.get("options") if isinstance(tool_args.get("options"), dict) else None,
                extras=tool_args.get("extras") if isinstance(tool_args.get("extras"), list) else None,
                special_instructions=tool_args.get("special_instructions"),
            )
        elif tool_name == "edit_item":
            return self.edit_item(
                line_id=str(tool_args.get("line_id") or ""),
                target_quantity=_safe_int(tool_args.get("target_quantity")),
                new_quantity=_safe_int(tool_args.get("new_quantity")),
                options=tool_args.get("options") if isinstance(tool_args.get("options"), dict) else None,
                add_extras=tool_args.get("add_extras") if isinstance(tool_args.get("add_extras"), list) else None,
                remove_extras=tool_args.get("remove_extras") if isinstance(tool_args.get("remove_extras"), list) else None,
                special_instructions=tool_args.get("special_instructions"),
            )
        elif tool_name == "remove_item":
            raw_rem_qty = tool_args.get("quantity")
            rem_qty = None if raw_rem_qty is None else _safe_int(raw_rem_qty, default=-1)
            return self.remove_item(
                line_id=str(tool_args.get("line_id") or ""),
                quantity=rem_qty,
            )
        elif tool_name == "clear_cart":
            return self.clear_cart()
        else:
            return {"success": False, "error": f"Unknown cart tool '{tool_name}'."}

    def to_mcp_items(self) -> list[dict[str, Any]]:
        """
        Serializes the cart lines into the exact schema required by the Moveo MCP `submit_order` tool:
        `[{"item_id": ..., "quantity": ..., "options": {...}, "extras": [...]}]`
        """
        return [
            {
                "item_id": line.item_id,
                "quantity": line.quantity,
                "options": dict(line.options),
                "extras": list(line.extras),
            }
            for line in self.lines
        ]

    def get_state_snapshot(self) -> dict[str, Any]:
        """
        Returns a complete JSON-serializable snapshot of the cart for:
          1. Tool response payloads fed back to the LLM.
          2. Structured turn logging (logger.py).
          3. WebSocket/SSE pushes to the React OrderTicketHUD.tsx component.
        """
        return {
            "lines": [line.model_dump() for line in self.lines],
            "total_item_count": self.total_item_count,
            "total_price": self.total_price,
            "exceeds_50_limit": self.exceeds_limit,
            "special_instructions": self.special_instructions,
            "is_submitted": self.is_submitted,
            "last_submitted_order_id": self.last_submitted_order_id,
            "modified_in_current_turn": self.modified_in_current_turn,
        }

    def format_for_prompt(self) -> str:
        """
        Formats the live cart state for injection into the user context wrapper
        on every turn of the agentic loop—eliminating the need for a separate list_items tool call.
        """
        if self.is_submitted:
            return (
                f"CURRENT CART STATE: Order #{self.last_submitted_order_id or 'SUBMITTED'} "
                "was already submitted. If the user orders again, adding an item will start a fresh cart."
            )

        if not self.lines:
            return "CURRENT CART STATE: Empty (0 items, Total: $0.00)."

        header = [f"CURRENT CART STATE ({self.total_item_count} item(s), Total: ${self.total_price:.2f}):"]
        for line in self.lines:
            header.append(f"  - {line.format_description()}")

        if self.special_instructions:
            header.append(f"  - Special Instructions: \"{self.special_instructions}\"")

        if self.exceeds_limit:
            header.append(
                f"  - WARNING: Cart total (${self.total_price:.2f}) exceeds the $50.00 restaurant limit! "
                "Inform the user they are over the $50.00 cap."
            )
        elif Decimal(str(self.total_price)) >= Decimal("40.00"):
            remaining = Decimal("50.00") - Decimal(str(self.total_price))
            header.append(
                f"  - BUDGET NOTICE: Cart total (${self.total_price:.2f}) is nearing the $50.00 limit "
                f"(only ${remaining:.2f} remaining). Do NOT ask 'Anything else?'—ask if they are ready to checkout."
            )

        if self.modified_in_current_turn:
            header.append(
                "  - TURN STATUS: Cart was modified during this turn. You MUST state the updated order/total "
                "and ask the user for confirmation before calling `submit_order`."
            )

        return "\n".join(header)