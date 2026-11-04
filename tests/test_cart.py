# tests/test_cart.py
"""
Unit tests for OrderCart state machine and transitions (src/core/cart.py).
Tests auto-merge, signature differentiation, stack splitting, consolidation, atomic rollback,
and special instructions persistence.
"""
from __future__ import annotations

import pytest
from src.core.cart import OrderCart
from src.core.menu import MenuCatalog


@pytest.fixture
def menu() -> MenuCatalog:
    return MenuCatalog.from_yaml("menu.yaml")


@pytest.fixture
def cart(menu: MenuCatalog) -> OrderCart:
    return OrderCart(menu)


# =====================================================================
# 1. Auto-Merge on Add & Signature Differentiation
# =====================================================================

def test_auto_merge_identical_items_on_add(cart: OrderCart) -> None:
    res1 = cart.add_item("classic_burger", quantity=1)
    assert res1["success"] is True
    assert len(cart.lines) == 1
    assert cart.lines[0].quantity == 1
    assert cart.total_price == 8.50

    res2 = cart.add_item("classic_burger", quantity=1)
    assert res2["success"] is True
    assert len(cart.lines) == 1  # Merged into line_1
    assert cart.lines[0].line_id == "line_1"
    assert cart.lines[0].quantity == 2
    assert cart.lines[0].line_total == 17.00
    assert cart.total_price == 17.00


def test_signature_differentiation(cart: OrderCart) -> None:
    cart.add_item("classic_burger", quantity=1, options={"size": "regular"})
    cart.add_item("classic_burger", quantity=1, options={"size": "large"})

    assert len(cart.lines) == 2
    assert cart.lines[0].line_id == "line_1"
    assert cart.lines[0].options["size"] == "regular"
    assert cart.lines[1].line_id == "line_2"
    assert cart.lines[1].options["size"] == "large"
    assert cart.total_item_count == 2
    assert cart.total_price == 8.50 + 10.50  # 19.00


def test_extras_signature_differentiation(cart: OrderCart) -> None:
    cart.add_item("classic_burger", extras=["cheese"])
    cart.add_item("classic_burger", extras=["bacon"])

    assert len(cart.lines) == 2
    assert cart.lines[0].extras == ["cheese"]
    assert cart.lines[1].extras == ["bacon"]


# =====================================================================
# 2. Atomic Stack Splitting (edit_item)
# =====================================================================

def test_atomic_stack_splitting(cart: OrderCart) -> None:
    # 3 Spicy Burgers in line_1
    cart.add_item("spicy_burger", quantity=3)
    assert len(cart.lines) == 1
    assert cart.lines[0].quantity == 3
    assert cart.lines[0].line_total == 28.50

    # User says "add bacon to just one of them"
    res = cart.edit_item(line_id="line_1", target_quantity=1, add_extras=["bacon"])
    assert res["success"] is True
    assert res["action"] == "split_and_updated"

    assert len(cart.lines) == 2
    # Original line decremented to 2
    assert cart.lines[0].line_id == "line_1"
    assert cart.lines[0].quantity == 2
    assert cart.lines[0].extras == []
    assert cart.lines[0].line_total == 19.00

    # New split line created with quantity 1
    assert cart.lines[1].line_id == "line_2"
    assert cart.lines[1].quantity == 1
    assert cart.lines[1].extras == ["bacon"]
    assert cart.lines[1].line_total == 11.00

    assert cart.total_price == 30.00


# =====================================================================
# 3. Consolidation on Edit (_consolidate_lines)
# =====================================================================

def test_consolidation_on_edit(cart: OrderCart) -> None:
    cart.add_item("classic_burger", extras=["cheese"])
    cart.add_item("classic_burger")
    assert len(cart.lines) == 2

    # Edit line_2 to add cheese -> now identical to line_1
    res = cart.edit_item(line_id="line_2", add_extras=["cheese"])
    assert res["success"] is True

    # Auto-consolidates back into 1 line with quantity 2
    assert len(cart.lines) == 1
    assert cart.lines[0].quantity == 2
    assert cart.lines[0].extras == ["cheese"]
    assert cart.lines[0].line_total == 19.00
    assert cart.total_price == 19.00


# =====================================================================
# 4. Atomic Rollback on Invalid Edit
# =====================================================================

def test_atomic_rollback_on_invalid_edit(cart: OrderCart) -> None:
    cart.add_item("classic_burger", quantity=2, options={"size": "regular"})
    snapshot_before = cart.get_state_snapshot()

    # Attempt to edit with an invalid extra
    res = cart.edit_item(line_id="line_1", add_extras=["pineapple"])
    assert res["success"] is False
    assert "Invalid extra" in res["error"]

    # State must be completely intact
    snapshot_after = cart.get_state_snapshot()
    assert snapshot_before == snapshot_after
    assert len(cart.lines) == 1
    assert cart.lines[0].quantity == 2
    assert cart.lines[0].extras == []


def test_atomic_rollback_on_invalid_split(cart: OrderCart) -> None:
    cart.add_item("spicy_burger", quantity=3)
    snapshot_before = cart.get_state_snapshot()

    # Attempt split with invalid cross-item option (patty on spicy burger)
    res = cart.edit_item(line_id="line_1", target_quantity=1, options={"patty": "chicken"})
    assert res["success"] is False

    # Original line was not decremented
    assert len(cart.lines) == 1
    assert cart.lines[0].quantity == 3
    assert cart.get_state_snapshot() == snapshot_before


# =====================================================================
# 5. Quantity Resizing & Removals
# =====================================================================

def test_quantity_resize_and_zero_removal(cart: OrderCart) -> None:
    cart.add_item("fries", quantity=1)
    assert cart.lines[0].quantity == 1

    # Resize to 4
    cart.edit_item(line_id="line_1", new_quantity=4)
    assert cart.lines[0].quantity == 4
    assert cart.lines[0].line_total == 14.00

    # Resize to 0 -> removes item
    cart.edit_item(line_id="line_1", new_quantity=0)
    assert len(cart.lines) == 0
    assert cart.total_price == 0.00


def test_partial_and_full_item_removal(cart: OrderCart) -> None:
    cart.add_item("soda", quantity=3)
    assert cart.lines[0].quantity == 3

    # Partial decrement: remove 1
    res1 = cart.remove_item(line_id="line_1", quantity=1)
    assert res1["success"] is True
    assert res1["action"] == "decremented"
    assert cart.lines[0].quantity == 2

    # Full removal
    res2 = cart.remove_item(line_id="line_1")
    assert res2["success"] is True
    assert res2["action"] == "removed_line"
    assert len(cart.lines) == 0


def test_remove_invalid_line_returns_error(cart: OrderCart) -> None:
    res = cart.remove_item(line_id="line_999")
    assert res["success"] is False
    assert "not found" in res["error"]


# =====================================================================
# 6. Special Instructions Accumulation
# =====================================================================

def test_special_instructions_accumulation(cart: OrderCart) -> None:
    cart.update_special_instructions("no onions")
    assert cart.special_instructions == "no onions"

    cart.update_special_instructions("extra napkins")
    assert cart.special_instructions == "no onions; extra napkins"

    # Does not duplicate already present instructions
    cart.update_special_instructions("no onions")
    assert cart.special_instructions == "no onions; extra napkins"


# =====================================================================
# 7. $50 Limit & Budget Cap Properties
# =====================================================================

def test_exceeds_limit_flag(cart: OrderCart) -> None:
    cart.add_item("margherita", quantity=2, options={"size": "large"})
    assert cart.exceeds_limit is False

    # Add 2 more Large Pizzas ($32.00 + $32.00 = $64.00 > $50)
    cart.add_item("margherita", quantity=2, options={"size": "large"})
    assert cart.total_price == 64.00
    assert cart.exceeds_limit is True


# =====================================================================
# 8. Clear Cart & Post-Submission Auto-Reset
# =====================================================================

def test_clear_cart(cart: OrderCart) -> None:
    cart.add_item("classic_burger")
    cart.update_special_instructions("pack carefully")
    assert len(cart.lines) == 1

    cart.clear_cart()
    assert len(cart.lines) == 0
    assert cart.special_instructions is None
    assert cart.total_price == 0.00


def test_auto_reset_on_add_after_submission(cart: OrderCart) -> None:
    cart.add_item("classic_burger")
    cart.mark_submitted("ORD-12345")
    assert cart.is_submitted is True

    # Next add_item should automatically clear old order and start fresh
    cart.add_item("fries")
    assert cart.is_submitted is False
    assert len(cart.lines) == 1
    assert cart.lines[0].item_id == "fries"
    assert cart.lines[0].line_id == "line_1"


# =====================================================================
# 9. MCP Serialization (to_mcp_items)
# =====================================================================

def test_to_mcp_items_format(cart: OrderCart) -> None:
    cart.add_item("classic_burger", quantity=2, options={"size": "large", "patty": "chicken"}, extras=["cheese", "bacon"])
    mcp_items = cart.to_mcp_items()

    assert len(mcp_items) == 1
    assert mcp_items[0] == {
        "item_id": "classic_burger",
        "quantity": 2,
        "options": {"size": "large", "patty": "chicken"},
        "extras": ["cheese", "bacon"],
    }