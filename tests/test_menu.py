# tests/test_menu.py
"""
Unit tests for MenuCatalog and menu schema validation (src/core/menu.py).
Tests deterministic option resolution, milkshake trap, cross-item hints, and Decimal math.
"""
from __future__ import annotations

import pytest
from src.core.menu import MenuCatalog, MenuValidationError


@pytest.fixture
def menu() -> MenuCatalog:
    return MenuCatalog.from_yaml("menu.yaml")


# =====================================================================
# 1. Default Option Fill Verification
# =====================================================================

def test_default_fill_classic_burger(menu: MenuCatalog) -> None:
    item = menu.validate_and_resolve("classic_burger")
    assert item.item_id == "classic_burger"
    assert item.name == "Classic Burger"
    assert item.options == {"size": "regular", "patty": "beef"}
    assert item.extras == []
    assert item.unit_price == 8.50
    assert item.line_total == 8.50


def test_default_fill_margherita(menu: MenuCatalog) -> None:
    item = menu.validate_and_resolve("margherita")
    assert item.options == {"size": "medium", "crust": "regular"}
    assert item.extras == []
    assert item.unit_price == 12.00
    assert item.line_total == 12.00


def test_default_fill_sides_and_drinks(menu: MenuCatalog) -> None:
    fries = menu.validate_and_resolve("fries")
    assert fries.options == {"size": "medium"}
    assert fries.unit_price == 3.50

    rings = menu.validate_and_resolve("onion_rings")
    assert rings.options == {"size": "medium"}
    assert rings.unit_price == 4.50

    soda = menu.validate_and_resolve("soda")
    assert soda.options == {"size": "medium", "flavor": "cola"}
    assert soda.unit_price == 2.00

    spicy = menu.validate_and_resolve("spicy_burger")
    assert spicy.options == {"size": "regular", "spice_level": "medium"}
    assert spicy.unit_price == 9.50


# =====================================================================
# 2. The "Milkshake Trap" (Missing Required Option)
# =====================================================================

def test_milkshake_trap_missing_flavor(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc_info:
        menu.validate_and_resolve("milkshake")
    
    err = str(exc_info.value).lower()
    assert "flavor" in err
    assert "missing required option" in err
    for flavor in ["vanilla", "chocolate", "strawberry", "oreo"]:
        assert flavor in err


def test_milkshake_resolved_with_flavor(menu: MenuCatalog) -> None:
    shake = menu.validate_and_resolve("milkshake", options={"flavor": "chocolate"})
    assert shake.options == {"size": "regular", "flavor": "chocolate"}
    assert shake.unit_price == 5.50
    assert shake.line_total == 5.50


# =====================================================================
# 3. Cross-Item Hallucination Guard
# =====================================================================

def test_cross_item_option_patty_on_spicy_burger(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc_info:
        menu.validate_and_resolve("spicy_burger", options={"patty": "chicken"})
    
    err = str(exc_info.value)
    assert "Invalid option" in err
    assert "patty" in err
    assert "classic_burger" in err  # Cross-item hint


def test_cross_item_option_crust_on_burger(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc_info:
        menu.validate_and_resolve("classic_burger", options={"crust": "thin"})
    
    err = str(exc_info.value)
    assert "crust" in err
    assert "margherita" in err


def test_cross_item_extra_on_fries(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc_info:
        menu.validate_and_resolve("fries", extras=["extra_patty"])
    
    err = str(exc_info.value)
    assert "extra_patty" in err
    assert "classic_burger" in err


def test_cross_item_extra_on_spicy_burger(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc_info:
        menu.validate_and_resolve("spicy_burger", extras=["avocado"])
    
    err = str(exc_info.value)
    assert "avocado" in err
    assert "classic_burger" in err


def test_extras_on_item_without_extras_rejected(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc_info:
        menu.validate_and_resolve("soda", extras=["cheese"])
    
    err = str(exc_info.value)
    assert "does not support any extras" in err


# =====================================================================
# 4. Decimal Math Precision & Price Modifiers
# =====================================================================

def test_negative_price_modifiers(menu: MenuCatalog) -> None:
    small_fries = menu.validate_and_resolve("fries", options={"size": "small"})
    # Base $3.50 - $1.00 = $2.50
    assert small_fries.unit_price == 2.50

    small_pizza = menu.validate_and_resolve("margherita", options={"size": "small"})
    # Base $12.00 - $2.00 = $10.00
    assert small_pizza.unit_price == 10.00

    small_soda = menu.validate_and_resolve("soda", options={"size": "small"})
    # Base $2.00 - $0.50 = $1.50
    assert small_soda.unit_price == 1.50


def test_positive_price_modifiers_and_extras(menu: MenuCatalog) -> None:
    # Large classic burger ($8.50 + $2.00 = $10.50)
    # + cheese ($1.00) + bacon ($1.50) + avocado ($2.00) + extra_patty ($3.00) = $18.00
    burger = menu.validate_and_resolve(
        "classic_burger",
        quantity=3,
        options={"size": "large"},
        extras=["cheese", "bacon", "avocado", "extra_patty"],
    )
    assert burger.unit_price == 18.00
    assert burger.line_total == 54.00


def test_fractional_prices_no_drift(menu: MenuCatalog) -> None:
    # Onion rings with ranch_dip ($4.50 + $0.75 = $5.25)
    rings = menu.validate_and_resolve("onion_rings", extras=["ranch_dip"])
    assert rings.unit_price == 5.25

    # Milkshake with cherry ($5.50 + $0.25 = $5.75)
    shake = menu.validate_and_resolve("milkshake", options={"flavor": "oreo"}, extras=["cherry_on_top"])
    assert shake.unit_price == 5.75


# =====================================================================
# 5. Extras Deduplication & Canonical Order
# =====================================================================

def test_extras_deduplicated_and_canonically_sorted(menu: MenuCatalog) -> None:
    burger = menu.validate_and_resolve(
        "classic_burger",
        extras=["bacon", "cheese", "bacon", "cheese"],
    )
    # Menu order: cheese, bacon, avocado, extra_patty
    assert burger.extras == ["cheese", "bacon"]


# =====================================================================
# 6. Invalid Inputs & Quantities
# =====================================================================

def test_invalid_quantity_rejected(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc:
        menu.validate_and_resolve("classic_burger", quantity=0)
    assert "Quantity must be at least 1" in str(exc.value)

    with pytest.raises(MenuValidationError) as exc2:
        menu.validate_and_resolve("classic_burger", quantity=-3)
    assert "Quantity must be at least 1" in str(exc2.value)


def test_unknown_item_id_rejected(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc:
        menu.validate_and_resolve("tacos")
    assert "Unknown item_id 'tacos'" in str(exc.value)


def test_invalid_choice_rejected(menu: MenuCatalog) -> None:
    with pytest.raises(MenuValidationError) as exc:
        menu.validate_and_resolve("fries", options={"size": "extra_large"})
    assert "Invalid choice 'extra_large'" in str(exc.value)