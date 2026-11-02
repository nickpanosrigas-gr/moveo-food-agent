from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


# =====================================================================
# 1. Menu Schema Models (Parses menu.yaml)
# =====================================================================

class OptionDefinition(BaseModel):
    """Represents a single option category (e.g., size, patty, crust, flavor)."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["single_choice"] = "single_choice"
    required: bool = False
    choices: list[str]
    default: str | None = None
    price_modifier: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_default_and_modifiers(self) -> OptionDefinition:
        if self.default is not None and self.default not in self.choices:
            raise ValueError(f"Default '{self.default}' must be one of {self.choices}")
        for choice_key in self.price_modifier:
            if choice_key not in self.choices:
                raise ValueError(f"Price modifier key '{choice_key}' not in choices {self.choices}")
        return self

    def get_modifier(self, choice: str) -> float:
        return float(self.price_modifier.get(choice, 0.0))


class ExtraChoice(BaseModel):
    """Represents a single extra add-on (e.g., cheese, bacon, truffle_oil)."""
    model_config = ConfigDict(extra="forbid")

    id: str
    price: float = Field(ge=0.0)


class ExtrasDefinition(BaseModel):
    """Represents the multi_choice extras block for a menu item."""
    model_config = ConfigDict(extra="forbid")

    type: Literal["multi_choice"] = "multi_choice"
    choices: list[ExtraChoice] = Field(default_factory=list)

    @property
    def price_map(self) -> dict[str, float]:
        return {extra.id: extra.price for extra in self.choices}

    @property
    def valid_ids(self) -> list[str]:
        return [extra.id for extra in self.choices]


class MenuItem(BaseModel):
    """Represents a single food or drink item in menu.yaml."""
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    base_price: float = Field(ge=0.0)
    options: dict[str, OptionDefinition] = Field(default_factory=dict)
    extras: ExtrasDefinition | None = None

    @property
    def valid_extra_ids(self) -> list[str]:
        return self.extras.valid_ids if self.extras else []

    @property
    def extra_prices(self) -> dict[str, float]:
        return self.extras.price_map if self.extras else {}


class MenuValidationError(ValueError):
    """Raised when an order item fails menu schema validation (invalid option, missing required flavor, etc.)."""
    pass


class ValidatedOrderItem(BaseModel):
    """The canonical, fully-resolved result of validating an item against the menu."""
    item_id: str
    name: str
    quantity: int = Field(ge=1)
    options: dict[str, str]
    extras: list[str]
    unit_price: float
    line_total: float


# =====================================================================
# 2. Menu Engine (Validation, Defaults, Price Math & Dynamic Tool Types)
# =====================================================================

class MenuCatalog:
    """
    Loads `menu.yaml` and provides:
      1. Deterministic option resolution (fills defaults, catches missing required options like Milkshake flavor).
      2. Exact unit and line price calculation (using Decimal to prevent floating-point drift).
      3. Dynamic Pydantic & Gemini tool schema generation so only valid item_ids and item-specific
         options/extras are allowed.
    """

    def __init__(self, items: list[MenuItem]):
        self.items: list[MenuItem] = items
        self.by_id: dict[str, MenuItem] = {item.id: item for item in items}
        if not self.by_id:
            raise ValueError("MenuCatalog cannot be initialized with an empty menu.")

    @classmethod
    def from_yaml(cls, path: str | Path = "menu.yaml") -> MenuCatalog:
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Menu file not found at: {yaml_path.resolve()}")

        with yaml_path.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not isinstance(raw_data, dict) or "menu" not in raw_data:
            raise ValueError("Invalid menu.yaml format: top-level 'menu' key is required.")

        parsed_items = [MenuItem.model_validate(entry) for entry in raw_data["menu"]]
        return cls(parsed_items)

    def get_item(self, item_id: str) -> MenuItem:
        if item_id not in self.by_id:
            valid_ids = ", ".join(self.by_id.keys())
            raise MenuValidationError(
                f"Unknown item_id '{item_id}'. Valid menu item IDs are: [{valid_ids}]."
            )
        return self.by_id[item_id]

    # -----------------------------------------------------------------
    # Option Resolution & Strict Per-Item Validation
    # -----------------------------------------------------------------
    def validate_and_resolve(
        self,
        item_id: str,
        quantity: int = 1,
        options: dict[str, Any] | None = None,
        extras: list[str] | None = None,
    ) -> ValidatedOrderItem:
        """
        Validates an item request against the exact menu item definition:
          - Rejects unknown item_ids or quantities < 1.
          - Rejects options that do not belong to this specific item (e.g., 'patty' on 'spicy_burger').
          - Applies default values for omitted options where a default exists.
          - Raises a helpful MenuValidationError if a required option has no default (e.g., milkshake 'flavor').
          - Rejects invalid extras for this item, deduplicates, and preserves canonical menu order.
          - Computes exact unit_price and line_total.
        """
        if quantity < 1:
            raise MenuValidationError(f"Quantity must be at least 1 (got {quantity}).")

        menu_item = self.get_item(item_id)
        provided_options = {k: str(v).lower().strip() for k, v in (options or {}).items() if v is not None}
        provided_extras = [str(e).lower().strip() for e in (extras or []) if e]

        # 1. Check for options that don't exist on this specific item
        unknown_opts = set(provided_options.keys()) - set(menu_item.options.keys())
        if unknown_opts:
            valid_opts = list(menu_item.options.keys())
            hint = self._build_cross_item_option_hint(unknown_opts)
            raise MenuValidationError(
                f"Invalid option(s) {sorted(unknown_opts)} for '{menu_item.name}' ({item_id}). "
                f"Allowed options for '{item_id}' are: {valid_opts}.{hint}"
            )

        # 2. Resolve each defined option on the menu item (user choice -> default -> error if required)
        resolved_options: dict[str, str] = {}
        for opt_name, opt_def in menu_item.options.items():
            user_choice = provided_options.get(opt_name)

            if user_choice is not None:
                if user_choice not in opt_def.choices:
                    raise MenuValidationError(
                        f"Invalid choice '{user_choice}' for option '{opt_name}' on '{menu_item.name}' ({item_id}). "
                        f"Valid choices for '{opt_name}' are: {opt_def.choices}."
                    )
                resolved_options[opt_name] = user_choice
            elif opt_def.default is not None:
                resolved_options[opt_name] = opt_def.default
            elif opt_def.required:
                raise MenuValidationError(
                    f"Missing required option '{opt_name}' for '{menu_item.name}' ({item_id}). "
                    f"Please ask the user to choose one of: {opt_def.choices}."
                )

        # 3. Validate extras against this specific item's allowed extras
        if provided_extras and not menu_item.extras:
            raise MenuValidationError(
                f"'{menu_item.name}' ({item_id}) does not support any extras (received {provided_extras})."
            )

        valid_extra_set = set(menu_item.valid_extra_ids)
        invalid_extras = [e for e in provided_extras if e not in valid_extra_set]
        if invalid_extras:
            hint = self._build_cross_item_extra_hint(invalid_extras)
            raise MenuValidationError(
                f"Invalid extra(s) {invalid_extras} for '{menu_item.name}' ({item_id}). "
                f"Allowed extras for '{item_id}' are: {menu_item.valid_extra_ids}.{hint}"
            )

        # Deduplicate extras and sort them in canonical menu order so identical items merge cleanly
        canonical_extras = [e_id for e_id in menu_item.valid_extra_ids if e_id in set(provided_extras)]

        # 4. Calculate unit price and line total using exact Decimal arithmetic
        unit_price = self.calculate_unit_price(item_id, resolved_options, canonical_extras)
        line_total = float(
            (Decimal(str(unit_price)) * Decimal(quantity)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        )

        return ValidatedOrderItem(
            item_id=item_id,
            name=menu_item.name,
            quantity=quantity,
            options=resolved_options,
            extras=canonical_extras,
            unit_price=unit_price,
            line_total=line_total,
        )

    def calculate_unit_price(
        self,
        item_id: str,
        resolved_options: dict[str, str],
        extras: list[str] | None = None,
    ) -> float:
        """Calculates the exact single-unit price of an item given its options and extras."""
        menu_item = self.get_item(item_id)
        total = Decimal(str(menu_item.base_price))

        for opt_name, choice_val in resolved_options.items():
            if opt_name in menu_item.options:
                modifier = menu_item.options[opt_name].get_modifier(choice_val)
                total += Decimal(str(modifier))

        extra_price_map = menu_item.extra_prices
        for extra_id in extras or []:
            if extra_id in extra_price_map:
                total += Decimal(str(extra_price_map[extra_id]))

        return float(total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    # -----------------------------------------------------------------
    # Dynamic Per-Item Pydantic Models & Gemini Tool Schema Generation
    # -----------------------------------------------------------------
    def build_per_item_models(self) -> dict[str, type[BaseModel]]:
        """
        Dynamically builds a strict Pydantic model for each item in `menu.yaml`.
        For example:
          - `classic_burger` gets `size: Literal['regular', 'large'] = 'regular'`,
            `patty: Literal['beef', 'chicken', 'veggie'] = 'beef'`,
            `extras: list[Literal['cheese', 'bacon', 'avocado', 'extra_patty']] = []`
          - `milkshake` gets `size: Literal['regular', 'large'] = 'regular'`,
            `flavor: Literal['vanilla', 'chocolate', 'strawberry', 'oreo'] | None = None`
        """
        per_item_models: dict[str, type[BaseModel]] = {}

        for item in self.items:
            opt_fields: dict[str, Any] = {}
            for opt_name, opt_def in item.options.items():
                choice_literal = Literal[tuple(opt_def.choices)]  # type: ignore[valid-type]
                # Make all option fields optional in the input model so defaults can be applied
                # or missing required options (like milkshake flavor) can return a clean tool message
                opt_fields[opt_name] = (
                    choice_literal | None,
                    Field(
                        default=opt_def.default,
                        description=(
                            f"{'REQUIRED' if opt_def.required and not opt_def.default else 'Optional'} "
                            f"{opt_name} for {item.name}. Allowed: {opt_def.choices}. Default: {opt_def.default}."
                        ),
                    ),
                )

            OptionsSubModel = create_model(
                f"{item.id.title().replace('_', '')}Options",
                __config__=ConfigDict(extra="forbid"),
                **opt_fields,
            )

            item_fields: dict[str, Any] = {
                "item_id": (Literal[item.id], Field(default=item.id, description=f"Must be '{item.id}' ({item.name})")),  # type: ignore[valid-type]
                "quantity": (int, Field(default=1, ge=1, description="Number of this item to add (1 or more)")),
                "options": (
                    OptionsSubModel,
                    Field(default_factory=OptionsSubModel, description=f"Valid options for {item.name}"),
                ),
            }

            if item.valid_extra_ids:
                extra_literal = Literal[tuple(item.valid_extra_ids)]  # type: ignore[valid-type]
                item_fields["extras"] = (
                    list[extra_literal],  # type: ignore[valid-type]
                    Field(default_factory=list, description=f"Valid extras for {item.name}: {item.valid_extra_ids}"),
                )
            else:
                item_fields["extras"] = (
                    list[NeverExtra] if False else list[str],
                    Field(default_factory=list, max_length=0, description=f"{item.name} does not allow any extras."),
                )

            per_item_models[item.id] = create_model(
                f"Add_{item.id.title().replace('_', '')}",
                __config__=ConfigDict(extra="forbid"),
                **item_fields,
            )

        return per_item_models

    def build_add_item_tool_model(self) -> type[BaseModel]:
        """
        Builds the unified `AddItemArgs` Pydantic model used by the `add_item` LLM tool.
        Uses strict `Literal` enums derived directly from `menu.yaml` for `item_id`, each option,
        and `extras`, and validates item-specific option compatibility automatically.
        """
        valid_item_ids = tuple(self.by_id.keys())
        ItemIdLiteral = Literal[valid_item_ids]  # type: ignore[valid-type]

        # Collect the union of all option names across the menu, annotated with which items support them
        option_to_choices: dict[str, list[str]] = {}
        option_to_items: dict[str, list[str]] = {}
        for item in self.items:
            for opt_name, opt_def in item.options.items():
                option_to_items.setdefault(opt_name, []).append(item.id)
                for c in opt_def.choices:
                    if c not in option_to_choices.setdefault(opt_name, []):
                        option_to_choices[opt_name].append(c)

        opt_fields: dict[str, Any] = {}
        for opt_name, choices in option_to_choices.items():
            ChoiceLit = Literal[tuple(choices)]  # type: ignore[valid-type]
            supported_by = ", ".join(option_to_items[opt_name])
            # Detail exact per-item allowed values in the field description so the LLM sees the dependency
            per_item_desc = "; ".join(
                f"{it.id}: {it.options[opt_name].choices} (default={it.options[opt_name].default})"
                for it in self.items
                if opt_name in it.options
            )
            opt_fields[opt_name] = (
                ChoiceLit | None,
                Field(
                    default=None,
                    description=f"Only valid for [{supported_by}]. Allowed values per item -> {per_item_desc}",
                ),
            )

        DynamicOptionsModel = create_model(
            "MenuItemOptionsInput",
            __config__=ConfigDict(extra="forbid"),
            **opt_fields,
        )

        # Collect all valid extra IDs across the menu with per-item mapping in description
        all_extras: list[str] = []
        extra_per_item_desc: list[str] = []
        for item in self.items:
            if item.valid_extra_ids:
                extra_per_item_desc.append(f"{item.id}: {item.valid_extra_ids}")
                for e_id in item.valid_extra_ids:
                    if e_id not in all_extras:
                        all_extras.append(e_id)

        ExtraIdLiteral = Literal[tuple(all_extras)]  # type: ignore[valid-type]
        catalog_ref = self

        class AddItemToolInput(BaseModel):
            model_config = ConfigDict(extra="forbid")

            item_id: ItemIdLiteral = Field(  # type: ignore[valid-type]
                ...,
                description=f"The menu item ID to add. Must be one of: {list(valid_item_ids)}",
            )
            quantity: int = Field(
                default=1,
                ge=1,
                description="Quantity of this item to add (1 or more).",
            )
            options: DynamicOptionsModel | None = Field(  # type: ignore[valid-type]
                default=None,
                description="Optional item customizations (only set fields supported by the chosen item_id).",
            )
            extras: list[ExtraIdLiteral] = Field(  # type: ignore[valid-type]
                default_factory=list,
                description=f"Optional extra add-ons. Valid extras by item -> {'; '.join(extra_per_item_desc)}",
            )

            def to_validated_item(self) -> ValidatedOrderItem:
                raw_opts = self.options.model_dump(exclude_none=True) if self.options else {}
                return catalog_ref.validate_and_resolve(
                    item_id=str(self.item_id),
                    quantity=self.quantity,
                    options=raw_opts,
                    extras=list(self.extras),
                )

        return AddItemToolInput

    # -----------------------------------------------------------------
    # Helpful Cross-Item Error Hints & Prompt Formatting
    # -----------------------------------------------------------------
    def _build_cross_item_option_hint(self, unknown_opts: set[str]) -> str:
        hints: list[str] = []
        for opt in sorted(unknown_opts):
            owners = [it.id for it in self.items if opt in it.options]
            if owners:
                hints.append(f" Note: option '{opt}' is only available on {owners}.")
        return "".join(hints)

    def _build_cross_item_extra_hint(self, invalid_extras: list[str]) -> str:
        hints: list[str] = []
        for ext in invalid_extras:
            owners = [it.id for it in self.items if ext in it.valid_extra_ids]
            if owners:
                hints.append(f" Note: extra '{ext}' is only available on {owners}.")
        return "".join(hints)

    def format_for_prompt(self) -> str:
        """Renders a compact, unambiguous reference sheet of the menu for the Gemini system prompt."""
        lines: list[str] = ["MENU CATALOG (All prices in USD):"]
        for item in self.items:
            lines.append(f"- {item.name} (item_id: '{item.id}', base_price: ${item.base_price:.2f})")
            for opt_name, opt_def in item.options.items():
                mods = ", ".join(
                    f"{c} ({opt_def.get_modifier(c):+.2f})" if opt_def.get_modifier(c) != 0 else c
                    for c in opt_def.choices
                )
                req_tag = (
                    "REQUIRED - MUST ASK USER (no default)"
                    if (opt_def.required and opt_def.default is None)
                    else f"default='{opt_def.default}'"
                )
                lines.append(f"    • option '{opt_name}' [{req_tag}]: [{mods}]")
            if item.extras and item.extras.choices:
                exts = ", ".join(f"{e.id} (+${e.price:.2f})" for e in item.extras.choices)
                lines.append(f"    • extras: [{exts}]")
            else:
                lines.append("    • extras: none")
        return "\n".join(lines)