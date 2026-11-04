# test_conversations_medium.py
"""
Medium conversation tests for FoodOrderAgent.
Covers honest user mistakes, omissions, and modifications (Scenarios 6 to 15).

Validation Rules:
- If final output is wrong -> Test FAILS (x)
- If final order is correct but intermediate turn tool calls failed -> PARTIAL PASS (~)
- If final order and all intermediate turn tool calls are correct -> FULL PASS (✓)
"""
from __future__ import annotations

from test_conversations_hard import (
    ExpectedToolCall,
    Scenario,
    Turn,
    run_all_tests,
)

MEDIUM_SCENARIOS: list[Scenario] = [
    # Scenario 6: The Milkshake Trap (Missing Required Flavor)
    Scenario(
        id=6,
        name="The Milkshake Trap (Missing Required Flavor)",
        turns=[
            Turn(
                user_message="Can I get a classic burger and a milkshake?",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "beef"}})
                ],  # Milkshake must be rejected until flavor specified
            ),
            Turn(
                user_message="Chocolate please, and make it large.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "milkshake", "quantity": 1, "options": {"size": "large", "flavor": "chocolate"}})
                ],
            ),
            Turn(
                user_message="That's everything.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "patty": "beef"},
                    "extras": [],
                },
                {
                    "item_id": "milkshake",
                    "quantity": 1,
                    "options": {"size": "large", "flavor": "chocolate"},
                    "extras": [],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 7: Full Line Item Removal
    Scenario(
        id=7,
        name="Full Line Item Removal",
        turns=[
            Turn(
                user_message="Give me a medium margherita pizza and large onion rings with spicy mayo.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "margherita", "quantity": 1, "options": {"size": "medium"}}),
                    ExpectedToolCall("add_item", {"item_id": "onion_rings", "quantity": 1, "options": {"size": "large"}, "extras": ["spicy_mayo"]}),
                ],
            ),
            Turn(
                user_message="Actually, remove the onion rings.",
                expected_tools=[
                    ExpectedToolCall("remove_item", {})
                ],
            ),
            Turn(
                user_message="No, that's all.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "margherita",
                    "quantity": 1,
                    "options": {"size": "medium", "crust": "regular"},
                    "extras": [],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 8: Replacing One Item Category With Another
    Scenario(
        id=8,
        name="Replacing One Item Category With Another",
        turns=[
            Turn(
                user_message="I'd like a classic burger with cheese.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular"}, "extras": ["cheese"]})
                ],
            ),
            Turn(
                user_message="Actually change that to a spicy jalapeño burger with bacon.",
                expected_tools=[
                    ExpectedToolCall("remove_item", {}),
                    ExpectedToolCall("add_item", {"item_id": "spicy_burger", "quantity": 1, "options": {"size": "regular", "spice_level": "medium"}, "extras": ["bacon"]}),
                ],
            ),
            Turn(
                user_message="I'm done.",
                expected_tools=[],
            ),
            Turn(
                user_message="Submit.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "spicy_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "spice_level": "medium"},
                    "extras": ["bacon"],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 9: Multi-Item Natural Language Addition
    Scenario(
        id=9,
        name="Multi-Item Natural Language Addition",
        turns=[
            Turn(
                user_message="Give me a regular classic burger, large fries with cheese sauce, and an orange soda.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular"}}),
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "large"}, "extras": ["cheese_sauce"]}),
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "medium", "flavor": "orange"}}),
                ],
            ),
            Turn(
                user_message="That is all.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "patty": "beef"},
                    "extras": [],
                },
                {
                    "item_id": "fries",
                    "quantity": 1,
                    "options": {"size": "large"},
                    "extras": ["cheese_sauce"],
                },
                {
                    "item_id": "soda",
                    "quantity": 1,
                    "options": {"size": "medium", "flavor": "orange"},
                    "extras": [],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 10: Option Modifications on Existing Cart Item
    Scenario(
        id=10,
        name="Option Modifications on Existing Cart Item",
        turns=[
            Turn(
                user_message="I want a small margherita pizza.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "margherita", "quantity": 1, "options": {"size": "small"}})
                ],
            ),
            Turn(
                user_message="Actually, make it large with thick crust.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"options": {"size": "large", "crust": "thick"}})
                ],
            ),
            Turn(
                user_message="That's it.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "margherita",
                    "quantity": 1,
                    "options": {"size": "large", "crust": "thick"},
                    "extras": [],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 11: Extra Topping Delta Removal
    Scenario(
        id=11,
        name="Extra Topping Delta Removal",
        turns=[
            Turn(
                user_message="I want a spicy jalapeño burger with cheese and bacon.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "spicy_burger", "quantity": 1, "options": {"size": "regular", "spice_level": "medium"}, "extras": ["cheese", "bacon"]})
                ],
            ),
            Turn(
                user_message="Actually take off the bacon.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"remove_extras": ["bacon"]})
                ],
            ),
            Turn(
                user_message="Nothing else.",
                expected_tools=[],
            ),
            Turn(
                user_message="Go ahead.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "spicy_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "spice_level": "medium"},
                    "extras": ["cheese"],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 12: Partial Stack Modification (Splitting Quantities)
    Scenario(
        id=12,
        name="Partial Stack Modification (Splitting Quantities)",
        turns=[
            Turn(
                user_message="I want two classic burgers.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 2, "options": {"size": "regular", "patty": "beef"}})
                ],
            ),
            Turn(
                user_message="Put cheese on just one of them.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"target_quantity": 1, "add_extras": ["cheese"]})
                ],
            ),
            Turn(
                user_message="That's all.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes, submit.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "patty": "beef"},
                    "extras": [],
                },
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "patty": "beef"},
                    "extras": ["cheese"],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 13: Decrementing Stacked Quantity
    Scenario(
        id=13,
        name="Decrementing Stacked Quantity",
        turns=[
            Turn(
                user_message="Add three medium soft drinks with diet cola.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 3, "options": {"size": "medium", "flavor": "diet_cola"}})
                ],
            ),
            Turn(
                user_message="Remove one of the sodas.",
                expected_tools=[
                    ExpectedToolCall("remove_item", {"quantity": 1})
                ],
            ),
            Turn(
                user_message="That's it.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "soda",
                    "quantity": 2,
                    "options": {"size": "medium", "flavor": "diet_cola"},
                    "extras": [],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 14: Ambiguous Burger Disambiguation
    Scenario(
        id=14,
        name="Ambiguous Burger Disambiguation",
        turns=[
            Turn(
                user_message="I want a burger.",
                expected_tools=[],  # Disambiguation question required
            ),
            Turn(
                user_message="Classic burger with a veggie patty, please.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "veggie"}})
                ],
            ),
            Turn(
                user_message="No, that's it.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "patty": "veggie"},
                    "extras": [],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 15: Special Instructions Persistence Across Turns
    Scenario(
        id=15,
        name="Special Instructions Persistence Across Turns",
        turns=[
            Turn(
                user_message="I want a large classic burger, no onions please.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "large"}, "special_instructions": "no onions please"})
                ],
            ),
            Turn(
                user_message="Add a small fries.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "small"}})
                ],
            ),
            Turn(
                user_message="Checkout.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "large", "patty": "beef"},
                    "extras": [],
                },
                {
                    "item_id": "fries",
                    "quantity": 1,
                    "options": {"size": "small"},
                    "extras": [],
                },
            ],
            "special_instructions": "no onions please",
        },
    ),
]

if __name__ == "__main__":
    run_all_tests(MEDIUM_SCENARIOS, suite_name="Medium Conversation Tests")
