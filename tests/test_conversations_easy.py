# test_conversations_easy.py
"""
Easy conversation tests for FoodOrderAgent.
Covers exact menu phrasing and standard flow (Scenarios 1 to 5).

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

EASY_SCENARIOS: list[Scenario] = [
    # Scenario 1: Single Customized Classic Burger Order
    Scenario(
        id=1,
        name="Single Customized Classic Burger Order",
        turns=[
            Turn(
                user_message="I'd like a large classic burger with cheese and bacon.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "large"}, "extras": ["cheese", "bacon"]})
                ],
            ),
            Turn(
                user_message="That's it.",
                expected_tools=[],  # Order summary required
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
                    "extras": ["cheese", "bacon"],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 2: Classic Multi-Category Combo
    Scenario(
        id=2,
        name="Classic Multi-Category Combo (Burger, Fries, Drink)",
        turns=[
            Turn(
                user_message="I want a spicy jalapeño burger without any extras.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "spicy_burger", "quantity": 1, "options": {"size": "regular", "spice_level": "medium"}, "extras": []})
                ],
            ),
            Turn(
                user_message="Add a small french fries.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "small"}})
                ],
            ),
            Turn(
                user_message="And a regular cola.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "medium", "flavor": "cola"}})
                ],
            ),
            Turn(
                user_message="Checkout.",
                expected_tools=[],  # Summary
            ),
            Turn(
                user_message="Confirm.",
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
                    "extras": [],
                },
                {
                    "item_id": "fries",
                    "quantity": 1,
                    "options": {"size": "small"},
                    "extras": [],
                },
                {
                    "item_id": "soda",
                    "quantity": 1,
                    "options": {"size": "medium", "flavor": "cola"},
                    "extras": [],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 3: Customized Margherita Pizza Order
    Scenario(
        id=3,
        name="Customized Margherita Pizza Order",
        turns=[
            Turn(
                user_message="Can I get a large margherita pizza with thin crust, mushrooms, and pepperoni?",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "margherita", "quantity": 1, "options": {"size": "large", "crust": "thin"}, "extras": ["mushrooms", "pepperoni"]})
                ],
            ),
            Turn(
                user_message="Nothing else, I am ready.",
                expected_tools=[],  # Summary
            ),
            Turn(
                user_message="Submit order.",
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
                    "options": {"size": "large", "crust": "thin"},
                    "extras": ["mushrooms", "pepperoni"],
                }
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 4: Sides and Beverage Order
    Scenario(
        id=4,
        name="Sides and Beverage Order",
        turns=[
            Turn(
                user_message="I would like medium onion rings with ranch dip.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "onion_rings", "quantity": 1, "options": {"size": "medium"}, "extras": ["ranch_dip"]})
                ],
            ),
            Turn(
                user_message="Also add a large soft drink with lemon-lime flavor.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "large", "flavor": "lemon_lime"}})
                ],
            ),
            Turn(
                user_message="I'm all set.",
                expected_tools=[],  # Summary
            ),
            Turn(
                user_message="Yes, please.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "onion_rings",
                    "quantity": 1,
                    "options": {"size": "medium"},
                    "extras": ["ranch_dip"],
                },
                {
                    "item_id": "soda",
                    "quantity": 1,
                    "options": {"size": "large", "flavor": "lemon_lime"},
                    "extras": [],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 5: Multi-Item Single-Turn Order
    Scenario(
        id=5,
        name="Multi-Item Single-Turn Order",
        turns=[
            Turn(
                user_message="I want a classic burger with chicken patty and an oreo milkshake.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "chicken"}, "extras": []}),
                    ExpectedToolCall("add_item", {"item_id": "milkshake", "quantity": 1, "options": {"size": "regular", "flavor": "oreo"}, "extras": []}),
                ],
            ),
            Turn(
                user_message="No, please submit.",
                expected_tools=[],  # Summary
            ),
            Turn(
                user_message="Yes, go ahead.",
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
                    "options": {"size": "regular", "patty": "chicken"},
                    "extras": [],
                },
                {
                    "item_id": "milkshake",
                    "quantity": 1,
                    "options": {"size": "regular", "flavor": "oreo"},
                    "extras": [],
                },
            ],
            "special_instructions": None,
        },
    ),
]

if __name__ == "__main__":
    run_all_tests(EASY_SCENARIOS, suite_name="Easy Conversation Tests")
