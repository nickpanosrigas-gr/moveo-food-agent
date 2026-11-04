# test_conversations_hard.py
"""
Hard conversation tests for FoodOrderAgent.
Covers complex stress tests, adversarial traps, and long-horizon dialogues (up to 15 turns).

Validation Rules:
- If final output is wrong -> Test FAILS (x)
- If final order is correct but intermediate turn tool calls failed -> PARTIAL PASS (~)
- If final order and all intermediate turn tool calls are correct -> FULL PASS (✓)
"""
from __future__ import annotations

import copy
import json
import re
import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))

from agent import FoodOrderAgent
from config.settings import settings


# =====================================================================
# Data Structures
# =====================================================================

@dataclass
class ExpectedToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Turn:
    user_message: str
    expected_tools: list[ExpectedToolCall] = field(default_factory=list)
    description: str = ""


@dataclass
class Scenario:
    id: int
    name: str
    turns: list[Turn]
    expected_final_order: dict[str, Any]


# =====================================================================
# Comparison Helpers
# =====================================================================

DEFAULT_OPTIONS_MAP = {
    "classic_burger": {"size": "regular", "patty": "beef"},
    "spicy_burger": {"size": "regular", "spice_level": "medium"},
    "margherita": {"size": "medium", "crust": "regular"},
    "fries": {"size": "medium"},
    "onion_rings": {"size": "medium"},
    "soda": {"size": "medium", "flavor": "cola"},
    "milkshake": {"size": "regular"},
}


def _normalize_extras(extras: list[str] | None) -> set[str]:
    return {str(e).strip().lower() for e in (extras or [])}


def _normalize_options(options: dict[str, Any] | None) -> dict[str, str]:
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in (options or {}).items() if v is not None}


def _item_matches(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    if expected.get("item_id") != actual.get("item_id"):
        return False
    if expected.get("quantity", 1) != actual.get("quantity", 1):
        return False

    item_id = expected.get("item_id", "")
    default_opts = DEFAULT_OPTIONS_MAP.get(item_id, {})

    exp_opts = _normalize_options(expected.get("options"))
    act_opts = _normalize_options(actual.get("options"))

    for k, v in exp_opts.items():
        act_val = act_opts.get(k, default_opts.get(k))
        if act_val != v:
            return False

    if _normalize_extras(expected.get("extras")) != _normalize_extras(actual.get("extras")):
        return False

    return True


def compare_orders(expected: dict[str, Any], actual: dict[str, Any] | None) -> tuple[bool, str]:
    if actual is None:
        return False, "Order was never submitted (no submit_order call was made)."

    exp_items = expected.get("items", [])
    act_items = actual.get("items", [])

    if len(exp_items) != len(act_items):
        return False, f"Expected {len(exp_items)} item(s), but got {len(act_items)} item(s)."

    matched_actual_indices = set()
    for exp_item in exp_items:
        found = False
        for idx, act_item in enumerate(act_items):
            if idx in matched_actual_indices:
                continue
            if _item_matches(exp_item, act_item):
                matched_actual_indices.add(idx)
                found = True
                break
        if not found:
            return False, f"Could not find matching actual item for expected: {json.dumps(exp_item, ensure_ascii=False)}"

    exp_instructions = (expected.get("special_instructions") or "").strip().lower()
    act_instructions = (actual.get("special_instructions") or "").strip().lower()
    if exp_instructions:
        if not act_instructions:
            return (
                False,
                f"Special instructions missing: expected keywords from '{exp_instructions}', but actual was empty.",
            )

        stop_words = {
            "and", "the", "for", "with", "please", "make", "sure",
            "that", "all", "into", "it", "is", "to", "in", "on", "a", "an",
        }
        exp_words = [w for w in re.findall(r"[a-zA-Z0-9]+", exp_instructions) if w not in stop_words]

        if exp_words:
            def _stem_match(kw: str, text: str) -> bool:
                if kw in text:
                    return True
                for suffix in ("ed", "ing", "es", "s"):
                    if kw.endswith(suffix) and len(kw) - len(suffix) >= 3:
                        if kw[:-len(suffix)] in text:
                            return True
                return False

            matched = [w for w in exp_words if _stem_match(w, act_instructions)]
            match_ratio = len(matched) / len(exp_words)
            if match_ratio < 0.6:
                return (
                    False,
                    f"Special instructions keyword mismatch: matched {len(matched)}/{len(exp_words)} "
                    f"keywords ({matched}). Expected keywords from '{exp_instructions}', got '{act_instructions}'",
                )

    return True, "Match"


def compare_turn_tools(expected_tools: list[ExpectedToolCall], actual_tools: list[dict[str, Any]]) -> tuple[bool, str]:
    real_actual = [
        t for t in actual_tools
        if t.get("name") not in ("submit_order_blocked_for_confirmation",)
    ]

    if not expected_tools:
        if real_actual:
            tool_names = [t.get("name") for t in real_actual]
            return False, f"Expected no cart/submit tool calls, but agent called: {tool_names}"
        return True, "Match"

    if len(expected_tools) != len(real_actual):
        return False, f"Expected {len(expected_tools)} tool call(s) ({[t.name for t in expected_tools]}), got {len(real_actual)} ({[t.get('name') for t in real_actual]})"

    matched_indices = set()
    for exp in expected_tools:
        found = False
        for idx, act in enumerate(real_actual):
            if idx in matched_indices:
                continue
            if act.get("name") != exp.name:
                continue

            act_args = act.get("arguments", {})
            item_id = str(act_args.get("item_id") or exp.arguments.get("item_id") or "")
            default_opts = DEFAULT_OPTIONS_MAP.get(item_id, {})
            args_match = True

            for arg_k, arg_v in exp.arguments.items():
                if arg_k == "extras":
                    if _normalize_extras(arg_v) != _normalize_extras(act_args.get("extras")):
                        args_match = False
                        break
                elif arg_k == "options":
                    exp_o = _normalize_options(arg_v)
                    act_o = _normalize_options(act_args.get("options"))
                    for ok, ov in exp_o.items():
                        val = act_o.get(ok, default_opts.get(ok))
                        if val != ov:
                            args_match = False
                            break
                elif arg_k in ("add_extras", "remove_extras"):
                    if _normalize_extras(arg_v) != _normalize_extras(act_args.get(arg_k)):
                        args_match = False
                        break
                elif arg_k == "item_id":
                    if str(act_args.get("item_id")) != str(arg_v):
                        args_match = False
                        break
                elif arg_k in ("quantity", "target_quantity", "new_quantity"):
                    if act_args.get(arg_k) is not None and int(act_args.get(arg_k)) != int(arg_v):
                        args_match = False
                        break

            if args_match:
                matched_indices.add(idx)
                found = True
                break

        if not found:
            return False, f"Expected tool call {exp.name}({exp.arguments}) was not matched in actual calls: {real_actual}"

    return True, "Match"


# =====================================================================
# Test Runner Logic
# =====================================================================

def run_scenario(scenario: Scenario, mock_mcp: bool = True) -> dict[str, Any]:
    agent = FoodOrderAgent()
    submitted_payload: dict[str, Any] | None = None

    if mock_mcp:
        def _mock_submit_order(items: list[dict[str, Any]], special_instructions: str | None = None, force_failure: str | None = None):
            nonlocal submitted_payload
            submitted_payload = {
                "items": copy.deepcopy(items),
                "special_instructions": special_instructions,
            }
            if force_failure == "kitchen_busy":
                return {"success": False, "error": "Kitchen is currently busy. Please try again."}
            total = float(agent.cart.total_price)
            if total > 50.0:
                return {"success": False, "error": f"Order total (${total:.2f}) exceeds $50 limit."}
            return {
                "success": True,
                "order_id": "ORD-TEST-99999",
                "total": total,
                "estimated_time": "15-20 minutes",
            }
        agent.mcp_client.submit_order = _mock_submit_order  # type: ignore[assignment]

    wrong_turns = []
    total_turns = len(scenario.turns)

    for turn_idx, turn in enumerate(scenario.turns, start=1):
        active_force_failure = "kitchen_busy" if "kitchen_busy" in turn.description.lower() else None
        response = agent.send(turn.user_message, force_failure=active_force_failure)

        actual_internal = []
        if agent.last_turn_log:
            actual_internal = agent.last_turn_log.get("internal_cart_actions") or agent.last_turn_log.get("internal_tool_calls") or []

        actual_exposed = response.get("tool_calls", [])
        actual_turn_tools = actual_internal + actual_exposed

        turn_passed, reason = compare_turn_tools(turn.expected_tools, actual_turn_tools)
        if not turn_passed:
            wrong_turns.append({
                "turn": turn_idx,
                "user_message": turn.user_message,
                "expected": [{"name": t.name, "arguments": t.arguments} for t in turn.expected_tools],
                "actual": [{"name": t.get("name"), "arguments": t.get("arguments")} for t in actual_turn_tools],
                "reason": reason,
                "agent_response": response.get("message", ""),
            })

    actual_final = submitted_payload or (agent.last_mcp_payload.get("arguments") if agent.last_mcp_payload else None)
    final_passed, final_reason = compare_orders(scenario.expected_final_order, actual_final)

    if not final_passed:
        status = "FAIL"
    elif wrong_turns:
        status = "PARTIAL"
    else:
        status = "FULL"

    return {
        "scenario_id": scenario.id,
        "name": scenario.name,
        "status": status,
        "total_turns": total_turns,
        "wrong_turns": wrong_turns,
        "final_passed": final_passed,
        "final_reason": final_reason,
        "expected_final_order": scenario.expected_final_order,
        "actual_final_order": actual_final,
    }


def print_banner(suite_name: str, agent_model: str) -> None:
    print("====================================================")
    print("  Moveo AI Food Ordering Agent (Test Suite)")
    print(f"  Backend: LLM={settings.LLM_PROVIDER} ({agent_model})")
    print(f"           Speech={settings.SPEECH_PROVIDER}")
    print(f"  Suite:   {suite_name}")
    print("====================================================")


def run_all_tests(scenarios: list[Scenario], suite_name: str = "Hard Conversation Tests") -> None:
    temp_agent = FoodOrderAgent()
    print_banner(suite_name=suite_name, agent_model=temp_agent.llm.model)

    print(f"Loading and warming up model ({temp_agent.llm.model})... ", end="", flush=True)
    temp_agent.warmup_sync()
    print("Done!\n")

    unloaded = False

    def _cleanup_resources(*_args: Any) -> None:
        nonlocal unloaded
        if not unloaded:
            unloaded = True
            print("\nUnloading model...", end=" ", flush=True)
            temp_agent.unload_sync()
            print("Done!")

    try:
        signal.signal(signal.SIGTERM, lambda *_a: (_cleanup_resources(), sys.exit(0)))
    except (ValueError, AttributeError):
        pass

    total_tests = len(scenarios)
    full_passes = 0
    partial_passes = 0
    failures = 0

    total_turns_count = 0
    wrong_turns_count = 0

    try:
        for scenario in scenarios:
            res = run_scenario(scenario)
            total_turns_count += res["total_turns"]
            wrong_turns_count += len(res["wrong_turns"])

            status = res["status"]
            if status == "FULL":
                full_passes += 1
                print(f"✓ Scenario {res['scenario_id']}: {res['name']}")
            elif status == "PARTIAL":
                partial_passes += 1
                print(f"~ Scenario {res['scenario_id']}: {res['name']}")
                print("    [Wrong Turns]:")
                for wt in res["wrong_turns"]:
                    print(f"      - Turn {wt['turn']} (User: '{wt['user_message']}'): {wt['reason']}")
            else:  # FAIL
                failures += 1
                print(f"x Scenario {res['scenario_id']}: {res['name']}")
                if res["wrong_turns"]:
                    print("    [Wrong Turns]:")
                    for wt in res["wrong_turns"]:
                        print(f"      - Turn {wt['turn']} (User: '{wt['user_message']}'): {wt['reason']}")
                print("    [Final Order Mismatch]:")
                print(f"      Reason: {res['final_reason']}")
                print(f"      Expected Order: {json.dumps(res['expected_final_order'], indent=8)}")
                print(f"      Actual Order:   {json.dumps(res['actual_final_order'], indent=8)}")
            print()

        print("=================================================================")
        print("                     TEST SUMMARY REPORT")
        print("=================================================================")
        print(f"Total Tests Run:     {total_tests}")
        print(f"Full Pass (✓):       {full_passes}")
        print(f"Partial Pass (~):    {partial_passes}")
        print(f"Failed (x):          {failures}")
        print("-----------------------------------------------------------------")
        print(f"Total Turns Run:     {total_turns_count}")
        print(f"Total Correct Turns: {total_turns_count - wrong_turns_count}")
        print(f"Total Wrong Turns:   {wrong_turns_count}")
        print("=================================================================")
    finally:
        _cleanup_resources()


# =====================================================================
# Scenarios (16 to 25)
# =====================================================================

HARD_SCENARIOS: list[Scenario] = [
    # Scenario 16: 7 Turns
    Scenario(
        id=16,
        name="The Topping Migration & Stack-Splitting Trap",
        turns=[
            Turn(
                user_message="Add two medium margherita pizzas with thin crust.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "margherita", "quantity": 2, "options": {"size": "medium", "crust": "thin"}, "extras": []})
                ],
            ),
            Turn(
                user_message="And a large french fries with mushrooms.",
                expected_tools=[],
            ),
            Turn(
                user_message="Fine, no extras on the fries, but put the mushrooms on one of the pizzas instead.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "large"}}),
                    ExpectedToolCall("edit_item", {"target_quantity": 1, "add_extras": ["mushrooms"]}),
                ],
            ),
            Turn(
                user_message="Actually, add pepperoni to the plain pizza too.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"add_extras": ["pepperoni"]}),
                ],
            ),
            Turn(
                user_message="Take the fries off the order.",
                expected_tools=[
                    ExpectedToolCall("remove_item", {}),
                ],
            ),
            Turn(
                user_message="That's it.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes, submit.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {}),
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "margherita",
                    "quantity": 1,
                    "options": {"size": "medium", "crust": "thin"},
                    "extras": ["mushrooms"],
                },
                {
                    "item_id": "margherita",
                    "quantity": 1,
                    "options": {"size": "medium", "crust": "thin"},
                    "extras": ["pepperoni"],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 17: 8 Turns
    Scenario(
        id=17,
        name="Dietary Allergen Navigation & Special Instructions Lifecycle",
        turns=[
            Turn(
                user_message="I want a spicy burger with a gluten-free bun and an extra patty.",
                expected_tools=[],
            ),
            Turn(
                user_message="Okay, make it a classic burger with veggie patty and extra patty, and note that I'm allergic to sesame.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "veggie"}, "extras": ["extra_patty"]}),
                ],
            ),
            Turn(
                user_message="Add a large milkshake.",
                expected_tools=[],
            ),
            Turn(
                user_message="Oreo milkshake with whipped cream, and make sure everything is sealed in separate bags.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "milkshake", "quantity": 1, "options": {"size": "large", "flavor": "oreo"}, "extras": ["whipped_cream"]}),
                ],
            ),
            Turn(
                user_message="Actually remove the whipped cream from the shake.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"remove_extras": ["whipped_cream"]}),
                ],
            ),
            Turn(
                user_message="Forget the sesame allergy, my friend left, but keep the separate bags instruction.",
                expected_tools=[
                    ExpectedToolCall("set_special_instructions", {"special_instructions": "Sealed in separate bags"}),
                ],
            ),
            Turn(
                user_message="I'm all done.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes, go ahead.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {}),
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "patty": "veggie"},
                    "extras": ["extra_patty"],
                },
                {
                    "item_id": "milkshake",
                    "quantity": 1,
                    "options": {"size": "large", "flavor": "oreo"},
                    "extras": [],
                },
            ],
            "special_instructions": "Sealed in separate bags",
        },
    ),

    # Scenario 18: 9 Turns
    Scenario(
        id=18,
        name="Pronoun Ambiguity & Oscillating Upgrade Trap",
        turns=[
            Turn(
                user_message="I want a classic burger and a spicy burger.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1}),
                    ExpectedToolCall("add_item", {"item_id": "spicy_burger", "quantity": 1}),
                ],
            ),
            Turn(
                user_message="Make it large.",
                expected_tools=[],
            ),
            Turn(
                user_message="Make both of them large.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"options": {"size": "large"}}),
                    ExpectedToolCall("edit_item", {"options": {"size": "large"}}),
                ],
            ),
            Turn(
                user_message="Add cheese to it.",
                expected_tools=[],
            ),
            Turn(
                user_message="Add cheese to the spicy one, and add avocado to the other.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"add_extras": ["cheese"]}),
                    ExpectedToolCall("edit_item", {"add_extras": ["avocado"]}),
                ],
            ),
            Turn(
                user_message="Wait, switch the avocado over to the spicy burger instead!",
                expected_tools=[],
            ),
            Turn(
                user_message="Keep avocado on the classic burger, but add bacon to the spicy burger.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"add_extras": ["bacon"]}),
                ],
            ),
            Turn(
                user_message="That's everything.",
                expected_tools=[],
            ),
            Turn(
                user_message="Submit.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {}),
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "large", "patty": "beef"},
                    "extras": ["avocado"],
                },
                {
                    "item_id": "spicy_burger",
                    "quantity": 1,
                    "options": {"size": "large", "spice_level": "medium"},
                    "extras": ["cheese", "bacon"],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 19: 12 Turns
    Scenario(
        id=19,
        name="The Budget Oscillating Gauntlet ($50 Cap Stress-Test)",
        turns=[
            Turn(
                user_message="I want 2 large margherita pizzas with extra cheese.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "margherita", "quantity": 2, "options": {"size": "large"}, "extras": ["extra_cheese"]})
                ],
            ),
            Turn(
                user_message="Add 2 large vanilla milkshakes with whipped cream.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "milkshake", "quantity": 2, "options": {"size": "large", "flavor": "vanilla"}, "extras": ["whipped_cream"]})
                ],
            ),
            Turn(
                user_message="Submit right now, I don't care about the limit.",
                expected_tools=[],
            ),
            Turn(
                user_message="Fine, make the milkshakes regular size instead of large.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"options": {"size": "regular"}})
                ],
            ),
            Turn(
                user_message="Add a small french fries.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "small"}})
                ],
            ),
            Turn(
                user_message="Ugh, remove the fries.",
                expected_tools=[
                    ExpectedToolCall("remove_item", {})
                ],
            ),
            Turn(
                user_message="Can I add ranch dip?",
                expected_tools=[],
            ),
            Turn(
                user_message="Okay, change one of the pizzas to medium instead of large.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"target_quantity": 1, "options": {"size": "medium"}})
                ],
            ),
            Turn(
                user_message="Now add a large diet cola.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "large", "flavor": "diet_cola"}})
                ],
            ),
            Turn(
                user_message="That is finally it.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes, and add cherry on top to both shakes.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"add_extras": ["cherry_on_top"]})
                ],
            ),
            Turn(
                user_message="Yes, confirm and submit.",
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
                    "options": {"size": "large", "crust": "regular"},
                    "extras": ["extra_cheese"],
                },
                {
                    "item_id": "margherita",
                    "quantity": 1,
                    "options": {"size": "medium", "crust": "regular"},
                    "extras": ["extra_cheese"],
                },
                {
                    "item_id": "milkshake",
                    "quantity": 2,
                    "options": {"size": "regular", "flavor": "vanilla"},
                    "extras": ["whipped_cream", "cherry_on_top"],
                },
                {
                    "item_id": "soda",
                    "quantity": 1,
                    "options": {"size": "large", "flavor": "diet_cola"},
                    "extras": [],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 20: 15 Turns
    Scenario(
        id=20,
        name="The Ultimate 15-Turn Chaos & State-Reconciliation Marathon",
        turns=[
            Turn(
                user_message="I want a burger and a pizza.",
                expected_tools=[],
            ),
            Turn(
                user_message="Classic burger with cheese, medium margherita with olives, and a milkshake.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "beef"}, "extras": ["cheese"]}),
                    ExpectedToolCall("add_item", {"item_id": "margherita", "quantity": 1, "options": {"size": "medium", "crust": "regular"}, "extras": ["olives"]}),
                ],
            ),
            Turn(
                user_message="Make it pistachio flavor.",
                expected_tools=[],
            ),
            Turn(
                user_message="Strawberry milkshake with cherry on top.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "milkshake", "quantity": 1, "options": {"size": "regular", "flavor": "strawberry"}, "extras": ["cherry_on_top"]}),
                ],
            ),
            Turn(
                user_message="Add another classic burger with cheese, exactly identical.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "beef"}, "extras": ["cheese"]}),
                ],
            ),
            Turn(
                user_message="On one of the cheese burgers, add bacon.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"target_quantity": 1, "add_extras": ["bacon"]}),
                ],
            ),
            Turn(
                user_message="Actually, take the bacon off that burger, it was a mistake.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"remove_extras": ["bacon"]}),
                ],
            ),
            Turn(
                user_message="Add ranch dip to the pizza.",
                expected_tools=[],
            ),
            Turn(
                user_message="Never mind then. Just make sure the pizza crust is well-done and sliced into 8 pieces.",
                expected_tools=[
                    ExpectedToolCall("set_special_instructions", {"special_instructions": "Pizza crust well-done and sliced into 8 pieces"}),
                ],
            ),
            Turn(
                user_message="Add large fries with cheese sauce and a small Sprite.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "large"}, "extras": ["cheese_sauce"]}),
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "small", "flavor": "lemon_lime"}}),
                ],
            ),
            Turn(
                user_message="Remove the strawberry milkshake, we have ice cream at home.",
                expected_tools=[
                    ExpectedToolCall("remove_item", {}),
                ],
            ),
            Turn(
                user_message="Replace the fries with large onion rings with spicy mayo.",
                expected_tools=[
                    ExpectedToolCall("remove_item", {}),
                    ExpectedToolCall("add_item", {"item_id": "onion_rings", "quantity": 1, "options": {"size": "large"}, "extras": ["spicy_mayo"]}),
                ],
            ),
            Turn(
                user_message="How much is my order right now and what's in it?",
                expected_tools=[],
            ),
            Turn(
                user_message="Looks good, submit it.",
                expected_tools=[],
            ),
            Turn(
                user_message="Yes, place the order.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {}),
                ],
            ),
        ],
        expected_final_order={
            "items": [
                {
                    "item_id": "classic_burger",
                    "quantity": 2,
                    "options": {"size": "regular", "patty": "beef"},
                    "extras": ["cheese"],
                },
                {
                    "item_id": "margherita",
                    "quantity": 1,
                    "options": {"size": "medium", "crust": "regular"},
                    "extras": ["olives"],
                },
                {
                    "item_id": "onion_rings",
                    "quantity": 1,
                    "options": {"size": "large"},
                    "extras": ["spicy_mayo"],
                },
                {
                    "item_id": "soda",
                    "quantity": 1,
                    "options": {"size": "small", "flavor": "lemon_lime"},
                    "extras": [],
                },
            ],
            "special_instructions": "Pizza crust well-done and sliced into 8 pieces",
        },
    ),

    # Scenario 21: 5 Turns
    Scenario(
        id=21,
        name="Auto-Merge Signature Collapsing Under Delta Swaps",
        turns=[
            Turn(
                user_message="I want one regular classic burger with cheese, and one regular classic burger with bacon.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular"}, "extras": ["cheese"]}),
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular"}, "extras": ["bacon"]}),
                ],
            ),
            Turn(
                user_message="On the burger with bacon, remove the bacon and add cheese instead.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"add_extras": ["cheese"], "remove_extras": ["bacon"]}),
                ],
            ),
            Turn(
                user_message="Add a regular cola.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "medium", "flavor": "cola"}}),
                ],
            ),
            Turn(
                user_message="I'm done.",
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
                    "quantity": 2,
                    "options": {"size": "regular", "patty": "beef"},
                    "extras": ["cheese"],
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

    # Scenario 22: 5 Turns
    Scenario(
        id=22,
        name="Negative & Zero Quantity Schema Attack",
        turns=[
            Turn(
                user_message="Add 0 classic burgers and -2 sodas.",
                expected_tools=[],
            ),
            Turn(
                user_message="Fine, give me 1 classic burger with cheese.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular"}, "extras": ["cheese"]}),
                ],
            ),
            Turn(
                user_message="Now remove 0 of that burger.",
                expected_tools=[],
            ),
            Turn(
                user_message="Keep it, that's my final order.",
                expected_tools=[],
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
                    "item_id": "classic_burger",
                    "quantity": 1,
                    "options": {"size": "regular", "patty": "beef"},
                    "extras": ["cheese"],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 23: 5 Turns
    Scenario(
        id=23,
        name="Mid-Session Cart Clearing & Fresh Re-Ordering",
        turns=[
            Turn(
                user_message="Give me two large margherita pizzas and three chocolate milkshakes.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "margherita", "quantity": 2, "options": {"size": "large"}}),
                    ExpectedToolCall("add_item", {"item_id": "milkshake", "quantity": 3, "options": {"size": "regular", "flavor": "chocolate"}}),
                ],
            ),
            Turn(
                user_message="Actually, cancel everything and clear my cart, my friends bailed.",
                expected_tools=[
                    ExpectedToolCall("clear_cart", {}),
                ],
            ),
            Turn(
                user_message="Just give me a medium fries with truffle oil.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "medium"}, "extras": ["truffle_oil"]}),
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
                    "item_id": "fries",
                    "quantity": 1,
                    "options": {"size": "medium"},
                    "extras": ["truffle_oil"],
                },
            ],
            "special_instructions": None,
        },
    ),

    # Scenario 24: 6 Turns
    Scenario(
        id=24,
        name="Kitchen Busy Transient Error Followed by Immediate User Modification",
        turns=[
            Turn(
                user_message="I want a small french fries.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "fries", "quantity": 1, "options": {"size": "small"}}),
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
                description="force kitchen_busy error",
            ),
            Turn(
                user_message="Wait, while you retry, also add a regular cola.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "medium", "flavor": "cola"}}),
                ],
            ),
            Turn(
                user_message="Yes, submit both now.",
                expected_tools=[
                    ExpectedToolCall("submit_order", {})
                ],
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

    # Scenario 25: 5 Turns
    Scenario(
        id=25,
        name="The Multi-Item Ambiguous Cascade",
        turns=[
            Turn(
                user_message="I want a burger and a soda.",
                expected_tools=[],
            ),
            Turn(
                user_message="Classic burger with cheddar, and an orange soda.",
                expected_tools=[
                    ExpectedToolCall("add_item", {"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "beef"}, "extras": ["cheese"]}),
                    ExpectedToolCall("add_item", {"item_id": "soda", "quantity": 1, "options": {"size": "medium", "flavor": "orange"}}),
                ],
            ),
            Turn(
                user_message="Make the soda large and add bacon to the burger.",
                expected_tools=[
                    ExpectedToolCall("edit_item", {"options": {"size": "large"}}),
                    ExpectedToolCall("edit_item", {"add_extras": ["bacon"]}),
                ],
            ),
            Turn(
                user_message="That's all.",
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
                    "extras": ["cheese", "bacon"],
                },
                {
                    "item_id": "soda",
                    "quantity": 1,
                    "options": {"size": "large", "flavor": "orange"},
                    "extras": [],
                },
            ],
            "special_instructions": None,
        },
    ),
]


if __name__ == "__main__":
    run_all_tests(HARD_SCENARIOS, suite_name="Hard Conversation Tests")