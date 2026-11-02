# Agentic Food Ordering Backend · NLP Engineer Assignment

A deterministic, low-latency conversational food-ordering agent built in pure Python 3.13+. Designed without heavyweight orchestration frameworks (no LangChain/LangGraph), this backend implements a custom multi-step tool-calling state machine powered by a local **Ollama (`gemma4:12b`, 32k context)** LLM, strict **Pydantic** menu validation, **Decimal** price arithmetic, a **Streamable HTTP Model Context Protocol (MCP)** client, and a **FastAPI** streaming server with automated GPU VRAM lifecycle management.

---

## Table of Contents

1. [System Architecture](#1-system-architecture)
2. [Repository Structure](#2-repository-structure)
3. [Setup & Installation](#3-setup--installation)
4. [Usage & Entry Points](#4-usage--entry-points)
   - [Interactive CLI (`main.py`)](#1-interactive-cli-mode)
   - [Programmatic Python API (`agent.py`)](#2-programmatic-python-api)
   - [FastAPI Streaming Server (`server/app.py`)](#3-fastapi-streaming-server)
   - [Docker Deployment](#4-docker-deployment)
5. [Core Engineering & Design Decisions](#5-core-engineering--design-decisions)
   - [Pure-Python Agentic Loop & Prefix KV-Caching](#1-pure-python-agentic-loop--prefix-kv-caching)
   - [Internal vs. Exposed Tool Calls Contract](#2-internal-vs-exposed-tool-calls-contract)
   - [Two-Layer Menu Validation & Dynamic Schema Generation (`menu.py`)](#3-two-layer-menu-validation--dynamic-schema-generation-menupy)
   - [Line-Based Cart Engine with Atomic "Split & Merge" (`cart.py`)](#4-line-based-cart-engine-with-atomic-split--merge-cartpy)
   - [Streamable HTTP MCP Client & 3-Mode Failure Handling (`client.py`)](#5-streamable-http-mcp-client--3-mode-failure-handling-clientpy)
   - [GPU VRAM Lifecycle & Session Watchdog (`app.py` & `speech.py`)](#6-gpu-vram-lifecycle--session-watchdog-apppy--speechpy)
6. [Handled Edge Cases](#6-handled-edge-cases)
7. [FastAPI NDJSON Streaming Protocol](#7-fastapi-ndjson-streaming-protocol)
8. [Structured Logging & Observability](#8-structured-logging--observability)

---

## 1. System Architecture

```text
                       ┌───────────────────────────────────────────────┐
                       │              Entry Interfaces                 │
                       │  1. CLI (`python main.py`)                    │
                       │  2. Python SDK (`from agent import ...`)      │
                       │  3. FastAPI Server (`server/app.py` :8000)    │
                       └───────────────────────┬───────────────────────┘
                                               │
                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  `FoodOrderAgent` (`src/core/agent_loop.py`)                 │
│                                                                                              │
│  ┌──────────────────────────────┐     Tool Calls      ┌───────────────────────────────────┐  │
│  │   `OllamaClient`             │ ──────────────────► │  Internal Cart Tools (`cart.py`)  │  │
│  │   - Model: `gemma4:12b`      │                     │  - `add_item`                     │  │
│  │   - Context: 32,000 tokens   │ ◄────────────────── │  - `edit_item` (Split & Merge)    │  │
│  │   - Static Prefix KV-Cache   │     Tool Results    │  - `remove_item`                  │  │
│  │   - `think: False`           │                     │  - `set_special_instructions`     │  │
│  └──────────────────────────────┘                     └─────────────────┬─────────────────┘  │
│                 │                                                       │                    │
│                 │ Confirmed `submit_order`                              │ Validates against  │
│                 ▼                                                       ▼                    │
│  ┌──────────────────────────────┐                     ┌───────────────────────────────────┐  │
│  │   `MCPOrderClient`           │                     │  `MenuCatalog` (`menu.py`)        │  │
│  │   - Streamable HTTP          │                     │  - Parses `menu.yaml`             │  │
│  │   - `X-Applicant-Email`      │                     │  - Resolves option defaults       │  │
│  │   - Handles `-32602`,        │                     │  - Enforces required choices      │  │
│  │     business refusals (>$50),│                     │  - Exact `Decimal` price math     │  │
│  │     and `kitchen_busy`       │                     │  - Generates strict JSON schemas  │  │
│  └──────────────┬───────────────┘                     └───────────────────────────────────┘  │
└─────────────────┼────────────────────────────────────────────────────────────────────────────┘
                  │
                  ▼
   [ Moveo Remote MCP Server ]
```

---

## 2. Repository Structure

```text
moveo-food-agent/
├── agent.py                     # Required root export: `from agent import FoodOrderAgent`
├── main.py                      # Required interactive CLI entry point: `python main.py`
├── menu.yaml                    # Canonical restaurant menu schema (burgers, pizzas, sides, drinks)
├── pyproject.toml               # Python 3.13+ project metadata & dependencies (managed via `uv`)
├── Dockerfile                   # Production container definition for the FastAPI backend
├── docker-compose.yml           # Orchestrates backend & web containers with host port bindings
├── .env                         # Environment configuration (ports, URLs, credentials)
├── README.md                    # Backend architecture, setup, and design documentation
│
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized, immutable `.env` loader & path resolver
│
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── agent_loop.py        # `FoodOrderAgent` state machine, tool router & confirmation guardrails
│   │   ├── menu.py              # Pydantic `menu.yaml` parser, validator, Decimal calculator & schema builder
│   │   ├── cart.py              # Stateful `OrderCart` engine with line signatures & Split/Merge logic
│   │   ├── ollama_client.py     # Persistent HTTP Ollama client with KV-cache warmup & VRAM offloading
│   │   └── prompts.py           # Static prefix system prompt & dynamic JSON-Schema tool declarations
│   ├── mcp/
│   │   └── client.py            # Async/Sync MCP Streamable HTTP client with 3-tier error normalization
│   └── utils/
│       └── logger.py            # Unified console formatter & JSONL structured turn logger (`TurnLogger`)
│
├── server/
│   ├── app.py                   # FastAPI NDJSON streaming server & 15-minute session/VRAM watchdog
│   └── speech.py                # Async client for hardware-accelerated STT (Whisper) & TTS (Kokoro)
│
└── logs/
    └── conversation_turns.jsonl # Structured turn-by-turn audit logs generated at runtime
```

*(Note: The `/web` frontend application and `/speech_service` GPU container maintain their own dedicated `README.md` files inside their respective directories.)*

---

## 3. Setup & Installation

### Prerequisites
* **Python:** `3.13+`
* **Package Manager:** [`uv`](https://docs.astral.sh/uv/) (or standard `pip`)
* **LLM Backend:** An active **Ollama** instance serving `gemma4:12b`
* **Speech Service *(Optional — required only for Voice endpoints in `server/app.py`)*:** Running instance of `speech_service`

### 1. Install Dependencies
```bash
# Clone the repository
git clone <your-repo-url>
cd moveo-food-agent

# Install dependencies into an isolated virtual environment using uv
uv sync
```

### 2. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory:

```env
# -------------------------------------------------------------------
# Network & Port Bindings (Docker / Reverse Proxy)
# -------------------------------------------------------------------
WEB_EXTERNAL_PORT=3000
WEB_INTERNAL_PORT=3000
BACKEND_EXTERNAL_PORT=8000
BACKEND_INTERNAL_PORT=8000

# -------------------------------------------------------------------
# LLM Backend (Ollama)
# -------------------------------------------------------------------
OLLAMA_URL=[http://192.168.1.220:11434](http://192.168.1.220:11434)
OLLAMA_MODEL=gemma4:12b
OLLAMA_CTX=32000

# -------------------------------------------------------------------
# Moveo MCP Server & Applicant Identification
# -------------------------------------------------------------------
APPLICANT_EMAIL=your_email_from_cv@example.com
MCP_SERVER_URL=[https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp](https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp)

# -------------------------------------------------------------------
# Speech Service (Faster-Whisper STT + Kokoro-82M TTS)
# -------------------------------------------------------------------
SPEECH_URL=http://localhost:8000
```

> **Important:** Ensure `APPLICANT_EMAIL` matches the email address on your CV. Every request to Moveo's MCP server automatically attaches this value via the `X-Applicant-Email` HTTP header.

---

## 4. Usage & Entry Points

### 1. Interactive CLI Mode
Run the agent directly in your terminal. On startup, `main.py` automatically pre-loads `gemma4:12b` into GPU VRAM and evaluates the static system prompt and tool schemas into Ollama's KV cache. When you exit (`quit`, `exit`, `Ctrl+C`, or `SIGTERM`), it automatically offloads the model from VRAM (`keep_alive=0`).

```bash
uv run main.py
```

### 2. Programmatic Python API
Import `FoodOrderAgent` directly from the root `agent.py` module as required by the specification:

```python
from agent import FoodOrderAgent

agent = FoodOrderAgent()

# Turn 1: Add item (local cart mutation -> no exposed `tool_calls` key)
r1 = agent.send("I want a large burger with cheese")
# -> {"message": "I've added a Large Classic Burger with cheese for $11.50. Anything else?"}

# Turn 2: Modify item (local cart mutation -> no exposed `tool_calls` key)
r2 = agent.send("Actually make it regular")
# -> {"message": "Updated to a Regular Classic Burger with cheese for $9.50. Anything else?"}

# Turn 3: Request checkout (triggers confirmation summary before MCP submission)
r3 = agent.send("Submit")
# -> {"message": "Your order is one Regular Classic Burger with cheese for a total of $9.50. Ready to submit?"}

# Turn 4: Confirm submission (invokes remote MCP `submit_order` tool)
r4 = agent.send("Yes")
# -> {
#      "message": "Order submitted! Order #ORD-12345. Estimated time: 15-20 minutes.",
#      "tool_calls": [
#        {
#          "name": "submit_order",
#          "arguments": {
#            "items": [{"item_id": "classic_burger", "quantity": 1, "options": {"size": "regular", "patty": "beef"}, "extras": ["cheese"]}],
#            "special_instructions": null
#          },
#          "result": {"success": true, "order_id": "ORD-12345", "total": 9.5, "estimated_time": "15-20 minutes"}
#        }
#      ]
#    }
```

### 3. FastAPI Streaming Server
Launch the FastAPI backend for the Web/Voice UI:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. Docker Deployment
```bash
docker compose up -d --build
```

---

## 5. Core Engineering & Design Decisions

### 1. Pure-Python Agentic Loop & Prefix KV-Caching
Instead of relying on LangGraph, `src/core/agent_loop.py` implements a bounded, deterministic `for step in range(self.max_steps)` state machine (`max_steps = 6`).
* **Prefix KV-Cache Optimization:** A common mistake in stateful agents is injecting the changing cart state directly into the top `system` prompt on every turn. Doing so invalidates the LLM's KV cache on every message, forcing full re-evaluation of the system instructions and tool definitions (~1,400+ tokens).
* **Our Solution:** `self.history[0]` (`role: "system"`) and `self.tools` are **100% static and immutable** across the entire session (`build_static_system_prompt()`). The live cart snapshot and confirmation flags are injected into the user turn wrapper (`[LIVE CART CONTEXT]...[/LIVE CART CONTEXT]`). Consequently, Ollama achieves a **100% prefix KV-cache hit** on every turn, and setting `"think": False` on `/api/chat` eliminates hidden reasoning latency prior to tool emission.
* **No `list_items` Tool Needed:** Because the live cart state (including `line_id`s, options, extras, unit prices, and total price) is deterministically injected into every turn's context, the agent never wastes an extra LLM round-trip calling a `list_items` tool and never hallucinates arithmetic.

### 2. Internal vs. Exposed `tool_calls` Contract
Per the assignment specification, `agent.send()` must only include the `"tool_calls"` key when the remote MCP `submit_order` tool is invoked (add/edit/remove turns return `{"message": "..."}`).
* Internally, the LLM is equipped with five tools: `add_item`, `edit_item`, `remove_item`, `set_special_instructions`, and `submit_order`.
* `agent_loop.py` routes the four cart management tools to the local Python `OrderCart` engine (`internal_tool_trace`, which is saved to the JSONL audit log for observability) while keeping them hidden from the returned `send()` dictionary.
* Only `submit_order` executions are appended to `exposed_tool_calls` and returned in `response["tool_calls"]`.

### 3. Two-Layer Menu Validation & Dynamic Schema Generation (`menu.py`)
`src/core/menu.py` parses `menu.yaml` into strict Pydantic models (`MenuCatalog`, `MenuItem`, `OptionDefinition`, `ExtrasDefinition`) and enforces validity at two distinct layers:
1. **Schema-Level (Pre-Inference):** `build_ollama_tools(menu)` dynamically inspects `menu.yaml` at startup to generate JSON Schema `enum` constraints for every `item_id`, option category, and extra ID.
2. **Runtime Execution-Level (`validate_and_resolve`):**
   * Rejects unknown `item_id`s or `quantity < 1`.
   * Rejects cross-item option hallucinations (e.g., passing `patty` on `spicy_burger`) and automatically generates a **cross-item hint** (`"Invalid option ['patty'] for 'Spicy Jalapeño Burger'. Note: option 'patty' is only available on ['classic_burger']."`).
   * Resolves default option values when omitted by the user (e.g., `size="regular"`, `patty="beef"`).
   * Enforces required options that have **no default** (specifically `milkshake` -> `flavor`), returning an actionable error to the LLM so it prompts the user to pick a flavor (`vanilla`, `chocolate`, `strawberry`, or `oreo`).
   * Deduplicates extras and sorts them in canonical menu order.
   * Performs all unit and line price calculations using Python's `decimal.Decimal` (`ROUND_HALF_UP`) to prevent IEEE-754 floating-point drift.

### 4. Line-Based Cart Engine with Atomic "Split & Merge" (`cart.py`)
`src/core/cart.py` manages the session's `OrderCart` using discrete `CartLineItem` objects (`line_1`, `line_2`, etc.). Each line computes a canonical identity `signature` tuple: `(item_id, sorted(options.items()), tuple(extras))`.
* **Auto-Merge on Add & Edit:** If a user orders a Regular Classic Burger and later says *"Add another regular classic burger"*, `_merge_or_append()` detects the matching signature and increments `line_1.quantity` to `2` rather than creating duplicate lines. Similarly, if editing `line_2` makes it identical to `line_1`, `_consolidate_lines()` automatically merges them back into a single line.
* **Partial Stack Split (`target_quantity < line.quantity`):** If `line_1` has `quantity=3` (e.g., 3 Spicy Burgers) and the user says *"Add bacon to just one of them"*, the LLM calls `edit_item(line_id="line_1", target_quantity=1, add_extras=["bacon"])`. `OrderCart` validates the new configuration first (for atomic rollback safety), decrements `line_1` to `quantity=2`, and splits the modified unit into `line_2` (`quantity=1` with `extras=["bacon"]`).
* **Delta Extra Updates:** Rather than overwriting the entire `extras` array on edit (which risks accidentally erasing existing toppings when removing one), `edit_item` uses explicit delta parameters (`add_extras` and `remove_extras`).

### 5. Streamable HTTP MCP Client & 3-Mode Failure Handling (`client.py`)
`src/mcp/client.py` (`MCPOrderClient`) connects to Moveo's MCP server (`https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp`) using the Streamable HTTP transport and includes `X-Applicant-Email` on every request.
* **Dual Transport Support:** Uses the official `mcp` Python SDK (`streamablehttp_client` + `ClientSession`) if installed, and includes a built-in, zero-dependency `httpx` JSON-RPC 2.0 Streamable HTTP implementation (handling `initialize`, `Mcp-Session-Id`, `notifications/initialized`, `tools/call`, and both `application/json` and `text/event-stream` SSE payloads).
* **Event-Loop Safe:** Detects whether an `asyncio` loop is already running (e.g., inside FastAPI) and dispatches safely in both synchronous (`main.py`) and asynchronous (`server/app.py`) contexts.
* **Complete Failure Path Coverage:**
  1. **Schema Validation (`isError: true`, JSON-RPC `-32602`):** Checks `result.isError` *before* reading `structuredContent`, extracting the JSON-RPC error message cleanly.
  2. **Transient Failures (`kitchen_busy` / `server_error`):** Supports the `X-Demo-Force-Failure` header. When a submission fails transiently, `agent_loop.py` sets `self.can_retry_submission = True` so if the user replies *"Yes"* to retry, the next turn immediately re-submits without repeating the confirmation prompt.
  3. **Business Refusal (`>$50.00 limit`):** Reads `structuredContent` (or parses the JSON text block) when `success: false` is returned, allowing the agent to inform the user of the $50 limit and ask which item to remove.

### 6. GPU VRAM Lifecycle & Session Watchdog (`app.py` & `speech.py`)
Because local LLM (`gemma4:12b`) and speech models (`Faster-Whisper` + `Kokoro-82M`) consume significant GPU VRAM, `server/app.py` implements an automated `SessionManager`:
* **Parallel Warmup on Connect:** When a user opens the web app (`/api/session/start`), `SessionManager` warms up Faster-Whisper, Kokoro-82M, and Ollama's 32k KV cache in parallel via `asyncio.gather` (with strict deduplication so React StrictMode double-mounts do not trigger duplicate loads).
* **15-Minute Inactivity Watchdog:** Every request refreshes a 900-second keep-alive timer (`INACTIVITY_TIMEOUT_SECONDS = 900`). A background watchdog task checks every 30 seconds and unloads all models from VRAM if all sessions have been idle for 15 minutes.
* **Tab-Close Offload (`5s` Grace Period):** When the browser tab closes (`POST /api/session/close`), `SessionManager` waits 5 seconds (to avoid thrashing VRAM on a simple browser refresh `F5`) and, if no active sessions remain, sends `keep_alive=0` to both Ollama and the Speech Service to free 100% of GPU memory.

---

## 6. Handled Edge Cases

| # | Edge Case Scenario | How the Backend Handles It |
| :- | :--- | :--- |
| **1** | **The "Milkshake Trap" (Required option with `default: null`)** | Every menu item has defaults for required options except `milkshake` (`flavor` is `required: true` with no default). If a user orders *"a burger and a milkshake"*, `add_item` succeeds for the burger and returns a validation error for the milkshake, prompting the agent to ask for a flavor (`vanilla`, `chocolate`, `strawberry`, `oreo`). |
| **2** | **Cross-Item Option & Extra Hallucinations** | If a user asks for a *"Veggie Spicy Burger"* or *"extra patty on a Spicy Burger"*, `MenuCatalog` catches that `patty` and `extra_patty` only exist on `classic_burger` and returns a cross-item hint so the LLM suggests the `classic_burger` or valid `spicy_burger` extras (`cheese`, `bacon`, `jalapeños`). |
| **3** | **Partial Modification of Stacked Items** | Ordering *"3 spicy burgers"* and then saying *"add bacon to one of them"* uses `edit_item(..., target_quantity=1, add_extras=["bacon"])` to atomically split the line into `2x Regular Spicy Burger` and `1x Regular Spicy Burger with bacon`. |
| **4** | **Two-Phase Confirmation Guardrail** | In `agent_loop.py`, if the LLM tries to call `submit_order` before the user has been shown the final summary and total price (`was_ready_to_submit == False`), the loop intercepts the call, returns `CONFIRMATION_REQUIRED_FIRST` to the model, and forces it to summarize the order and ask *"Ready to submit?"* first. |
| **5** | **Post-Confirmation "Cold Feet" (`"Yes, and add a cola"`)** | If the agent asks *"Ready to submit?"* and the user replies *"Yes, and add a cola"*, `self.cart.modified_in_current_turn` flips to `True`. The loop blocks `submit_order` for that turn, adds the cola, announces the new total, and asks for confirmation again. |
| **6** | **Duplicate `submit_order` Calls in a Single Turn** | `submitted_in_this_turn` guarantees that even if an LLM emits two tool calls in the same turn, `submit_order` only executes once against the MCP server. |
| **7** | **Persistent `special_instructions` Aggregation** | Dietary notes mentioned on Turn 1 (*"no onions please"*) are stored deterministically in `OrderCart.special_instructions` and automatically attached to the final `submit_order` MCP payload on Turn N. |
| **8** | **Over-$50 Orders & Live MCP Failure Testing** | Orders exceeding $50.00 are allowed to reach `submit_order` so the MCP server's business refusal (`success: false`) is captured in `tool_calls` and explained to the user. Additionally, passing `force_failure` (`kitchen_busy`, `server_error`, `schema_error`, `business_refusal`) allows one-click testing of all MCP error paths. |

---

## 7. FastAPI NDJSON Streaming Protocol

`server/app.py` exposes streaming endpoints returning `application/x-ndjson` (Newline-Delimited JSON) so the client can synchronize text rendering and audio playback sentence-by-sentence.

### Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/session/start` | Initializes session `FoodOrderAgent` and warms up Whisper, Kokoro, and Ollama KV cache. |
| `POST` | `/api/session/close` | Closes session; offloads GPU VRAM (`keep_alive=0`) after a 5s grace period if no tabs remain. |
| `POST` | `/api/session/reset` | Resets conversation history and clears the cart for a fresh order without unloading VRAM. |
| `POST` | `/api/chat/text` | Accepts JSON `{session_id, message, force_failure}` and streams NDJSON turn events. |
| `POST` | `/api/chat/voice` | Accepts `multipart/form-data` (`file`, `session_id`, `force_failure`), transcribes via Whisper, and streams NDJSON turn events. |

### NDJSON Stream Event Sequence

1. **`transcription`** *(Emitted first on `/api/chat/voice` only, immediately after Faster-Whisper finishes)*:
   ```json
   {"type": "transcription", "text": "I want a large classic burger with bacon"}
   ```
2. **`turn_metadata`** *(Emitted immediately after `agent.send()` completes tool calls)*:
   ```json
   {
     "type": "turn_metadata",
     "full_message": "I've added a Large Classic Burger with bacon for $12.00. Anything else?",
     "tool_calls": [],
     "cart_state": {
       "lines": [
         {
           "line_id": "line_1",
           "item_id": "classic_burger",
           "name": "Classic Burger",
           "quantity": 1,
           "options": {"size": "large", "patty": "beef"},
           "extras": ["bacon"],
           "unit_price": 12.0,
           "line_total": 12.0
         }
       ],
       "total_item_count": 1,
       "total_price": 12.0,
       "exceeds_50_limit": false,
       "special_instructions": null,
       "is_submitted": false,
       "last_submitted_order_id": null,
       "modified_in_current_turn": true
     },
     "turn_log": { "...structured turn log..." },
     "mcp_payload": null
   }
   ```
3. **`agent_chunk`** *(Emitted 1..N times as `SpeechServiceClient.synthesize_stream()` splits the response into sentences and synthesizes 24kHz WAV audio via Kokoro-82M)*:
   ```json
   {
     "type": "agent_chunk",
     "chunk_index": 0,
     "total_chunks": 2,
     "is_final": false,
     "text_chunk": "I've added a Large Classic Burger with bacon for $12.00.",
     "audio_b64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
     "sample_rate": 24000
   }
   ```
4. **`turn_complete`** *(Signals end of stream)*:
   ```json
   {"type": "turn_complete"}
   ```

---

## 8. Structured Logging & Observability

Requirement #5 is satisfied via `src/utils/logger.py`, which provides two layers of observability:

1. **Unified Console Telemetry (`setup_logging`):**
   Standardizes log formatting across `main.py`, `server/app.py`, and Uvicorn while silencing noisy low-level `httpx`/`httpcore` traces. Every LLM call, local tool invocation, argument payload, cart total update, and MCP transaction is logged clearly:
   ```text
   2026-11-02 19:30:10 [INFO] [src.core.ollama_client]: [Ollama] Calling gemma4:12b (ctx=32000)...
   2026-11-02 19:30:11 [INFO] [src.core.agent_loop]: [TOOL CALL] add_item | Args: {"item_id": "classic_burger", "quantity": 1, "options": {"size": "large"}, "extras": ["cheese"]}
   2026-11-02 19:30:11 [INFO] [src.core.agent_loop]: [TOOL RESULT] add_item -> success=True | Action: added_or_merged | Cart Total: $11.50
   ```

2. **Structured JSONL Turn Logs (`TurnLogger` -> `logs/conversation_turns.jsonl`):**
   Every completed turn appends a structured JSON record to disk (and to an in-memory ring buffer for the UI debug console) capturing the user message, agent response, exposed MCP `tool_calls`, internal cart tool traces (`internal_cart_actions`), and the resulting `cart_state` snapshot:
   ```json
   {
     "timestamp": "2026-11-02T19:30:12.410284+00:00",
     "session_id": "sess_a8f91c2b",
     "turn_id": 1,
     "user_message": "I want a large classic burger with cheese",
     "agent_message": "I've added a Large Classic Burger with cheese for $11.50. Anything else?",
     "tool_calls": [],
     "internal_cart_actions": [
       {
         "name": "add_item",
         "arguments": {"item_id": "classic_burger", "quantity": 1, "options": {"size": "large"}, "extras": ["cheese"]},
         "result": {"success": true, "action": "added_or_merged", "line": {"line_id": "line_1", "unit_price": 11.5, "line_total": 11.5}}
       }
     ],
     "cart_state": {
       "lines": [{"line_id": "line_1", "item_id": "classic_burger", "name": "Classic Burger", "quantity": 1, "options": {"size": "large", "patty": "beef"}, "extras": ["cheese"], "unit_price": 11.5, "line_total": 11.5}],
       "total_item_count": 1,
       "total_price": 11.5,
       "exceeds_50_limit": false,
       "special_instructions": null,
       "is_submitted": false
     },
     "metadata": {"model": "gemma4:12b", "num_ctx": 32000}
   }
   ```