# Moveo AI — Agentic Food Ordering System (NLP Engineer Assignment)

A production-ready conversational food ordering agent built in **Python 3.13+**. The system manages the complete food ordering lifecycle through natural language dialogue (both text and voice)—parsing and validating items against `menu.yaml`, managing complex multi-item modifications and stack splitting, enforcing a strict two-step confirmation protocol, and submitting final orders via the **Model Context Protocol (MCP)** Streamable HTTP transport.

> ### 🚀 Live Production Deployment & Zero-Config Testing
>
> **1. Instant Browser Testing (No Local Setup Required):**  
> The complete application stack (React Voice/Text Web UI + FastAPI Streaming Backend) is containerized with Docker and live on my server:  
> 🔗 **[https://moveo-food-agent.pali.autos](https://moveo-food-agent.pali.autos)**
>
> **2. Zero-Config Local Execution (`git clone` & run):**  
> **No `.env` tinkering or API keys are required to run this repository locally.** All default endpoints are pre-configured directly inside `config/settings.py` to connect out-of-the-box to my live GPU infrastructure:
> - **Default LLM Server (`LLAMA_URL`):** `https://llama.pali.autos` — running `llama-server` with **Qwen 3.6 35B (`Qwen3.6-35B-A3B`)**.
> - **Default Voice STT/TTS Server (`SPEECH_SERVICE_URL`):** `https://speech.pali.autos` — running my custom GPU `speech_service` container (Faster-Whisper `large-v3-turbo` STT + Kokoro-82M TTS; see `speech_service/README.md` for architecture details).
>
> **Quick Reviewer Walkthrough on the Web UI:**
> 1. **Voice or Text Ordering:** Click the central **Voice Orb** to speak your order naturally (or type in the right-hand chat bar). Watch the **Thermal Delivery Ticket HUD** on the left update in real time with exact prices and the `$50.00` restaurant cap meter.
> 2. **Inspect Logs & MCP Payloads:** Click **Developer Console** at the bottom of the screen to inspect real-time structured JSONL turn logs, cart snapshots, and MCP `submit_order` JSON-RPC payloads.
> 3. **Test MCP Failure Modes Live:** Use the **Mode** selector in the Developer Console (`kitchen_busy`, `server_error`, `schema_error`, `business_refusal`) to inject failure headers or malformed payloads on the next checkout and verify how the agent recovers.

---

## Table of Contents

1. [Assignment Compliance & Core Capabilities](#1-assignment-compliance--core-capabilities)
2. [System Architecture & Request Lifecycle](#2-system-architecture--request-lifecycle)
   - [End-to-End Turn Flow (`agent_loop.py`)](#end-to-end-turn-flow-agent_looppy)
   - [Internal vs. Exposed Tool Separation (Requirement #6)](#internal-vs-exposed-tool-separation-requirement-6)
   - [KV-Cache Optimization & Dynamic User Context Injection (`prompts.py`)](#kv-cache-optimization--dynamic-user-context-injection-promptspy)
3. [Core Domain Engine (`src/core/`)](#3-core-domain-engine-srccore)
   - [Menu Catalog & Schema Validation (`menu.py`)](#1-menu-catalog--schema-validation-menupy)
   - [Transactional Order Cart Engine (`cart.py`)](#2-transactional-order-cart-engine-cartpy)
4. [LLM & Speech Routing Layer (`src/clients/`, `server/` & `speech_service/`)](#4-llm--speech-routing-layer-srcclients-server--speech_service)
   - [Environment-Driven Model Router (`router.py`)](#environment-driven-model-router-routerpy)
   - [Default GPU LLM Backend: `llama.cpp` (`llama-server`) with Qwen 3.6 35B](#default-gpu-llm-backend-llamacpp-llama-server-with-qwen-36-35b)
   - [Default GPU Voice Backend: `speech_service` (Faster-Whisper + Kokoro-82M)](#default-gpu-voice-backend-speech_service-faster-whisper--kokoro-82m)
   - [Optional Cloud Backend: Mistral AI API (LLM & Voice Comparison)](#optional-cloud-backend-mistral-ai-api-llm--voice-comparison)
   - [Resilience & Exponential Backoff Retry Engine (`errors.py`)](#resilience--exponential-backoff-retry-engine-errorspy)
5. [MCP Client & Specification Rejection Handling (`src/mcp/client.py`)](#5-mcp-client--specification-rejection-handling-srcmcpclientpy)
   - [Authentication & Failure Injection](#authentication--failure-injection)
   - [Handling All Three Specification Rejection Paths](#handling-all-three-specification-rejection-paths)
6. [Repository Structure](#6-repository-structure)
7. [Zero-Config Defaults (`config/settings.py`) & Optional `.env` Overrides](#7-zero-config-defaults-configsettingspy--optional-env-overrides)
8. [Running the Application Locally (Fresh Install Walkthrough)](#8-running-the-application-locally-fresh-install-walkthrough)
   - [1. Zero-Config Quickstart (`uv run main.py`)](#1-zero-config-quickstart-uv-run-mainpy)
   - [2. Virtual Environment Setup (`uv sync --extra dev`)](#2-virtual-environment-setup-uv-sync---extra-dev)
   - [3. Programmatic Python API (`agent.py`)](#3-programmatic-python-api-agentpy)
   - [4. FastAPI Streaming Backend Server (`server/app.py`)](#4-fastapi-streaming-backend-server-serverapppy)
   - [5. React Voice & Web UI (Launched from the Root Directory)](#5-react-voice--web-ui-launched-from-the-root-directory)
9. [Live Server & Docker Compose Deployment](#9-live-server--docker-compose-deployment)
10. [Test Suites, Evaluation Results & Analysis (`tests/`)](#10-test-suites-evaluation-results--analysis-tests)
    - [Instant Unit Tests (`pytest`) — 53/53 Passed](#1-instant-unit-tests-pytest--5353-passed)
    - [End-to-End Conversational Evaluation Suites (25 Scenarios)](#2-end-to-end-conversational-evaluation-suites-25-scenarios)
    - [Evaluation Benchmark Results & Behavioral Analysis](#3-evaluation-benchmark-results--behavioral-analysis)
11. [Structured Turn Logging (`src/utils/logger.py`)](#11-structured-turn-logging-srcutilsloggerpy)

---

## 1. Assignment Compliance & Core Capabilities

| # | Assignment Requirement | Implementation Mechanism | Reference Files |
| :--- | :--- | :--- | :--- |
| **1** | **Parse `menu.yaml` & Validate Orders** | Parses `menu.yaml` into strict Pydantic models with automatic default resolution, cross-item hallucination detection, missing required option catches (the "Milkshake Trap"), and exact `Decimal` price math. | `src/core/menu.py` |
| **2** | **Add, Modify & Remove Items** | Transactional `OrderCart` supporting `add_item`, delta-based `edit_item`, `remove_item`, and `clear_cart`. Merges identical items by canonical signature, splits stacked quantities on partial edits (`target_quantity`), and guarantees atomic rollback on invalid edits. | `src/core/cart.py` |
| **3** | **Confirm Order Before Submission** | Dual-layer protection: static system prompt directives + a deterministic intercept in `FoodOrderAgent.send()` that blocks `submit_order` if the cart was modified in the current turn or if confirmation was not requested beforehand. | `src/core/agent_loop.py`<br>`src/core/prompts.py` |
| **4** | **Call `submit_order` via MCP & Handle Responses** | Connects to Moveo's Streamable HTTP MCP server using `X-Applicant-Email`. Normalizes all three rejection modes: JSON-RPC `-32602` schema errors (`isError: true`), server errors (`kitchen_busy`), and business refusals (`> $50.00` limit). | `src/mcp/client.py` |
| **5** | **Structured Turn Logging** | Thread-safe `TurnLogger` recording every turn in newline-delimited JSON (`logs/conversation_turns.jsonl` or `logs/session_<id>.jsonl`) with ISO timestamps, session ID, user/agent text, exposed tool calls, internal cart actions, and cart snapshots. | `src/utils/logger.py` |
| **6** | **Two Entry Points (`main.py` & `agent.py`)** | Interactive terminal CLI (`python main.py` / `uv run main.py`) and programmatic API (`from agent import FoodOrderAgent` with `agent.send(msg) -> dict`). Strictly separates internal cart tools from exposed `response["tool_calls"]`. | `main.py`<br>`agent.py` |
| **7** | **Bonus: Full Voice Agent (STT + TTS)** | Real-time NDJSON streaming pipeline pairing Faster-Whisper STT (`large-v3-turbo`) with Kokoro-82M TTS at `https://speech.pali.autos` (or Mistral Realtime STT/TTS), backed by a Whisper silence-hallucination filter and VRAM watchdog. | `server/app.py`<br>`server/speech.py`<br>`speech_service/` |

---

## 2. System Architecture & Request Lifecycle

```text
                            User Input (CLI / Python API / Web UI)
                                               │
                                               ▼
                      ┌──────────────────────────────────────────────────┐
                      │           FoodOrderAgent.send(message)           │
                      │             (src/core/agent_loop.py)             │
                      └────────────────────────┬─────────────────────────┘
                                               │
               1. Prepend [LIVE CART CONTEXT] & AWAITING_FINAL_CONFIRMATION flag
                                               │
                                               ▼
                      ┌──────────────────────────────────────────────────┐
                      │          LLM Router (src/clients/router.py)      │
                      ├────────────────────────┬─────────────────────────┤
                      │ LLM_PROVIDER=          │ LLM_PROVIDER=           │
                      │ llama-server (DEFAULT) │ mistral (OPTIONAL)      │
                      │ LlamaClient            │ MistralClient           │
                      │ (Qwen 3.6 35B @        │ (Mistral Cloud API)     │
                      │  llama.pali.autos)     │                         │
                      └────────────────────────┴────────────┬────────────┘
                                                            │
                               2. Multi-Step Tool Loop (max_steps=6)
                                                            │
                   ┌────────────────────────────────────────┴────────────────────────────────────────┐
                   ▼                                                                                 ▼
    [ Internal Cart & Menu Tools ]                                                    [ External MCP Tool ]
    (`add_item`, `edit_item`, `remove_item`,                                            (`submit_order`)
     `set_special_instructions`, `clear_cart`)                                               │
                   │                                                         3. Confirmation & Guardrail Intercepts
                   ▼                                                                         │
         OrderCart & MenuCatalog                                                             ▼
    - Validates against `menu.yaml`                                                    MCPOrderClient
    - Auto-merges / splits item stacks                                     (Streamable HTTP + X-Applicant-Email)
    - Resets `awaiting_confirmation = False`                                                 │
    - Hidden from `response["tool_calls"]`                                    - Exposed in `response["tool_calls"]`
                   │                                                          - Normalizes -32602, >$50, kitchen_busy
                   └────────────────────────────────────────┬────────────────────────────────┘
                                                            │
                               4. Prune Duplicate Failed Cycles & Log Turn
                                                            │
                                                            ▼
                      ┌──────────────────────────────────────────────────┐
                      │         TurnLogger (src/utils/logger.py)         │
                      │       Appends turn JSONL record to disk          │
                      └────────────────────────┬─────────────────────────┘
                                               │
                                               ▼
                          Returns `{"message": str, "tool_calls": [...]}`
```

### End-to-End Turn Flow (`agent_loop.py`)

When `agent.send(message, force_failure=None)` is called, `FoodOrderAgent` coordinates a deterministic 7-stage pipeline:

1. **Input Sanitization & Turn Initialization:**
   - Strips whitespace and verifies non-empty alphanumeric content. Empty strings or audio silence artifacts instantly return `"Sorry, I couldn't hear you."` without wasting LLM tokens or modifying history.
   - Invokes `self.cart.start_turn()`, resetting `modified_in_current_turn = False`.
   - Snapshots whether the agent was ready to submit entering this turn: `was_ready_to_submit = self.awaiting_confirmation or self.can_retry_submission`.

2. **Dynamic Cart Context Wrapping:**
   - Rather than mutating the static system prompt on every turn (which destroys LLM KV prompt caching), `format_user_turn_with_context()` prepends a `[LIVE CART CONTEXT]` block directly to the user message. This block contains every active line item with prices, options, extras, special instructions, `$50.00` budget status, and the `AWAITING_FINAL_CONFIRMATION` boolean.

3. **Multi-Step Tool Loop (`max_steps = 6`):**
   - Dispatches `self.history` and `self.tools` to the active LLM client at `temperature=0.1`.
   - If the model returns natural language with no tool calls, the loop terminates.
   - If the model emits tool calls, each call is processed sequentially:
     - **Duplicate Failed Call Breaker:** If the model repeats the exact same tool name and JSON arguments that failed on the immediately preceding step, the loop intercepts the call with `DUPLICATE_CALL_BLOCKED`, forcing the LLM to either repair its arguments or reply in plain text.
     - **Internal Cart Tools (`add_item`, `edit_item`, `remove_item`, `clear_cart`, `set_special_instructions`):** Executed locally against `self.cart`. On any successful modification, `awaiting_confirmation`, `can_retry_submission`, and `was_ready_to_submit` are immediately reset to `False` (**"Cold Feet" protection**: if a user says *"Yes, and add a cola"*, the agent adds the cola and asks for confirmation of the new total rather than prematurely submitting).
     - **External MCP Tool (`submit_order`):** Protected by three programmatic guardrails:
       1. *Duplicate Guard:* Blocks multiple `submit_order` executions in the same turn.
       2. *Empty Cart Guard:* Rejects submission if `cart.lines` is empty.
       3. *Premature Submission Guard:* If `cart.modified_in_current_turn` is `True` or `was_ready_to_submit` is `False`, the call is intercepted with `CONFIRMATION_REQUIRED_FIRST`. The agent sets `awaiting_confirmation = True` and prompts the user with the exact total price.
       - When valid, `self.mcp_client.submit_order()` executes over Streamable HTTP. The resulting payload is appended to `exposed_tool_calls`.

4. **Thinking Tag Sanitization (`_sanitize_agent_text`):**
   - Strips reasoning traces (`<think>...</think>`) emitted by reasoning models, leaving clean conversational text for chat display and TTS synthesis.

5. **Confirmation State Transition (`_update_confirmation_state`):**
   - Automatically sets `awaiting_confirmation = True` for the subsequent turn if the agent explicitly asked a confirmation question (e.g., *"Ready to submit?"*, *"Shall I place your order?"*) or if the user indicated checkout intent on an unmodified cart.

6. **History Pruning (`_prune_history_failures`):**
   - Collapses consecutive duplicate failed tool cycles in `self.history` into a single entry to prevent conversational context degradation over long multi-turn interactions.

7. **Structured Turn Persistence:**
   - Appends the full turn snapshot (user message, agent message, exposed tools, internal cart trace, and cart state) to the active JSONL log file via `TurnLogger`.

### Internal vs. Exposed Tool Separation (Requirement #6)

The assignment specification requires that modifying or adding items returns:
```json
{
  "message": "Large Classic Burger with cheese. Anything else?"
}
```
Only when the user explicitly confirms and the order is submitted via the MCP server should `tool_calls` appear in the response:
```json
{
  "message": "Order submitted! Order #ORD-12345. Estimated time: 15-20 minutes.",
  "tool_calls": [
    {
      "name": "submit_order",
      "arguments": {
        "items": [
          {
            "item_id": "classic_burger",
            "quantity": 1,
            "options": {"size": "regular", "patty": "beef"},
            "extras": ["cheese"]
          }
        ],
        "special_instructions": null
      },
      "result": {
        "success": true,
        "order_id": "ORD-12345",
        "total": 9.50,
        "estimated_time": "15-20 minutes"
      }
    }
  ]
}
```

`FoodOrderAgent` strictly enforces this boundary: all internal catalog and cart modifications (`add_item`, `edit_item`, `remove_item`, `clear_cart`, `set_special_instructions`) are recorded under `internal_cart_actions` for structured telemetry, while `response["tool_calls"]` is populated exclusively when `submit_order` is executed.

### KV-Cache Optimization & Dynamic User Context Injection (`prompts.py`)

35B models can suffer substantial Time-To-First-Token (TTFT) latency if system prompts mutate between dialogue turns. `build_static_system_prompt()` constructs an immutable system prompt prefix containing:
- **Action-First Directives:** Mandatory tool calling rules (never claim an item was added without executing `add_item`).
- **Voice & TTS Delivery Rules:** Concise 1–2 sentence responses with zero Markdown (no asterisks, bullet points, or tables).
- **Category Playbooks:** Disambiguation logic for generic burger requests, the required Milkshake flavor rule, and soda flavor mappings (e.g., "Sprite" $\rightarrow$ `lemon_lime`, "Coke" $\rightarrow$ `cola`).
- **Budget Threshold Protocols:** Rules governing totals under `$40.00` (*"Anything else?"*), between `$40.00–$50.00` (*near limit, offer checkout*), and above `$50.00` (*restaurant cap exceeded*).
- **Menu Reference:** A clean, plain-text representation of `menu.yaml`.

Because this system prompt (`history[0]`) and the tool definitions never change, `llama-server` pins them in slot `0` (`cache_prompt: true`), caching 5,462 prompt tokens on startup for near-instant processing on subsequent turns.

---

## 3. Core Domain Engine (`src/core/`)

### 1. Menu Catalog & Schema Validation (`menu.py`)

`MenuCatalog.from_yaml("menu.yaml")` serves as the authoritative source of truth for items, options, extras, and pricing:

- **Deterministic Default Fills:** Automatically resolves omitted optional fields to their configured defaults (e.g., `classic_burger` defaults to `size="regular"`, `patty="beef"`; `margherita` defaults to `size="medium"`, `crust="regular"`).
- **The "Milkshake Trap" Protection:** In `menu.yaml`, `milkshake` defines `flavor` as `required: true` with **no default value**. If an LLM attempts to call `add_item(item_id="milkshake")` without a flavor, `validate_and_resolve()` raises a `MenuValidationError` detailing `[vanilla, chocolate, strawberry, oreo]`, prompting the agent to ask the user.
- **Cross-Item Hallucination Guard:** If the model attempts to apply an option or extra belonging to another item (e.g., specifying `patty="chicken"` on a `spicy_burger` or requesting `extra_patty` on `fries`), the validator rejects the call and provides a hint indicating which item supports that customization.
- **Exact Decimal Price Arithmetic:** Base prices, negative size discounts (e.g., `-$2.00` for small pizza, `-$1.00` for small fries), and extras are calculated using `decimal.Decimal` with `ROUND_HALF_UP` quantization to eliminate floating-point drift.
- **Canonical Extra Ordering:** Extras are deduplicated and sorted in the exact sequence declared in `menu.yaml`, ensuring that identical customizations always generate identical signatures.

### 2. Transactional Order Cart Engine (`cart.py`)

`OrderCart` manages session items as `CartLineItem` records (`line_1`, `line_2`, ...):

- **Signature Auto-Merging:** Each line computes a canonical tuple: `(item_id, tuple(sorted(options.items())), tuple(extras))`. Adding an item with a matching signature increments the existing line's quantity and recalculates its line total.
- **Atomic Stack Splitting (`target_quantity`):** When modifying a subset of stacked items (e.g., the user has two burgers in `line_1` and states *"put cheese on just one of them"*), `edit_item(line_id="line_1", target_quantity=1, add_extras=["cheese"])` validates the target configuration first, decrements `line_1` to `quantity=1`, and creates `line_2` with `quantity=1` and `extras=["cheese"]`.
- **Auto-Consolidation (`_consolidate_lines`):** If modifying `line_2` makes its customizations identical to `line_1`, the cart merges them back into a single line.
- **Atomic Rollback Guarantee:** Before mutating any line, candidate options and extras are validated against `MenuCatalog`. If validation fails, the cart remains untouched.
- **Persistent Special Instructions:** `update_special_instructions()` accumulates dietary and kitchen notes across turns (separated by `; `) and passes them to `submit_order` at checkout.
- **Budget Tracking:** Dynamically evaluates `total_price > 50.00`, exposing `exceeds_limit = True` to the agent and HUD.

---

## 4. LLM & Speech Routing Layer (`src/clients/`, `server/` & `speech_service/`)

### Environment-Driven Model Router (`router.py`)

`src/clients/router.py` acts as an abstract factory routing LLM and Speech operations based on `config/settings.py` (which defaults to `llama-server` and `speech_service` with zero configuration required):

| Subsystem | Setting / `.env` Variable | Default Hosted GPU Option (`llama-server` / `speech_service`) | Optional Cloud Option (`mistral`) |
| :--- | :--- | :--- | :--- |
| **LLM Reasoning** | `LLM_PROVIDER` | `LlamaClient` (`https://llama.pali.autos` · `Qwen3.6-35B-A3B`) | `MistralClient` (`ministral-14b-2512`) |
| **Voice STT / TTS** | `SPEECH_PROVIDER` | `SpeechServiceClient` (`https://speech.pali.autos` · Whisper + Kokoro) | `MistralSpeechClient` (Voxtral STT/TTS) |

When the GPU services are active (`any_service_uses_vram() == True`), `server/app.py` and `main.py` manage GPU memory automatically:
- Pre-warms STT, TTS, and the 5,462-token static system prompt KV cache on session startup.
- Automatically offloads models from GPU VRAM on CLI exit (`SIGTERM` / `quit`), 5 seconds after the last browser tab closes (`/api/session/close`), or after 15 minutes of inactivity.

### Default GPU LLM Backend: `llama.cpp` (`llama-server`) with Qwen 3.6 35B

Hosted live at `https://llama.pali.autos`, the default inference engine runs `llama-server` serving **Qwen 3.6 35B (`Qwen3.6-35B-A3B`)** with prompt caching pinned to slot `0`, flash attention enabled, and reasoning tags disabled.

#### Systemd Service Unit File (`/etc/systemd/system/llama-server.service`):

```ini
[Unit]
Description=Llama.cpp Dynamic Router Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/llama.cpp
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="CUDA_MODULE_LOADING=EAGER"
ExecStart=/opt/llama.cpp/build/bin/llama-server \
  --models-dir /opt/models \
  --models-autoload \
  --models-max 1 \
  --host 0.0.0.0 \
  --port 8080 \
  -np 1 \
  -c 16384 \
  -fa on \
  --no-warmup \
  --cache-prompt \
  --cache-reuse 256 \
  --fit on \
  --fit-target 3072 \
  --jinja \
  -b 2048 \
  -ub 1024 \
  --poll 0 \
  --poll-batch 0 \
  --reasoning off \
  --no-mmproj

Restart=always
RestartSec=5
LimitMEMLOCK=infinity
KillSignal=SIGINT
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
```

### Default GPU Voice Backend: `speech_service` (Faster-Whisper + Kokoro-82M)

Hosted live at `https://speech.pali.autos` (with full source code and deployment documentation in `speech_service/`), this dedicated CUDA microservice handles all Speech-to-Text (`POST /stt` via Faster-Whisper `large-v3-turbo`) and Text-to-Speech (`POST /tts` via Kokoro-82M at 24kHz WAV), alongside process-isolated VRAM keep-alive management (`POST /manage/stt`, `POST /manage/tts`).

### Optional Cloud Backend: Mistral AI API (LLM & Voice Comparison)

If you wish to test the system using Mistral's cloud API instead of the default hosted GPU services, add your `MISTRAL_API_KEY` to `.env` and set `LLM_PROVIDER=mistral` and/or `SPEECH_PROVIDER=mistral`:
- **Mistral Voice (`voxtral-mini-transcribe-realtime-3` STT & `voxtral-mini-tts-3-0` TTS):** Performs excellently with low latency over both Realtime WebSockets and REST.
- **Mistral LLM (`ministral-14b-2512`):** Note that `ministral-14b-2512` (the model available on my tier) exhibits noticeably weaker multi-turn tool-calling and state-tracking accuracy on complex cart modifications compared to **Qwen 3.6 35B (`Qwen3.6-35B-A3B`)**. Therefore, keeping the default `LLM_PROVIDER=llama-server` is strongly recommended for the best agentic performance.

### Resilience & Exponential Backoff Retry Engine (`errors.py`)

All HTTP network requests pass through `request_with_retry_sync` and `request_with_retry_async`, providing:
- Full jitter exponential backoff on transient errors (`408`, `429`, `500`, `502`, `503`, `504`) and network timeouts, honoring `Retry-After` headers.
- Mapping into domain exceptions: `BadRequestError` (400/422), `AuthenticationError` (401/403), `EndpointOrModelNotFoundError` (404), `RateLimitError` (429), and `UpstreamServerError` (5xx).

---

## 5. MCP Client & Specification Rejection Handling (`src/mcp/client.py`)

`MCPOrderClient` connects to the Moveo Streamable HTTP MCP server (`https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp`) using the official `mcp` SDK (`streamablehttp_client` + `ClientSession`), backed by a fallback `httpx` JSON-RPC implementation.

### Authentication & Failure Injection

- **Authentication Header:** Injects `"X-Applicant-Email": "nickpanosrigas@gmail.com"` on every request.
- **Failure Injection Testing (`force_failure`):**
  - `kitchen_busy`: Injects HTTP header `X-Demo-Force-Failure: kitchen_busy` (MCP returns `success: false` with a retryable message).
  - `server_error`: Injects HTTP header `X-Demo-Force-Failure: server_error` (MCP returns `isError: true` with a `success: false` body).
  - `schema_error`: Modifies the outbound payload's first item quantity to `0` to trigger a genuine JSON-RPC `-32602` error.
  - `business_refusal`: Inflates the outbound quantity to exceed the `$50.00` restaurant limit.

### Handling All Three Specification Rejection Paths

Per the assignment specification, `_parse_mcp_sdk_result()` **checks `isError` before reading `structuredContent`**:

1. **Schema Validation (`isError: true`, JSON-RPC `-32602`):** No `structuredContent` exists on this path. The client extracts the text block, parses the `-32602` error code, and returns `{"success": False, "error": "Schema validation error (-32602): ..."}`.
2. **Server Error / Transient Busy (`kitchen_busy`):** Extracts the error text and returns `{"success": False, "error": "..."}`. `FoodOrderAgent` marks `can_retry_submission = True` and prompts the user to retry.
3. **Business Refusal (`isError: false`, `success: false`, e.g., `> $50.00` limit):** Reads `structuredContent` (or fallback text) and returns the refusal reason so the agent can help the user adjust their order.

---

## 6. Repository Structure

```text
moveo-food-agent/
├── .env.example                    # Optional environment variable template (defaults baked into settings.py)
├── Dockerfile                      # Production container build for FastAPI backend (using uv)
├── docker-compose.yml              # Multi-container orchestration (backend + web services)
├── pyproject.toml                  # Python 3.13+ project metadata & dependencies
├── uv.lock                         # Deterministic dependency lockfile for uv
├── agent.py                        # Programmatic API entry point exporting FoodOrderAgent
├── main.py                         # Interactive terminal CLI entry point
├── menu.yaml                       # Restaurant catalog schema: items, prices, options, extras
│
├── config/
│   └── settings.py                 # Centralized settings with pre-configured working defaults
├── logs/                           # Directory storing structured conversation_turns.jsonl files
├── speech_service/                 # Isolated GPU STT/TTS Docker service (Faster-Whisper + Kokoro-82M)
│   ├── Dockerfile                  # CUDA 12.3 + cuDNN 9 runtime container with pre-baked weights
│   ├── docker-compose.yml          # GPU reservation compose file for speech.pali.autos
│   ├── main.py                     # Multiprocessing worker queues & VRAM keep-alive endpoints
│   └── README.md                   # Documentation for the speech_service container
├── web/                            # React Voice Web UI (see web/README.md for frontend internals)
│
├── server/
│   ├── app.py                      # FastAPI streaming endpoints, session manager & VRAM watchdog
│   ├── speech.py                   # Client for GPU Speech Service (https://speech.pali.autos)
│   └── mistral_speech.py           # Client for Mistral Cloud Speech (Realtime WebSocket/REST STT + TTS)
│
├── src/
│   ├── clients/
│   │   ├── router.py               # Master factory routing LLM & Speech calls based on settings
│   │   ├── llama_client.py         # Llama-server client (Qwen 3.6 35B) with KV caching
│   │   ├── mistral_client.py       # Mistral Chat Completions API client
│   │   └── errors.py               # HTTP error mapping & exponential backoff engine
│   │
│   ├── core/
│   │   ├── agent_loop.py           # FoodOrderAgent state machine, tool loop & guardrails
│   │   ├── cart.py                 # OrderCart engine (auto-merge, stack splitting, Decimal math)
│   │   ├── menu.py                 # MenuCatalog parser, validator & dynamic tool schemas
│   │   └── prompts.py              # Static system prompt & dynamic user context injection
│   │
│   ├── mcp/
│   │   └── client.py               # Streamable HTTP MCP client for submit_order
│   │
│   └── utils/
│       └── logger.py               # Application logging formatter & TurnLogger JSONL recorder
│
└── tests/
    ├── test_menu.py                # Unit tests for menu parsing, defaults & decimal pricing
    ├── test_cart.py                # Unit tests for cart auto-merge, splitting, rollback & budget
    ├── test_mcp.py                 # Unit tests for MCP headers, failure injection & rejections
    ├── test_agent_loop.py          # Unit tests for confirmation guardrails & tool contracts
    ├── test_conversations_easy.py  # E2E Evaluation Suite: Scenarios 1–5 (Standard ordering)
    ├── test_conversations_medium.py# E2E Evaluation Suite: Scenarios 6–15 (Omissions & edits)
    └── test_conversations_hard.py  # E2E Evaluation Suite: Scenarios 16–25 (Stress & 15-turn chaos)
```

---

## 7. Zero-Config Defaults (`config/settings.py`) & Optional `.env` Overrides

**You do not need to create or edit a `.env` file to run this project.** `config/settings.py` is pre-configured with working production defaults pointing directly to my live GPU servers (`https://llama.pali.autos` and `https://speech.pali.autos`) and Moveo's MCP endpoint:

| Setting | Default Value in `config/settings.py` | Description |
| :--- | :--- | :--- |
| `LLM_PROVIDER` | `"llama-server"` | Uses the hosted Qwen 3.6 35B server by default |
| `SPEECH_PROVIDER` | `"speech_service"` | Uses the hosted Faster-Whisper + Kokoro-82M server by default |
| `LLAMA_URL` | `"https://llama.pali.autos"` | Public HTTPS endpoint for `llama-server` |
| `LLAMA_MODEL` | `"Qwen3.6-35B-A3B"` | Active 35B MoE model on `llama-server` |
| `LLAMA_CTX` | `16384` | Context window size |
| `SPEECH_SERVICE_URL` | `"https://speech.pali.autos"` | Public HTTPS endpoint for `speech_service` (STT & TTS) |
| `APPLICANT_EMAIL` | `"nickpanosrigas@gmail.com"` | Sent in `X-Applicant-Email` header to Moveo MCP |
| `MCP_SERVER_URL` | `"https://webviews-git-moveo-food-nlp-assignment-moveo-ai.vercel.app/api/demo/mcp"` | Moveo Streamable HTTP MCP Server |

### Optional: Switching to Mistral API
If you want to switch from the default hosted GPU servers to the Mistral API, copy `.env.example` to `.env`, provide your `MISTRAL_API_KEY`, and change `LLM_PROVIDER` and/or `SPEECH_PROVIDER` to `mistral`:

```bash
cp .env.example .env
```

```env
LLM_PROVIDER=mistral
SPEECH_PROVIDER=mistral
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_LLM=ministral-14b-2512
MISTRAL_STT=voxtral-mini-transcribe-realtime-3
MISTRAL_TTS=voxtral-mini-tts-3-0
```

---

## 8. Running the Application Locally (Fresh Install Walkthrough)

### Prerequisites
- **Python:** `3.13+`
- **`uv` Package Manager:** Recommended for instant dependency resolution (or standard `venv` + `pip`)
- **Node.js:** `20+` or `22+` (only if running the React Web UI locally outside Docker)

### 1. Zero-Config Quickstart (`uv run main.py`)

From a fresh clone on any network, running `uv run main.py` automatically creates `.venv`, installs all 37 dependencies from `uv.lock`, warms up the remote Qwen 3.6 35B KV cache at `https://llama.pali.autos`, and launches the interactive agent:

```bash
git clone https://github.com/nickpanosrigas-gr/moveo-food-agent
cd moveo-food-agent
uv run main.py
```

Example terminal output on a fresh install:
```text
Using CPython 3.13.7 interpreter at: /usr/bin/python3
Creating virtual environment at: .venv
      Built moveo-food-agent @ file:///home/nick/Desktop/moveo-food-agent
Installed 37 packages in 5ms
2026-11-05 08:53:30 [INFO] [src.clients.router]: [Router] Using Llama-Server Client (model=Qwen3.6-35B-A3B)
2026-11-05 08:53:30 [INFO] [src.clients.llama_client]: [llama-server] Warming up KV cache for static system prompt...
2026-11-05 08:53:45 [INFO] [src.clients.llama_client]: [llama-server] KV cache ready (5462 prompt tokens cached).
====================================================
  Moveo AI Food Ordering Agent (Interactive Mode)
  Backend: LLM=llama-server (Qwen3.6-35B-A3B)
           Speech=speech_service
  Type 'quit' or 'exit' to stop.
====================================================

You: hi

2026-11-05 08:53:48 [INFO] [src.clients.llama_client]: [llama-server] Calling Qwen3.6-35B-A3B via /v1/chat/completions...

Agent: Hello! What can I get for you today?

You: exit
2026-11-05 08:53:50 [INFO] [src.clients.llama_client]: [llama-server] Offloading model from GPU VRAM via /models/unload...
2026-11-05 08:53:50 [INFO] [src.clients.llama_client]: [llama-server] Model successfully offloaded from VRAM.
Goodbye!
```

### 2. Virtual Environment Setup (`uv sync --extra dev`)

If you want to activate the virtual environment directly and include `pytest` for running the unit test suite:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
uv sync --extra dev
```

*(Alternatively, without `uv`, you can run `pip install -e ".[dev]"`).*

### 3. Programmatic Python API (`agent.py`)

Use `FoodOrderAgent` directly in Python scripts or evaluation harnesses:

```python
from agent import FoodOrderAgent

agent = FoodOrderAgent()

# 1. Add item with customization
res1 = agent.send("I want a large classic burger with cheese")
print(res1["message"])
# -> "Large Classic Burger with cheese. Anything else?"

# 2. Modify an item in the cart
res2 = agent.send("Actually make it regular")
print(res2["message"])
# -> "Updated: Regular Classic Burger with cheese. Anything else?"

# 3. Request submission (triggers two-step confirmation summary)
res3 = agent.send("Submit")
print(res3["message"])
# -> "Your order: Regular Classic Burger with cheese — $9.50. Ready to submit?"

# 4. Confirm submission (calls MCP submit_order tool)
res4 = agent.send("Yes")
print(res4["message"])
# -> "Order submitted! Order #ORD-12345. Estimated time: 15-20 minutes."
print(res4.get("tool_calls"))
# -> [{'name': 'submit_order', 'arguments': {...}, 'result': {'success': True, ...}}]
```

### 4. FastAPI Streaming Backend Server (`server/app.py`)

Launch the backend streaming server locally on port `8000` (connects automatically to `https://llama.pali.autos` and `https://speech.pali.autos`):

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 5. React Voice & Web UI (Launched from the Root Directory)

To run the interactive React frontend in development mode, install and start it directly from the root repository directory (do not `cd web`):

```bash
# Install frontend dependencies from root
npm --prefix web install

# Start Vite dev server on http://localhost:3000 (proxies /api to localhost:8000)
npm --prefix web run dev
```

*(For architectural details on the 5-state Voice Orb canvas, Web Audio FIFO queuing, and thermal receipt HUD, refer to `web/README.md`.)*

---

## 9. Live Server & Docker Compose Deployment

### 1. Live Hosted Production Instance (Zero Local Setup)

For maximum convenience during evaluation, the entire Dockerized application is deployed and running live on my server at:

🔗 **[https://moveo-food-agent.pali.autos](https://moveo-food-agent.pali.autos)**

- **What is running:** The `moveo-food-agent-web` frontend container and `moveo-food-agent-backend` FastAPI container orchestrated via `docker-compose.yml`, connected to my GPU inference nodes (`https://llama.pali.autos` running `Qwen3.6-35B-A3B` and `https://speech.pali.autos` running Faster-Whisper + Kokoro-82M) and Moveo's live MCP server.
- **Reviewer Features Built Into the UI:**
  - **Voice & Text Modes:** Click the central orb to speak with automatic Voice Activity Detection (Silero VAD), or type in the chat panel on the right.
  - **Live Order Ticket HUD:** Watch items, options, extras, stack splits, and the `$50.00` limit progress bar update in real time on the left receipt.
  - **Developer Console & Failure Injection:** Open the bottom **Developer Console** drawer to inspect structured turn logs (`Turn Logs` tab), raw MCP `submit_order` payloads (`MCP Payloads` tab), or use the **Mode** dropdown to force `kitchen_busy`, `server_error`, `schema_error`, or `business_refusal` on your next order submission.

### 2. Local Docker Compose Deployment

To build and run the containers on your own machine using `Dockerfile` and `web/Dockerfile`:

#### Port Bindings:
- **Web UI:** `http://localhost:3005` (mapped to internal container port `3000`)
- **FastAPI Backend:** `http://localhost:8005` (mapped to internal container port `8000`)

#### Build and Run Containers:

```bash
# Build and start all services in detached mode (uses default endpoints out of the box)
docker compose up -d --build

# View real-time container logs
docker compose logs -f

# Stop and remove containers
docker compose down
```

---

## 10. Test Suites, Evaluation Results & Analysis (`tests/`)

Testing is separated into two tiers: **Instant Unit Tests** (zero external network calls) and **End-to-End Conversational Evaluation Suites** (25 multi-turn scenarios executed against the live LLM backend).

### 1. Instant Unit Tests (`pytest`) — 53/53 Passed

The four unit test modules verify `MenuCatalog` parsing, `OrderCart` state transitions, `MCPOrderClient` rejection normalization, and `FoodOrderAgent` confirmation guardrails using deterministic mocks.

Run all 53 unit tests with verbose output (**executes in ~0.42 seconds**):

```bash
pytest tests/test_menu.py tests/test_cart.py tests/test_mcp.py tests/test_agent_loop.py -v
```

---

### 2. End-to-End Conversational Evaluation Suites (25 Scenarios)

The three conversational evaluation suites execute 25 multi-turn dialogues (134 total conversation turns) against the active LLM backend (`Qwen3.6-35B-A3B` via `llama-server`).

> **Note:** These scripts automatically warm up the model KV cache before Scenario 1 and offload the model from GPU VRAM upon completion.

#### Evaluation Grading Scale:
- **FULL PASS (`✓`):** Final submitted order payload **and** all intermediate turn tool calls match expected behavior.
- **PARTIAL PASS (`~`):** Final submitted order payload is **100% accurate**, and the order succeeded end-to-end, with a minor variation in an intermediate turn's tool path.
- **FAIL (`x`):** Final submitted order payload did not match expected items, options, extras, or instructions.

#### Run Each Suite Independently:

- **Easy Suite (Scenarios 1–5 · Runtime: ~1.5 minutes):**  
  Tests exact menu phrasing, single customized items, standard multi-category combos, pizzas, sides, and single-turn multi-item orders.
  ```bash
  uv run python tests/test_conversations_easy.py
  ```

- **Medium Suite (Scenarios 6–15 · Runtime: ~4 minutes):**  
  Tests user mistakes, omissions, the "Milkshake Trap" (missing required flavor), item category swaps, option upgrades, extra delta removals, partial stack splitting (`target_quantity`), quantity decrements, generic burger disambiguation, and special instruction persistence across turns.
  ```bash
  uv run python tests/test_conversations_medium.py
  ```

- **Hard Suite (Scenarios 16–25 · Runtime: ~6 minutes):**  
  Tests complex conversational challenges: topping migrations, allergen instructions, pronoun ambiguity (*"make both of them large"*), the `$50.00` budget limit gauntlet, a 15-turn state-reconciliation stress test, auto-merge signature collapsing, zero/negative quantity attacks, mid-session cart wipes, and `kitchen_busy` transient failure recovery with mid-retry edits.
  ```bash
  uv run python tests/test_conversations_hard.py
  ```

---

### 3. Evaluation Benchmark Results & Behavioral Analysis

Below is the summary of results running all 25 scenarios (134 turns) on `llama-server` (`Qwen3.6-35B-A3B`):

| Test Suite | Scenarios | Full Pass (`✓`) | Partial Pass (`~`) | Failed (`x`) | Effective Order Accuracy (`✓` + `~`) | Correct Turns |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Easy (`Scenarios 1–5`)** | 5 | 5 | 0 | 0 | **5 / 5 (100%)** | 18 / 18 (100%) |
| **Medium (`Scenarios 6–15`)** | 10 | 7 | 1 | 2 | **8 / 10 (80%)** | 33 / 39 (84.6%) |
| **Hard (`Scenarios 16–25`)** | 10 | 3 | 4 | 3 | **7 / 10 (70%)** | 54 / 77 (70.1%) |
| **Total Overall** | **25** | **15** | **5** | **5** | **20 / 25 (80%)** | **105 / 134 (78.4%)** |

#### Key Behavioral Observations:

1. **Easy Suite (100% Pass Rate):**
   - The agent achieved a perfect score across all 5 scenarios and all 18 turns, handling single and multi-item orders, option customizations, and two-step confirmations without error.

2. **Medium Suite (80% Effective Pass Rate & The "Margherita Cola" Schema Artifact):**
   - Achieved 8/10 successful final orders (including complex stack splitting in Scenario 12, quantity decrements in Scenario 13, item category replacement in Scenario 8, and cross-turn special instructions in Scenario 15).
   - **Interesting Model Quirk in Scenario 10 (`margherita` + `flavor: "cola"`):** When modifying an existing `margherita` pizza in Scenario 10 (*"Actually, make it large with thick crust"*), `Qwen3.6-35B-A3B` persistently attempts to populate `"flavor": "cola"` inside the `edit_item` `options` dictionary alongside `"crust": "thick"`. Because the unified JSON Schema for `options` across all menu items includes `flavor` (which has `default='cola'` for `soda`), the model occasionally over-eagerly fills default sub-properties from the shared `options` schema even when editing a pizza—despite explicit system prompt rules forbidding cross-item options. Our deterministic `MenuCatalog` validator catches this cross-item hallucination (`Invalid option ['flavor'] for 'Margherita Pizza'`) and the `DUPLICATE_CALL_BLOCKED` circuit breaker prevents an infinite loop, keeping the cart state uncorrupted.

3. **Hard Suite (70% Effective Pass Rate on Adversarial & Long-Horizon Stress Tests):**
   - The Hard suite is intentionally designed to push the 35B model to its limits with 7-to-15-turn state mutations, pronoun ambiguity traps, and `$50.00` cap oscillations.
   - The agent successfully passed some of the toughest scenarios in the benchmark—including **Scenario 20** (the 15-turn chaos and state-reconciliation marathon), **Scenario 21** (auto-merge signature collapsing), **Scenario 22** (negative/zero quantity attacks), **Scenario 23** (mid-session cart clearing), **Scenario 24** (`kitchen_busy` recovery with mid-retry cart edits), and **Scenario 25** (multi-item ambiguous cascade).
   - As expected on adversarial edge cases, a few failures occurred when the model defaulted to `size: "regular"` instead of preserving `"large"` across multi-turn milkshake/soda additions (Scenarios 17 & 19) or eagerly applied an upgrade to both burgers before asking for pronoun disambiguation (Scenario 18). Overall, achieving 80% end-to-end order accuracy across all 25 suites demonstrates strong domain resilience.

---

## 11. Structured Turn Logging (`src/utils/logger.py`)

Every dialogue turn is recorded by `TurnLogger` into newline-delimited JSON (`logs/conversation_turns.jsonl` for CLI/API, or `logs/session_<id>.jsonl` for web sessions):

```json
{
  "timestamp": "2026-09-15T14:22:10.412891+00:00",
  "session_id": "sess_a8f91c02",
  "turn_id": 1,
  "user_message": "I'd like a large classic burger with cheese and bacon.",
  "agent_message": "Added a Large Classic Burger with cheese and bacon ($13.00). Anything else?",
  "tool_calls": [],
  "internal_cart_actions": [
    {
      "name": "add_item",
      "arguments": {
        "item_id": "classic_burger",
        "quantity": 1,
        "options": {
          "size": "large"
        },
        "extras": [
          "cheese",
          "bacon"
        ]
      },
      "result": {
        "success": true,
        "action": "added_or_merged"
      }
    }
  ],
  "cart_state": {
    "lines": [
      {
        "line_id": "line_1",
        "item_id": "classic_burger",
        "name": "Classic Burger",
        "quantity": 1,
        "options": {
          "size": "large",
          "patty": "beef"
        },
        "extras": [
          "cheese",
          "bacon"
        ],
        "unit_price": 13.0,
        "line_total": 13.0
      }
    ],
    "total_item_count": 1,
    "total_price": 13.0,
    "exceeds_50_limit": false,
    "special_instructions": null,
    "is_submitted": false
  },
  "metadata": {
    "llm_provider": "llama-server",
    "model": "Qwen3.6-35B-A3B",
    "num_ctx": 16384
  }
}
```