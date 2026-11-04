# Voice & Web UI (`/web`) — Moveo Food Ordering Agent

A real-time, voice- and text-driven conversational ordering interface built with **React 19**, **TypeScript**, **Tailwind CSS**, **Web Audio API**, and client-side **Silero VAD (`@ricky0123/vad-web`)**.

This frontend serves as the interactive cockpit for the Food Ordering Agent. It delivers millisecond-accurate audio/text synchronized responses, a 5-state HTML5 Canvas Voice Orb with envelope-clamped waveforms, a realistic thermal kitchen ticket HUD with a live $50 limit meter, and an integrated developer console for real-time observability and MCP failure injection.

---

## Table of Contents

1. [Architecture & Information Flow](#1-architecture--information-flow)
2. [Frontend-to-Backend Communication Protocol](#2-frontend-to-backend-communication-protocol)
3. [The 5-State Visual Machine](#3-the-5-state-visual-machine)
4. [Core Component Breakdown](#4-core-component-breakdown)
   - [Voice Orb (`VoiceOrb.tsx`)](#voice-orb-voiceorbtsx)
   - [Chat Panel (`ChatPanel.tsx`)](#chat-panel-chatpaneltsx)
   - [Order Ticket HUD (`OrderTicketHUD.tsx`)](#order-ticket-hud-ordertickethudtsx)
   - [Developer Console (`DeveloperConsole.tsx`)](#developer-console-developerconsoletsx)
5. [Client Hooks & Audio Engineering](#5-client-hooks--audio-engineering)
   - [Silero VAD & In-Browser WAV Encoding (`useVAD.ts`)](#silero-vad--in-browser-wav-encoding-usevadts)
   - [Web Audio FIFO Playback & Text Sync (`useAudioPlayer.ts`)](#web-audio-fifo-playback--text-sync-useaudioplayerts)
   - [Stream Consumer & Lifecycle Management (`useAgentStream.ts`)](#stream-consumer--lifecycle-management-useagentstreamts)
   - [Autoplay Policy Unlocker (`main.tsx`)](#autoplay-policy-unlocker-maintsx)
6. [Getting Started & Development](#6-getting-started--development)

---

## 1. Architecture & Information Flow

The frontend operates as a state-reactive loop connecting user input (voice or text), network streaming, real-time audio playback, and synchronous visual updates:

```text
                     ┌──────────────────────────────────────────────┐
                     │                   App.tsx                    │
                     │   (Master Stage Coordinator & Backdrop)      │
                     └──────┬───────────────────┬────────────┬──────┘
                            │                   │            │
            ┌───────────────┴────────┐          │   ┌────────┴──────────────┐
            ▼                        ▼          ▼   ▼                       ▼
   OrderTicketHUD.tsx        VoiceOrb.tsx   ChatPanel.tsx     DeveloperConsole.tsx
  (Thermal Receipt &      (5-State 2D Canvas (Chat Feed &       (Telemetry, Logs &
   $50 Limit Gauge)        Waveform Engine)   Input Bar)        MCP Failure Injection)
            ▲                        ▲          ▲                           ▲
            │                        │          │                           │
            └───────────────┬────────┴──────────┴───────────────────────────┘
                            │
               ┌────────────┴────────────────────────┐
               │          useAgentStream.ts          │
               │   (NDJSON Stream & Lifecycle Sync)  │
               └───────────┬──────────────────┬──────┘
                           │                  │
                           ▼                  ▼
                     useVAD.ts         useAudioPlayer.ts
               (Silero Mic Capture &   (FIFO Audio Queue &
                 Float32 PCM to WAV)    AnalyserNode Engine)
```

### End-to-End Flow Summary

1. **User Voice Input:** The user speaks. `useVAD.ts` detects speech start in real time via client-side WebAssembly. When speech ends, it converts raw audio samples into a 16kHz mono WAV file and dispatches it.
2. **User Text Input:** The user types into `ChatPanel.tsx` and hits send. The message is immediately appended to the feed, and an HTTP POST request is dispatched.
3. **Backend Processing:** The backend runs STT (for voice), LLM tool orchestration, MCP server execution (`submit_order`), and TTS synthesis.
4. **Streaming Response (NDJSON):** As events resolve on the server, they stream back to `useAgentStream.ts` line-by-line.
5. **UI & Audio Fan-out:**
   - Transcriptions immediately appear as emerald user chat bubbles.
   - Cart modifications update the thermal receipt and total limit progress bar.
   - Incoming audio chunks queue into `useAudioPlayer.ts`, playing speech through the speakers while streaming text word-by-word into purple agent bubbles.
   - Audio analysis drives live waveform vibrations across the glowing Voice Orb.

---

## 2. Frontend-to-Backend Communication Protocol

The web UI communicates with the FastAPI backend over HTTP using a reverse-proxy route (`/api/*`). In development, Vite proxies `/api` directly to `http://localhost:8000`. In production, Nginx or Docker routing serves the same role.

### Communication Endpoints

| Endpoint | Method | Payload / Format | Purpose |
| :--- | :--- | :--- | :--- |
| `/api/session/start` | `POST` | `application/json`<br>`{ "session_id": string }` | Warms up backend models (LLM, STT, TTS) and restores any existing cart or log state for the session. |
| `/api/session/reset` | `POST` | `application/json`<br>`{ "session_id": string }` | Clears conversation history, resets the cart, and restores initial provider settings. |
| `/api/session/close` | `POST` | `application/json` (via Beacon)<br>`{ "session_id": string }` | Dispatched via `navigator.sendBeacon` when the browser tab closes so the backend can release GPU resources. |
| `/api/chat/text` | `POST` | `application/json`<br>`{ "session_id", "message", "force_failure" }` | Sends typed text and initiates a chunked NDJSON streaming response. |
| `/api/chat/voice` | `POST` | `multipart/form-data`<br>`file` (WAV), `session_id`, `force_failure` | Uploads recorded audio and initiates the same chunked NDJSON streaming response. |

### The NDJSON Streaming Protocol

Both `/api/chat/text` and `/api/chat/voice` return chunked transfer-encoded streams using **Newline-Delimited JSON (`application/x-ndjson`)**. 

Because network packets can split JSON objects mid-line, `useAgentStream.ts` maintains a string buffer. It splits received chunks by newline (`\n`), processes only complete lines, and leaves any incomplete fragment in the buffer for the next network packet:

```typescript
buffer += decoder.decode(value, { stream: true });
const lines = buffer.split('\n');
buffer = lines.pop() ?? ''; // Keep incomplete fragment
```

#### Event Lifecycle

1. **`transcription`** (Voice turns only)
   ```json
   { "type": "transcription", "text": "I want a classic burger" }
   ```
   *Action:* Displays the user's speech transcript immediately in an emerald bubble and switches the orb to **orange** (thinking) while the LLM generates the response.

2. **`turn_metadata`**
   ```json
   {
     "type": "turn_metadata",
     "cart_state": { "lines": [...], "total_price": 12.50, ... },
     "turn_log": { "turn_id": 1, "tool_calls": [...], ... },
     "mcp_payload": { "name": "submit_order", "arguments": {...}, "result": {...} }
   }
   ```
   *Action:* Re-renders the thermal delivery ticket (`OrderTicketHUD`), updates the Developer Console log list, and saves the latest MCP payload.

3. **`agent_chunk`** (Repeated for each synthesized sentence)
   ```json
   {
     "type": "agent_chunk",
     "chunk_index": 0,
     "total_chunks": 2,
     "is_final": false,
     "text_chunk": "I added a classic burger to your order.",
     "audio_b64": "<base64-encoded WAV>",
     "sample_rate": 24000
   }
   ```
   *Action:* Enqueues into the Web Audio FIFO queue. When playback starts, the orb turns **purple** and text streams into the chat panel in sync with the audio.

4. **`turn_complete`**
   ```json
   { "type": "turn_complete" }
   ```
   *Action:* Signals that all server processing has completed. Once the audio queue finishes playing, the orb returns to idle (**blue** or **gray**).

5. **`error`**
   ```json
   { "type": "error", "error": "Detailed error message" }
   ```
   *Action:* Catches runtime errors, releases input locks, and logs diagnostic information.

---

## 3. The 5-State Visual Machine

The entire UI synchronizes across five visual states. The active state governs the full-screen radial backdrop, the Voice Orb canvas, and the chat input bar:

| State | Color | Background Radial Glow | Meaning & Trigger Condition | Chat Input Bar Behavior |
| :---: | :---: | :--- | :--- | :--- |
| **`gray`** | Slate Gray<br>`#6B7280` | `rgba(107, 114, 128, 0.28)` on `#090d16` | **Muted / Standby:** Initial state on page load, or after the user clicks the orb to cancel/mute listening. | Enabled. Clean gray border and send button. |
| **`blue`** | Electric Blue<br>`#3B82F6` | `rgba(59, 130, 246, 0.30)` on `#040b1e` | **Listening Standby:** Microphone is active and Silero VAD is waiting for speech. | Enabled. Blue border, focus ring, and button glow. |
| **`green`** | Emerald<br>`#10B981` | `rgba(16, 185, 129, 0.32)` on `#021812` | **User Active:** User is either actively speaking (`VAD onSpeechStart`) or focused on the text input box. | Highlighted in emerald with an active glow. |
| **`orange`** | Amber<br>`#F59E0B` | `rgba(245, 158, 11, 0.30)` on `#180d02` | **Agent Thinking:** Backend STT, LLM inference, or MCP tool execution is running. | Locked (`disabled`). Shows spinner; placeholder says *"Agent is thinking..."*. |
| **`purple`** | Violet<br>`#8B5CF6` | `rgba(139, 92, 246, 0.34)` on `#0f0520` | **Agent Speaking:** Web Audio is playing TTS speech (or the local welcome greeting). | Locked (`disabled`). Shows spinner; placeholder says *"Agent is speaking..."*. |

---

## 4. Core Component Breakdown

### Voice Orb (`VoiceOrb.tsx`)

A 60 FPS HTML5 Canvas component that serves as the visual centerpiece and primary voice control touchpoint.

* **What it displays:** A glowing sphere featuring concentric radial gradients, an outer glow, a responsive bounding border, dynamic waveform lines, and state-specific subtext captions.
* **What it does:** Visually renders live microphone audio (when the user speaks) and synthesizer frequencies (when the agent speaks). It also acts as the primary click button to wake up the assistant or cancel/mute speech.
* **How it works:**
  * **Retina / High-DPI Sharpness:** Canvas dimensions are multiplied by `window.devicePixelRatio` on every frame to eliminate blurriness on modern displays.
  * **Dynamic Radius Math:**
    $$R(t) = R_{\text{base}} + (\text{smoothedRMS} \times \text{scale}) + A_{\text{pulse}} \times \sin(\omega t)$$
    * Uses an exponential moving average (`smoothedRMS += (rawRMS - smoothedRMS) * 0.15`) so volume expansion feels organic rather than jittery.
    * In `green` (user speaking), volume scale is `65`. In `purple` (agent speaking), volume scale is `35`.
  * **Envelope-Clamped Waveforms:** In `blue`, `green`, and `purple` states, a horizontal line is drawn across the circle diameter. To prevent the waveform peaks from clipping outside the orb boundary, displacement is multiplied by a sine windowing envelope:
    $$u = \frac{x - (x_{\text{center}} - R)}{2R}, \quad E(u) = \sin(u \cdot \pi)$$
    Because $E(0) = 0$ and $E(1) = 0$, displacement naturally tapers to zero at the outer edges, keeping the waveform contained within the circle.
  * **Thinking Animation (`orange`):** Instead of a waveform, three horizontal dots oscillate along an offset sine wave:
    $$y_i(t) = y_{\text{center}} + \sin(6t + 0.8i) \times 12 \quad \text{for } i \in \{-1, 0, 1\}$$
  * **Interaction Logic:**
    * In `gray`: Clicking unlocks browser audio, requests microphone permission, triggers the local welcome greeting, and transitions to `purple` $\rightarrow$ `blue`.
    * In `blue` or `green`: Clicking immediately stops recording, discards any in-flight voice buffer, and drops to `gray` without sending data to the server.
    * In `orange` or `purple`: Clicks are ignored while the agent is busy.

---

### Chat Panel (`ChatPanel.tsx`)

A minimalist, floating conversation panel docked on the right side of the screen.

* **What it displays:** A chronological message history with distinct role bubbles, timestamps, agent/user avatars, an animated streaming bubble with a blinking cursor, a thinking indicator, and an adaptive input bar.
* **What it does:** Provides instant visual feedback of recognized speech, displays progressive text as audio plays, and allows full order interaction via typing.
* **How it works:**
  * **Bubble Styling:** User messages render in emerald green on the right (`bg-emerald-600/30`); agent messages render in purple on the left (`bg-purple-600/30`).
  * **Live Stream Preview:** While the agent is speaking, incoming text accumulates inside an ephemeral streaming bubble with a flashing cursor (`span.animate-pulse`). When audio finishes, the text commits permanently into a message bubble.
  * **Thinking State:** In the `orange` state before text arrives, a temporary bubble renders with a spinning `Loader2` icon and the text *"Agent is thinking..."*.
  * **Input Bar Synchronization:** The text input bar dynamically adopts the colors and ring glows of the active `orbState`. When the agent is thinking or speaking, the input field and send button are disabled to prevent out-of-order requests.
  * **Auto-Scroll:** Uses an invisible anchor ref (`messagesEndRef`) that triggers smooth scrolling whenever new messages, streaming tokens, or processing states change.

---

### Order Ticket HUD (`OrderTicketHUD.tsx`)

A skeuomorphic kitchen delivery ticket docked on the left side of the screen.

* **What it displays:** A realistic dot-matrix receipt on off-white paper (`#F7F5F0`) with serrated zigzag tear edges, paper fiber texture, a spindle punch hole, single-line items with prices, hierarchical options and extras, persistent notes, and a live $50 limit progress bar.
* **What it does:** Mirrors the current server-side cart state in real time. It visually highlights customizations and enforces the $50 order limit constraint before checkout.
* **How it works:**
  * **Paper Texture & Construction:** Built using SVG zigzag tear paths at the top and bottom, an SVG `feTurbulence` fractal noise filter (`mix-blend-multiply`) for realistic paper fiber, a dark spindle punch hole, and the `Share Tech Mono` monospace font.
  * **Single-Line Item Alignment:** Constrained to `370px` width. Item rows place `<qty> x <name>` (with flex truncation) and `$<total>` on a single line (`whitespace-nowrap`) with identical baseline alignment.
  * **Hierarchical Indentation:**
    * Options are indented (`pl-[36px]`) and mapped to friendly labels (e.g., `beef` $\rightarrow$ `Beef Patty`, `mild` $\rightarrow$ `Mild Spice`).
    * Extras are indented (`pl-[36px]`), mapped via `EXTRA_LABELS`, and prefixed with `+ ` (e.g., `+ Bacon`, `+ Truffle Oil`).
    * Special instructions appear at the bottom in a `NOTE:` section.
  * **$50.00 Limit Meter:**
    * Computes `progressPercent = min(100, (total / 50.0) * 100)`.
    * **$0.00 – $40.00:** Black progress bar; displays remaining budget (e.g., `$15.50 LEFT`).
    * **$40.01 – $50.00:** Amber progress bar (`bg-amber-500`); displays `NEAR LIMIT`.
    * **Over $50.00:** Red progress bar (`bg-red-600`); displays `LIMIT EXCEEDED` alongside an `OVER $50 RESTAURANT CAP` warning badge.
  * **Status Stamps:** Once `cartState.is_submitted` is true, a bordered `SUBMITTED #<order_id>` stamp appears across the bottom of the receipt.
  * **Auto-Scroll Fade:** When cart items exceed the ticket height, the body scrolls smoothly to the newest item and displays a paper fade gradient (`▼ MORE ITEMS ▼`) above the pinned totals footer.

---

### Developer Console (`DeveloperConsole.tsx`)

A collapsible, monochrome inspection drawer docked at the bottom of the center column.

* **What it displays:** Active session metrics (turn count, MCP status, active LLM and speech provider), an injectable failure mode menu, raw conversation turn logs with cart snapshots, and raw JSON-RPC payloads for MCP calls.
* **What it does:** Gives engineers full observability into the agent's internal reasoning, tool calls, and cart transitions, while enabling one-click failure mode testing for MCP robustness evaluation.
* **How it works:**
  * **Collapsible Drawer:** Smoothly toggles between a compact 48px control bar (`h-12`) and an expanded 320px inspection deck (`h-[320px]`).
  * **Directional Dropdown Menu:** To avoid clipping outside the viewport:
    * When collapsed at the bottom of the screen, the failure mode menu drops **upward** (`bottom-full mb-2`).
    * When expanded, the menu drops **downward** (`top-full mt-2`).
  * **5 Injectable Failure Modes (`force_failure`):**
    1. `none`: Standard execution against the MCP server.
    2. `kitchen_busy`: Injects a `503 Service Unavailable` response to test transient error retry handling.
    3. `server_error`: Injects an MCP `500 Internal Error` response.
    4. `schema_error`: Triggers a JSON-RPC `-32602` invalid argument error (e.g., zero quantity).
    5. `business_refusal`: Simulates an order total exceeding the $50 limit to test business refusal handling.
  * **Inspection Tabs:**
    * **Turn Logs:** Displays sequential turn records including user speech, agent reply, executed tool calls with parameters, and a collapsible JSON cart snapshot.
    * **MCP Payloads:** Displays the exact arguments sent to `submit_order`, injected HTTP headers (`X-Demo-Force-Failure`), and the returned MCP response.
  * **One-Click Clipboard Export:** Includes a `Copy JSON` button that formats the active tab's data with 2-space indentation and displays a temporary confirmation checkmark.

---

## 5. Client Hooks & Audio Engineering

### Silero VAD & In-Browser WAV Encoding (`useVAD.ts`)

Manages client-side microphone capture and Voice Activity Detection using `@ricky0123/vad-web`:

* **WebAssembly VAD:** Runs the Silero VAD ONNX model in a browser WebAssembly runtime at 16,000 Hz mono with browser echo cancellation and noise suppression enabled.
* **Pause Tolerance (`redemptionFrames: 19`):** Standard VAD settings cut off audio after ~750ms of silence. This hook configures 19 frames (~1.82 seconds of silence tolerance), allowing users to pause mid-sentence to think about an order item without premature cutoff.
* **Zero-Dependency WAV Encoder (`encodeWAV`):** Converts the `Float32Array` PCM audio emitted by Silero VAD directly into a standard 44-byte RIFF/WAVE Blob (`audio/wav`, 16-bit linear PCM) in browser memory before sending it to `/api/chat/voice`:
  ```typescript
  // Clamps 32-bit float samples (-1.0 to 1.0) to 16-bit signed PCM integers (-32768 to 32767)
  const s = Math.max(-1, Math.min(1, samples[i]));
  view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  ```
* **Live RMS Metering:** Directs the microphone stream through a Web Audio `AnalyserNode` to compute real-time root-mean-square volume for the `green` Voice Orb animation.

---

### Web Audio FIFO Playback & Text Sync (`useAudioPlayer.ts`)

Solves the common desynchronization problem where agent text appears all at once while voice audio lags behind.

* **Ordered FIFO Queue:** Incoming `agent_chunk` events contain sentence text and Base64-encoded audio. Audio is decoded into an `AudioBuffer` via `ctx.decodeAudioData()` and placed in an ordered queue (`queueRef`).
* **Playback Pipeline:** Plays chunks sequentially through an `AudioBufferSourceNode` connected to an `AnalyserNode` (`fftSize = 512`, `smoothingTimeConstant = 0.8`).
* **Synchronized Display:** The instant an audio chunk begins playing (`onChunkStart`), its corresponding sentence text is appended to the chat bubble.
* **Fallback Animation:** If an audio chunk is missing or fails to decode, an automatic fallback delay calculates duration from character length:
  $$\text{duration} = \max(1200\text{ms}, \text{text\_length} \times 55\text{ms})$$
* **Visual Data Hooks:** Exposes `getByteFrequencyData()`, `getByteTimeDomainData()`, and `getRMSVolume()` to animate the purple Voice Orb in real time.

---

### Stream Consumer & Lifecycle Management (`useAgentStream.ts`)

The central state coordinator connecting the network, audio engine, and React UI:

* **State Ownership:** Manages `messages`, `cartState`, `turnLogs`, `lastMcpPayload`, `orbState`, and error handling.
* **Cancellation Support:** Wraps requests in an `AbortController`. When the user resets the session, in-flight HTTP streams and audio queues are aborted immediately.
* **Microphone State Memory:** Tracks whether the microphone was ever activated via `isMicReadyRef`. When agent speech completes, `getIdleState()` returns the UI to `blue` if the mic was active, or `gray` if the user has only typed.

---

### Autoplay Policy Unlocker (`main.tsx`)

Modern browsers restrict audio playback until a user gesture occurs. `AudioContextProvider` creates a root `AudioContext` and registers one-time listeners for `pointerdown`, `keydown`, and `touchstart`. On the very first tap or click anywhere on the page, the audio context resumes, ensuring subsequent assistant greetings and voice responses play without browser security warnings.

---

## 6. Getting Started & Development

### Prerequisites

* **Node.js:** `v20+` or `v22+`
* **Backend:** FastAPI server running on `http://localhost:8000`

### 1. Installation

```bash
cd web
npm install
```

### 2. Audio Assets

Ensure the greeting audio asset exists in the public directory:
```text
web/public/audio/mistral.wav   # (or speech.wav depending on provider)
```
*(If omitted, `useAudioPlayer.ts` automatically uses a timed text fallback.)*

### 3. Local Development

Start the Vite development server on `http://localhost:3000`:
```bash
npm run dev
```
All calls to `/api/*` are automatically proxied to `http://localhost:8000`.

### 4. Production Build & Docker

To create an optimized production bundle:
```bash
npm run build
```

To run using Docker (Node 22 Alpine serving the bundle via `serve` on port 3000):
```bash
docker build -t moveo-food-web .
docker run -p 3000:3000 -e WEB_INTERNAL_PORT=3000 moveo-food-web
```