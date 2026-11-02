# Agentic Food Ordering Web & Voice UI (`/web`) · NLP Engineer Assignment

A real-time, voice- and text-enabled conversational interface built with **React 19**, **TypeScript**, **Tailwind CSS**, **Web Audio API**, and **Client-Side Silero VAD (`@ricky0123/vad-web`)**. 

Designed as the interactive frontend for the Moveo Food Ordering Agent (satisfying the **Voice Agent Bonus** and providing a live observability suite for the **MCP `submit_order` tool**), this application features a 5-state HTML5 Canvas Voice Orb with mathematical envelope-clamped waveforms, millisecond-accurate TTS audio/text synchronization, a realistic dot-matrix thermal delivery receipt HUD, and an integrated developer console for one-click testing of all MCP failure modes.

---

## Table of Contents

1. [System Topology & Architecture](#1-system-topology--architecture)
2. [Directory Structure](#2-directory-structure)
3. [Visual Theme & 5-State Machine](#3-visual-theme--5-state-machine)
4. [UI Layout & Component Breakdown](#4-ui-layout--component-breakdown)
   - [Left HUD: Thermal Delivery Receipt (`OrderTicketHUD.tsx`)](#1-left-hud-thermal-delivery-receipt-ordertickethudtsx)
   - [Center Stage: 5-State Canvas Voice Orb (`VoiceOrb.tsx`)](#2-center-stage-5-state-canvas-voice-orb-voiceorbtsx)
   - [Center Bottom: Monochrome Developer Console (`DeveloperConsole.tsx`)](#3-center-bottom-monochrome-developer-console-developerconsoletsx)
   - [Right Panel: Synchronized Chat & State-Reactive Input (`ChatPanel.tsx`)](#4-right-panel-synchronized-chat--state-reactive-input-chatpaneltsx)
5. [Core Frontend Engineering & Hooks](#5-core-frontend-engineering--hooks)
   - [Client-Side Silero VAD & 16kHz RIFF/WAV Encoder (`useVAD.ts`)](#1-client-side-silero-vad--16khz-riffwav-encoder-usevadts)
   - [Web Audio FIFO Sequencer & Millisecond Text Sync (`useAudioPlayer.ts`)](#2-web-audio-fifo-sequencer--millisecond-text-sync-useaudioplayerts)
   - [NDJSON Stream Consumer & Session Lifecycle (`useAgentStream.ts`)](#3-ndjson-stream-consumer--session-lifecycle-useagentstreamts)
6. [Audio & Text Synchronization Protocol](#6-audio--text-synchronization-protocol)
7. [Setup, Development & Production Build](#7-setup-development--production-build)

---

## 1. System Topology & Architecture

Traffic is routed under a single domain (`https://moveo-food-agent.pali.autos`) through a Cloudflare Tunnel into an Nginx reverse proxy. In local development, Vite proxies `/api/*` directly to the FastAPI backend (`http://localhost:8000`), eliminating Cross-Origin Resource Sharing (CORS) overhead.

```text
[ End User Browser (React 19 + Web Audio API + Silero WASM VAD) ]
        │  [https://moveo-food-agent.pali.autos](https://moveo-food-agent.pali.autos)
        ▼
[ Cloudflare Secure Tunnel / Host Nginx Proxy ]
   ├─── location /api/ ───► [http://127.0.0.1:8000](http://127.0.0.1:8000) (FastAPI NDJSON Streaming Backend)
   │                                │
   │                                ├──► [ Ollama LLM ] ([http://192.168.1.220:11434](http://192.168.1.220:11434) - gemma4:12b)
   │                                ├──► [ Speech Service ] (Faster-Whisper STT + Kokoro-82M TTS)
   │                                └──► [ Moveo MCP Server ] (Streamable HTTP `submit_order`)
   │
   └─── location /     ───► [http://127.0.0.1:3000](http://127.0.0.1:3000) (React Web UI Container)
```

---

## 2. Directory Structure

```text
web/
├── package.json                   # React 19, TypeScript, Tailwind, Lucide, @ricky0123/vad-web
├── vite.config.ts                 # Dev server config proxying `/api` to `http://localhost:8000`
├── tailwind.config.js             # Custom styling, animations, and color definitions
├── Dockerfile                     # Multi-stage production build served via `serve -s dist -l 3000`
├── README.md                      # Frontend architecture and engineering documentation
│
├── public/
│   └── audio/
│       └── greeting.wav           # Local 24kHz welcome audio ("Hello, what do you want to eat today?")
│
└── src/
    ├── main.tsx                   # Root React 19 mount with `AudioContextProvider` autoplay unlocker
    ├── App.tsx                    # Master 3-column stage, full-screen state-reactive backdrop & VAD glue
    ├── index.css                  # Tailwind directives, custom scrollbars, and SVG micro-grain overlay
    │
    ├── types/
    │   └── index.ts               # Strict TypeScript interfaces: OrbState, CartState, TurnLog,
    │                              # MCPPayload, FailureMode, and NDJSONEvent discriminated unions
    │
    ├── components/
    │   ├── VoiceOrb.tsx           # 5-state HTML5 Canvas with envelope-clamped waveform math
    │   ├── OrderTicketHUD.tsx     # Realistic dot-matrix thermal delivery receipt & $50 limit bar
    │   ├── ChatPanel.tsx          # Minimalist chat feed with state-synchronized input bar
    │   └── DeveloperConsole.tsx   # Monochrome collapsible drawer for Turn Logs & MCP failure testing
    │
    └── hooks/
        ├── useVAD.ts              # Silero VAD mic listener, RMS analyzer & Float32-to-WAV encoder
        ├── useAudioPlayer.ts      # Web Audio FIFO queue, AnalyserNode & text/audio sync engine
        └── useAgentStream.ts      # NDJSON stream consumer, local greeting trigger & beacon lifecycle
```

---

## 3. Visual Theme & 5-State Machine

The entire application shares a unified, full-screen canvas (`100vw x 100vh`) layered beneath an SVG fractal-noise micro-grain texture (`mix-blend-mode: overlay`). The full-screen radial background, the center Voice Orb, and the right-hand chat input bar transition synchronously across five semantic states:

| State | Core Hex | Full-Screen Ambient Backdrop | Semantic Meaning & Trigger Condition |
| :--- | :--- | :--- | :--- |
| **`gray`** | `#6B7280` *(Gray-500)* | `rgba(107, 114, 128, 0.28)` on `#090d16` | **Mic Muted / Standby:** Initial load state before the orb is clicked, or when the user clicks an active (`blue`/`green`) orb to mute/cancel voice capture. |
| **`blue`** | `#3B82F6` *(Blue-500)* | `rgba(59, 130, 246, 0.30)` on `#040b1e` | **Listening Standby:** Microphone permission is active and Silero VAD is monitoring for user speech. |
| **`green`** | `#10B981` *(Emerald-500)* | `rgba(16, 185, 129, 0.32)` on `#021812` | **User Active (Speaking or Typing):** Entered automatically when Silero VAD fires `onSpeechStart()` **or** when the user clicks/focuses the text input bar to type. |
| **`orange`** | `#F59E0B` *(Amber-500)* | `rgba(245, 158, 11, 0.30)` on `#180d02` | **Agent Thinking:** Entered when VAD detects speech end or a text message is submitted. Indicates Faster-Whisper STT, Ollama `gemma4:12b` tool calling, or MCP `submit_order` execution is running. |
| **`purple`** | `#8B5CF6` *(Violet-500)* | `rgba(139, 92, 246, 0.34)` on `#0f0520` | **Agent Responding:** Entered the exact millisecond `AudioBufferSourceNode.start(0)` begins playing Kokoro-82M TTS audio (or the initial local greeting). Reverts to `blue` (or `gray` if mic was never enabled) when playback finishes. |

---

## 4. UI Layout & Component Breakdown

```text
+--------------------------------------------------------------------------------------------------+
|                        Master Full-Screen State-Reactive Canvas (`App.tsx`)                      |
|  - Full-viewport radial gradient shifting dynamically across gray / blue / green / orange / purple|
|  - SVG fractal-noise micro-grain overlay (`mix-blend-mode: overlay`)                             |
+------------------------------+-----------------------------------+-------------------------------+
|     LEFT HUD (370px)         |           CENTER STAGE            |      RIGHT PANEL (420px)      |
|    `OrderTicketHUD.tsx`      |    Top:    [ Reset Session ]      |        `ChatPanel.tsx`        |
|                              |                                   |                               |
|  -Zigzag torn paper edges    |    Middle:   `VoiceOrb.tsx`       |  - Borderless floating feed   |
|  - "DELIVERY" header & hole  |      - High-DPI HTML5 Canvas      |  - Emerald user bubbles       |
|  - Dot-matrix black font     |      - Envelope-clamped waveform  |  - Purple agent bubbles       |
|  - Single-line `Qty x Item`  |      - 3-dot sine wave thinking   |  - Live cursor TTS streaming  |
|  - Indented options & extras |      - Click-to-mute / cancel     |  - State-colored input bar    |
|  - Scrollable body w/ fade   |                                   |  - Auto-locked during orange  |
|  - Pinned Hovering Total     |    Bottom: `DeveloperConsole.tsx` |    & purple states            |
|  - $50.00 Limit Progress Bar |      - Monochrome zinc UI         |                               |
|  - Submitted / Cap Stamps    |      - Smart up/down Mode menu    |                               |
|                              |      - Turn Logs & MCP Inspector  |                               |
+------------------------------+-----------------------------------+-------------------------------+
```

### 1. Left HUD: Thermal Delivery Receipt (`OrderTicketHUD.tsx`)
Designed to look and behave like a physical dot-matrix kitchen ticket (`#F7F5F0` off-white paper with radial grain dots, an SVG `<feTurbulence>` fiber texture, a realistic spindle punch hole, and SVG serrated zigzag tear edges at the top and bottom):
- **All-Black Dot-Matrix Typography:** Uses `Share Tech Mono` in solid black (`#000000`) with full-width dashed dividers (`----`) aligned with the right-hand price column.
- **Strict Single-Line Item Formatting:** Engineered specifically around the assignment's `menu.yaml` dimensions at `370px` width so even the longest item (`1 x Spicy Jalapeño Burger` + `$11.50`) fits on a single line (`whitespace-nowrap`) with its price aligned on the exact same baseline.
- **Hierarchical Indentation:**
  - Parent row: `<quantity> x <Item Name>` on the left, `$<line_total>` on the right.
  - Indented below (`pl-[36px]`): Resolved option values (e.g., `Large`, `Beef Patty`, `Hot Spice`, `Thin Crust`), one per line.
  - Indented below options (`pl-[36px]`): Selected extras prefixed with `+` (e.g., `+ Cheese`, `+ Bacon`, `+ Truffle Oil`), one per line.
  - Bottom of list: Any persistent `special_instructions` note (e.g., `NOTE: "No onions please"`).
- **Fixed Ticket Area with Scroll Fade & Pinned Footer:** When an order contains more items than fit vertically, the middle item list scrolls independently and displays a smooth paper fade gradient (`▼ MORE ITEMS ▼`) above the pinned footer. The `TOTAL` and `$50.00 MAX LIMIT` progress bar remain permanently visible at the bottom of the ticket.
- **$50.00 Limit Progress Bar & Status Stamps:** Tracks `cartState.total_price` against the $50.00 restaurant cap. Renders in black under $40.00, turns amber (`bg-amber-500`) between $40.01 and $50.00 (`NEAR LIMIT`), and turns crimson red (`bg-red-600`) above $50.00 (`LIMIT EXCEEDED`) with an `OVER $50 RESTAURANT CAP` warning stamp. When `submit_order` succeeds, renders a boxed `SUBMITTED #<order_id>` stamp.

### 2. Center Stage: 5-State Canvas Voice Orb (`VoiceOrb.tsx`)
Rendered at 60 FPS inside a `requestAnimationFrame` loop with automatic `window.devicePixelRatio` scaling for Retina/4K sharpness:
- **Dynamic Radius Equation:**
  $$R(t) = R_{\text{base}} + (\text{rmsVolume} \times \text{scale}) + A_{\text{pulse}} \times \sin(\omega t)$$
  Uses an exponential moving average (`smoothedRMS += (rawRMS - smoothedRMS) * 0.15`) so the orb expands and contracts fluidly with live voice volume.
- **Envelope-Clamped Waveform (`blue`, `green`, `purple`):**
  Draws a horizontal waveform across the diameter of the circle from $x = x_{\text{center}} - R(t)$ to $x = x_{\text{center}} + R(t)$. To guarantee that high-amplitude peaks never clip through the circular border, every sample point is multiplied by a sine windowing envelope:
  $$E(x) = \sin\left(\frac{x - (x_{\text{center}} - R(t))}{2R(t)} \times \pi\right)$$
  Because $E(x) = 0$ at both endpoints ($u = 0$ and $u = 1$) and $E(x) = 1$ at the center ($u = 0.5$), the waveform tapers smoothly into the perimeter stroke.
- **Orange State Thinking Dots:**
  During LLM/STT processing (`orange`), the center line transforms into three oscillating dots driven by an offset sine wave:
  $$y_i(t) = y_{\text{center}} + \sin(6t + 0.8i) \times 12 \quad \text{for } i \in \{-1, 0, 1\}$$
- **Interactive Click Behavior:**
  - **First Click (`gray` -> `purple` -> `blue`):** Requests microphone access, unlocks the browser's `AudioContext`, and triggers a frontend-only welcome greeting (`"Hello, what do you want to eat today?"`) using `/audio/greeting.wav`. The orb turns `purple` and animates to the greeting audio while streaming the text into the chat panel, then lands in `blue` listening standby.
  - **Click While Active (`blue` or `green` -> `gray`):** Clicking the orb during `blue` or `green` immediately pauses Silero VAD, sets `discardRecordingRef.current = true` to drop any in-flight voice buffer **without** sending a request to the backend, and returns the app to `gray` muted standby.
  - **Locked During Processing (`orange` & `purple`):** Clicks are disabled while the agent is thinking or speaking.

### 3. Center Bottom: Monochrome Developer Console (`DeveloperConsole.tsx`)
Positioned at the bottom of the center column directly beneath the Voice Orb (with a matching `Reset Session` button at the top of the center column):
- **Clean Monochrome Aesthetic:** Styled in a relaxed system sans-serif font (`font-sans`) over a translucent zinc background (`bg-zinc-900/95`, `border-white/15`) with pure white/zinc typography and bold structural labels (`Turn #`, `User:`, `Agent:`, `Tool Calls:`, `Method:`, `Status:`).
- **Smart Direction Custom Failure-Mode Dropdown:** Replaces ugly native OS `<select>` menus (especially on Linux Firefox) with a custom React dropdown menu. When the console drawer is collapsed at the bottom of the screen, the menu **drops UP** (`bottom-full mb-2`); when the drawer is expanded, the menu automatically **drops DOWN** (`top-full mt-2`).
- **5 Selectable Failure Modes (`force_failure`):**
  1. `none` — Standard execution against the Moveo MCP server.
  2. `kitchen_busy` — Injects `X-Demo-Force-Failure: kitchen_busy` on `submit_order` to test transient failure handling and instant user retry.
  3. `server_error` — Injects `X-Demo-Force-Failure: server_error` to test MCP `isError: true` + `success: false` handling.
  4. `schema_error` — Forces a malformed `quantity: 0` payload on `submit_order` to test JSON-RPC `-32602` validation error handling.
  5. `business_refusal` — Temporarilyinflates the outbound payload above $50.00 to test MCP business refusal handling.
- **Two Real-Time Inspection Tabs:**
  - **`Turn Logs (N)`:** Displays every conversation turn's timestamp, session ID, user/agent messages, internal cart tool calls (`add_item`, `edit_item`, `remove_item`, `set_special_instructions`), and a collapsible JSON `Cart Snapshot`.
  - **`MCP Payloads (submit_order)`:** Displays the exact JSON-RPC `submit_order` tool arguments, active `X-Demo-Force-Failure` headers, and the returned MCP response (`order_id`, `total`, `estimated_time`, or `error`). Includes a one-click `Copy JSON` button.

### 4. Right Panel: Synchronized Chat & State-Reactive Input (`ChatPanel.tsx`)
A minimalist, borderless conversation feed floating directly over the full-screen vignette background:
- **Role-Specific Bubbles:**
  - **User Messages:** Emerald green bubbles (`bg-emerald-600/30 border-emerald-500/40 text-emerald-100`) aligned right with user avatar and timestamp.
  - **Agent Messages:** Purple bubbles (`bg-purple-600/30 border-purple-500/40 text-purple-100`) aligned left with bot avatar and timestamp.
- **Voice-to-Chat Consistency:** On voice turns, as soon as Faster-Whisper finishes transcription, the backend flushes the `transcription` NDJSON packet *before* starting LLM inference. The user's green bubble appears immediately so they can verify what was heard while the orb is still in `orange` (thinking).
- **State-Synchronized Input Bar:** The message input box, border glow, send button, and loading spinner dynamically match the active `orbState` color (`gray`, `blue`, `green`, `orange`, `purple`). Clicking into the text input automatically shifts the app's visual state to `green`; blurring without sending returns to `blue` (if the mic was activated) or `gray` (if the user typed without ever enabling the mic).

---

## 5. Core Frontend Engineering & Hooks

### 1. Client-Side Silero VAD & 16kHz RIFF/WAV Encoder (`useVAD.ts`)
- Runs the Silero VAD ONNX model directly in the browser via `@ricky0123/vad-web` (`MicVAD.new`) at a 16,000 Hz mono sample rate with echo cancellation and noise suppression enabled.
- **Natural Thinking Pause Tolerance (`redemptionFrames: 19`):** Standard VAD configurations cut off speech after ~750ms of silence, which prematurely submits orders when a user pauses mid-sentence to think about their next item. We configure `redemptionFrames: 19` (~1.82 seconds of silence tolerance), giving the user a full extra second to pause and think before `onSpeechEnd` fires.
- **Zero-Dependency PCM-to-WAV Encoder (`encodeWAV`):** Converts the raw `Float32Array` 16kHz mono PCM audio buffer emitted by Silero VAD into a canonical 44-byte RIFF/WAVE binary `Blob` (`audio/wav`, 16-bit PCM) ready for `POST /api/chat/voice`.
- **Live RMS Metering:** Taps the microphone `MediaStream` into a Web Audio `AnalyserNode` (`fftSize = 512`) to compute real-time root-mean-square (RMS) volume for the `green` VoiceOrb animation.

### 2. Web Audio FIFO Sequencer & Millisecond Text Sync (`useAudioPlayer.ts`)
- Manages a single shared `AudioContext` (`latencyHint: 'interactive'`) and `AnalyserNode` (`fftSize = 512`, `smoothingTimeConstant = 0.8`).
- Decodes incoming Base64 WAV packets (`audio_b64`) from Kokoro-82M TTS via `audioContext.decodeAudioData()` and queues them in an ordered FIFO buffer (`queueRef`).
- Exposes `enqueueLocalGreeting(audioUrl, text)` to load `/audio/greeting.wav` into the exact same playback pipeline on initial orb activation (with an automatic timed text-stream fallback if the audio file is absent).
- Provides `getByteFrequencyData()`, `getByteTimeDomainData()`, and `getRMSVolume()` to drive the `purple` VoiceOrb waveform in real time.

### 3. NDJSON Stream Consumer & Session Lifecycle (`useAgentStream.ts`)
- **Streaming `ReadableStream` Parser (`processNDJSONStream`):** Reads chunked `application/x-ndjson` responses from `/api/chat/text` and `/api/chat/voice` using `TextDecoder`, splits on newline boundaries (`\n`), buffers incomplete trailing fragments across network chunks, and dispatches strongly typed `NDJSONEvent` objects.
- **Mic-Aware State Restoration (`isMicReadyRef`):** Tracks whether the user has actually activated the microphone orb. If a user types a message in the chat panel or clicks `Reset Session` *before* ever clicking the gray orb, `getIdleState()` ensures the UI returns to `gray` rather than falsely indicating `blue` (listening).
- **VRAM Lifecycle Integration:**
  - Fires `POST /api/session/start` on mount to pre-warm Faster-Whisper, Kokoro-82M, and Ollama's 32k KV cache.
  - Registers a `beforeunload` listener that dispatches `navigator.sendBeacon('/api/session/close', blob)` when the browser tab closes, allowing the backend to offload all models from GPU VRAM after a 5-second refresh grace period.

---

## 6. Audio & Text Synchronization Protocol

A common pitfall in voice agents is displaying the full LLM text response immediately while TTS audio lags behind, breaking conversational immersion. Our frontend enforces strict sentence-by-sentence synchronization:

```text
[ Incoming NDJSON Stream (`agent_chunk` 0..N) ]
          │
          ├──► `text_chunk` ("I've added a Large Classic Burger...")
          └──► `audio_b64`  ──► `audioContext.decodeAudioData()`
                                         │
                             [ FIFO Chunk Sequencer ]
                                         │
          ┌──────────────────────────────┴──────────────────────────────┐
          ▼                                                             ▼
`source.start(0)` executes                             `onChunkStart(chunk)` fires
  - Orb state switches to `purple`                       - Appends `text_chunk` to active
  - Audio routes through `AnalyserNode`                    purple streaming bubble with
  - Drives envelope-clamped orb waveform                   blinking cursor
          │
          ▼
`source.onended` triggers next chunk in FIFO queue
          │
          ▼
When `chunk.is_final === true` finishes playing:
  - `onPlaybackComplete()` commits full message to chat history
  - Orb returns to `blue` (or `gray`) and unlocks input controls
```

---

## 7. Setup, Development & Production Build

### Prerequisites
- **Node.js:** `20+` (or `22+`)
- **Backend Server:** FastAPI server (`server/app.py`) running on `http://localhost:8000`

### 1. Install Dependencies
```bash
cd web
npm install
```

### 2. Local Greeting Audio Asset
Place your 1-sentence welcome recording (`"Hello, what do you want to eat today?"`) at:
```text
web/public/audio/greeting.wav
```
*(Note: If this file is not present, `useAudioPlayer.ts` automatically falls back to a timed text-streaming simulation so the UI still transitions through the `purple` greeting state cleanly.)*

### 3. Run Development Server
Starts Vite on `http://localhost:3000` with `/api` requests proxied to `http://localhost:8000`:
```bash
npm run dev
```

### 4. Production Build & Docker Container
To compile TypeScript and bundle production assets into `web/dist/`:
```bash
npm run build
```

Or build and run via the root `docker-compose.yml` (which builds the multi-stage `web/Dockerfile` and serves the static bundle on port `3000`):
```bash
docker compose up -d --build
```