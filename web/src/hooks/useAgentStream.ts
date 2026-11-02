import { useState, useRef, useCallback, useEffect } from 'react';
import type {
  OrbState,
  CartState,
  ChatMessage,
  TurnLog,
  MCPPayload,
  FailureMode,
  NDJSONEvent,
} from '../types';
import { useAudioPlayer, type AgentChunkEvent } from './useAudioPlayer';

interface UseAgentStreamOptions {
  sessionId?: string;
  initialFailureMode?: FailureMode;
}

export interface UseAgentStreamReturn {
  orbState: OrbState;
  setOrbState: (state: OrbState) => void;
  setMicReady: (ready: boolean) => void;
  isProcessing: boolean;
  failureMode: FailureMode;
  setFailureMode: (mode: FailureMode) => void;

  messages: ChatMessage[];
  streamingAgentMessage: string;
  cartState: CartState;
  turnLogs: TurnLog[];
  lastMcpPayload: MCPPayload | null;

  playGreeting: () => Promise<void>;
  sendTextMessage: (text: string) => Promise<void>;
  sendVoiceMessage: (audioBlob: Blob) => Promise<void>;
  resetSession: () => Promise<void>;

  audioPlayer: ReturnType<typeof useAudioPlayer>;
}

const INITIAL_CART_STATE: CartState = {
  lines: [],
  total_item_count: 0,
  total_price: 0.0,
  exceeds_50_limit: false,
  special_instructions: null,
  is_submitted: false,
};

const GREETING_TEXT = 'Hello, what do you want to eat today?';
const GREETING_AUDIO_URL = '/audio/greeting.wav';

export function useAgentStream({
  sessionId = 'sess_default',
  initialFailureMode = 'none',
}: UseAgentStreamOptions = {}): UseAgentStreamReturn {
  const [orbState, setOrbState] = useState<OrbState>('gray');
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [failureMode, setFailureMode] = useState<FailureMode>(initialFailureMode);

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingAgentMessage, setStreamingAgentMessage] = useState<string>('');
  const [cartState, setCartState] = useState<CartState>(INITIAL_CART_STATE);
  const [turnLogs, setTurnLogs] = useState<TurnLog[]>([]);
  const [lastMcpPayload, setLastMcpPayload] = useState<MCPPayload | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const activeAgentMessageRef = useRef<string>('');
  const isMicReadyRef = useRef<boolean>(false);

  const setMicReady = useCallback((ready: boolean) => {
    isMicReadyRef.current = ready;
  }, []);

  const getIdleState = useCallback((): OrbState => {
    return isMicReadyRef.current ? 'blue' : 'gray';
  }, []);

  const audioPlayer = useAudioPlayer({
    onPlaybackStart: () => {
      setOrbState('purple');
    },
    onChunkStart: (chunk: AgentChunkEvent) => {
      activeAgentMessageRef.current = activeAgentMessageRef.current
        ? `${activeAgentMessageRef.current} ${chunk.text_chunk}`
        : chunk.text_chunk;
      setStreamingAgentMessage(activeAgentMessageRef.current);
    },
    onPlaybackComplete: () => {
      if (activeAgentMessageRef.current.trim().length > 0) {
        const finalizedMsg: ChatMessage = {
          id: `agent_${Date.now()}`,
          role: 'agent',
          text: activeAgentMessageRef.current.trim(),
          timestamp: new Date().toISOString(),
        };
        setMessages((prev) => [...prev, finalizedMsg]);
      }
      activeAgentMessageRef.current = '';
      setStreamingAgentMessage('');

      setIsProcessing(false);
      // Return to blue only if mic is active, otherwise return to gray
      setOrbState(getIdleState());
    },
    onError: (err) => {
      console.error('[AudioPlayer Error]', err);
    },
  });

  const playGreeting = useCallback(async () => {
    setIsProcessing(true);
    activeAgentMessageRef.current = '';
    setStreamingAgentMessage('');
    await audioPlayer.resumeAudioContext();
    await audioPlayer.enqueueLocalGreeting(GREETING_AUDIO_URL, GREETING_TEXT);
  }, [audioPlayer]);

  const processNDJSONStream = useCallback(
    async (response: Response) => {
      if (!response.body) {
        throw new Error('ReadableStream not supported on this response');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed) continue;

            const event: NDJSONEvent = JSON.parse(trimmed);

            switch (event.type) {
              case 'transcription': {
                const userMsg: ChatMessage = {
                  id: `user_${Date.now()}`,
                  role: 'user',
                  text: event.text,
                  timestamp: new Date().toISOString(),
                };
                setMessages((prev) => [...prev, userMsg]);
                setOrbState('orange');
                break;
              }

              case 'turn_metadata': {
                setCartState(event.cart_state);
                setTurnLogs((prev) => [...prev, event.turn_log]);
                if (event.mcp_payload) {
                  setLastMcpPayload(event.mcp_payload);
                }
                break;
              }

              case 'agent_chunk': {
                await audioPlayer.enqueueChunk(event);
                break;
              }

              case 'turn_complete': {
                break;
              }

              case 'error': {
                throw new Error(event.error);
              }
            }
          }
        }
      } catch (err: unknown) {
        if ((err as Error)?.name !== 'AbortError') {
          console.error('[Stream Parsing Error]', err);
          setIsProcessing(false);
          setOrbState(getIdleState());
        }
      } finally {
        reader.releaseLock();
      }
    },
    [audioPlayer, getIdleState]
  );

  const sendTextMessage = useCallback(
    async (messageText: string) => {
      const cleanText = messageText.trim();
      if (!cleanText || isProcessing) return;

      setIsProcessing(true);
      setOrbState('orange');
      activeAgentMessageRef.current = '';
      setStreamingAgentMessage('');

      const userMsg: ChatMessage = {
        id: `user_${Date.now()}`,
        role: 'user',
        text: cleanText,
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMsg]);

      abortControllerRef.current = new AbortController();

      try {
        await audioPlayer.resumeAudioContext();
        const response = await fetch('/api/chat/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: sessionId,
            message: cleanText,
            force_failure: failureMode,
          }),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        await processNDJSONStream(response);
      } catch (err) {
        console.error('[Text Chat Error]', err);
        setIsProcessing(false);
        setOrbState(getIdleState());
      }
    },
    [sessionId, failureMode, isProcessing, audioPlayer, processNDJSONStream, getIdleState]
  );

  const sendVoiceMessage = useCallback(
    async (audioBlob: Blob) => {
      if (isProcessing) return;

      setIsProcessing(true);
      setOrbState('orange');
      activeAgentMessageRef.current = '';
      setStreamingAgentMessage('');

      const formData = new FormData();
      formData.append('file', audioBlob, 'speech.wav');
      formData.append('session_id', sessionId);
      formData.append('force_failure', failureMode);

      abortControllerRef.current = new AbortController();

      try {
        await audioPlayer.resumeAudioContext();
        const response = await fetch('/api/chat/voice', {
          method: 'POST',
          body: formData,
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        await processNDJSONStream(response);
      } catch (err) {
        console.error('[Voice Chat Error]', err);
        setIsProcessing(false);
        setOrbState(getIdleState());
      }
    },
    [sessionId, failureMode, isProcessing, audioPlayer, processNDJSONStream, getIdleState]
  );

  const resetSession = useCallback(async () => {
    audioPlayer.stopPlayback();
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    try {
      await fetch('/api/session/reset', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });

      setMessages([]);
      setStreamingAgentMessage('');
      setCartState(INITIAL_CART_STATE);
      setTurnLogs([]);
      setLastMcpPayload(null);
      setIsProcessing(false);
      // Preserve gray if the orb was never clicked; only go to blue if mic was enabled
      setOrbState(getIdleState());
    } catch (err) {
      console.error('[Session Reset Failed]', err);
    }
  }, [sessionId, audioPlayer, getIdleState]);

  useEffect(() => {
    fetch('/api/session/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId }),
    })
      .then((res) => res.json())
      .then((data) => {
        if (data.cart_state) setCartState(data.cart_state);
        if (data.logs) setTurnLogs(data.logs);
      })
      .catch((err) => console.error('[Session Warmup Error]', err));

    const handleBeforeUnload = () => {
      const payload = JSON.stringify({ session_id: sessionId });
      const blob = new Blob([payload], { type: 'application/json' });
      navigator.sendBeacon('/api/session/close', blob);
    };

    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      handleBeforeUnload();
    };
  }, [sessionId]);

  return {
    orbState,
    setOrbState,
    setMicReady,
    isProcessing,
    failureMode,
    setFailureMode,
    messages,
    streamingAgentMessage,
    cartState,
    turnLogs,
    lastMcpPayload,
    playGreeting,
    sendTextMessage,
    sendVoiceMessage,
    resetSession,
    audioPlayer,
  };
}