import { useRef, useState, useCallback, useEffect } from 'react';
import type { NDJSONEvent } from '../types';

export type AgentChunkEvent = Extract<NDJSONEvent, { type: 'agent_chunk' }>;

type AnalyserDataArray = Parameters<AnalyserNode['getByteFrequencyData']>[0];

interface AudioQueueItem {
  chunk: AgentChunkEvent;
  audioBuffer: AudioBuffer | null;
}

interface UseAudioPlayerOptions {
  onPlaybackStart?: () => void;
  onChunkStart?: (chunk: AgentChunkEvent) => void;
  onChunkEnd?: (chunk: AgentChunkEvent) => void;
  onPlaybackComplete?: () => void;
  onError?: (error: Error) => void;
}

export interface UseAudioPlayerReturn {
  audioContext: AudioContext | null;
  analyserNode: AnalyserNode | null;
  isPlaying: boolean;
  activeChunkIndex: number | null;
  enqueueChunk: (chunk: AgentChunkEvent) => Promise<void>;
  enqueueLocalGreeting: (audioUrl: string, text: string) => Promise<void>;
  stopPlayback: () => void;
  resumeAudioContext: () => Promise<void>;
  getByteFrequencyData: () => Uint8Array | null;
  getByteTimeDomainData: () => Uint8Array | null;
  getRMSVolume: () => number;
}

function base64ToArrayBuffer(base64: string): ArrayBuffer {
  const binaryString = window.atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}

export function useAudioPlayer({
  onPlaybackStart,
  onChunkStart,
  onChunkEnd,
  onPlaybackComplete,
  onError,
}: UseAudioPlayerOptions = {}): UseAudioPlayerReturn {
  const [isPlaying, setIsPlaying] = useState(false);
  const [activeChunkIndex, setActiveChunkIndex] = useState<number | null>(null);

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const currentSourceRef = useRef<AudioBufferSourceNode | null>(null);

  const queueRef = useRef<AudioQueueItem[]>([]);
  const isProcessingQueueRef = useRef(false);
  const isPlayingRef = useRef(false);
  const frequencyDataArrayRef = useRef<AnalyserDataArray | null>(null);
  const timeDomainDataArrayRef = useRef<AnalyserDataArray | null>(null);

  const callbacksRef = useRef({
    onPlaybackStart,
    onChunkStart,
    onChunkEnd,
    onPlaybackComplete,
    onError,
  });

  useEffect(() => {
    callbacksRef.current = {
      onPlaybackStart,
      onChunkStart,
      onChunkEnd,
      onPlaybackComplete,
      onError,
    };
  }, [onPlaybackStart, onChunkStart, onChunkEnd, onPlaybackComplete, onError]);

  const initAudio = useCallback(() => {
    if (!audioContextRef.current) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx({ latencyHint: 'interactive' });

      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.8;

      analyser.connect(ctx.destination);

      audioContextRef.current = ctx;
      analyserNodeRef.current = analyser;

      frequencyDataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
      timeDomainDataArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
    }

    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }
  }, []);

  const resumeAudioContext = useCallback(async () => {
    initAudio();
    if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
      await audioContextRef.current.resume();
    }
  }, [initAudio]);

  const playNextInQueue = useCallback(async () => {
    if (queueRef.current.length === 0) {
      isProcessingQueueRef.current = false;
      isPlayingRef.current = false;
      setIsPlaying(false);
      setActiveChunkIndex(null);
      return;
    }

    isProcessingQueueRef.current = true;
    const item = queueRef.current.shift()!;
    const { chunk, audioBuffer } = item;

    if (!isPlayingRef.current) {
      isPlayingRef.current = true;
      setIsPlaying(true);
      callbacksRef.current.onPlaybackStart?.();
    }

    setActiveChunkIndex(chunk.chunk_index);
    callbacksRef.current.onChunkStart?.(chunk);

    if (!audioBuffer) {
      const fallbackDuration = Math.max(1200, chunk.text_chunk.length * 55);
      setTimeout(() => {
        callbacksRef.current.onChunkEnd?.(chunk);
        if (chunk.is_final) {
          callbacksRef.current.onPlaybackComplete?.();
        }
        playNextInQueue();
      }, fallbackDuration);
      return;
    }

    const ctx = audioContextRef.current!;
    const analyser = analyserNodeRef.current!;

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(analyser);
    currentSourceRef.current = source;

    source.onended = () => {
      currentSourceRef.current = null;
      callbacksRef.current.onChunkEnd?.(chunk);

      if (chunk.is_final) {
        callbacksRef.current.onPlaybackComplete?.();
      }

      playNextInQueue();
    };

    source.start(0);
  }, []);

  const enqueueChunk = useCallback(
    async (chunk: AgentChunkEvent) => {
      initAudio();
      const ctx = audioContextRef.current!;

      let decodedBuffer: AudioBuffer | null = null;

      if (chunk.audio_b64) {
        try {
          const rawBuffer = base64ToArrayBuffer(chunk.audio_b64);
          decodedBuffer = await ctx.decodeAudioData(rawBuffer);
        } catch (err) {
          const decodeError = err instanceof Error ? err : new Error('Audio decoding failed');
          callbacksRef.current.onError?.(decodeError);
        }
      }

      queueRef.current.push({
        chunk,
        audioBuffer: decodedBuffer,
      });

      if (!isProcessingQueueRef.current) {
        playNextInQueue();
      }
    },
    [initAudio, playNextInQueue]
  );

  /**
   * Fetches a local audio file from web/public (e.g. /audio/greeting.wav)
   * and plays it through the exact same synchronized purple orb & bubble pipeline.
   */
  const enqueueLocalGreeting = useCallback(
    async (audioUrl: string, text: string) => {
      initAudio();
      const ctx = audioContextRef.current!;

      let decodedBuffer: AudioBuffer | null = null;

      try {
        const res = await fetch(audioUrl);
        if (res.ok) {
          const rawBuffer = await res.arrayBuffer();
          decodedBuffer = await ctx.decodeAudioData(rawBuffer);
        } else {
          console.warn(`[Greeting Audio] ${audioUrl} not found, using timed text fallback.`);
        }
      } catch (err) {
        console.warn('[Greeting Audio] Failed to decode local greeting file, using fallback:', err);
      }

      const syntheticChunk: AgentChunkEvent = {
        type: 'agent_chunk',
        chunk_index: 0,
        total_chunks: 1,
        is_final: true,
        text_chunk: text,
        audio_b64: null,
        sample_rate: 24000,
      };

      queueRef.current.push({
        chunk: syntheticChunk,
        audioBuffer: decodedBuffer,
      });

      if (!isProcessingQueueRef.current) {
        playNextInQueue();
      }
    },
    [initAudio, playNextInQueue]
  );

  const stopPlayback = useCallback(() => {
    if (currentSourceRef.current) {
      try {
        currentSourceRef.current.stop();
        currentSourceRef.current.disconnect();
      } catch {
        // Ignore already stopped source
      }
      currentSourceRef.current = null;
    }

    queueRef.current = [];
    isProcessingQueueRef.current = false;
    isPlayingRef.current = false;
    setIsPlaying(false);
    setActiveChunkIndex(null);
  }, []);

  const getByteFrequencyData = useCallback((): Uint8Array | null => {
    const analyser = analyserNodeRef.current;
    const data = frequencyDataArrayRef.current;
    if (!analyser || !data) return null;

    analyser.getByteFrequencyData(data);
    return data;
  }, []);

  const getByteTimeDomainData = useCallback((): Uint8Array | null => {
    const analyser = analyserNodeRef.current;
    const data = timeDomainDataArrayRef.current;
    if (!analyser || !data) return null;

    analyser.getByteTimeDomainData(data);
    return data;
  }, []);

  const getRMSVolume = useCallback((): number => {
    const data = getByteTimeDomainData();
    if (!data || !isPlayingRef.current) return 0;

    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const normalized = (data[i] - 128) / 128;
      sum += normalized * normalized;
    }
    return Math.sqrt(sum / data.length);
  }, [getByteTimeDomainData]);

  useEffect(() => {
    return () => {
      stopPlayback();
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, [stopPlayback]);

  return {
    audioContext: audioContextRef.current,
    analyserNode: analyserNodeRef.current,
    isPlaying,
    activeChunkIndex,
    enqueueChunk,
    enqueueLocalGreeting,
    stopPlayback,
    resumeAudioContext,
    getByteFrequencyData,
    getByteTimeDomainData,
    getRMSVolume,
  };
}