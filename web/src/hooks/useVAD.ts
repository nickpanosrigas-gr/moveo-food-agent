import { useState, useEffect, useRef, useCallback } from 'react';
import { MicVAD } from '@ricky0123/vad-web';

type AnalyserDataArray = Parameters<AnalyserNode['getByteTimeDomainData']>[0];

interface UseVADOptions {
  onSpeechStart?: () => void;
  onSpeechEnd?: (audioBlob: Blob) => void;
  onVADMisfire?: () => void;
  enabled?: boolean;
}

export interface UseVADReturn {
  isListening: boolean;
  isSpeaking: boolean;
  hasPermission: boolean;
  error: string | null;
  analyserNode: AnalyserNode | null;
  start: () => Promise<void>;
  pause: () => void;
  resume: () => void;
  getRMSVolume: () => number;
}

/**
 * Encodes 16kHz mono Float32Array PCM samples directly into a canonical RIFF/WAVE Blob.
 */
function encodeWAV(samples: Float32Array, sampleRate = 16000): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, 'WAVE');

  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); // 16-bit

  writeString(view, 36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string): void {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

export function useVAD({
  onSpeechStart,
  onSpeechEnd,
  onVADMisfire,
  enabled = true,
}: UseVADOptions = {}): UseVADReturn {
  const [hasPermission, setHasPermission] = useState<boolean>(false);
  const [isListening, setIsListening] = useState<boolean>(false);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const vadRef = useRef<Awaited<ReturnType<typeof MicVAD.new>> | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserNodeRef = useRef<AnalyserNode | null>(null);
  const timeDomainArrayRef = useRef<AnalyserDataArray | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);

  const callbacksRef = useRef({ onSpeechStart, onSpeechEnd, onVADMisfire });
  useEffect(() => {
    callbacksRef.current = { onSpeechStart, onSpeechEnd, onVADMisfire };
  }, [onSpeechStart, onSpeechEnd, onVADMisfire]);

  const setupAudioAnalyser = useCallback((stream: MediaStream) => {
    if (!audioContextRef.current) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AudioCtx();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      analyser.smoothingTimeConstant = 0.4;

      const sourceNode = ctx.createMediaStreamSource(stream);
      sourceNode.connect(analyser);

      audioContextRef.current = ctx;
      analyserNodeRef.current = analyser;
      timeDomainArrayRef.current = new Uint8Array(analyser.frequencyBinCount);
    }
  }, []);

  const start = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
        },
      });

      mediaStreamRef.current = stream;
      setupAudioAnalyser(stream);
      setHasPermission(true);

      const vad = await MicVAD.new({
        stream,
        positiveSpeechThreshold: 0.8,
        negativeSpeechThreshold: 0.65,
        // 19 frames (~1.82 seconds total silence tolerance) gives 1 extra full second to think mid-sentence
        redemptionFrames: 19,
        onSpeechStart: () => {
          setIsSpeaking(true);
          callbacksRef.current.onSpeechStart?.();
        },
        onSpeechEnd: (audio: Float32Array) => {
          setIsSpeaking(false);
          const wavBlob = encodeWAV(audio, 16000);
          callbacksRef.current.onSpeechEnd?.(wavBlob);
        },
        onVADMisfire: () => {
          setIsSpeaking(false);
          callbacksRef.current.onVADMisfire?.();
        },
      });

      vadRef.current = vad;
      vad.start();
      setIsListening(true);
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Microphone access denied';
      setError(msg);
      setHasPermission(false);
    }
  }, [setupAudioAnalyser]);

  const pause = useCallback(() => {
    vadRef.current?.pause();
    setIsListening(false);
    setIsSpeaking(false);
  }, []);

  const resume = useCallback(() => {
    if (enabled && hasPermission && vadRef.current) {
      vadRef.current.start();
      setIsListening(true);
    }
  }, [enabled, hasPermission]);

  const getRMSVolume = useCallback((): number => {
    const analyser = analyserNodeRef.current;
    const data = timeDomainArrayRef.current;
    if (!analyser || !data) return 0;

    analyser.getByteTimeDomainData(data);

    let sum = 0;
    for (let i = 0; i < data.length; i++) {
      const normalized = (data[i] - 128) / 128;
      sum += normalized * normalized;
    }
    return Math.sqrt(sum / data.length);
  }, []);

  useEffect(() => {
    if (!vadRef.current || !hasPermission) return;
    if (enabled) {
      vadRef.current.start();
      setIsListening(true);
    } else {
      vadRef.current.pause();
      setIsListening(false);
      setIsSpeaking(false);
    }
  }, [enabled, hasPermission]);

  useEffect(() => {
    return () => {
      vadRef.current?.destroy();
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      }
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, []);

  return {
    isListening,
    isSpeaking,
    hasPermission,
    error,
    analyserNode: analyserNodeRef.current,
    start,
    pause,
    resume,
    getRMSVolume,
  };
}