import React, { useRef, useEffect, useMemo } from 'react';
import type { OrbState } from '../types';

interface VoiceOrbProps {
  state: OrbState;
  isLoading?: boolean;
  rmsVolume?: number;
  getByteTimeDomainData?: () => Uint8Array | null;
  getByteFrequencyData?: () => Uint8Array | null;
  onOrbClick?: () => void;
  className?: string;
}

interface ColorConfig {
  core: string;
  glow: string;
  ambient: string;
  waveform: string;
  border: string;
}

const STATE_COLORS: Record<OrbState, ColorConfig> = {
  gray: {
    core: '#6B7280',
    glow: 'rgba(107, 114, 128, 0.15)',
    ambient: 'rgba(107, 114, 128, 0.05)',
    waveform: 'rgba(156, 163, 175, 0.4)',
    border: 'rgba(156, 163, 175, 0.6)',
  },
  blue: {
    core: '#3B82F6',
    glow: 'rgba(59, 130, 246, 0.18)',
    ambient: 'rgba(59, 130, 246, 0.06)',
    waveform: 'rgba(96, 165, 250, 0.85)',
    border: 'rgba(59, 130, 246, 0.75)',
  },
  green: {
    core: '#10B981',
    glow: 'rgba(16, 185, 129, 0.25)',
    ambient: 'rgba(16, 185, 129, 0.08)',
    waveform: 'rgba(52, 211, 153, 0.95)',
    border: 'rgba(16, 185, 129, 0.85)',
  },
  orange: {
    core: '#F59E0B',
    glow: 'rgba(245, 158, 11, 0.22)',
    ambient: 'rgba(245, 158, 11, 0.07)',
    waveform: 'rgba(251, 191, 36, 0.9)',
    border: 'rgba(245, 158, 11, 0.8)',
  },
  purple: {
    core: '#8B5CF6',
    glow: 'rgba(139, 92, 246, 0.25)',
    ambient: 'rgba(139, 92, 246, 0.08)',
    waveform: 'rgba(167, 139, 250, 0.95)',
    border: 'rgba(139, 92, 246, 0.85)',
  },
};

const STATE_CAPTIONS: Record<OrbState, string> = {
  gray: 'Click the orb or enable microphone to start ordering.',
  blue: 'Listening... Click the orb to stop listening.',
  green: 'Listening to your order... Click to cancel.',
  orange: 'Thinking...',
  purple: 'Responding...',
};

export const VoiceOrb: React.FC<VoiceOrbProps> = ({
  state,
  isLoading = false,
  rmsVolume = 0,
  getByteTimeDomainData,
  getByteFrequencyData,
  onOrbClick,
  className = '',
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const isClickable = !isLoading && (state === 'blue' || state === 'green' || state === 'gray');

  const propsRef = useRef({
    state,
    isLoading,
    rmsVolume,
    getByteTimeDomainData,
    getByteFrequencyData,
  });

  useEffect(() => {
    propsRef.current = {
      state,
      isLoading,
      rmsVolume,
      getByteTimeDomainData,
      getByteFrequencyData,
    };
  }, [state, isLoading, rmsVolume, getByteTimeDomainData, getByteFrequencyData]);

  const activeColors = useMemo(() => STATE_COLORS[state], [state]);
  const activeCaption = useMemo(() => {
    if (isLoading && state === 'gray') {
      return 'Initializing AI models... Please wait.';
    }
    return STATE_CAPTIONS[state];
  }, [state, isLoading]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let startTime: number | null = null;
    let smoothedRMS = 0;

    const render = (timestamp: number) => {
      if (startTime === null) startTime = timestamp;
      const t = (timestamp - startTime) / 1000;

      const {
        state: currentState,
        isLoading: currentIsLoading,
        rmsVolume: rawRMS,
        getByteTimeDomainData: getTimeDomain,
        getByteFrequencyData: getFrequency,
      } = propsRef.current;

      const colors = STATE_COLORS[currentState];

      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;

      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr;
        canvas.height = height * dpr;
      }

      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;
      const baseRadius = Math.min(width, height) * 0.28;

      smoothedRMS += (rawRMS - smoothedRMS) * 0.15;

      let pulseOffset = 0;
      if (currentState === 'gray') {
        if (currentIsLoading) {
          // Visible, smooth breathing enlargement and shrinking (12% base radius swing)
          pulseOffset = Math.sin(t * 2.2) * (baseRadius * 0.12);
        } else {
          const cycle = t % 2.5;
          if (cycle < 0.5) {
            pulseOffset = Math.sin((cycle / 0.25) * Math.PI * 2) * 4;
          }
        }
      } else if (currentState === 'blue') {
        pulseOffset = Math.sin(t * 1.8) * 5;
      } else if (currentState === 'green') {
        pulseOffset = Math.sin(t * 4) * 2;
      } else if (currentState === 'orange') {
        pulseOffset = Math.sin(t * 3) * 3;
      } else if (currentState === 'purple') {
        pulseOffset = Math.sin(t * 2.5) * 4;
      }

      const volumeScale = currentState === 'green' ? 65 : 35;
      const R = Math.max(
        baseRadius * 0.7,
        baseRadius + smoothedRMS * volumeScale + pulseOffset
      );

      const glowGradient = ctx.createRadialGradient(cx, cy, R * 0.1, cx, cy, R * 1.8);
      glowGradient.addColorStop(0, colors.glow);
      glowGradient.addColorStop(0.6, colors.ambient);
      glowGradient.addColorStop(1, 'rgba(0, 0, 0, 0)');

      ctx.fillStyle = glowGradient;
      ctx.beginPath();
      ctx.arc(cx, cy, R * 1.8, 0, Math.PI * 2);
      ctx.fill();

      const coreGradient = ctx.createRadialGradient(
        cx - R * 0.2,
        cy - R * 0.2,
        R * 0.1,
        cx,
        cy,
        R
      );
      coreGradient.addColorStop(0, `${colors.core}33`);
      coreGradient.addColorStop(0.8, `${colors.core}11`);
      coreGradient.addColorStop(1, 'rgba(15, 23, 42, 0.4)');

      ctx.fillStyle = coreGradient;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.fill();

      ctx.lineWidth = 1.8;
      ctx.strokeStyle = colors.border;
      ctx.shadowColor = colors.core;
      ctx.shadowBlur = currentState === 'gray' ? 6 : 14;
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.stroke();

      ctx.shadowBlur = 0;

      if (currentState === 'gray') {
        // Resting center line that resizes with the breathing orb
        ctx.beginPath();
        ctx.strokeStyle = colors.waveform;
        ctx.lineWidth = 2;
        ctx.moveTo(cx - R, cy);
        ctx.lineTo(cx + R, cy);
        ctx.stroke();

        if (currentIsLoading) {
          // Orbital arc orbiting outside the orb at a clean, consistent 12px gap
          const loadingGap = 12;
          const loadingRadius = R + loadingGap;
          const spinSpeed = 2.6;
          const startAngle = (t * spinSpeed) % (Math.PI * 2);
          const arcLength = Math.PI * 0.65;

          ctx.save();
          ctx.beginPath();
          ctx.lineWidth = 2.4;
          ctx.lineCap = 'round';
          ctx.strokeStyle = colors.border;
          ctx.shadowColor = colors.core;
          ctx.shadowBlur = 8;
          ctx.arc(cx, cy, loadingRadius, startAngle, startAngle + arcLength);
          ctx.stroke();
          ctx.restore();
        }
      } else if (currentState === 'orange') {
        const dotSpacing = 22;
        ctx.fillStyle = colors.waveform;

        const indices = [-1, 0, 1];
        indices.forEach((i) => {
          const dotX = cx + i * dotSpacing;
          const dotY = cy + Math.sin(6 * t + 0.8 * i) * 12;

          ctx.beginPath();
          ctx.arc(dotX, dotY, 4.5, 0, Math.PI * 2);
          ctx.fill();
        });
      } else {
        ctx.beginPath();
        ctx.strokeStyle = colors.waveform;
        ctx.lineWidth = 2.2;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        const step = 2;
        const startX = cx - R;
        const endX = cx + R;

        const freqData = currentState === 'purple' ? getFrequency?.() : null;
        const timeData = currentState === 'green' ? getTimeDomain?.() : null;

        for (let x = startX; x <= endX; x += step) {
          const u = (x - startX) / (2 * R);
          const envelope = Math.sin(u * Math.PI);

          let displacement = 0;

          if (currentState === 'blue') {
            displacement =
              Math.sin(u * 10 + t * 3.5) * 9 +
              Math.sin(u * 18 - t * 2) * 5;
          } else if (currentState === 'green') {
            if (timeData && timeData.length > 0) {
              const dataIdx = Math.floor(u * (timeData.length - 1));
              const normalizedSample = (timeData[dataIdx] - 128) / 128;
              displacement = normalizedSample * 55;
            } else {
              displacement =
                (Math.sin(u * 16 + t * 8) * 18 +
                  Math.sin(u * 32 - t * 12) * 10) *
                (smoothedRMS * 2.5 + 0.2);
            }
          } else if (currentState === 'purple') {
            if (freqData && freqData.length > 0) {
              const dataIdx = Math.floor(u * Math.min(freqData.length, 128));
              const freqMagnitude = freqData[dataIdx] / 255;
              displacement =
                freqMagnitude * 45 * Math.sin(u * 20 + t * 10);
            } else {
              displacement =
                (Math.sin(u * 14 + t * 6) * 16 +
                  Math.sin(u * 28 - t * 10) * 8) *
                (smoothedRMS * 2.2 + 0.3);
            }
          }

          const y = cy + displacement * envelope;

          if (x === startX) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }
        }

        ctx.stroke();
      }

      ctx.restore();
      animationFrameId = requestAnimationFrame(render);
    };

    animationFrameId = requestAnimationFrame(render);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <div
      className={`relative flex flex-col items-center justify-center select-none ${className}`}
    >
      <div
        onClick={isClickable ? onOrbClick : undefined}
        role="button"
        tabIndex={isClickable ? 0 : -1}
        aria-label={`Voice Orb - ${isLoading ? 'loading' : state} state`}
        className={`relative flex items-center justify-center transition-transform duration-300 outline-none ${
          isClickable
            ? 'cursor-pointer hover:scale-[1.02] active:scale-[0.98]'
            : 'cursor-default'
        } ${state === 'gray' && !isLoading ? 'animate-pulse' : ''}`}
      >
        <canvas
          ref={canvasRef}
          className="w-[280px] h-[280px] sm:w-[320px] sm:h-[320px] md:w-[380px] md:h-[380px]"
        />
      </div>

      <p
        className="mt-6 text-sm md:text-base font-medium tracking-wide text-center transition-colors duration-500 max-w-sm px-4"
        style={{ color: activeColors.core }}
      >
        {activeCaption}
      </p>
    </div>
  );
};