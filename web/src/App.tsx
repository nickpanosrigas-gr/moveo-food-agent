import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { RotateCcw } from 'lucide-react';
import { useAgentStream } from './hooks/useAgentStream';
import { useVAD } from './hooks/useVAD';
import { useGlobalAudio } from './main';
import { VoiceOrb } from './components/VoiceOrb';
import { OrderTicketHUD } from './components/OrderTicketHUD';
import { ChatPanel } from './components/ChatPanel';
import { DeveloperConsole } from './components/DeveloperConsole';
import type { OrbState } from './types';

const FULLSCREEN_BACKGROUNDS: Record<OrbState, { primary: string; secondary: string; base: string }> = {
  gray: {
    primary: 'rgba(107, 114, 128, 0.28)',
    secondary: 'rgba(75, 85, 99, 0.14)',
    base: '#090d16',
  },
  blue: {
    primary: 'rgba(59, 130, 246, 0.30)',
    secondary: 'rgba(29, 78, 216, 0.16)',
    base: '#040b1e',
  },
  green: {
    primary: 'rgba(16, 185, 129, 0.32)',
    secondary: 'rgba(5, 150, 105, 0.18)',
    base: '#021812',
  },
  orange: {
    primary: 'rgba(245, 158, 11, 0.30)',
    secondary: 'rgba(180, 83, 9, 0.16)',
    base: '#180d02',
  },
  purple: {
    primary: 'rgba(139, 92, 246, 0.34)',
    secondary: 'rgba(109, 40, 217, 0.18)',
    base: '#0f0520',
  },
};

export default function App() {
  const { resumeAudioContext } = useGlobalAudio();

  const {
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
  } = useAgentStream({
    sessionId: 'sess_default',
    initialFailureMode: 'none',
  });

  const [liveVolume, setLiveVolume] = useState<number>(0);
  const [isChatInputFocused, setIsChatInputFocused] = useState<boolean>(false);
  const [hasGreeted, setHasGreeted] = useState<boolean>(false);

  // When true, speech emitted while stopping the orb is discarded and never sent to the backend
  const discardRecordingRef = useRef<boolean>(false);

  const vad = useVAD({
    enabled: hasGreeted && (orbState === 'blue' || orbState === 'green') && !isProcessing,
    onSpeechStart: () => {
      if (orbState === 'blue') {
        setOrbState('green');
      }
    },
    onSpeechEnd: (audioBlob: Blob) => {
      if (discardRecordingRef.current) {
        discardRecordingRef.current = false;
        return;
      }
      sendVoiceMessage(audioBlob);
    },
    onVADMisfire: () => {
      if (orbState === 'green' && !isChatInputFocused) {
        setOrbState(vad.hasPermission && hasGreeted ? 'blue' : 'gray');
      }
    },
  });

  useEffect(() => {
    setMicReady(vad.hasPermission && hasGreeted && orbState !== 'gray');
  }, [vad.hasPermission, hasGreeted, orbState, setMicReady]);

  // Orb click behavior:
  // - In blue or green: click stops recording, drops audio, and reverts to gray
  // - In gray: click starts/resumes microphone
  const handleOrbClick = useCallback(async () => {
    if (orbState === 'orange' || orbState === 'purple') {
      return;
    }

    if (orbState === 'blue' || orbState === 'green') {
      discardRecordingRef.current = true;
      vad.pause();
      setOrbState('gray');
      setMicReady(false);
      return;
    }

    // Gray state: wake up the orb
    await resumeAudioContext();

    if (!hasGreeted) {
      await vad.start();
      setHasGreeted(true);
      setMicReady(true);
      await playGreeting();
    } else {
      if (!vad.hasPermission) {
        await vad.start();
      } else {
        vad.resume();
      }
      discardRecordingRef.current = false;
      setMicReady(true);
      setOrbState('blue');
    }
  }, [orbState, hasGreeted, vad, playGreeting, setOrbState, setMicReady, resumeAudioContext]);

  const handleChatFocusChange = useCallback(
    (focused: boolean) => {
      setIsChatInputFocused(focused);
      if (!focused && orbState === 'green' && !vad.isSpeaking) {
        setOrbState(vad.hasPermission && hasGreeted ? 'blue' : 'gray');
      }
    },
    [orbState, vad.isSpeaking, vad.hasPermission, hasGreeted, setOrbState]
  );

  useEffect(() => {
    let animId: number;

    const pollVolume = () => {
      if (orbState === 'green') {
        setLiveVolume(vad.getRMSVolume());
      } else if (orbState === 'purple') {
        setLiveVolume(audioPlayer.getRMSVolume());
      } else {
        setLiveVolume(0);
      }
      animId = requestAnimationFrame(pollVolume);
    };

    animId = requestAnimationFrame(pollVolume);
    return () => cancelAnimationFrame(animId);
  }, [orbState, vad, audioPlayer]);

  const effectiveVisualState: OrbState =
    isChatInputFocused && (orbState === 'blue' || orbState === 'gray') ? 'green' : orbState;

  const activeBg = useMemo(
    () => FULLSCREEN_BACKGROUNDS[effectiveVisualState],
    [effectiveVisualState]
  );

  return (
    <div
      className="relative w-screen h-screen flex flex-col text-slate-100 overflow-hidden select-none transition-colors duration-700"
      style={{ backgroundColor: activeBg.base }}
    >
      {/* Full-Screen State-Reactive Ambient Background */}
      <div
        className="absolute inset-0 pointer-events-none transition-all duration-700 ease-out"
        style={{
          background: `
            radial-gradient(circle 110vw at 50% 50%, ${activeBg.primary} 0%, ${activeBg.secondary} 55%, ${activeBg.base} 100%)
          `,
        }}
      />

      {/* Micro-grain CSS Texture Overlay */}
      <div className="micro-grain-texture" />

      {/* 3-Column Split Master Stage */}
      <main className="relative z-10 flex-1 min-h-0 flex flex-row overflow-hidden">
        {/* LEFT HUD: Delivery Ticket */}
        <OrderTicketHUD cartState={cartState} />

        {/* CENTER STAGE: Reset Button (Top), VoiceOrb (Center), DeveloperConsole (Bottom) */}
        <section className="flex-1 min-w-0 h-full flex flex-col items-center justify-between pt-6 px-4 relative">
          {/* Top Middle: Reset Session Button (Styled identical to Developer Console Mode selector) */}
          <div className="z-20 flex-shrink-0">
            <button
              onClick={resetSession}
              type="button"
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-white/20 text-white font-normal font-sans text-xs transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5 text-zinc-300" />
              <span>RESET SESSION</span>
            </button>
          </div>

          {/* Center: 5-State Voice Orb */}
          <div className="flex-1 flex items-center justify-center w-full min-h-0 my-2">
            <VoiceOrb
              state={effectiveVisualState}
              rmsVolume={liveVolume}
              getByteTimeDomainData={
                orbState === 'purple'
                  ? audioPlayer.getByteTimeDomainData
                  : undefined
              }
              getByteFrequencyData={
                orbState === 'purple'
                  ? audioPlayer.getByteFrequencyData
                  : undefined
              }
              onOrbClick={handleOrbClick}
            />
          </div>

          {/* Bottom Middle: Developer Console */}
          <div className="w-full max-w-3xl flex-shrink-0">
            <DeveloperConsole
              failureMode={failureMode}
              onFailureModeChange={setFailureMode}
              turnLogs={turnLogs}
              lastMcpPayload={lastMcpPayload}
            />
          </div>
        </section>

        {/* RIGHT PANEL: Chat Bubbles & State-Colored Input Bar */}
        <ChatPanel
          messages={messages}
          streamingAgentMessage={streamingAgentMessage}
          orbState={effectiveVisualState}
          isProcessing={isProcessing}
          onSendMessage={sendTextMessage}
          onInputFocusChange={handleChatFocusChange}
        />
      </main>
    </div>
  );
}