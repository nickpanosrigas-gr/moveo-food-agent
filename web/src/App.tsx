import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
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
    llmProvider,
    speechProvider,
    orbState,
    setOrbState,
    setMicReady,
    isProcessing,
    isModelsLoaded,
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
  const [isWaitingForModels, setIsWaitingForModels] = useState<boolean>(false);

  const discardRecordingRef = useRef<boolean>(false);

  const vad = useVAD({
    enabled: hasGreeted && (orbState === 'blue' || orbState === 'green') && !isProcessing && !isWaitingForModels,
    onSpeechStart: () => {
      setOrbState('green');
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

  const wakeUpOrb = useCallback(async () => {
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
  }, [resumeAudioContext, hasGreeted, vad, playGreeting, setMicReady, setOrbState]);

  useEffect(() => {
    if (isModelsLoaded && isWaitingForModels) {
      setIsWaitingForModels(false);
      wakeUpOrb();
    }
  }, [isModelsLoaded, isWaitingForModels, wakeUpOrb]);

  const handleOrbClick = useCallback(async () => {
    if (isWaitingForModels || orbState === 'orange' || orbState === 'purple') {
      return;
    }

    if (orbState === 'blue' || orbState === 'green') {
      discardRecordingRef.current = true;
      vad.pause();
      setOrbState('gray');
      setMicReady(false);
      return;
    }

    if (!isModelsLoaded) {
      setIsWaitingForModels(true);
      return;
    }

    await wakeUpOrb();
  }, [isWaitingForModels, orbState, isModelsLoaded, vad, setOrbState, setMicReady, wakeUpOrb]);

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
      if (effectiveVisualState === 'green') {
        setLiveVolume(vad.getRMSVolume());
      } else if (effectiveVisualState === 'purple') {
        setLiveVolume(audioPlayer.getRMSVolume());
      } else {
        setLiveVolume(0);
      }
      animId = requestAnimationFrame(pollVolume);
    };

    animId = requestAnimationFrame(pollVolume);
    return () => cancelAnimationFrame(animId);
  }, [vad, audioPlayer]);

  // Directly derive green visual state from active speech detection
  const effectiveVisualState: OrbState =
    isChatInputFocused && (orbState === 'blue' || orbState === 'gray')
      ? 'green'
      : vad.isSpeaking && (orbState === 'blue' || orbState === 'green')
      ? 'green'
      : orbState;

  const activeBg = useMemo(
    () => FULLSCREEN_BACKGROUNDS[effectiveVisualState],
    [effectiveVisualState]
  );

  return (
    <div
      className="relative w-screen h-screen flex flex-col text-slate-100 overflow-hidden select-none transition-colors duration-700"
      style={{ backgroundColor: activeBg.base }}
    >
      <div
        className="absolute inset-0 pointer-events-none transition-all duration-700 ease-out"
        style={{
          background: `
            radial-gradient(circle 110vw at 50% 50%, ${activeBg.primary} 0%, ${activeBg.secondary} 55%, ${activeBg.base} 100%)
          `,
        }}
      />

      <div className="micro-grain-texture" />

      <main className="relative z-10 flex-1 min-h-0 flex flex-row overflow-hidden">
        <OrderTicketHUD cartState={cartState} />

        <section className="flex-1 min-w-0 h-full flex flex-col items-center justify-between pt-6 px-4 relative">
          <div className="z-20 flex-shrink-0">
            <button
              onClick={() => {
                setIsWaitingForModels(false);
                resetSession();
              }}
              type="button"
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-white/20 text-white font-normal font-sans text-xs transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5 text-zinc-300" />
              <span>RESET SESSION</span>
            </button>
          </div>

          <div className="flex-1 flex items-center justify-center w-full min-h-0 my-2">
            <VoiceOrb
              state={effectiveVisualState}
              isLoading={isWaitingForModels}
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

          <div className="w-full max-w-3xl flex-shrink-0">
            <DeveloperConsole
              failureMode={failureMode}
              onFailureModeChange={setFailureMode}
              turnLogs={turnLogs}
              lastMcpPayload={lastMcpPayload}
              llmProvider={llmProvider}
              speechProvider={speechProvider}
            />
          </div>
        </section>

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