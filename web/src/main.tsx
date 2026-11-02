import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

interface GlobalAudioContextValue {
  audioContext: AudioContext | null;
  resumeAudioContext: () => Promise<void>;
  isUnlocked: boolean;
}

const GlobalAudioContext = createContext<GlobalAudioContextValue>({
  audioContext: null,
  resumeAudioContext: async () => {},
  isUnlocked: false,
});

export const useGlobalAudio = () => useContext(GlobalAudioContext);

export const AudioContextProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isUnlocked, setIsUnlocked] = useState(false);
  const audioContextRef = useRef<AudioContext | null>(null);

  const getOrCreateAudioContext = useCallback((): AudioContext => {
    if (!audioContextRef.current) {
      const AudioCtx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      audioContextRef.current = new AudioCtx({ latencyHint: 'interactive' });
    }
    return audioContextRef.current;
  }, []);

  const resumeAudioContext = useCallback(async () => {
    const ctx = getOrCreateAudioContext();
    if (ctx.state === 'suspended') {
      await ctx.resume();
    }
    setIsUnlocked(true);
  }, [getOrCreateAudioContext]);

  // Unlock audio automatically on the first user interaction (click, tap, or keypress)
  useEffect(() => {
    const handleFirstGesture = async () => {
      try {
        await resumeAudioContext();
      } catch (err) {
        console.warn('[AudioContext] Auto-resume on gesture deferred:', err);
      } finally {
        window.removeEventListener('pointerdown', handleFirstGesture);
        window.removeEventListener('keydown', handleFirstGesture);
        window.removeEventListener('touchstart', handleFirstGesture);
      }
    };

    window.addEventListener('pointerdown', handleFirstGesture);
    window.addEventListener('keydown', handleFirstGesture);
    window.addEventListener('touchstart', handleFirstGesture);

    return () => {
      window.removeEventListener('pointerdown', handleFirstGesture);
      window.removeEventListener('keydown', handleFirstGesture);
      window.removeEventListener('touchstart', handleFirstGesture);
      if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, [resumeAudioContext]);

  return (
    <GlobalAudioContext.Provider
      value={{
        audioContext: audioContextRef.current,
        resumeAudioContext,
        isUnlocked,
      }}
    >
      {children}
    </GlobalAudioContext.Provider>
  );
};

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Failed to find root element with id "root".');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <AudioContextProvider>
      <App />
    </AudioContextProvider>
  </React.StrictMode>
);