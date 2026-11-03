import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Bot, User } from 'lucide-react';
import type { ChatMessage, OrbState } from '../types';

interface ChatPanelProps {
  messages: ChatMessage[];
  streamingAgentMessage?: string;
  orbState: OrbState;
  isProcessing: boolean;
  onSendMessage: (text: string) => void | Promise<void>;
  onInputFocusChange?: (isFocused: boolean) => void;
  className?: string;
}

const INPUT_STATE_STYLES: Record<
  OrbState,
  {
    input: string;
    button: string;
    loader: string;
  }
> = {
  gray: {
    input:
      'bg-gray-500/15 border-gray-500/40 text-gray-100 placeholder-gray-400/70 focus:border-emerald-400/70 focus:ring-1 focus:ring-emerald-400/40',
    button:
      'bg-gray-500/30 border-gray-400/40 text-gray-200 hover:bg-gray-500/50 hover:text-white',
    loader: 'text-gray-300',
  },
  blue: {
    input:
      'bg-blue-500/15 border-blue-500/40 text-blue-100 placeholder-blue-300/60 focus:border-blue-400/70 focus:ring-1 focus:ring-blue-400/40',
    button:
      'bg-blue-600/40 border-blue-400/50 text-blue-200 hover:bg-blue-500/60 hover:text-white shadow-[0_0_12px_rgba(59,130,246,0.3)]',
    loader: 'text-blue-400',
  },
  green: {
    input:
      'bg-emerald-500/15 border-emerald-500/50 text-emerald-100 placeholder-emerald-300/60 ring-1 ring-emerald-400/30',
    button:
      'bg-emerald-600/50 border-emerald-400/60 text-emerald-100 hover:bg-emerald-500/70 hover:text-white shadow-[0_0_12px_rgba(16,185,129,0.35)]',
    loader: 'text-emerald-400',
  },
  orange: {
    input:
      'bg-amber-500/15 border-amber-500/40 text-amber-100 placeholder-amber-300/70 cursor-not-allowed',
    button:
      'bg-amber-600/30 border-amber-500/40 text-amber-300 cursor-not-allowed shadow-[0_0_12px_rgba(245,158,11,0.25)]',
    loader: 'text-amber-400',
  },
  purple: {
    input:
      'bg-purple-500/15 border-purple-500/40 text-purple-100 placeholder-purple-300/70 cursor-not-allowed',
    button:
      'bg-purple-600/30 border-purple-500/40 text-purple-300 cursor-not-allowed shadow-[0_0_12px_rgba(139,92,246,0.25)]',
    loader: 'text-purple-400',
  },
};

export const ChatPanel: React.FC<ChatPanelProps> = ({
  messages,
  streamingAgentMessage = '',
  orbState,
  isProcessing,
  onSendMessage,
  onInputFocusChange,
  className = '',
}) => {
  const [inputText, setInputText] = useState('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const isInputLocked = orbState === 'orange' || orbState === 'purple' || isProcessing;
  const activeStyle = INPUT_STATE_STYLES[orbState];

  const scrollToBottom = (behavior: ScrollBehavior = 'smooth') => {
    messagesEndRef.current?.scrollIntoView({ behavior });
  };

  useEffect(() => {
    scrollToBottom('smooth');
  }, [messages, streamingAgentMessage, isInputLocked]);

  const handleFocus = () => {
    onInputFocusChange?.(true);
  };

  const handleBlur = () => {
    onInputFocusChange?.(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputText.trim();
    if (!trimmed || isInputLocked) return;

    onSendMessage(trimmed);
    setInputText('');
    inputRef.current?.blur();
    onInputFocusChange?.(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const formatTimestamp = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  return (
    <aside
      className={`w-full lg:w-[420px] flex flex-col h-full bg-transparent ${className}`}
    >
      {/* Message Feed */}
      <div className="flex-1 overflow-y-auto px-5 py-6 space-y-4 scroll-smooth">
        {messages.map((msg) => {
          const isUser = msg.role === 'user';

          return (
            <div
              key={msg.id}
              className={`flex flex-col ${isUser ? 'items-end' : 'items-start'} transition-all`}
            >
              <div className="flex items-end gap-2 max-w-[85%]">
                {!isUser && (
                  <div className="w-6 h-6 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-purple-300 mb-1 flex-shrink-0">
                    <Bot className="w-3.5 h-3.5" />
                  </div>
                )}

                <div
                  className={`px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm transition-all ${
                    isUser
                      ? 'bg-emerald-600/30 border border-emerald-500/40 text-emerald-100 rounded-br-xs'
                      : 'bg-purple-600/30 border border-purple-500/40 text-purple-100 rounded-bl-xs'
                  }`}
                >
                  <p className="whitespace-pre-wrap break-words">{msg.text}</p>
                </div>

                {isUser && (
                  <div className="w-6 h-6 rounded-full bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-300 mb-1 flex-shrink-0">
                    <User className="w-3.5 h-3.5" />
                  </div>
                )}
              </div>

              <span className="text-[10px] text-slate-500 mt-1 px-1">
                {formatTimestamp(msg.timestamp)}
              </span>
            </div>
          );
        })}

        {/* Live Synchronized Agent Streaming Bubble */}
        {streamingAgentMessage && (
          <div className="flex flex-col items-start transition-all">
            <div className="flex items-end gap-2 max-w-[85%]">
              <div className="w-6 h-6 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center text-purple-300 mb-1 flex-shrink-0 animate-pulse">
                <Bot className="w-3.5 h-3.5" />
              </div>

              <div className="px-4 py-3 rounded-2xl text-sm leading-relaxed shadow-sm bg-purple-600/30 border border-purple-500/40 text-purple-100 rounded-bl-xs">
                <p className="whitespace-pre-wrap break-words">{streamingAgentMessage}</p>
                <span className="inline-block w-1.5 h-3.5 ml-1 bg-purple-300 animate-pulse align-middle" />
              </div>
            </div>
            <span className="text-[10px] text-purple-400/70 mt-1 px-8">Speaking...</span>
          </div>
        )}

        {/* Agent Thinking Indicator */}
          {orbState === 'orange' && !streamingAgentMessage && (
            <div className="flex flex-col items-start transition-all">
              <div className="flex items-end gap-2 max-w-[85%]">
                <div className="w-6 h-6 rounded-full bg-amber-500/20 border border-amber-500/30 flex items-center justify-center text-amber-300 mb-1 flex-shrink-0 animate-pulse">
                  <Bot className="w-3.5 h-3.5" />
                </div>

                <div className="flex items-center gap-2.5 px-4 py-3 rounded-2xl rounded-bl-xs text-sm leading-relaxed shadow-sm bg-amber-500/10 border border-amber-500/30 text-amber-200">
                  <Loader2 className="w-4 h-4 animate-spin text-amber-400 flex-shrink-0" />
                  <span>Agent is thinking...</span>
                </div>
              </div>
            </div>
          )}

        <div ref={messagesEndRef} />
      </div>

      {/* State-Synchronized Input Bar */}
      <div className="p-4 bg-transparent">
        <form onSubmit={handleSubmit} className="relative flex items-center">
          <input
            ref={inputRef}
            type="text"
            value={inputText}
            onFocus={handleFocus}
            onBlur={handleBlur}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isInputLocked}
            placeholder={
              isInputLocked
                ? orbState === 'orange'
                  ? 'Agent is thinking...'
                  : 'Agent is speaking...'
                : 'Type your order or request...'
            }
            className={`w-full px-4 py-3 pr-12 rounded-xl text-sm border backdrop-blur-md transition-all duration-300 outline-none ${activeStyle.input}`}
          />

          <button
            type="submit"
            disabled={isInputLocked || !inputText.trim()}
            aria-label="Send message"
            className={`absolute right-2 p-2 rounded-lg border transition-all duration-300 flex items-center justify-center ${
              !isInputLocked && !inputText.trim()
                ? `${activeStyle.button} opacity-60 cursor-default`
                : `${activeStyle.button} cursor-pointer`
            }`}
          >
            {isInputLocked ? (
              <Loader2 className={`w-4 h-4 animate-spin ${activeStyle.loader}`} />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </form>
      </div>
    </aside>
  );
};