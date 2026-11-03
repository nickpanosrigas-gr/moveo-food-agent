import React, { useState, useRef, useEffect } from 'react';
import {
  Terminal,
  ChevronUp,
  ChevronDown,
  Copy,
  Check,
  SlidersHorizontal,
} from 'lucide-react';
import type { FailureMode, TurnLog, MCPPayload } from '../types';

interface DeveloperConsoleProps {
  failureMode: FailureMode;
  onFailureModeChange: (mode: FailureMode) => void;
  turnLogs: TurnLog[];
  lastMcpPayload: MCPPayload | null;
  className?: string;
}

type ConsoleTab = 'turn_logs' | 'mcp_payloads';

const FAILURE_MODES: Array<{ value: FailureMode; label: string; short: string }> = [
  { value: 'none', label: 'none (Standard Execution)', short: 'none' },
  { value: 'kitchen_busy', label: 'kitchen_busy (MCP 503 Transient)', short: 'kitchen_busy' },
  { value: 'server_error', label: 'server_error (MCP 500 Failure)', short: 'server_error' },
  { value: 'schema_error', label: 'schema_error (JSON-RPC -32602)', short: 'schema_error' },
  { value: 'business_refusal', label: 'business_refusal (> €50.00 Limit)', short: 'business_refusal' },
];

export const DeveloperConsole: React.FC<DeveloperConsoleProps> = ({
  failureMode,
  onFailureModeChange,
  turnLogs,
  lastMcpPayload,
  className = '',
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(false);
  const [activeTab, setActiveTab] = useState<ConsoleTab>('turn_logs');
  const [copied, setCopied] = useState<boolean>(false);
  const [isDropdownOpen, setIsDropdownOpen] = useState<boolean>(false);

  const consoleContentRef = useRef<HTMLDivElement | null>(null);
  const dropdownRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (isExpanded && consoleContentRef.current) {
      consoleContentRef.current.scrollTop = consoleContentRef.current.scrollHeight;
    }
  }, [turnLogs, lastMcpPayload, isExpanded, activeTab]);

  // Close custom dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleCopyCurrentView = () => {
    const dataToCopy =
      activeTab === 'turn_logs'
        ? JSON.stringify(turnLogs, null, 2)
        : JSON.stringify(lastMcpPayload ?? { status: 'No MCP call recorded' }, null, 2);

    navigator.clipboard.writeText(dataToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const currentModeObj =
    FAILURE_MODES.find((m) => m.value === failureMode) || FAILURE_MODES[0];

  return (
    <div
      className={`relative w-full bg-zinc-900/95 backdrop-blur-md rounded-t-xl border border-b-0 border-white/15 text-white font-sans transition-all duration-300 z-30 select-none shadow-[0_-10px_30px_rgba(0,0,0,0.6)] ${
        isExpanded ? 'h-[320px]' : 'h-12'
      } flex flex-col ${className}`}
    >
      {/* TOP CONTROL BAR */}
      <div className="h-12 px-4 flex items-center justify-between border-b border-white/10 text-xs flex-shrink-0 gap-3">
        {/* Left: Collapse/Expand Toggle & Status Counters */}
        <div className="flex items-center gap-3 min-w-0">
          <button
            onClick={() => setIsExpanded((prev) => !prev)}
            className="flex items-center gap-2 text-white hover:text-zinc-300 transition-colors focus:outline-none cursor-pointer"
          >
            <div className="p-1.5 rounded-lg bg-white/10 border border-white/15">
              <Terminal className="w-3.5 h-3.5 text-white" />
            </div>
            <span className="font-bold tracking-wide text-xs text-white">
              Developer Console
            </span>
            {isExpanded ? (
              <ChevronDown className="w-4 h-4 text-zinc-300" />
            ) : (
              <ChevronUp className="w-4 h-4 text-zinc-300" />
            )}
          </button>

          <div className="hidden sm:flex items-center gap-2 text-[11px] text-zinc-300">
            <span className="px-2.5 py-0.5 rounded-md bg-white/5 border border-white/10">
              <strong className="font-bold text-white">Turns:</strong> {turnLogs.length}
            </span>
            <span className="px-2.5 py-0.5 rounded-md bg-white/5 border border-white/10">
              <strong className="font-bold text-white">MCP:</strong>{' '}
              {lastMcpPayload ? 'Submitted' : 'Standby'}
            </span>
          </div>
        </div>

        {/* Right: Custom Cross-Browser Dropdown Menu (Drops UP when collapsed, Drops DOWN when expanded) */}
        <div ref={dropdownRef} className="relative flex-shrink-0">
          <button
            type="button"
            onClick={() => setIsDropdownOpen((prev) => !prev)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 border border-white/20 text-white text-xs transition-colors cursor-pointer"
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-zinc-300" />
            <span className="text-zinc-300">Mode:</span>
            <span className="font-bold text-white">{currentModeObj.short}</span>
            {isExpanded ? (
              <ChevronDown
                className={`w-3.5 h-3.5 text-zinc-300 transition-transform duration-200 ${
                  isDropdownOpen ? 'rotate-180' : ''
                }`}
              />
            ) : (
              <ChevronUp
                className={`w-3.5 h-3.5 text-zinc-300 transition-transform duration-200 ${
                  isDropdownOpen ? 'rotate-180' : ''
                }`}
              />
            )}
          </button>

          {isDropdownOpen && (
            <div
              className={`absolute right-0 w-64 bg-zinc-800 border border-white/20 rounded-xl shadow-2xl py-1.5 z-50 overflow-hidden ${
                isExpanded ? 'top-full mt-2' : 'bottom-full mb-2'
              }`}
            >
              {FAILURE_MODES.map((mode) => {
                const isSelected = mode.value === failureMode;
                return (
                  <button
                    key={mode.value}
                    type="button"
                    onClick={() => {
                      onFailureModeChange(mode.value);
                      setIsDropdownOpen(false);
                    }}
                    className={`w-full text-left px-3.5 py-2 text-xs flex items-center justify-between transition-colors cursor-pointer ${
                      isSelected
                        ? 'bg-white/15 text-white font-bold'
                        : 'text-zinc-200 hover:bg-white/10 hover:text-white font-normal'
                    }`}
                  >
                    <span className="truncate">{mode.label}</span>
                    {isSelected && <Check className="w-3.5 h-3.5 text-white flex-shrink-0 ml-2" />}
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* EXPANDABLE CONSOLE BODY */}
      {isExpanded && (
        <div className="flex-1 min-h-0 flex flex-col bg-zinc-900/90 rounded-b-none overflow-hidden">
          {/* TABS HEADER */}
          <div className="flex items-center justify-between px-4 bg-zinc-950/50 border-b border-white/10 text-xs">
            <div className="flex items-center gap-1">
              <button
                onClick={() => setActiveTab('turn_logs')}
                className={`px-3 py-2 border-b-2 text-xs transition-colors cursor-pointer ${
                  activeTab === 'turn_logs'
                    ? 'border-white text-white font-bold bg-white/5'
                    : 'border-transparent text-zinc-400 hover:text-white'
                }`}
              >
                Turn Logs ({turnLogs.length})
              </button>

              <button
                onClick={() => setActiveTab('mcp_payloads')}
                className={`px-3 py-2 border-b-2 text-xs transition-colors cursor-pointer ${
                  activeTab === 'mcp_payloads'
                    ? 'border-white text-white font-bold bg-white/5'
                    : 'border-transparent text-zinc-400 hover:text-white'
                }`}
              >
                MCP Payloads (submit_order)
              </button>
            </div>

            <button
              onClick={handleCopyCurrentView}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-white/5 hover:bg-white/10 border border-white/15 text-white text-[11px] transition-colors cursor-pointer"
            >
              {copied ? (
                <>
                  <Check className="w-3 h-3 text-white" />
                  <span className="font-bold">Copied</span>
                </>
              ) : (
                <>
                  <Copy className="w-3 h-3 text-zinc-300" />
                  <span>Copy JSON</span>
                </>
              )}
            </button>
          </div>

          {/* LOGS & PAYLOADS CONTENT */}
          <div
            ref={consoleContentRef}
            className="flex-1 min-h-0 overflow-y-auto p-4 text-xs leading-relaxed space-y-3 select-text"
          >
            {activeTab === 'turn_logs' ? (
              turnLogs.length === 0 ? (
                <div className="text-zinc-400 py-8 text-center select-none">
                  <p className="font-bold text-white">No turn logs recorded yet</p>
                  <p className="text-xs text-zinc-400 mt-1">
                    Speak or send a message to inspect turn events and cart snapshots.
                  </p>
                </div>
              ) : (
                turnLogs.map((log: TurnLog, idx: number) => (
                  <div
                    key={`log_${log.turn_id}_${idx}`}
                    className="border border-white/15 bg-zinc-800/60 rounded-lg p-3 text-white space-y-2"
                  >
                    <div className="flex items-center justify-between text-[11px] text-zinc-300 pb-1.5 border-b border-white/10">
                      <span>
                        <strong className="font-bold text-white">Turn #{log.turn_id}</strong> — {log.timestamp}
                      </span>
                      <span>
                        <strong className="font-bold text-white">Session:</strong> {log.session_id}
                      </span>
                    </div>

                    <div className="space-y-1 text-xs">
                      <div>
                        <strong className="font-bold text-white">User:</strong>{' '}
                        <span className="text-zinc-200">"{log.user_message}"</span>
                      </div>
                      <div>
                        <strong className="font-bold text-white">Agent:</strong>{' '}
                        <span className="text-zinc-200">"{log.agent_message}"</span>
                      </div>
                    </div>

                    {log.tool_calls && log.tool_calls.length > 0 && (
                      <div className="bg-zinc-950/60 p-2.5 rounded-md border border-white/10">
                        <strong className="font-bold text-white text-[11px] block mb-1">
                          Tool Calls:
                        </strong>
                        <pre className="font-sans text-[11px] text-zinc-200 whitespace-pre-wrap overflow-x-auto">
                          {JSON.stringify(log.tool_calls, null, 2)}
                        </pre>
                      </div>
                    )}

                    <details className="pt-1">
                      <summary className="text-[11px] font-bold text-white hover:text-zinc-300 cursor-pointer">
                        Cart Snapshot (Total: €{log.cart_state?.total_price?.toFixed(2) || '0.00'})
                      </summary>
                      <pre className="mt-1.5 font-sans text-[11px] text-zinc-200 bg-zinc-950/60 p-2.5 rounded-md overflow-x-auto border border-white/10">
                        {JSON.stringify(log.cart_state, null, 2)}
                      </pre>
                    </details>
                  </div>
                ))
              )
            ) : (
              /* TAB 2: MCP PAYLOADS */
              <div>
                {!lastMcpPayload ? (
                  <div className="text-zinc-400 py-8 text-center select-none">
                    <p className="font-bold text-white">No MCP submit_order calls yet</p>
                    <p className="text-xs text-zinc-400 mt-1">
                      Complete an order or trigger a failure mode to inspect the JSON-RPC payload.
                    </p>
                  </div>
                ) : (
                  <div className="border border-white/15 bg-zinc-800/60 rounded-lg p-3 text-white space-y-2.5">
                    <div className="flex items-center justify-between text-[11px] pb-1.5 border-b border-white/10">
                      <span>
                        <strong className="font-bold text-white">Method:</strong> {lastMcpPayload.name}
                      </span>
                      <span>
                        <strong className="font-bold text-white">Status:</strong>{' '}
                        {lastMcpPayload.result?.success ? 'Success (200 OK)' : 'Failed / Refused'}
                      </span>
                    </div>

                    {failureMode !== 'none' && (
                      <div className="p-2 rounded-md bg-white/5 border border-white/15 text-[11px] text-white">
                        <strong className="font-bold">Injected Header:</strong> X-Demo-Force-Failure: {failureMode}
                      </div>
                    )}

                    <div>
                      <strong className="font-bold text-white text-[11px] block mb-1">
                        Arguments (JSON-RPC Params):
                      </strong>
                      <pre className="font-sans text-[11px] text-zinc-200 bg-zinc-950/60 p-2.5 rounded-md border border-white/10 overflow-x-auto">
                        {JSON.stringify(lastMcpPayload.arguments, null, 2)}
                      </pre>
                    </div>

                    <div>
                      <strong className="font-bold text-white text-[11px] block mb-1">
                        Result (MCP Response):
                      </strong>
                      <pre className="font-sans text-[11px] text-zinc-200 bg-zinc-950/60 p-2.5 rounded-md border border-white/10 overflow-x-auto">
                        {JSON.stringify(lastMcpPayload.result, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};