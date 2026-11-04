export type OrbState = 'gray' | 'blue' | 'green' | 'orange' | 'purple';

export type FailureMode =
  | 'none'
  | 'kitchen_busy'
  | 'server_error'
  | 'schema_error'
  | 'business_refusal';

export type LLMProvider = 'mistral' | 'llama-server';
export type SpeechProvider = 'mistral' | 'speech_service';

export interface CartLineItem {
  line_id: string;
  item_id: string;
  name: string;
  quantity: number;
  options: Record<string, string>;
  extras: string[];
  unit_price: number;
  line_total: number;
}

export interface CartState {
  lines: CartLineItem[];
  total_item_count: number;
  total_price: number;
  exceeds_50_limit: boolean;
  special_instructions: string | null;
  is_submitted: boolean;
  last_submitted_order_id?: string | null;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'agent';
  text: string;
  timestamp: string;
}

export interface MCPPayload {
  name: string;
  arguments: {
    items: Array<{
      item_id: string;
      quantity: number;
      options?: Record<string, string>;
      extras?: string[];
    }>;
    special_instructions?: string | null;
  };
  result: {
    success: boolean;
    order_id?: string;
    total?: number;
    estimated_time?: string;
    error?: string;
  };
  forced_failure_mode?: string | null;
}

export interface TurnLog {
  timestamp: string;
  turn_id: number;
  session_id: string;
  user_message: string;
  agent_message: string;
  tool_calls: Array<{
    name: string;
    arguments: Record<string, any>;
    result: Record<string, any>;
  }>;
  cart_state: CartState;
  metadata?: Record<string, any>;
}

export type NDJSONEvent =
  | { type: 'transcription'; text: string; empty_speech?: boolean }
  | {
      type: 'turn_metadata';
      full_message: string;
      tool_calls: any[];
      cart_state: CartState;
      turn_log: TurnLog;
      mcp_payload: MCPPayload | null;
    }
  | {
      type: 'agent_chunk';
      chunk_index: number;
      total_chunks: number;
      is_final: boolean;
      text_chunk: string;
      audio_b64: string | null;
      sample_rate: number;
    }
  | { type: 'turn_complete' }
  | { type: 'error'; error: string };