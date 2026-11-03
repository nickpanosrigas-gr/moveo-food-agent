import React, { useRef, useState, useEffect, useCallback } from 'react';
import { CheckCircle2, AlertTriangle } from 'lucide-react';
import type { CartState, CartLineItem } from '../types';

interface OrderTicketHUDProps {
  cartState: CartState;
  className?: string;
}

const EXTRA_LABELS: Record<string, string> = {
  cheese: 'Cheese',
  bacon: 'Bacon',
  avocado: 'Avocado',
  extra_patty: 'Extra Patty',
  jalapeños: 'Jalapeños',
  extra_cheese: 'Extra Cheese',
  olives: 'Olives',
  mushrooms: 'Mushrooms',
  pepperoni: 'Pepperoni',
  truffle_oil: 'Truffle Oil',
  parmesan: 'Parmesan',
  cheese_sauce: 'Cheese Sauce',
  ranch_dip: 'Ranch Dip',
  spicy_mayo: 'Spicy Mayo',
  whipped_cream: 'Whipped Cream',
  cherry_on_top: 'Cherry On Top',
};

const OPTION_VALUE_LABELS: Record<string, string> = {
  regular: 'Regular',
  large: 'Large',
  small: 'Small',
  medium: 'Medium',
  beef: 'Beef Patty',
  chicken: 'Chicken Patty',
  veggie: 'Veggie Patty',
  mild: 'Mild Spice',
  hot: 'Hot Spice',
  extra_hot: 'Extra Hot',
  thin: 'Thin Crust',
  thick: 'Thick Crust',
  cola: 'Cola',
  diet_cola: 'Diet Cola',
  lemon_lime: 'Lemon-Lime',
  orange: 'Orange',
  vanilla: 'Vanilla',
  chocolate: 'Chocolate',
  strawberry: 'Strawberry',
  oreo: 'Oreo',
};

function formatOptionLine(key: string, value: string): string {
  if (key === 'spice_level' && value === 'medium') return 'Medium Spice';
  if (key === 'crust' && value === 'regular') return 'Regular Crust';
  return OPTION_VALUE_LABELS[value] || value.replace(/_/g, ' ');
}

function formatExtraLine(extraId: string): string {
  return EXTRA_LABELS[extraId] || extraId.replace(/_/g, ' ');
}

/**
 * Full-width thermal dashed divider that spans right up to the end of the Total price column.
 */
const FullWidthDashes: React.FC<{ spaced?: boolean }> = ({ spaced = false }) => (
  <div className="w-full overflow-hidden whitespace-nowrap text-black/85 select-none text-[14px] leading-none tracking-normal">
    {spaced
      ? '- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -'
      : '------------------------------------------------------------------------'}
  </div>
);

export const OrderTicketHUD: React.FC<OrderTicketHUDProps> = ({
  cartState,
  className = '',
}) => {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const [canScrollDown, setCanScrollDown] = useState(false);
  const [canScrollUp, setCanScrollUp] = useState(false);

  const checkScroll = useCallback(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const { scrollTop, scrollHeight, clientHeight } = el;
    setCanScrollUp(scrollTop > 4);
    setCanScrollDown(scrollTop + clientHeight < scrollHeight - 4);
  }, []);

  useEffect(() => {
    checkScroll();
    const el = scrollContainerRef.current;
    if (el) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
      setTimeout(checkScroll, 300);
    }
  }, [cartState.lines, cartState.special_instructions, checkScroll]);

  useEffect(() => {
    window.addEventListener('resize', checkScroll);
    return () => window.removeEventListener('resize', checkScroll);
  }, [checkScroll]);

  const total = cartState.total_price || 0;
  const limitCeiling = 50.0;
  const progressPercent = Math.min(100, (total / limitCeiling) * 100);
  const isAmberWarning = total > 40.0 && total <= 50.0;
  const isExceeded = total > 50.0 || cartState.exceeds_50_limit;

  return (
    <aside
      className={`w-[370px] flex-shrink-0 h-full flex flex-col p-4 select-none z-10 ${className}`}
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
        .font-receipt {
          font-family: 'Share Tech Mono', 'Courier New', Courier, monospace;
          letter-spacing: -0.02em;
        }
        .receipt-scrollbar::-webkit-scrollbar {
          width: 4px;
        }
        .receipt-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .receipt-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(0, 0, 0, 0.25);
          border-radius: 2px;
        }
      `}</style>

      {/* Physical Receipt Container (Wider & Zoomed In) */}
      <div className="relative flex-1 flex flex-col min-h-0 filter drop-shadow-[0_14px_28px_rgba(0,0,0,0.7)]">
        {/* TOP ZIGZAG TEAR EDGE */}
        <div className="w-full h-3.5 overflow-hidden flex-shrink-0">
          <svg
            viewBox="0 0 300 12"
            preserveAspectRatio="none"
            className="w-full h-full block"
          >
            <path
              d="M0,12 L0,8 L6,0 L12,8 L18,0 L24,8 L30,0 L36,8 L42,0 L48,8 L54,0 L60,8 L66,0 L72,8 L78,0 L84,8 L90,0 L96,8 L102,0 L108,8 L114,0 L120,8 L126,0 L132,8 L138,0 L144,8 L150,0 L156,8 L162,0 L168,8 L174,0 L180,8 L186,0 L192,8 L198,0 L204,8 L210,0 L216,8 L222,0 L228,8 L234,0 L240,8 L246,0 L252,8 L258,0 L264,8 L270,0 L276,8 L282,0 L288,8 L294,0 L300,8 L300,12 Z"
              fill="#F7F5F0"
            />
          </svg>
        </div>

        {/* MAIN RECEIPT PAPER BODY */}
        <div
          className="relative flex-1 flex flex-col min-h-0 bg-[#F7F5F0] text-black font-receipt px-4 pt-1 pb-2 overflow-hidden"
          style={{
            backgroundImage: `
              radial-gradient(rgba(0, 0, 0, 0.035) 1px, transparent 0),
              linear-gradient(to right, rgba(0,0,0,0.025), transparent 8%, transparent 92%, rgba(0,0,0,0.035))
            `,
            backgroundSize: '12px 12px, 100% 100%',
          }}
        >
          {/* Paper Fiber Overlay */}
          <svg className="pointer-events-none absolute inset-0 w-full h-full opacity-[0.045] mix-blend-multiply">
            <filter id="receiptPaperNoise">
              <feTurbulence
                type="fractalNoise"
                baseFrequency="0.85"
                numOctaves="3"
                stitchTiles="stitch"
              />
            </filter>
            <rect width="100%" height="100%" filter="url(#receiptPaperNoise)" />
          </svg>

          {/* Ticket Header: "DELIVERY" */}
          <div className="relative z-10 flex-shrink-0 text-center pt-1 pb-2.5">
            <FullWidthDashes />
            <div className="my-2 flex items-center justify-center relative">
              <span className="text-xl font-bold tracking-[0.22em] text-black uppercase">
                DELIVERY
              </span>
              <span
                className="w-4 h-4 rounded-full bg-slate-900/90 border border-black/25 shadow-inner ml-2.5 inline-block"
                title="Ticket Spindle Hole"
              />
            </div>
            <FullWidthDashes />
          </div>

          {/* SCROLLABLE CART ITEMS AREA */}
          <div className="relative flex-1 min-h-0 flex flex-col">
            {canScrollUp && (
              <div
                className="pointer-events-none absolute top-0 left-0 right-0 h-7 z-20 transition-opacity duration-200"
                style={{
                  background:
                    'linear-gradient(to bottom, #F7F5F0 20%, rgba(247, 245, 240, 0))',
                }}
              />
            )}

            <div
              ref={scrollContainerRef}
              onScroll={checkScroll}
              className="flex-1 overflow-y-auto receipt-scrollbar pr-0.5 space-y-3.5 py-1.5"
            >
              {cartState.lines.length === 0 ? (
                <div className="h-full flex flex-col items-center justify-center text-center py-12 text-black/50">
                  <p className="text-sm uppercase tracking-widest font-bold">
                    *** NO ITEMS ***
                  </p>
                  <p className="text-xs mt-1">
                    Speak or type to add items
                  </p>
                </div>
              ) : (
                cartState.lines.map((item: CartLineItem) => {
                  const optionEntries = Object.entries(item.options || {});
                  const extrasList = item.extras || [];

                  return (
                    <div key={item.line_id} className="text-black text-[15.5px] leading-snug">
                      {/* Primary Line: Quantity x Name & Price at exact same height, strictly 1 line */}
                      <div className="flex items-baseline justify-between gap-2 whitespace-nowrap">
                        <div className="flex items-baseline min-w-0 flex-1 overflow-hidden">
                          <span className="font-bold flex-shrink-0">
                            {item.quantity} x&nbsp;
                          </span>
                          <span className="font-bold truncate">
                            {item.name}
                          </span>
                        </div>
                        <span className="font-bold flex-shrink-0 pl-1">
                          €{item.line_total.toFixed(2)}
                        </span>
                      </div>

                      {/* Option Lines */}
                      {optionEntries.map(([optKey, optVal]) => (
                        <div
                          key={`${item.line_id}_opt_${optKey}`}
                          className="pl-[36px] text-[14px] text-black whitespace-nowrap truncate"
                        >
                          {formatOptionLine(optKey, optVal)}
                        </div>
                      ))}

                      {/* Extra Lines */}
                      {extrasList.map((extra, idx) => (
                        <div
                          key={`${item.line_id}_ext_${idx}`}
                          className="pl-[36px] text-[14px] text-black whitespace-nowrap truncate"
                        >
                          + {formatExtraLine(extra)}
                        </div>
                      ))}
                    </div>
                  );
                })
              )}

              {/* Special Instructions */}
              {cartState.special_instructions && (
                <div className="pt-2 text-black text-[14px]">
                  <div className="mb-1.5">
                    <FullWidthDashes spaced />
                  </div>
                  <div className="font-bold uppercase text-[13px]">
                    NOTE:
                  </div>
                  <div className="pl-2 break-words leading-tight">
                    "{cartState.special_instructions}"
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Scroll Fade Overlay */}
            {canScrollDown && (
              <div
                className="pointer-events-none absolute bottom-0 left-0 right-0 h-11 z-20 flex items-end justify-center pb-0.5"
                style={{
                  background:
                    'linear-gradient(to top, #F7F5F0 30%, rgba(247, 245, 240, 0))',
                }}
              >
                <span className="text-[10px] font-bold uppercase tracking-widest text-black/70 bg-[#F7F5F0]/95 px-2 rounded">
                  ▼ MORE ITEMS ▼
                </span>
              </div>
            )}
          </div>

          {/* PINNED HOVERING TOTAL FOOTER */}
          <div className="relative z-30 flex-shrink-0 pt-1.5 bg-[#F7F5F0]">
            <FullWidthDashes />

            <div className="flex items-baseline justify-between py-1.5 text-black">
              <span className="text-sm font-bold uppercase">
                ITEMS ({cartState.total_item_count || 0})
              </span>
              <span className="text-lg font-bold tracking-tight">
                TOTAL: €{total.toFixed(2)}
              </span>
            </div>

            {/* €50 Limit Progress Meter */}
            <div className="mt-1 mb-1.5">
              <div className="flex items-center justify-between text-xs text-black mb-1">
                <span className="uppercase font-bold">€50.00 MAX LIMIT</span>
                <span className="font-bold">
                  {isExceeded
                    ? 'LIMIT EXCEEDED'
                    : isAmberWarning
                    ? 'NEAR LIMIT'
                    : `€${Math.max(0, limitCeiling - total).toFixed(2)} LEFT`}
                </span>
              </div>

              <div className="w-full h-2.5 border border-black bg-black/5 p-[1px]">
                <div
                  className={`h-full transition-all duration-300 ${
                    isExceeded
                      ? 'bg-red-600'
                      : isAmberWarning
                      ? 'bg-amber-500'
                      : 'bg-black'
                  }`}
                  style={{ width: `${progressPercent}%` }}
                />
              </div>
            </div>

            {/* Order Submitted Stamp */}
            {cartState.is_submitted && (
              <div className="mt-2 py-1.5 border-2 border-black text-center font-bold text-sm uppercase tracking-widest flex items-center justify-center gap-1.5 bg-black/5">
                <CheckCircle2 className="w-4 h-4 text-black" />
                <span>
                  SUBMITTED{' '}
                  {cartState.last_submitted_order_id
                    ? `#${cartState.last_submitted_order_id}`
                    : ''}
                </span>
              </div>
            )}

            {/* Over-Limit Warning Stamp */}
            {isExceeded && !cartState.is_submitted && (
              <div className="mt-1.5 py-1 border border-black text-center font-bold text-xs uppercase tracking-wider flex items-center justify-center gap-1 bg-black/5">
                <AlertTriangle className="w-3.5 h-3.5 text-black" />
                <span>OVER €50 RESTAURANT CAP</span>
              </div>
            )}

            <div className="mt-1">
              <FullWidthDashes />
            </div>
          </div>
        </div>

        {/* BOTTOM ZIGZAG TEAR EDGE */}
        <div className="w-full h-3.5 overflow-hidden flex-shrink-0">
          <svg
            viewBox="0 0 300 12"
            preserveAspectRatio="none"
            className="w-full h-full block"
          >
            <path
              d="M0,0 L300,0 L300,4 L294,12 L288,4 L282,12 L276,4 L270,12 L264,4 L258,12 L252,4 L246,12 L240,4 L234,12 L228,4 L222,12 L216,4 L210,12 L204,4 L198,12 L192,4 L186,12 L180,4 L174,12 L168,4 L162,12 L156,4 L150,12 L144,4 L138,12 L132,4 L126,12 L120,4 L114,12 L108,4 L102,12 L96,4 L90,12 L84,4 L78,12 L72,4 L66,12 L60,4 L54,12 L48,4 L42,12 L36,4 L30,12 L24,4 L18,12 L12,4 L6,12 L0,4 Z"
              fill="#F7F5F0"
            />
          </svg>
        </div>
      </div>
    </aside>
  );
};