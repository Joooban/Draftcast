import { useState } from 'react'
import type { GameState, StateFeature } from '../types'
import { STATE_FEATURES, STATE_LABELS, STATE_RANGES, STATE_STEPS } from '../types'

interface Props {
  state: GameState
  onChange: (state: GameState) => void
}

const FIRSTS = new Set<StateFeature>(['first_blood_blue', 'first_dragon_blue', 'first_tower_blue'])
const FIRST_OPTIONS: { value: number; label: string; cls: string }[] = [
  { value: 1, label: 'Blue', cls: 'blue' },
  { value: 0, label: 'None', cls: '' },
  { value: -1, label: 'Red', cls: 'red' },
]

function describe(v: number): string {
  if (v === 0) return 'Even'
  return `${v > 0 ? 'Blue' : 'Red'} +${Math.abs(v).toLocaleString()}`
}

const clamp = (v: number, [min, max]: [number, number]) => Math.min(max, Math.max(min, Math.round(v)))

// Typed entry for use mid-game: keeps the raw text while typing (so "-" or "" are allowed),
// and commits any number it can parse, clamped to the feature's range.
function NumberBox({ feature, value, onCommit }: { feature: StateFeature; value: number; onCommit: (v: number) => void }) {
  // null when not editing, so the box follows the slider; the raw text only lives while focused.
  const [text, setText] = useState<string | null>(null)
  const range = STATE_RANGES[feature]
  const commit = (raw: string) => {
    const n = Number(raw.replace(/[,\s]/g, ''))
    if (raw.trim() !== '' && Number.isFinite(n)) onCommit(clamp(n, range))
  }
  return (
    <input
      className="num"
      type="text"
      inputMode="numeric"
      aria-label={`${STATE_LABELS[feature]}, Blue minus Red`}
      value={text ?? String(value)}
      onFocus={(e) => {
        setText(String(value))
        e.target.select()
      }}
      onChange={(e) => {
        setText(e.target.value)
        commit(e.target.value)
      }}
      onBlur={() => setText(null)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') (e.target as HTMLInputElement).blur()
      }}
    />
  )
}

export default function StateInput({ state, onChange }: Props) {
  const set = (f: StateFeature, v: number) => onChange({ ...state, [f]: v })
  return (
    <div className="state-grid">
      <p className="muted small full">
        Values at 15:00, Blue minus Red. Drag a slider or type a number: positive = Blue ahead, negative = Red ahead
        (e.g. -1500 when Red leads by 1,500 gold). Tab moves to the next box.
      </p>
      {STATE_FEATURES.map((f) => {
        if (FIRSTS.has(f)) {
          return (
            <div key={f} className="slider">
              <span className="slider-head">{STATE_LABELS[f]}</span>
              <div className="seg" role="radiogroup" aria-label={STATE_LABELS[f]}>
                {FIRST_OPTIONS.map((o) => (
                  <button
                    key={o.value}
                    type="button"
                    role="radio"
                    aria-checked={state[f] === o.value}
                    className={state[f] === o.value ? `seg-btn active ${o.cls}` : 'seg-btn'}
                    onClick={() => set(f, o.value)}
                  >
                    {o.label}
                  </button>
                ))}
              </div>
            </div>
          )
        }
        const [min, max] = STATE_RANGES[f]
        return (
          <div key={f} className="slider">
            <span className="slider-head">
              <span>{STATE_LABELS[f]}</span>
              <span className={state[f] > 0 ? 'blue' : state[f] < 0 ? 'red' : 'muted'}>{describe(state[f])}</span>
            </span>
            <div className="slider-row">
              <input
                type="range"
                aria-label={STATE_LABELS[f]}
                min={min}
                max={max}
                step={STATE_STEPS[f] ?? 1}
                value={state[f]}
                onChange={(e) => set(f, Number(e.target.value))}
              />
              <NumberBox feature={f} value={state[f]} onCommit={(v) => set(f, v)} />
            </div>
          </div>
        )
      })}
    </div>
  )
}
