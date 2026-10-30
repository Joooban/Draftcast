import type { GameState, StateFeature } from '../types'
import { STATE_FEATURES, STATE_LABELS, STATE_RANGES, STATE_STEPS } from '../types'

interface Props {
  state: GameState
  onChange: (state: GameState) => void
}

const FIRSTS = new Set<StateFeature>(['first_blood_blue', 'first_dragon_blue', 'first_tower_blue'])

function describe(f: StateFeature, v: number): string {
  if (FIRSTS.has(f)) return v === 1 ? 'Blue' : v === -1 ? 'Red' : 'None'
  if (v === 0) return 'Even'
  return `${v > 0 ? 'Blue' : 'Red'} +${Math.abs(v).toLocaleString()}`
}

export default function StateInput({ state, onChange }: Props) {
  return (
    <div className="state-grid">
      <p className="muted small full">Values at 15:00, Blue minus Red. Right = Blue ahead.</p>
      {STATE_FEATURES.map((f) => {
        const [min, max] = STATE_RANGES[f]
        return (
          <label key={f} className="slider">
            <span className="slider-head">
              <span>{STATE_LABELS[f]}</span>
              <span className={state[f] > 0 ? 'blue' : state[f] < 0 ? 'red' : 'muted'}>{describe(f, state[f])}</span>
            </span>
            <input
              type="range"
              min={min}
              max={max}
              step={STATE_STEPS[f] ?? 1}
              value={state[f]}
              onChange={(e) => onChange({ ...state, [f]: Number(e.target.value) })}
            />
          </label>
        )
      })}
    </div>
  )
}
