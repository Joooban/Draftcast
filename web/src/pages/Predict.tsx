import { useEffect, useState } from 'react'
import { getChampions, predict } from '../api'
import DraftInput from '../components/DraftInput'
import StateInput from '../components/StateInput'
import type { Champion, GameState, Mode, PartialTeam, PredictRequest, PredictResponse, Role, Team } from '../types'
import { ROLES, emptyState } from '../types'

const MODES: { id: Mode; label: string }[] = [
  { id: 'draft', label: 'Draft' },
  { id: 'state', label: 'Game state' },
  { id: 'both', label: 'Both' },
]

const isFull = (t: PartialTeam): t is Team => ROLES.every((r) => t[r] !== undefined)

export default function Predict() {
  const [champions, setChampions] = useState<Champion[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>('draft')
  const [blue, setBlue] = useState<PartialTeam>({})
  const [red, setRed] = useState<PartialTeam>({})
  const [state, setState] = useState<GameState>(emptyState)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    getChampions().then(setChampions, (e: Error) => setLoadError(e.message))
  }, [])

  const needsDraft = mode !== 'state'
  const needsState = mode !== 'draft'
  const draftReady = isFull(blue) && isFull(red)
  const ready = !needsDraft || draftReady

  const setPick = (side: 'blue' | 'red', role: Role, id: number | undefined) => {
    const update = (t: PartialTeam) => ({ ...t, [role]: id })
    if (side === 'blue') setBlue(update)
    else setRed(update)
  }

  const run = async () => {
    const req: PredictRequest = { mode }
    if (needsDraft) {
      req.blue = blue as Team
      req.red = red as Team
    }
    if (needsState) req.state = state
    setBusy(true)
    setError(null)
    try {
      setResult(await predict(req))
    } catch (e) {
      setResult(null)
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const missing = needsDraft ? 10 - ROLES.filter((r) => blue[r] !== undefined).length - ROLES.filter((r) => red[r] !== undefined).length : 0

  return (
    <div className="page">
      <h1>Who wins this game?</h1>
      <p className="lead">
        Pick a draft, set the game state at 15:00, or both. The model returns Blue's win probability and shows what
        moved it.
      </p>

      <div className="tabs" role="tablist">
        {MODES.map((m) => (
          <button
            key={m.id}
            role="tab"
            aria-selected={mode === m.id}
            className={mode === m.id ? 'tab active' : 'tab'}
            onClick={() => {
              setMode(m.id)
              setResult(null)
              setError(null)
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {loadError && <p className="error">Couldn't load champions: {loadError}</p>}

      {needsDraft && <DraftInput champions={champions} blue={blue} red={red} onChange={setPick} />}
      {needsState && <StateInput state={state} onChange={setState} />}

      <div className="actions">
        <button className="primary" disabled={!ready || busy} onClick={run}>
          {busy ? 'Predicting…' : 'Predict'}
        </button>
        {needsDraft && !draftReady && <span className="muted small">{missing} pick{missing === 1 ? '' : 's'} left</span>}
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {result && (
        <section className="result">
          <div className="prob">
            Blue wins <strong>{(result.probability * 100).toFixed(1)}%</strong>
          </div>
          <ul>
            {result.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </section>
      )}
    </div>
  )
}
