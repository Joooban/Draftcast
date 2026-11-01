import { useEffect, useState } from 'react'
import type { FormEvent } from 'react'
import { getChampions, predict } from '../api'
import DraftInput from '../components/DraftInput'
import ResultPanel from '../components/ResultPanel'
import StateInput from '../components/StateInput'
import type { LiveEntry, Side } from '../live'
import { emptyEntry, toState } from '../live'
import type { Champion, Minute, Mode, PartialTeam, PredictRequest, PredictResponse, Role, Team } from '../types'
import { ROLES } from '../types'

const MODES: { id: Mode; label: string }[] = [
  { id: 'draft', label: 'Draft' },
  { id: 'state', label: 'Game state' },
  { id: 'both', label: 'Both' },
]

const SIDE_OPTIONS: { id: Side | null; label: string }[] = [
  { id: 'blue', label: 'Blue' },
  { id: 'red', label: 'Red' },
  { id: null, label: 'Neither' },
]

const isFull = (t: PartialTeam): t is Team => ROLES.every((r) => t[r] !== undefined)

export default function Predict() {
  const [champions, setChampions] = useState<Champion[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>('draft')
  const [blue, setBlue] = useState<PartialTeam>({})
  const [red, setRed] = useState<PartialTeam>({})
  const [entry, setEntry] = useState<LiveEntry>(emptyEntry)
  const [minute, setMinute] = useState<Minute>(15)
  const [side, setSide] = useState<Side | null>(null)
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    getChampions().then(setChampions, (e: Error) => setLoadError(e.message))
  }, [])

  // Any input change makes the shown result stale, so it goes.
  const changed = () => {
    setResult(null)
    setError(null)
  }

  const needsDraft = mode !== 'state'
  const needsState = mode !== 'draft'
  const draftReady = isFull(blue) && isFull(red)
  const state = toState(entry)
  const known = Object.keys(state).length
  const ready = (!needsDraft || draftReady) && (!needsState || known > 0)

  const setPick = (team: 'blue' | 'red', role: Role, id: number | undefined) => {
    const update = (t: PartialTeam) => ({ ...t, [role]: id })
    if (team === 'blue') setBlue(update)
    else setRed(update)
    changed()
  }

  const run = async (e?: FormEvent) => {
    e?.preventDefault()
    if (!ready || busy) return
    const req: PredictRequest = { mode, minute }
    if (needsDraft) {
      req.blue = blue as Team
      req.red = red as Team
    }
    if (needsState) req.state = state
    setBusy(true)
    setError(null)
    try {
      setResult(await predict(req))
    } catch (err) {
      setResult(null)
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const missing = needsDraft
    ? 10 - ROLES.filter((r) => blue[r] !== undefined).length - ROLES.filter((r) => red[r] !== undefined).length
    : 0

  return (
    <div className="page">
      <h1>Who wins this game?</h1>
      <p className="lead">
        Pick a draft, type the game state at 10:00 or 15:00, or both. Draftcast returns each team's win chance and
        explains what moved it.
      </p>

      <div className="tabs" role="tablist">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            role="tab"
            aria-selected={mode === m.id}
            className={mode === m.id ? 'tab active' : 'tab'}
            onClick={() => {
              setMode(m.id)
              changed()
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {loadError && <p className="error">Couldn't load champions: {loadError}</p>}

      <form onSubmit={run} className="predict-form">
        {needsDraft && <DraftInput champions={champions} blue={blue} red={red} onChange={setPick} />}
        {needsState && (
          <StateInput
            entry={entry}
            onChange={(e) => {
              setEntry(e)
              changed()
            }}
            minute={minute}
            onMinute={(m) => {
              setMinute(m)
              changed()
            }}
          />
        )}

        <div className="actions">
          <button className="primary" type="submit" disabled={!ready || busy}>
            {busy ? 'Predicting…' : 'Predict'}
          </button>
          {needsState && (
            <button
              type="button"
              className="secondary"
              onClick={() => {
                setEntry(emptyEntry())
                changed()
              }}
            >
              Reset game state
            </button>
          )}
          <div className="seg side-toggle" role="radiogroup" aria-label="I'm on">
            <span className="small muted seg-label">I'm on</span>
            {SIDE_OPTIONS.map((o) => (
              <button
                key={o.label}
                type="button"
                role="radio"
                aria-checked={side === o.id}
                className={side === o.id ? `seg-btn active ${o.id ?? ''}` : 'seg-btn'}
                onClick={() => setSide(o.id)}
              >
                {o.label}
              </button>
            ))}
          </div>
          {needsDraft && !draftReady && (
            <span className="muted small">
              {missing} pick{missing === 1 ? '' : 's'} left
            </span>
          )}
          {needsState && known === 0 && <span className="muted small">Enter at least one game-state value</span>}
        </div>
      </form>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {result && <ResultPanel result={result} side={side} />}
    </div>
  )
}
