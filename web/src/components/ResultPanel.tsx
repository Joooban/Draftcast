import { Suspense, lazy } from 'react'
import type { Side } from '../live'
import type { PredictResponse } from '../types'

const WaterfallChart = lazy(() => import('./WaterfallChart'))

const signed = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`

export default function ResultPanel({ result }: { result: PredictResponse; side: Side | null }) {
  const pct = result.probability * 100
  const { split } = result
  const shareTotal = split ? Math.abs(split.draft_pct) + Math.abs(split.state_pct) : 0

  return (
    <section className="result" aria-live="polite">
      <div className="prob">
        Blue wins <strong className={pct >= 50 ? 'blue' : 'red'}>{pct.toFixed(1)}%</strong>
        <span className="muted small"> · Red {(100 - pct).toFixed(1)}%</span>
      </div>

      <h3>Why</h3>
      <ul className="reasons">
        {result.reasons.map((r) => (
          <li key={r.text}>{r.text}</li>
        ))}
      </ul>

      {split && (
        <div className="split">
          <div className="split-bar" aria-hidden="true">
            <span className="seg-draft" style={{ flexGrow: shareTotal ? Math.abs(split.draft_pct) : 1 }} />
            <span className="seg-state" style={{ flexGrow: shareTotal ? Math.abs(split.state_pct) : 1 }} />
          </div>
          <div className="split-legend small">
            <span><i className="dot seg-draft" /> Draft {signed(split.draft_pct)}</span>
            <span><i className="dot seg-state" /> Game state {signed(split.state_pct)}</span>
          </div>
        </div>
      )}

      <p className="muted small">
        Start from an average game ({(result.baseline * 100).toFixed(1)}%), add each factor in order of size, end at
        the prediction. Blue bars help Blue, red bars help Red.
      </p>
      <Suspense fallback={<p className="muted small">Loading chart…</p>}>
        <WaterfallChart baseline={result.baseline} contributions={result.contributions} />
      </Suspense>
    </section>
  )
}
