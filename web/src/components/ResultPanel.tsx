import { Suspense, lazy } from 'react'
import type { Side } from '../live'
import type { PredictResponse, StateFeature } from '../types'
import { STATE_LABELS } from '../types'

const WaterfallChart = lazy(() => import('./WaterfallChart'))

const pts = (v: number) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(1)} pts`
const pct = (v: number) => `${(v * 100).toFixed(0)}%`

interface Props {
  result: PredictResponse
  side: Side | null
}

export default function ResultPanel({ result, side }: Props) {
  const blue = result.probability
  const leading = blue >= 0.5 ? 'blue' : 'red'
  const yours = side === 'blue' ? blue : side === 'red' ? 1 - blue : null
  const { split, calibration: cal } = result
  // Inputs left at 0 still shift the waterfall a little (they're compared with the average game),
  // so they're labelled as even rather than credited as a reason.
  const contributions = result.contributions.map((c) => (c.value === 0 ? { ...c, label: `${c.label} (even)` } : c))

  return (
    <section className="result" aria-live="polite">
      <p className={`verdict ${result.strength === 'toss-up' ? '' : leading}`}>{result.verdict}</p>
      <p className="muted small odds">
        Blue {(blue * 100).toFixed(1)}% · Red {((1 - blue) * 100).toFixed(1)}%
        {yours !== null && (
          <>
            {' · '}
            <strong className={side ?? ''}>
              Your team ({side === 'blue' ? 'Blue' : 'Red'}): {(yours * 100).toFixed(1)}%
            </strong>
          </>
        )}
      </p>
      {result.reasons.length === 0 && <p className="muted small">No single factor stands out; this is close to an average game.</p>}
      {result.reasons.length > 0 && (
        <>
          <h3>Why</h3>
          <ul className="reasons">
            {result.reasons.map((r) => (
              <li key={r.text}>
                <span>{r.text}</span>
                <span className={`pts ${r.value_pct >= 0 ? 'blue' : 'red'}`}>{pts(r.value_pct)}</span>
              </li>
            ))}
          </ul>
        </>
      )}
      <p className="muted small">
        Points are the change in Blue's win chance. {result.reliability.text}
      </p>

      <details className="full-reasoning">
        <summary>Show the full reasoning</summary>

        <p className="muted small">
          Start from an average game ({(result.baseline * 100).toFixed(1)}%), add each factor in order of size, end at
          the prediction. Blue bars help Blue, red bars help Red.
        </p>
        <Suspense fallback={<p className="muted small">Loading chart…</p>}>
          <WaterfallChart baseline={result.baseline} contributions={contributions} />
        </Suspense>

        {split && (
          <p>
            The draft moved it <strong>{pts(split.draft_pct)}</strong>, the game state{' '}
            <strong>{pts(split.state_pct)}</strong>.
          </p>
        )}

        <div className="table-scroll">
          <table className="factors">
            <thead>
              <tr>
                <th scope="col">Factor</th>
                <th scope="col">Your input</th>
                <th scope="col">Effect</th>
                <th scope="col">Log-odds</th>
              </tr>
            </thead>
            <tbody>
              {contributions.map((c) => (
                <tr key={c.feature}>
                  <th scope="row">{c.label}</th>
                  <td>{c.value === null ? '—' : c.value.toLocaleString()}</td>
                  <td className={c.value_pct >= 0 ? 'blue' : 'red'}>{pts(c.value_pct)}</td>
                  <td>{c.logit.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <h4>Reality check</h4>
        {cal ? (
          <p>
            When this model said Blue had {pct(cal.lo)}–{pct(cal.hi)}, Blue won{' '}
            <strong>{(cal.win_rate * 100).toFixed(0)}%</strong> of {cal.n.toLocaleString()} test games.
          </p>
        ) : (
          <p className="muted">Too few test games near this prediction to check it.</p>
        )}

        <h4>How far to trust it</h4>
        <p>
          On the test games it was right {pct(result.reliability.accuracy)} of the time, with an AUC of{' '}
          {result.reliability.auc.toFixed(2)} (0.5 is a coin flip, 1.0 is perfect). Test games come from the newest
          patch, which the model never trained on.
        </p>

        {result.omitted.length > 0 && (
          <p className="muted small">
            Left out (unknown): {result.omitted.map((f: StateFeature) => STATE_LABELS[f]).join(', ')}. The prediction
            uses a model trained without them, at {result.minute}:00.
          </p>
        )}
      </details>
    </section>
  )
}
