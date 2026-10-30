import { Suspense, lazy, useEffect, useMemo, useState } from 'react'
import { getInsights } from '../api'
import type { Insights as InsightsData } from '../types'

const Charts = lazy(() => import('./InsightsCharts'))

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export default function Insights() {
  const [data, setData] = useState<InsightsData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')

  useEffect(() => {
    getInsights().then(setData, (e: Error) => setError(e.message))
  }, [])

  const champions = useMemo(() => {
    if (!data) return []
    const q = query.trim().toLowerCase()
    if (q) return data.champions.filter((c) => c.name.toLowerCase().includes(q))
    // Default view: the 10 best and 10 worst by win rate.
    const all = data.champions
    return all.length <= 20 ? all : [...all.slice(0, 10), ...all.slice(-10)]
  }, [data, query])

  if (error) return <div className="page"><h1>Insights</h1><p className="error">{error}</p></div>
  if (!data) return <div className="page"><h1>Insights</h1><p className="muted">Loading…</p></div>

  const { blue_side: blue, objectives } = data
  const dvt = objectives.dragon_vs_tower

  return (
    <div className="page">
      <h1>Insights</h1>
      <p className="lead">What predicts wins in current SG2 ranked games (Emerald and above).</p>

      <div className="stats">
        <div className="stat">
          <span className="stat-value">{pct(blue.win_rate)}</span>
          <span className="muted small">
            Blue-side win rate over {blue.n.toLocaleString()} games (p = {blue.p_value.toPrecision(2)})
          </span>
        </div>
        <div className="stat">
          <span className="stat-value">{dvt.estimate_diff >= 0 ? 'Dragon' : 'Tower'}</span>
          <span className="muted small">
            worth more at 15:00 in log-odds, by {Math.abs(dvt.estimate_diff).toFixed(2)} (p ={' '}
            {dvt.p_value.toPrecision(2)})
          </span>
        </div>
      </div>

      <Suspense fallback={<p className="muted">Loading charts…</p>}>
        <Charts data={data} champions={champions} query={query} onQuery={setQuery} />
      </Suspense>
    </div>
  )
}
