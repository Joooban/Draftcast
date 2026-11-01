import { useMemo, useState } from 'react'
import type { Insights } from '../types'

type Row = Insights['champions'][number]
type SortKey = 'win_rate' | 'games' | 'name'

const SORTS: { key: SortKey; label: string }[] = [
  { key: 'win_rate', label: 'Win rate' },
  { key: 'games', label: 'Games' },
  { key: 'name', label: 'Name' },
]

const pct = (v: number) => `${(v * 100).toFixed(1)}%`

export default function ChampionTable({ champions }: { champions: Row[] }) {
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState<SortKey>('win_rate')
  const [minGames, setMinGames] = useState('30')

  const rows = useMemo(() => {
    const q = query.trim().toLowerCase()
    const min = Number(minGames) || 0
    const kept = champions.filter((c) => c.games >= min && (!q || c.name.toLowerCase().includes(q)))
    const by: Record<SortKey, (a: Row, b: Row) => number> = {
      win_rate: (a, b) => b.win_rate - a.win_rate,
      games: (a, b) => b.games - a.games,
      name: (a, b) => a.name.localeCompare(b.name),
    }
    return [...kept].sort(by[sort])
  }, [champions, query, sort, minGames])

  return (
    <div className="champ-table">
      <div className="champ-controls">
        <input
          className="search"
          placeholder="Search a champion…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Search champions"
        />
        <div className="seg" role="radiogroup" aria-label="Sort by">
          {SORTS.map((s) => (
            <button
              key={s.key}
              type="button"
              role="radio"
              aria-checked={sort === s.key}
              className={sort === s.key ? 'seg-btn active' : 'seg-btn'}
              onClick={() => setSort(s.key)}
            >
              {s.label}
            </button>
          ))}
        </div>
        <label className="small min-games">
          Min. games
          <input
            className="num"
            type="text"
            inputMode="numeric"
            value={minGames}
            onChange={(e) => setMinGames(e.target.value.replace(/\D/g, ''))}
          />
        </label>
      </div>
      <p className="muted small">
        {rows.length} champion{rows.length === 1 ? '' : 's'} · games in training patches · 95% Wilson intervals
      </p>
      <div className="table-scroll champ-scroll">
        <table className="factors">
          <thead>
            <tr>
              <th scope="col">Champion</th>
              <th scope="col">Games</th>
              <th scope="col">Win rate</th>
              <th scope="col">95% interval</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={c.id}>
                <th scope="row">{c.name}</th>
                <td>{c.games.toLocaleString()}</td>
                <td className={c.win_rate >= 0.5 ? 'blue' : 'red'}>{pct(c.win_rate)}</td>
                <td className="muted">
                  {pct(c.ci_low)}–{pct(c.ci_high)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
