import { useEffect, useMemo, useRef, useState } from 'react'
import type { Champion } from '../types'

interface Props {
  champions: Champion[]
  value: number | undefined
  onChange: (id: number | undefined) => void
  disabledIds: Set<number>
  label: string
}

export default function ChampionPicker({ champions, value, onChange, disabledIds, label }: Props) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const root = useRef<HTMLDivElement>(null)
  const selected = champions.find((c) => c.id === value)

  const matches = useMemo(() => {
    const q = query.trim().toLowerCase()
    return q ? champions.filter((c) => c.name.toLowerCase().includes(q)) : champions
  }, [champions, query])

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const choose = (c: Champion) => {
    onChange(c.id)
    setOpen(false)
    setQuery('')
  }

  return (
    <div className="picker" ref={root}>
      <button
        type="button"
        className="picker-button"
        aria-label={`${label}: ${selected ? selected.name : 'none'}`}
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
      >
        {selected ? (
          <>
            <img src={selected.icon_url} alt="" width={28} height={28} />
            <span>{selected.name}</span>
          </>
        ) : (
          <span className="muted">Pick champion</span>
        )}
      </button>
      {selected && (
        <button type="button" className="picker-clear" aria-label={`Clear ${label}`} onClick={() => onChange(undefined)}>
          ×
        </button>
      )}
      {open && (
        <div className="picker-menu" role="listbox">
          <input
            autoFocus
            placeholder="Search…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Escape') setOpen(false)
              if (e.key === 'Enter') {
                const first = matches.find((c) => !disabledIds.has(c.id))
                if (first) choose(first)
              }
            }}
          />
          <ul>
            {matches.map((c) => {
              const taken = disabledIds.has(c.id) && c.id !== value
              return (
                <li key={c.id}>
                  <button type="button" disabled={taken} onClick={() => choose(c)} role="option" aria-selected={c.id === value}>
                    <img src={c.icon_url} alt="" width={24} height={24} loading="lazy" />
                    <span>{c.name}</span>
                    {taken && <span className="muted small">picked</span>}
                  </button>
                </li>
              )
            })}
            {matches.length === 0 && <li className="muted small pad">No champion matches “{query}”</li>}
          </ul>
        </div>
      )}
    </div>
  )
}
