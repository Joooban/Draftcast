import type { Contribution } from '../types'
import Plot, { baseLayout, plotConfig, themeColors } from './Plot'

interface Props {
  baseline: number
  contributions: Contribution[]
}

const SHOWN = 8

export default function WaterfallChart({ baseline, contributions }: Props) {
  const c = themeColors()
  // Contributions arrive largest-first; small ones are folded into one bar so the chart stays
  // readable on a phone. The folded bar keeps the total exact.
  const shown = contributions.slice(0, SHOWN)
  const rest = contributions.slice(SHOWN)
  const bars = rest.length
    ? [...shown, { feature: 'other', label: `${rest.length} other features`, value_pct: rest.reduce((s, x) => s + x.value_pct, 0) }]
    : shown
  const final = baseline * 100 + bars.reduce((s, x) => s + x.value_pct, 0)

  const labels = ['Average game', ...bars.map((b) => b.label), 'Blue win chance']
  const values = [baseline * 100, ...bars.map((b) => b.value_pct), final]
  const text = [
    `${(baseline * 100).toFixed(1)}%`,
    ...bars.map((b) => `${b.value_pct >= 0 ? '+' : ''}${b.value_pct.toFixed(1)}`),
    `${final.toFixed(1)}%`,
  ]

  return (
    <Plot
      data={[
        {
          type: 'waterfall',
          orientation: 'h',
          y: labels,
          x: values,
          measure: ['absolute', ...bars.map(() => 'relative'), 'total'],
          text,
          textposition: 'outside',
          cliponaxis: false,
          hovertemplate: '%{y}: %{text}<extra></extra>',
          increasing: { marker: { color: c.blue } },
          decreasing: { marker: { color: c.red } },
          totals: { marker: { color: c.muted } },
          connector: { line: { color: c.grid } },
        },
      ]}
      layout={baseLayout({
        height: 60 + labels.length * 30,
        margin: { l: 10, r: 50, t: 10, b: 30 },
        yaxis: { autorange: 'reversed' },
        xaxis: { ticksuffix: '%', range: [0, 100] },
      })}
      config={plotConfig}
      useResizeHandler
      style={{ width: '100%' }}
    />
  )
}
