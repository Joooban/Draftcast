// Plotly is large; pages load this module lazily so the first paint stays fast.
import Plotly from 'plotly.js-dist-min'
import type { Layout } from 'plotly.js'
import createPlotlyComponent from 'react-plotly.js/factory'

const Plot = createPlotlyComponent(Plotly)
export default Plot

const cssVar = (name: string, fallback: string) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback

export function themeColors() {
  return {
    text: cssVar('--text', '#1c1d21'),
    muted: cssVar('--muted', '#6b6e76'),
    grid: cssVar('--border', '#e2e2de'),
    blue: cssVar('--blue', '#2f6fdf'),
    red: cssVar('--red', '#d2453c'),
    accent: cssVar('--accent', '#2f6fdf'),
  }
}

export function baseLayout(extra: Partial<Layout> = {}): Partial<Layout> {
  const c = themeColors()
  return {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { color: c.text, size: 12 },
    margin: { l: 10, r: 10, t: 10, b: 40 },
    autosize: true,
    showlegend: false,
    ...extra,
    xaxis: { gridcolor: c.grid, zerolinecolor: c.grid, automargin: true, ...extra.xaxis },
    yaxis: { gridcolor: c.grid, zerolinecolor: c.grid, automargin: true, ...extra.yaxis },
  }
}

export const plotConfig = { displayModeBar: false, responsive: true }
