import Plot, { baseLayout, plotConfig, themeColors } from '../components/Plot'
import type { Insights } from '../types'

interface Props {
  data: Insights
  champions: Insights['champions']
  query: string
  onQuery: (q: string) => void
}

const TERM_LABELS: Record<string, string> = {
  gold_k: '+1k gold',
  dragons_diff: '+1 dragon',
  towers_diff: '+1 tower',
  grubs_diff: '+1 voidgrub',
  kills_diff: '+1 kill',
}

const MODEL_LABELS: Record<string, string> = {
  baseline: 'Baseline',
  draft_lr: 'Draft (LR)',
  draft_lgbm: 'Draft (LGBM)',
  state_lr: 'State (LR)',
  state_lgbm: 'State (LGBM)',
  combined_lr: 'Combined (LR)',
  combined_lgbm: 'Combined (LGBM)',
}

const common = { config: plotConfig, useResizeHandler: true, style: { width: '100%' } }

export default function InsightsCharts({ data, champions, query, onQuery }: Props) {
  const c = themeColors()
  const models = Object.keys(MODEL_LABELS).filter((k) => k in data.model_comparison)
  const m = data.model_comparison

  return (
    <>
      <h2>Model accuracy on the newest patch</h2>
      <p className="muted small">Higher AUC is better; lower log loss is better. LR runs the app; LGBM is a benchmark.</p>
      <div className="chart-pair">
        {(['auc', 'log_loss'] as const).map((metric) => (
          <Plot
            key={metric}
            {...common}
            data={[
              {
                type: 'bar',
                orientation: 'h',
                y: models.map((k) => MODEL_LABELS[k]),
                x: models.map((k) => m[k][metric]),
                text: models.map((k) => m[k][metric].toFixed(3)),
                textposition: 'auto',
                marker: { color: models.map((k) => (k.endsWith('_lr') ? c.accent : c.muted)) },
                hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
              },
            ]}
            layout={baseLayout({
              height: 280,
              title: { text: metric === 'auc' ? 'ROC AUC' : 'Log loss', font: { size: 13 } },
              margin: { l: 10, r: 10, t: 30, b: 30 },
              yaxis: { autorange: 'reversed' },
              xaxis: metric === 'auc' ? { range: [0.5, 1] } : {},
            })}
          />
        ))}
      </div>

      <h2>Win rate vs gold lead at 15:00</h2>
      <Plot
        {...common}
        data={[
          {
            type: 'scatter',
            mode: 'lines+markers',
            x: data.gold_curve.map((b) => b.gold_bin_center),
            y: data.gold_curve.map((b) => b.win_rate * 100),
            customdata: data.gold_curve.map((b) => b.n),
            line: { color: c.accent },
            hovertemplate: 'Gold lead %{x}: %{y:.1f}% (%{customdata} games)<extra></extra>',
          },
        ]}
        layout={baseLayout({
          height: 300,
          xaxis: { title: { text: 'Blue gold lead' } },
          yaxis: { title: { text: 'Blue win rate' }, ticksuffix: '%', range: [0, 100] },
          shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 50, y1: 50, line: { color: c.muted, dash: 'dot' } }],
        })}
      />

      <h2>What an objective is worth</h2>
      <p className="muted small">
        Odds ratios from a logistic regression on the 15:00 state, each holding the others fixed, with 95% intervals.
        Association, not causation.
      </p>
      <Plot
        {...common}
        data={[
          {
            type: 'scatter',
            mode: 'markers',
            y: data.objectives.terms.map((t) => TERM_LABELS[t.term] ?? t.term),
            x: data.objectives.terms.map((t) => t.odds_ratio),
            error_x: {
              type: 'data',
              symmetric: false,
              array: data.objectives.terms.map((t) => t.ci_high - t.odds_ratio),
              arrayminus: data.objectives.terms.map((t) => t.odds_ratio - t.ci_low),
              color: c.muted,
            },
            marker: { color: c.accent, size: 9 },
            hovertemplate: '%{y}: odds ×%{x:.2f}<extra></extra>',
          },
        ]}
        layout={baseLayout({
          height: 280,
          xaxis: { type: 'log', title: { text: 'Odds ratio for Blue winning' } },
          yaxis: { autorange: 'reversed' },
          shapes: [{ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: 1, x1: 1, line: { color: c.muted, dash: 'dot' } }],
        })}
      />

      <h2>Champion win rates</h2>
      <input
        className="search"
        placeholder="Search a champion…"
        value={query}
        onChange={(e) => onQuery(e.target.value)}
        aria-label="Search champions"
      />
      <p className="muted small">
        {query ? `${champions.length} match${champions.length === 1 ? '' : 'es'}` : 'Ten highest and ten lowest'} ·
        champions with at least 30 games in training patches · 95% Wilson intervals
      </p>
      {champions.length > 0 && (
        <Plot
          {...common}
          data={[
            {
              type: 'scatter',
              mode: 'markers',
              y: champions.map((ch) => ch.name),
              x: champions.map((ch) => ch.win_rate * 100),
              customdata: champions.map((ch) => ch.games),
              error_x: {
                type: 'data',
                symmetric: false,
                array: champions.map((ch) => (ch.ci_high - ch.win_rate) * 100),
                arrayminus: champions.map((ch) => (ch.win_rate - ch.ci_low) * 100),
                color: c.muted,
              },
              marker: { color: champions.map((ch) => (ch.win_rate >= 0.5 ? c.blue : c.red)), size: 8 },
              hovertemplate: '%{y}: %{x:.1f}% over %{customdata} games<extra></extra>',
            },
          ]}
          layout={baseLayout({
            height: 60 + champions.length * 24,
            xaxis: { ticksuffix: '%', title: { text: 'Win rate' } },
            yaxis: { autorange: 'reversed' },
            shapes: [{ type: 'line', yref: 'paper', y0: 0, y1: 1, x0: 50, x1: 50, line: { color: c.muted, dash: 'dot' } }],
          })}
        />
      )}
    </>
  )
}
