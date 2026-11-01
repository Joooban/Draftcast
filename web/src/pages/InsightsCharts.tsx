import Plot, { baseLayout, plotConfig, themeColors } from '../components/Plot'
import type { Insights } from '../types'

interface Props {
  data: Insights
}

// The app's models whose predicted-vs-actual curves are worth comparing.
const CALIBRATED: { key: string; label: string }[] = [
  { key: 'state_lr', label: 'Game state, 15:00' },
  { key: 'combined_lr', label: 'Draft + game state, 15:00' },
  { key: 'state_lr_m10', label: 'Game state, 10:00' },
]

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

export default function InsightsCharts({ data }: Props) {
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

      <h2>Does the model mean what it says?</h2>
      <p className="muted small">
        Test-patch games grouped by the predicted chance for Blue. Points on the dotted line mean "when it says 70%,
        Blue wins about 70%". This is the reality check shown with each prediction.
      </p>
      <Plot
        {...common}
        data={CALIBRATED.filter((m) => data.calibration[m.key]).map((m, i) => ({
          type: 'scatter' as const,
          mode: 'lines+markers' as const,
          name: m.label,
          x: data.calibration[m.key].map((b) => b.mean_pred * 100),
          y: data.calibration[m.key].map((b) => b.win_rate * 100),
          customdata: data.calibration[m.key].map((b) => b.n),
          line: { color: [c.accent, c.blue, c.muted][i], dash: i === 2 ? ('dash' as const) : ('solid' as const) },
          hovertemplate: '%{fullData.name}: said %{x:.0f}%, Blue won %{y:.0f}% (%{customdata} games)<extra></extra>',
        }))}
        layout={baseLayout({
          height: 340,
          legend: { orientation: 'h', y: -0.25 },
          xaxis: { title: { text: 'Predicted chance for Blue' }, ticksuffix: '%', range: [0, 100] },
          yaxis: { title: { text: 'Blue actually won' }, ticksuffix: '%', range: [0, 100] },
          shapes: [{ type: 'line', x0: 0, x1: 100, y0: 0, y1: 100, line: { color: c.muted, dash: 'dot' } }],
        })}
      />
    </>
  )
}
