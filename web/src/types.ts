// Mirrors lolwp/api/schemas.py and lolwp/config.py.

export type Role = 'top' | 'jungle' | 'mid' | 'bot' | 'support'
export const ROLES: Role[] = ['top', 'jungle', 'mid', 'bot', 'support']
export const ROLE_NAMES: Record<Role, string> = {
  top: 'Top',
  jungle: 'Jungle',
  mid: 'Mid',
  bot: 'Bot',
  support: 'Support',
}

export type Mode = 'draft' | 'state' | 'both'

export interface Champion {
  id: number
  name: string
  icon_url: string
  tags: string[]
}

export type Team = Record<Role, number>
export type PartialTeam = Partial<Record<Role, number>>

export const STATE_FEATURES = [
  'gold_diff',
  'cs_diff',
  'level_diff',
  'kills_diff',
  'dragons_diff',
  'grubs_diff',
  'towers_diff',
  'plates_diff',
  'wards_placed_diff',
  'wards_killed_diff',
  'first_blood_blue',
  'first_dragon_blue',
  'first_tower_blue',
] as const
export type StateFeature = (typeof STATE_FEATURES)[number]
export type GameState = Record<StateFeature, number>

export const STATE_RANGES: Record<StateFeature, [number, number]> = {
  gold_diff: [-15000, 15000],
  cs_diff: [-200, 200],
  level_diff: [-20, 20],
  kills_diff: [-30, 30],
  dragons_diff: [-3, 3],
  grubs_diff: [-6, 6],
  towers_diff: [-6, 6],
  plates_diff: [-30, 30],
  wards_placed_diff: [-60, 60],
  wards_killed_diff: [-30, 30],
  first_blood_blue: [-1, 1],
  first_dragon_blue: [-1, 1],
  first_tower_blue: [-1, 1],
}

export const STATE_LABELS: Record<StateFeature, string> = {
  gold_diff: 'Gold lead',
  cs_diff: 'CS lead',
  level_diff: 'Total level lead',
  kills_diff: 'Kill lead',
  dragons_diff: 'Dragons',
  grubs_diff: 'Voidgrubs',
  towers_diff: 'Towers',
  plates_diff: 'Turret plates',
  wards_placed_diff: 'Wards placed',
  wards_killed_diff: 'Wards cleared',
  first_blood_blue: 'First blood',
  first_dragon_blue: 'First dragon',
  first_tower_blue: 'First tower',
}

export type Minute = 10 | 15
export const MINUTES: Minute[] = [10, 15]

export interface PredictRequest {
  mode: Mode
  minute: Minute
  blue?: Team
  red?: Team
  // Only the known fields; anything left out is treated as unknown.
  state?: Partial<GameState>
}

export interface Contribution {
  feature: string
  label: string
  value_pct: number
  value: number | null
  logit: number
}

export interface Reason {
  text: string
  value_pct: number
}

export type Strength = 'toss-up' | 'slight edge' | 'favored' | 'strong favorite'

export interface PredictResponse {
  probability: number
  baseline: number
  verdict: string
  strength: Strength
  contributions: Contribution[]
  reasons: Reason[]
  split: { draft_pct: number; state_pct: number } | null
  omitted: StateFeature[]
  calibration: { lo: number; hi: number; win_rate: number; n: number } | null
  reliability: { auc: number; accuracy: number; text: string }
  minute: Minute
}

export interface ModelMetrics {
  auc: number
  log_loss: number
  accuracy: number
  brier: number
  cv_auc_mean: number
  cv_auc_std: number
}

export interface Insights {
  champions: {
    id: number
    name: string
    games: number
    win_rate: number
    ci_low: number
    ci_high: number
  }[]
  gold_curve: { gold_bin_center: number; win_rate: number; n: number }[]
  objectives: {
    terms: { term: string; odds_ratio: number; ci_low: number; ci_high: number; p_value: number }[]
    dragon_vs_tower: { estimate_diff: number; p_value: number }
  }
  blue_side: { win_rate: number; n: number; p_value: number }
  model_comparison: Record<string, ModelMetrics>
  calibration: Record<string, CalibrationBin[]>
  data: DataSummary
}

export interface CalibrationBin {
  lo: number
  hi: number
  mean_pred: number
  win_rate: number
  n: number
}

export interface DataSummary {
  train_patches: string[]
  test_patch: string | null
  n_games: number
  dropped: Record<string, number>
}

export interface Meta extends DataSummary {
  minutes: Minute[]
}
