// Live-game entry as typed from the scoreboard, and its conversion to the API's game state.
// A blank box means "unknown": a summed stat is known only when every player's box parses,
// and a team stat only when both teams' boxes do. Unknown fields are left out, never sent as 0.
// Imports use .ts extensions so `npm run check:live` can run this file under Node directly.
import type { Role, StateFeature } from './types.ts'
import { ROLES } from './types.ts'

export type Side = 'blue' | 'red'
export const SIDES: Side[] = ['blue', 'red']

export type PlayerStat = 'level' | 'cs' | 'gold'
export type PlayerStats = Record<PlayerStat, string>

export const TEAM_STATS = ['kills', 'dragons', 'towers', 'grubs', 'plates', 'wards_placed', 'wards_killed'] as const
export type TeamStat = (typeof TEAM_STATS)[number]

export const FIRST_STATS = ['first_blood_blue', 'first_dragon_blue', 'first_tower_blue'] as const
export type FirstStat = (typeof FIRST_STATS)[number]
export type FirstValue = 'blue' | 'none' | 'red' | 'unknown'

export interface LiveEntry {
  players: Record<Side, Record<Role, PlayerStats>>
  team: Record<Side, Record<TeamStat, string>>
  firsts: Record<FirstStat, FirstValue>
}

const PLAYER_FEATURE: Record<PlayerStat, StateFeature> = { level: 'level_diff', cs: 'cs_diff', gold: 'gold_diff' }
const FIRST_VALUE: Record<Exclude<FirstValue, 'unknown'>, number> = { blue: 1, none: 0, red: -1 }

const blankPlayers = () =>
  Object.fromEntries(ROLES.map((r) => [r, { level: '', cs: '', gold: '' }])) as Record<Role, PlayerStats>
const blankTeam = () => Object.fromEntries(TEAM_STATS.map((s) => [s, ''])) as Record<TeamStat, string>

export function emptyEntry(): LiveEntry {
  return {
    players: { blue: blankPlayers(), red: blankPlayers() },
    team: { blue: blankTeam(), red: blankTeam() },
    firsts: { first_blood_blue: 'unknown', first_dragon_blue: 'unknown', first_tower_blue: 'unknown' },
  }
}

/** A non-negative whole number, ignoring commas and spaces; null for blank or anything else. */
export function parseCount(text: string): number | null {
  const t = text.replace(/[,\s]/g, '')
  return /^\d+$/.test(t) ? Number(t) : null
}

export function toState(e: LiveEntry): Partial<Record<StateFeature, number>> {
  const out: Partial<Record<StateFeature, number>> = {}
  for (const stat of Object.keys(PLAYER_FEATURE) as PlayerStat[]) {
    const totals = SIDES.map((side) => ROLES.map((r) => parseCount(e.players[side][r][stat])))
    if (totals.every((team) => team.every((v) => v !== null))) {
      const [blue, red] = totals.map((team) => team.reduce<number>((s, v) => s + (v ?? 0), 0))
      out[PLAYER_FEATURE[stat]] = blue - red
    }
  }
  for (const stat of TEAM_STATS) {
    const blue = parseCount(e.team.blue[stat])
    const red = parseCount(e.team.red[stat])
    if (blue !== null && red !== null) out[`${stat}_diff` as StateFeature] = blue - red
  }
  for (const f of FIRST_STATS) {
    const v = e.firsts[f]
    if (v !== 'unknown') out[f] = FIRST_VALUE[v]
  }
  return out
}
