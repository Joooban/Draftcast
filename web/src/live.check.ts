// No frontend test runner here: `npm run check:live` runs this with Node's type stripping,
// and `tsc -b` type-checks it with the rest of src.
import { createLatest } from './latest.ts'
import { emptyEntry, isInvalid, toState } from './live.ts'
import type { LiveEntry } from './live.ts'
import { ROLES } from './types.ts'

function eq(name: string, got: unknown, want: unknown) {
  const g = JSON.stringify(got)
  const w = JSON.stringify(want)
  if (g !== w) throw new Error(`${name}: got ${g}, want ${w}`)
}

function filled(stat: 'level' | 'cs' | 'gold', blue: string, red: string): LiveEntry {
  const e = emptyEntry()
  for (const r of ROLES) {
    e.players.blue[r][stat] = blue
    e.players.red[r][stat] = red
  }
  return e
}

eq('all blank', toState(emptyEntry()), {})

const nineCs = filled('cs', '100', '90')
nineCs.players.red.support.cs = ''
eq('nine CS filled', 'cs_diff' in toState(nineCs), false)

eq('levels summed', toState(filled('level', '9', '8')), { level_diff: 5 })

const kills = emptyEntry()
kills.team.blue.kills = '7'
kills.team.red.kills = '4'
eq('team kills', toState(kills), { kills_diff: 3 })

const oneSide = emptyEntry()
oneSide.team.blue.dragons = '2'
eq('one team only is unknown', toState(oneSide), {})

eq('gold with commas', toState(filled('gold', '1,200', '1000')).gold_diff, 1000)

const bad = filled('cs', '100', '90')
bad.players.blue.mid.cs = '-5'
eq('negative is unknown', 'cs_diff' in toState(bad), false)

const firsts = emptyEntry()
firsts.firsts.first_blood_blue = 'red'
firsts.firsts.first_tower_blue = 'none'
eq('firsts', toState(firsts), { first_blood_blue: -1, first_tower_blue: 0 })

// Team-total mode: the two typed totals replace the per-player sum, and per-player boxes are ignored.
const totals = filled('gold', '3000', '2500')
totals.modes.gold = 'team'
totals.totals.blue.gold = '15,400'
totals.totals.red.gold = '13,100'
eq('gold team totals', toState(totals), { gold_diff: 2300 })

const oneTotal = emptyEntry()
oneTotal.modes.cs = 'team'
oneTotal.totals.blue.cs = '300'
eq('one total only is unknown', toState(oneTotal), {})

// Switching back keeps the typed totals but uses the player boxes again.
totals.modes.gold = 'player'
eq('back to per player', toState(totals), { gold_diff: 2500 })

const mixed = filled('level', '9', '8')
mixed.modes.gold = 'team'
mixed.totals.blue.gold = '15000'
mixed.totals.red.gold = '16000'
eq('mixed modes', toState(mixed), { level_diff: 5, gold_diff: -1000 })

// A box with text that isn't a count is flagged, so it isn't silently treated as unknown.
eq('blank is fine', isInvalid(''), false)
eq('count is fine', isInvalid('1,200'), false)
eq('4.1k is invalid', isInvalid('4.1k'), true)
eq('minus is invalid', isInvalid('-'), true)

// Only the newest request may show its result; an input change also invalidates an in-flight one.
const latest = createLatest()
const first = latest.next()
eq('first is current', latest.isCurrent(first), true)
latest.invalidate()
eq('invalidated', latest.isCurrent(first), false)
const second = latest.next()
eq('newer request wins', [latest.isCurrent(first), latest.isCurrent(second)], [false, true])

console.log('live ok')
