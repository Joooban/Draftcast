// No frontend test runner here: `npm run check:live` runs this with Node's type stripping,
// and `tsc -b` type-checks it with the rest of src.
import { emptyEntry, toState } from './live.ts'
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

console.log('live ok')
