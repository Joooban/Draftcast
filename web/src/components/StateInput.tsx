import type { EntryMode, FirstStat, FirstValue, LiveEntry, PlayerStat, Side, TeamStat } from '../live'
import { FIRST_STATS, SIDES, TEAM_STATS, isInvalid } from '../live'
import type { Minute } from '../types'
import { MINUTES, ROLES, ROLE_NAMES, STATE_LABELS } from '../types'

interface Props {
  entry: LiveEntry
  onChange: (entry: LiveEntry) => void
  minute: Minute
  onMinute: (minute: Minute) => void
}

const PLAYER_STATS: { stat: PlayerStat; label: string }[] = [
  { stat: 'level', label: 'Level' },
  { stat: 'cs', label: 'CS' },
  { stat: 'gold', label: 'Gold' },
]

const TOTAL_LABELS: Record<PlayerStat, string> = { level: 'Total levels', cs: 'Total CS', gold: 'Total gold' }

const MODE_OPTIONS: { value: EntryMode; label: string }[] = [
  { value: 'player', label: 'Per player' },
  { value: 'team', label: 'Team total' },
]

const TEAM_LABELS: Record<TeamStat, string> = {
  kills: 'Kills',
  dragons: 'Dragons',
  towers: 'Towers',
  grubs: 'Voidgrubs',
  plates: 'Turret plates',
  wards_placed: 'Wards placed',
  wards_killed: 'Wards cleared',
}

const FIRST_OPTIONS: { value: FirstValue; label: string; cls: string }[] = [
  { value: 'unknown', label: '?', cls: '' },
  { value: 'blue', label: 'Blue', cls: 'blue' },
  { value: 'none', label: 'None', cls: '' },
  { value: 'red', label: 'Red', cls: 'red' },
]

const SIDE_NAMES: Record<Side, string> = { blue: 'Blue', red: 'Red' }

function NumberBox({ value, label, onChange }: { value: string; label: string; onChange: (v: string) => void }) {
  const invalid = isInvalid(value)
  return (
    <input
      className={invalid ? 'num invalid' : 'num'}
      aria-invalid={invalid}
      title={invalid ? 'Whole numbers only (commas are fine)' : undefined}
      type="text"
      inputMode="numeric"
      autoComplete="off"
      aria-label={label}
      value={value}
      onFocus={(e) => e.target.select()}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}

export default function StateInput({ entry, onChange, minute, onMinute }: Props) {
  const setPlayer = (side: Side, role: (typeof ROLES)[number], stat: PlayerStat, v: string) =>
    onChange({
      ...entry,
      players: { ...entry.players, [side]: { ...entry.players[side], [role]: { ...entry.players[side][role], [stat]: v } } },
    })
  const setTotal = (side: Side, stat: PlayerStat, v: string) =>
    onChange({ ...entry, totals: { ...entry.totals, [side]: { ...entry.totals[side], [stat]: v } } })
  const setMode = (stat: PlayerStat, m: EntryMode) => onChange({ ...entry, modes: { ...entry.modes, [stat]: m } })
  const perPlayer = PLAYER_STATS.filter(({ stat }) => entry.modes[stat] === 'player')
  const asTotals = PLAYER_STATS.filter(({ stat }) => entry.modes[stat] === 'team')

  const setTeam = (side: Side, stat: TeamStat, v: string) =>
    onChange({ ...entry, team: { ...entry.team, [side]: { ...entry.team[side], [stat]: v } } })
  const setFirst = (f: FirstStat, v: FirstValue) => onChange({ ...entry, firsts: { ...entry.firsts, [f]: v } })

  // Only boxes on screen count: a hidden box isn't used, so its text can't be wrong.
  const invalidCount = [
    ...SIDES.flatMap((side) => ROLES.flatMap((r) => perPlayer.map(({ stat }) => entry.players[side][r][stat]))),
    ...SIDES.flatMap((side) => asTotals.map(({ stat }) => entry.totals[side][stat])),
    ...SIDES.flatMap((side) => Object.values(entry.team[side])),
  ].filter(isInvalid).length

  return (
    <div className="live">
      <div className="live-head">
        <div className="seg" role="radiogroup" aria-label="Game time">
          {MINUTES.map((m) => (
            <button
              key={m}
              type="button"
              role="radio"
              aria-checked={minute === m}
              className={minute === m ? 'seg-btn active' : 'seg-btn'}
              onClick={() => onMinute(m)}
            >
              At {m}:00
            </button>
          ))}
        </div>
        <p className="muted small">
          Type what the scoreboard shows. Leave a box empty if you can't see it: the model then works without that
          number instead of assuming it's even. Watching a broadcast? Switch gold, CS or levels to team totals.
        </p>
      </div>

      <div className="modes">
        {PLAYER_STATS.map(({ stat, label }) => (
          <div key={stat} className="first">
            <span className="small">{label}</span>
            <div className="seg" role="radiogroup" aria-label={`Enter ${label === 'CS' ? 'CS' : label.toLowerCase()} as`}>
              {MODE_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  role="radio"
                  aria-checked={entry.modes[stat] === o.value}
                  className={entry.modes[stat] === o.value ? 'seg-btn active' : 'seg-btn'}
                  onClick={() => setMode(stat, o.value)}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {perPlayer.length > 0 && (
        <div className="table-scroll">
          <table className="entry">
            <caption>Players</caption>
            <thead>
              <tr>
                <th scope="col">Role</th>
                {SIDES.map((side) =>
                  perPlayer.map(({ stat, label }) => (
                    <th key={`${side}-${stat}`} scope="col" className={side}>
                      {SIDE_NAMES[side]} {label.toLowerCase()}
                    </th>
                  )),
                )}
              </tr>
            </thead>
            <tbody>
              {ROLES.map((role) => (
                <tr key={role}>
                  <th scope="row">{ROLE_NAMES[role]}</th>
                  {SIDES.map((side) =>
                    perPlayer.map(({ stat, label }) => (
                      <td key={`${side}-${stat}`} data-label={`${SIDE_NAMES[side]} ${label.toLowerCase()}`}>
                        <NumberBox
                          value={entry.players[side][role][stat]}
                          label={`${SIDE_NAMES[side]} ${ROLE_NAMES[role].toLowerCase()} ${label === 'CS' ? 'CS' : label.toLowerCase()}`}
                          onChange={(v) => setPlayer(side, role, stat, v)}
                        />
                      </td>
                    )),
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="table-scroll">
        <table className="entry entry-team">
          <caption>Teams</caption>
          <thead>
            <tr>
              <th scope="col">Stat</th>
              <th scope="col" className="blue">Blue</th>
              <th scope="col" className="red">Red</th>
            </tr>
          </thead>
          <tbody>
            {asTotals.map(({ stat }) => (
              <tr key={`total-${stat}`}>
                <th scope="row">{TOTAL_LABELS[stat]}</th>
                {SIDES.map((side) => (
                  <td key={side} data-label={SIDE_NAMES[side]}>
                    <NumberBox
                      value={entry.totals[side][stat]}
                      label={`${SIDE_NAMES[side]} ${TOTAL_LABELS[stat] === 'Total CS' ? 'total CS' : TOTAL_LABELS[stat].toLowerCase()}`}
                      onChange={(v) => setTotal(side, stat, v)}
                    />
                  </td>
                ))}
              </tr>
            ))}
            {TEAM_STATS.map((stat) => (
              <tr key={stat}>
                <th scope="row">{TEAM_LABELS[stat]}</th>
                {SIDES.map((side) => (
                  <td key={side} data-label={SIDE_NAMES[side]}>
                    <NumberBox
                      value={entry.team[side][stat]}
                      label={`${SIDE_NAMES[side]} ${TEAM_LABELS[stat].toLowerCase()}`}
                      onChange={(v) => setTeam(side, stat, v)}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {invalidCount > 0 && (
        <p className="error small" role="alert">
          {invalidCount} box{invalidCount === 1 ? '' : 'es'} {invalidCount === 1 ? "isn't a" : "aren't"} whole number
          {invalidCount === 1 ? '' : 's'} (outlined in red). They count as unknown until fixed.
        </p>
      )}

      <div className="firsts">
        {FIRST_STATS.map((f) => (
          <div key={f} className="first">
            <span className="small">{STATE_LABELS[f]}</span>
            <div className="seg" role="radiogroup" aria-label={STATE_LABELS[f]}>
              {FIRST_OPTIONS.map((o) => (
                <button
                  key={o.value}
                  type="button"
                  role="radio"
                  aria-checked={entry.firsts[f] === o.value}
                  aria-label={o.value === 'unknown' ? 'Unknown' : o.label}
                  className={entry.firsts[f] === o.value ? `seg-btn active ${o.cls}` : 'seg-btn'}
                  onClick={() => setFirst(f, o.value)}
                >
                  {o.label}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
