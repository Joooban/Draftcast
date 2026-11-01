import type { FirstStat, FirstValue, LiveEntry, PlayerStat, Side, TeamStat } from '../live'
import { FIRST_STATS, SIDES, TEAM_STATS } from '../live'
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
  return (
    <input
      className="num"
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
  const setTeam = (side: Side, stat: TeamStat, v: string) =>
    onChange({ ...entry, team: { ...entry.team, [side]: { ...entry.team[side], [stat]: v } } })
  const setFirst = (f: FirstStat, v: FirstValue) => onChange({ ...entry, firsts: { ...entry.firsts, [f]: v } })

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
          number instead of assuming it's even.
        </p>
      </div>

      <div className="table-scroll">
        <table className="entry">
          <caption>Players</caption>
          <thead>
            <tr>
              <th scope="col">Role</th>
              {SIDES.map((side) =>
                PLAYER_STATS.map(({ stat, label }) => (
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
                  PLAYER_STATS.map(({ stat, label }) => (
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
