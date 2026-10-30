import type { Champion, PartialTeam, Role } from '../types'
import { ROLES, ROLE_NAMES } from '../types'
import ChampionPicker from './ChampionPicker'

interface Props {
  champions: Champion[]
  blue: PartialTeam
  red: PartialTeam
  onChange: (side: 'blue' | 'red', role: Role, id: number | undefined) => void
}

export default function DraftInput({ champions, blue, red, onChange }: Props) {
  const picked = new Set([...Object.values(blue), ...Object.values(red)].filter((v): v is number => v !== undefined))

  return (
    <div className="draft">
      {(['blue', 'red'] as const).map((side) => {
        const team = side === 'blue' ? blue : red
        return (
          <section key={side} className={`team team-${side}`}>
            <h3>{side === 'blue' ? 'Blue side' : 'Red side'}</h3>
            {ROLES.map((role) => (
              <div className="slot" key={role}>
                <span className="role">{ROLE_NAMES[role]}</span>
                <ChampionPicker
                  champions={champions}
                  value={team[role]}
                  disabledIds={picked}
                  label={`${side} ${ROLE_NAMES[role]}`}
                  onChange={(id) => onChange(side, role, id)}
                />
              </div>
            ))}
          </section>
        )
      })}
    </div>
  )
}
