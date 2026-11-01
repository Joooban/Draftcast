import { useEffect, useState } from 'react'
import { getMeta } from '../api'
import { trainedOn } from '../format'
import type { Meta } from '../types'

export default function Footer() {
  const [meta, setMeta] = useState<Meta | null>(null)

  useEffect(() => {
    // The label is a nicety: if the models aren't loaded it simply doesn't show.
    getMeta().then(setMeta, () => setMeta(null))
  }, [])

  const label = meta && trainedOn(meta)

  return (
    <footer className="footer">
      {label && <p className="trained-on">{label}.</p>}
      This project isn't endorsed by Riot Games and doesn't reflect the views or opinions of Riot Games or anyone
      officially involved in producing or managing Riot Games properties. Riot Games and all associated properties are
      trademarks or registered trademarks of Riot Games, Inc.
    </footer>
  )
}
