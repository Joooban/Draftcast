import type { DataSummary } from './types'

/** "Trained on patches 16.18–16.20, tested on 16.21 (24,683 SG2 games)", or null if unknown. */
export function trainedOn(data: DataSummary): string | null {
  const patches = data.train_patches
  if (!patches.length || !data.test_patch) return null
  const range = patches.length === 1 ? patches[0] : `${patches[0]}–${patches[patches.length - 1]}`
  return `Trained on patches ${range}, tested on ${data.test_patch} (${data.n_games.toLocaleString()} SG2 games)`
}
