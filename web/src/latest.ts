// Tracks which in-flight request may still show its result. Starting a new request, or changing
// any input, makes older ones stale, so a slow response can't show a result for inputs that are gone.
export function createLatest() {
  let current = 0
  return {
    next: () => ++current,
    invalidate: () => {
      current += 1
    },
    isCurrent: (token: number) => token === current,
  }
}
