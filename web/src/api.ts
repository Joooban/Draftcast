import type { Champion, Insights, PredictRequest, PredictResponse } from './types'

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init)
  if (!res.ok) {
    let detail = `Request failed (${res.status})`
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // not JSON; keep the status message
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const getChampions = () => request<Champion[]>('/api/champions')

export const predict = (req: PredictRequest) =>
  request<PredictResponse>('/api/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })

export const getInsights = () => request<Insights>('/api/insights')
