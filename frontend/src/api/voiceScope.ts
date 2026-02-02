export type MetricKey = 'stability' | 'clarity' | 'resonance' | 'brightness' | 'voiced'

export type InsightKey = 'strong' | 'weak' | 'gap' | 'balance'

export type ReliabilityGrade = 'A' | 'B' | 'C'

export const METRIC_LABELS_JP: Record<MetricKey, string> = {
  stability: '安定性',
  clarity: 'クリアさ',
  resonance: '響き・抜け',
  brightness: '明るさ',
  voiced: '発声率',
}

export interface MetricsBundle {
  score: Record<MetricKey, number>
  raw: Record<string, unknown>
}

export interface Insight {
  key: InsightKey
  title: string
  body: string
  highlight: MetricKey | null
}

export interface Summary {
  overall: number
  match: number | null
  reliability_grade: ReliabilityGrade
  voiced_ratio: number
  pitch_range_note: string | null
}

export interface AnalyzeResponse {
  source: MetricsBundle
  target: MetricsBundle | null
  summary: Summary
  insights: Insight[]
  balance: number
  recommended_highlight_key: MetricKey | null
}

export interface FeedbackResponse {
  policy: { note: string }
  focus_key: MetricKey
  observations: string[]
  search_queries: string[]
  optional_queries: string[]
}

export async function analyzeVoice(params: {
  source: File
  target?: File
  sr: number
  consultationText?: string
  mode?: string
}): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append('source', params.source)
  if (params.target) form.append('target', params.target)
  form.append('sr', String(params.sr))
  if (params.consultationText) form.append('consultation_text', params.consultationText)
  if (params.mode) form.append('mode', params.mode)

  const res = await fetch('/v1/analyze', {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Analyze failed (${res.status})`)
  }
  return (await res.json()) as AnalyzeResponse
}

export async function getFeedback(params: {
  analysis: AnalyzeResponse
  consultationText?: string
  focusKey?: MetricKey
}): Promise<FeedbackResponse> {
  const res = await fetch('/v1/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      consultation_text: params.consultationText ?? null,
      focus_key: params.focusKey ?? null,
      analysis: params.analysis,
    }),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Feedback failed (${res.status})`)
  }
  return (await res.json()) as FeedbackResponse
}
