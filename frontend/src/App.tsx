import './App.css'
import { useMemo, useState } from 'react'
import {
  analyzeVoice,
  getFeedback,
  METRIC_LABELS_JP,
  type AnalyzeResponse,
  type FeedbackResponse,
  type MetricKey,
} from './api/voiceScope'
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from 'recharts'

function AngleTick(props: any) {
  const { x, y, payload, activeKey } = props
  const metricKey = payload?.payload?.key as MetricKey | undefined
  const isActive = Boolean(activeKey && metricKey === activeKey)

  const fill = isActive ? '#4c1d95' : '#334155'
  const fontWeight = isActive ? 800 : 600

  return (
    <g>
      <text x={x} y={y} textAnchor="middle" dominantBaseline="middle" fill={fill} fontWeight={fontWeight}>
        {payload?.value}
      </text>
      {isActive ? (
        <line x1={x - 22} y1={y + 10} x2={x + 22} y2={y + 10} stroke="#7C3AED" strokeWidth={3} />
      ) : null}
    </g>
  )
}

function App() {
  const [mode, setMode] = useState<'compare' | 'quick'>('compare')
  const [sr, setSr] = useState<number>(44100)
  const [consultationText, setConsultationText] = useState('')

  const [sourceFile, setSourceFile] = useState<File | null>(null)
  const [targetFile, setTargetFile] = useState<File | null>(null)

  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null)
  const [selectedInsightIndex, setSelectedInsightIndex] = useState(0)
  const [feedback, setFeedback] = useState<FeedbackResponse | null>(null)

  const selectedHighlightKey: MetricKey | null =
    analysis?.insights?.[selectedInsightIndex]?.highlight ?? analysis?.recommended_highlight_key ?? null

  const radarData = useMemo(() => {
    if (!analysis) return []
    const keys: MetricKey[] = ['stability', 'clarity', 'resonance', 'brightness', 'voiced']
    return keys.map((k) => ({
      key: k,
      label: METRIC_LABELS_JP[k],
      source: analysis.source.score[k] ?? 0,
      target: analysis.target?.score?.[k] ?? 70,
      highlight: selectedHighlightKey === k,
    }))
  }, [analysis, selectedHighlightKey])

  async function runAnalyze() {
    setError(null)
    setFeedback(null)
    if (!sourceFile) {
      setError('Source（自分の声）を選択してください')
      return
    }

    setIsLoading(true)
    try {
      const result = await analyzeVoice({
        source: sourceFile,
        target: mode === 'compare' ? targetFile ?? undefined : undefined,
        sr,
        consultationText: consultationText.trim() || undefined,
        mode,
      })
      setAnalysis(result)
      setSelectedInsightIndex(0)

      const fb = await getFeedback({
        analysis: result,
        consultationText: consultationText.trim() || undefined,
        focusKey: result.recommended_highlight_key ?? undefined,
      })
      setFeedback(fb)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setIsLoading(false)
    }
  }

  async function rerunFeedback(focusKey: MetricKey | null) {
    if (!analysis) return
    setError(null)
    try {
      const fb = await getFeedback({
        analysis,
        consultationText: consultationText.trim() || undefined,
        focusKey: focusKey ?? undefined,
      })
      setFeedback(fb)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const focusKey: MetricKey | null = selectedHighlightKey
  const focusLabel = focusKey ? METRIC_LABELS_JP[focusKey] : null
  const focusSource = focusKey && analysis ? analysis.source.score[focusKey] : null
  const focusTarget =
    focusKey && analysis ? (analysis.target?.score?.[focusKey] ?? 70) : null
  const focusDiff =
    typeof focusSource === 'number' && typeof focusTarget === 'number' ? focusSource - focusTarget : null

  const sourceStabilityNote = analysis?.source?.raw?.['stability_note']
  const targetStabilityNote = analysis?.target?.raw?.['stability_note']

  const sourceRaw = analysis?.source?.raw ?? null
  const targetRaw = analysis?.target?.raw ?? null

  const formatNumber = (v: unknown, digits = 2) => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return '—'
    const p = Math.pow(10, digits)
    return String(Math.round(v * p) / p)
  }

  const rawHintFor = (key: MetricKey, raw: Record<string, unknown> | null) => {
    if (!raw) return null
    if (key === 'brightness') return `centroid_hz_median: ${formatNumber(raw['centroid_hz_median'], 0)} Hz`
    if (key === 'resonance') return `rolloff85_hz_median: ${formatNumber(raw['rolloff85_hz_median'], 0)} Hz`
    if (key === 'clarity') return `flatness_median: ${formatNumber(raw['flatness_median'], 4)}`
    if (key === 'voiced') return `voiced_ratio: ${formatNumber(raw['voiced_ratio'], 3)}`
    if (key === 'stability') {
      const std = formatNumber(raw['f0_residual_std_cents'], 1)
      const used = formatNumber(raw['f0_used_frames'], 0)
      const note = raw['stability_note'] ? String(raw['stability_note']) : '—'
      return `residual_std: ${std} cents / used_frames: ${used} / note: ${note}`
    }
    return null
  }

  return (
    <div className="container">
      <header className="header">
        <div>
          <div className="title">Voice-Scope</div>
          <div className="subtitle">理想声とのギャップを可視化（MVP）</div>
        </div>
        <div className="headerRight">
          <span className="pill">Backend: /v1</span>
        </div>
      </header>

      <section className="card">
        <div className="grid2">
          <div>
            <div className="label">モード</div>
            <div className="segmented">
              <button
                className={mode === 'compare' ? 'seg active' : 'seg'}
                onClick={() => setMode('compare')}
                type="button"
              >
                比較
              </button>
              <button
                className={mode === 'quick' ? 'seg active' : 'seg'}
                onClick={() => setMode('quick')}
                type="button"
              >
                クイック
              </button>
            </div>
          </div>
          <div>
            <div className="label">サンプルレート</div>
            <select className="select" value={sr} onChange={(e) => setSr(Number(e.target.value))}>
              {[16000, 22050, 44100, 48000].map((v) => (
                <option key={v} value={v}>
                  {v} Hz
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="label" style={{ marginTop: 14 }}>
          悩み（任意）
        </div>
        <textarea
          className="textarea"
          value={consultationText}
          placeholder="例：高い声になると苦しい気がする / 声がこもる / 通らない など"
          onChange={(e) => setConsultationText(e.target.value)}
        />

        <div className="grid2" style={{ marginTop: 14 }}>
          <div>
            <div className="label">Source（必須）</div>
            <input
              className="file"
              type="file"
              accept="audio/*,video/*"
              onChange={(e) => setSourceFile(e.target.files?.[0] ?? null)}
            />
            <div className="hint">{sourceFile ? sourceFile.name : '音声ファイルを選択'}</div>
          </div>
          <div>
            <div className="label">Target（任意）</div>
            <input
              className="file"
              type="file"
              accept="audio/*,video/*"
              disabled={mode !== 'compare'}
              onChange={(e) => setTargetFile(e.target.files?.[0] ?? null)}
            />
            <div className="hint">{mode !== 'compare' ? 'クイックでは未使用' : targetFile ? targetFile.name : '未選択（基準70）'}</div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 12, marginTop: 16, alignItems: 'center' }}>
          <button className="primary" type="button" disabled={isLoading} onClick={runAnalyze}>
            {isLoading ? '解析中…' : '解析する'}
          </button>
          {error ? <div className="error">{error}</div> : null}
        </div>
      </section>

      {analysis ? (
        <section className="grid2" style={{ marginTop: 16 }}>
          <div className="card">
            <div className="cardTitle">比較レーダー</div>
            <div className="chartBox">
              <ResponsiveContainer width="100%" height={360}>
                <RadarChart data={radarData} outerRadius="78%">
                  <PolarGrid />
                  <PolarAngleAxis dataKey="label" tick={<AngleTick activeKey={selectedHighlightKey} />} />
                  <PolarRadiusAxis domain={[0, 100]} />
                  <Radar dataKey="target" stroke="#EC4899" fill="#EC4899" fillOpacity={0.10} />
                  <Radar dataKey="source" stroke="#7C3AED" fill="#7C3AED" fillOpacity={0.22} />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <div className="hint">
              推奨フォーカス：<b>{selectedHighlightKey ? METRIC_LABELS_JP[selectedHighlightKey] : '—'}</b>
            </div>
          </div>

          <div className="card">
            <div className="cardTitle">サマリー</div>
            <div className="stats">
              <div className="stat">
                <div className="statNum">{analysis.summary.overall}</div>
                <div className="statLabel">総合</div>
              </div>
              <div className="stat">
                <div className="statNum">{analysis.summary.match ?? '—'}</div>
                <div className="statLabel">一致度</div>
              </div>
              <div className="stat">
                <div className="statNum">{Math.round(analysis.summary.voiced_ratio * 100)}%</div>
                <div className="statLabel">発声率</div>
              </div>
              <div className="stat">
                <div className="statNum">{analysis.summary.pitch_range_note ?? '—'}</div>
                <div className="statLabel">音域(参考)</div>
              </div>
            </div>

            <div className="divider" />

            <div className="cardTitle">フォーカス詳細</div>
            <div className="focusCard">
              <div className="focusHead">
                <div>
                  <div className="focusTitle">{focusLabel ?? '—'}</div>
                  <div className="hint">要点クリックでこの表示も切り替わります</div>
                </div>
                <div className={focusDiff !== null && focusDiff >= 0 ? 'focusDiff plus' : 'focusDiff minus'}>
                  {focusDiff === null ? '—' : `${focusDiff >= 0 ? '+' : ''}${focusDiff}`}
                </div>
              </div>

              <div className="focusRows">
                <div className="focusRow">
                  <div className="focusKey">Source</div>
                  <div className="focusVal">{focusSource ?? '—'}</div>
                </div>
                <div className="focusRow">
                  <div className="focusKey">Target</div>
                  <div className="focusVal">{focusTarget ?? '—'}</div>
                </div>
              </div>
              {focusKey === 'stability' && sourceStabilityNote && sourceStabilityNote !== 'ok' ? (
                <div className="hint" style={{ marginTop: 8 }}>
                  Source stability: {String(sourceStabilityNote)}
                </div>
              ) : null}
              {focusKey === 'stability' && analysis.target && targetStabilityNote && targetStabilityNote !== 'ok' ? (
                <div className="hint" style={{ marginTop: 4 }}>
                  Target stability: {String(targetStabilityNote)}
                </div>
              ) : null}
              {analysis.target ? null : (
                <div className="hint" style={{ marginTop: 8 }}>
                  ※ Target未設定のときは基準70で差分計算
                </div>
              )}

              <div className="hint" style={{ marginTop: 10 }}>
                詳細（スコアが0/100に張り付くときの原因確認）
              </div>
              <div className="hint" style={{ marginTop: 4, lineHeight: 1.5 }}>
                <div>Source: {focusKey ? rawHintFor(focusKey, sourceRaw) ?? '—' : '—'}</div>
                <div>Target: {focusKey && analysis.target ? rawHintFor(focusKey, targetRaw) ?? '—' : '—'}</div>
                <div style={{ marginTop: 4 }}>
                  ※ 100は「一定以上で上限到達」、0は「算出不能」または「範囲外で下限到達」の可能性があります
                </div>
              </div>
            </div>

            <div className="divider" />

            <div className="cardTitle">要点（クリック）</div>
            <div className="insights">
              {analysis.insights.map((it, i) => (
                <button
                  key={`${it.key}-${i}`}
                  type="button"
                  className={i === selectedInsightIndex ? 'insight active' : 'insight'}
                  onClick={() => {
                    setSelectedInsightIndex(i)
                    void rerunFeedback(it.highlight)
                  }}
                >
                  <div className="insightTitle">{it.title}</div>
                  <div className="insightBody">{it.body}</div>
                </button>
              ))}
            </div>
          </div>
        </section>
      ) : null}

      {feedback ? (
        <section className="card" style={{ marginTop: 16 }}>
          <div className="cardTitle">AI返答（検索ナビ）</div>
          <div className="hint">{feedback.policy.note}</div>

          <div className="grid2" style={{ marginTop: 12 }}>
            <div>
              <div className="label">観察（断定しない）</div>
              <ul className="list">
                {feedback.observations.map((o, i) => (
                  <li key={i}>{o}</li>
                ))}
              </ul>
            </div>
            <div>
              <div className="label">調べ方（検索クエリ例）</div>
              <div className="chips">
                {feedback.search_queries.map((q) => (
                  <span className="chip" key={q}>
                    {q}
                  </span>
                ))}
              </div>
              <div className="label" style={{ marginTop: 10 }}>
                追加で良く使う
              </div>
              <div className="chips">
                {feedback.optional_queries.map((q) => (
                  <span className="chip" key={q}>
                    {q}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </section>
      ) : null}

      <footer className="footer">© Voice-Scope (local dev)</footer>
    </div>
  )
}

export default App
