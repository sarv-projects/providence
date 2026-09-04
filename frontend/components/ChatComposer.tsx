'use client'

import { Send, ChevronDown } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { apiGet } from '@/lib/api'
import { ModelMenu, type CatalogProvider } from './ModelMenu'

export type ComposerMode = 'chat' | 'research'
export type ResearchDepth = 'standard' | 'deep'
export interface ResearchLenses {
  recency: boolean
  academic: boolean
  compare: boolean
}

export const EMPTY_LENSES: ResearchLenses = { recency: false, academic: false, compare: false }

const LENS_META: { key: keyof ResearchLenses; label: string; hint: string }[] = [
  { key: 'recency', label: 'Recency', hint: 'Prefer 2024–2026 sources' },
  { key: 'academic', label: 'Academic', hint: 'Papers-first, wider arXiv pass' },
  { key: 'compare', label: 'Compare', hint: 'Structured comparison matrix' },
]

interface ChatComposerProps {
  input: string
  onInputChange: (value: string) => void
  onSubmit: (event: React.FormEvent) => void
  disabled?: boolean
  mode: ComposerMode
  onModeChange: (mode: ComposerMode) => void
  researchDepth: ResearchDepth
  onResearchDepthChange: (depth: ResearchDepth) => void
  lenses: ResearchLenses
  onLensesChange: (lenses: ResearchLenses) => void
  autonomy: string
  onAutonomyChange: (level: string) => void
  planFirst: boolean
  onPlanFirstChange: (value: boolean) => void
  selectedModel: string
  onModelChange: (model: string) => void
}


export function ChatComposer({
  input,
  onInputChange,
  onSubmit,
  disabled = false,
  mode,
  onModeChange,
  researchDepth,
  onResearchDepthChange,
  lenses,
  onLensesChange,
  autonomy,
  onAutonomyChange,
  planFirst,
  onPlanFirstChange,
  selectedModel,
  onModelChange,
}: ChatComposerProps) {
  const [modelGroups, setModelGroups] = useState<CatalogProvider[]>([])
  const [modelsLoading, setModelsLoading] = useState(true)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 300)}px`
  }, [input])

  useEffect(() => {
    apiGet<{ providers?: CatalogProvider[]; default_provider?: string; default_model?: string }>('/api/providers/catalog?discover=true')
      .then((data) => {
        setModelGroups((data.providers || []).filter((group) => group.models.length > 0))
        if (!selectedModel && data.default_provider && data.default_model) {
          onModelChange(`${data.default_provider}/${data.default_model}`)
        }
      })
      .catch(() => setModelGroups([]))
      .finally(() => setModelsLoading(false))
    // The catalog is fetched once per composer mount; selection changes must not refetch it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return (
    <div className="composer-wrap">
      <form onSubmit={onSubmit} className="composer max-w-4xl mx-auto">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          placeholder={mode === 'chat' ? 'Message Providence…' : 'What should I research?'}
          className="composer-input"
          disabled={disabled}
          rows={1}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              if (!disabled && input.trim()) onSubmit(e)
            }
          }}
        />
        <div className="composer-toolbar">
          <div className="toolbar-left">
            <ModelMenu
              selected={selectedModel}
              onSelect={(provider, model) => onModelChange(`${provider}/${model}`)}
              providers={modelGroups}
              loading={modelsLoading}
              disabled={disabled}
            />
            <div className="mode-segment" role="tablist" aria-label="Choose chat or research">
              {(['chat', 'research'] as ComposerMode[]).map((m) => (
                <button
                  key={m}
                  type="button"
                  role="tab"
                  aria-selected={mode === m}
                  disabled={disabled}
                  onClick={() => onModeChange(m)}
                  className={`mode-segment-btn${mode === m ? ' active' : ''}`}
                >
                  {m === 'chat' ? 'Chat' : 'Research'}
                </button>
              ))}
            </div>
            {mode === 'research' && (
              <>
                <div className="mode-segment subtle" role="tablist" aria-label="Research depth">
                  {(['standard', 'deep'] as ResearchDepth[]).map((d) => (
                    <button
                      key={d}
                      type="button"
                      role="tab"
                      aria-selected={researchDepth === d}
                      disabled={disabled}
                      onClick={() => onResearchDepthChange(d)}
                      className={`mode-segment-btn${researchDepth === d ? ' active' : ''}`}
                      title={d === 'standard' ? 'Default iterative research' : 'Heavy analysis with thinker + triangulation'}
                    >
                      {d === 'standard' ? 'Standard' : 'Deep'}
                    </button>
                  ))}
                </div>
                <div className="lens-pills" aria-label="Research lenses">
                  {LENS_META.map(({ key, label, hint }) => (
                    <button
                      key={key}
                      type="button"
                      disabled={disabled}
                      onClick={() => onLensesChange({ ...lenses, [key]: !lenses[key] })}
                      className={`lens-pill${lenses[key] ? ' active' : ''}`}
                      aria-pressed={lenses[key]}
                      title={hint}
                    >
                      {label}
                    </button>
                  ))}
                </div>
                <label className="toolbar-select subtle-select">
                  <span className="sr-only">Autonomy</span>
                  <select value={autonomy} onChange={(e) => onAutonomyChange(e.target.value)} disabled={disabled}>
                    <option value="L1">Auto</option>
                    <option value="L2">Plan review</option>
                    <option value="L3">Hard budget</option>
                  </select>
                  <ChevronDown size={14} />
                </label>
                <label className="plan-toggle">
                  <input type="checkbox" checked={planFirst || autonomy === 'L2'} disabled={disabled || autonomy === 'L2'} onChange={(e) => onPlanFirstChange(e.target.checked)} />
                  Edit plan
                </label>
              </>
            )}
          </div>
          <button type="submit" disabled={disabled || !input.trim()} className="send-button" aria-label="Send message">
            <Send size={17} />
          </button>
        </div>
      </form>
      <p className="composer-hint">Free models only · Enter to send · Shift + Enter for a new line</p>
    </div>
  )
}
