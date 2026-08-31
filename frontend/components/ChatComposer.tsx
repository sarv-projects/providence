'use client'

import { Send, ChevronDown } from 'lucide-react'
import { useEffect, useState } from 'react'
import { apiGet } from '@/lib/api'
import { ModelMenu, type CatalogProvider } from './ModelMenu'

export type ComposerMode = 'chat' | 'research'

interface ChatComposerProps {
  input: string
  onInputChange: (value: string) => void
  onSubmit: (event: React.FormEvent) => void
  disabled?: boolean
  mode: ComposerMode
  onModeChange: (mode: ComposerMode) => void
  researchMode: string
  onResearchModeChange: (mode: string) => void
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
  researchMode,
  onResearchModeChange,
  autonomy,
  onAutonomyChange,
  planFirst,
  onPlanFirstChange,
  selectedModel,
  onModelChange,
}: ChatComposerProps) {
  const [modelGroups, setModelGroups] = useState<CatalogProvider[]>([])
  const [modelsLoading, setModelsLoading] = useState(true)

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
            <label className="toolbar-select" title="Choose chat or research">
              <span className="sr-only">Mode</span>
              <select value={mode} onChange={(e) => onModeChange(e.target.value as ComposerMode)} disabled={disabled}>
                <option value="chat">Chat</option>
                <option value="research">Research</option>
              </select>
              <ChevronDown size={14} />
            </label>
            {mode === 'research' && (
              <>
                <label className="toolbar-select subtle-select">
                  <span className="sr-only">Research depth</span>
                  <select value={researchMode} onChange={(e) => onResearchModeChange(e.target.value)} disabled={disabled}>
                    <option value="quick">Quick</option>
                    <option value="standard">Standard</option>
                    <option value="deep">Deep</option>
                    <option value="academic">Academic</option>
                    <option value="recency">Recency</option>
                    <option value="compare">Compare</option>
                  </select>
                  <ChevronDown size={14} />
                </label>
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
