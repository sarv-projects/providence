'use client'

import { useEffect, useMemo, useState } from 'react'
import { Check, SlidersHorizontal } from 'lucide-react'
import { apiGet, apiPost } from '@/lib/api'
import { ProviderSettings, AppShell, EMPTY_LENSES } from '@/components'
import type { ResearchDepth, ResearchLenses } from '@/components'
import type { CatalogProvider } from '@/components'

interface Provider { name: string; base_url: string; has_auth: boolean; models: string[] }

const LENS_FIELDS: { key: keyof ResearchLenses; label: string; hint: string }[] = [
  { key: 'recency', label: 'Recency lens', hint: 'Prefer 2024–2026 sources' },
  { key: 'academic', label: 'Academic lens', hint: 'Papers-first, wider arXiv pass' },
  { key: 'compare', label: 'Compare lens', hint: 'Structured comparison matrix' },
]

// Stored modes from before the lens refactor migrate to depth + implied lens.
const LEGACY_MODE: Record<string, { depth: ResearchDepth; lens: keyof ResearchLenses | null }> = {
  recency: { depth: 'standard', lens: 'recency' },
  academic: { depth: 'deep', lens: 'academic' },
  compare: { depth: 'standard', lens: 'compare' },
  quick: { depth: 'standard', lens: null },
  'ultra-long': { depth: 'deep', lens: null },
}

export default function SettingsPage() {
  const [providers, setProviders] = useState<Provider[]>([])
  const [depth, setDepth] = useState<ResearchDepth>('standard')
  const [lenses, setLenses] = useState<ResearchLenses>({ ...EMPTY_LENSES })
  const [autonomy, setAutonomy] = useState('L1')
  const [maxCost, setMaxCost] = useState(5)
  const [maxIterations, setMaxIterations] = useState(3)
  const [selectedModel, setSelectedModel] = useState('opencode_free/nemotron-3-ultra-free')
  const [catalogModels, setCatalogModels] = useState<string[]>([])
  const [saved, setSaved] = useState(false)
  const [saveError, setSaveError] = useState('')

  useEffect(() => {
    apiGet<Provider[]>('/api/providers').then(setProviders).catch(() => setProviders([]))
    apiGet<{ providers?: CatalogProvider[] }>('/api/providers/catalog?discover=true')
      .then((data) => {
        const flat: string[] = []
        for (const p of data.providers || []) {
          for (const m of p.models || []) {
            if (m.free) flat.push(`${p.id}/${m.id}`)
          }
        }
        setCatalogModels(flat)
      })
      .catch(() => setCatalogModels([]))
    apiGet<any>('/api/settings').then((data) => {
      let base = { ...EMPTY_LENSES }
      if (data.mode) {
        const legacy = LEGACY_MODE[data.mode]
        if (legacy) {
          setDepth(legacy.depth)
          if (legacy.lens) base = { ...base, [legacy.lens]: true }
        } else if (data.mode === 'deep' || data.mode === 'standard') {
          setDepth(data.mode)
        }
      }
      base = {
        recency: !!data.lens_recency || base.recency,
        academic: !!data.lens_academic || base.academic,
        compare: !!data.lens_compare || base.compare,
      }
      setLenses(base)
      if (data.autonomy) setAutonomy(data.autonomy)
      if (data.max_cost != null) setMaxCost(Math.max(0, Number(data.max_cost) || 0))
      if (data.max_iterations != null) setMaxIterations(Math.min(50, Math.max(1, Number(data.max_iterations) || 3)))
      if (data.default_model) setSelectedModel(data.default_model)
    }).catch(() => {})
  }, [])

  const modelKnown = useMemo(
    () => catalogModels.length === 0 || catalogModels.includes(selectedModel),
    [catalogModels, selectedModel]
  )

  async function save() {
    setSaveError('')
    const cost = Math.max(0, Number(maxCost) || 0)
    const iters = Math.min(50, Math.max(1, Number(maxIterations) || 3))
    setMaxCost(cost)
    setMaxIterations(iters)
    try {
      await apiPost('/api/settings', {
        mode: depth,
        autonomy,
        max_cost: cost,
        max_iterations: iters,
        default_model: selectedModel,
        lens_recency: lenses.recency,
        lens_academic: lenses.academic,
        lens_compare: lenses.compare,
      })
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2200)
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : 'Failed to save settings')
    }
  }

  return (
    <AppShell title="Settings" description="Choose how Providence responds, researches, and connects to models." action={<button className="primary-button" onClick={save}>{saved ? <><Check size={15} /> Saved</> : 'Save changes'}</button>}>
      <div className="page-stack">
        {saveError && <div className="notice notice-error">{saveError}</div>}
        <section className="settings-section">
          <div className="section-title"><SlidersHorizontal size={18} /><div><h2>Research defaults</h2><p>These defaults apply when the composer does not override them.</p></div></div>
          <div className="settings-grid">
            <label>Research depth
              <select value={depth} onChange={(e) => setDepth(e.target.value as ResearchDepth)}>
                <option value="standard">Standard — default iterative research</option>
                <option value="deep">Deep — thinker + triangulation</option>
              </select>
            </label>
            <label>Autonomy<select value={autonomy} onChange={(e) => setAutonomy(e.target.value)}><option value="L1">Automatic</option><option value="L2">Review plan first</option><option value="L3">Hard budget cap</option></select></label>
            <label>Maximum budget<input type="number" min="0" step="0.5" value={maxCost} onChange={(e) => setMaxCost(Number(e.target.value))} /></label>
            <label>Maximum iterations<input type="number" min="1" max="50" value={maxIterations} onChange={(e) => setMaxIterations(Number(e.target.value))} /></label>
          </div>
          <div className="lens-checks">
            <span className="plan-field-label">Default lenses (combinable)</span>
            <div className="lens-check-row">
              {LENS_FIELDS.map(({ key, label, hint }) => (
                <label key={key} className={`lens-check${lenses[key] ? ' active' : ''}`} title={hint}>
                  <input
                    type="checkbox"
                    checked={lenses[key]}
                    onChange={(e) => setLenses((prev) => ({ ...prev, [key]: e.target.checked }))}
                  />
                  <span><strong>{label}</strong><small>{hint}</small></span>
                </label>
              ))}
            </div>
          </div>
          <label className="wide-field">Default model
            {catalogModels.length > 0 ? (
              <select value={modelKnown ? selectedModel : '__custom__'} onChange={(e) => { if (e.target.value !== '__custom__') setSelectedModel(e.target.value) }}>
                {catalogModels.map((m) => (
                  <option key={m} value={m}>{m}</option>
                ))}
                {!modelKnown && <option value="__custom__">{selectedModel} (unavailable — pick a replacement)</option>}
              </select>
            ) : (
              <input value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} placeholder="provider/model" />
            )}
            <small>
              {!modelKnown && catalogModels.length > 0
                ? 'The saved model is no longer in the catalog. Pick a replacement above.'
                : 'Validated against the runtime provider catalog.'}
            </small>
          </label>
        </section>

        <ProviderSettings providers={providers} onProvidersChange={setProviders} />
      </div>
    </AppShell>
  )
}
