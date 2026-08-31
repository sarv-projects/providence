'use client'

import { useEffect, useState } from 'react'
import { Check, SlidersHorizontal } from 'lucide-react'
import { apiGet, apiPost } from '@/lib/api'
import { ProviderSettings, AppShell } from '@/components'

interface Provider { name: string; base_url: string; has_auth: boolean; models: string[] }

export default function SettingsPage() {
  const [providers, setProviders] = useState<Provider[]>([])
  const [mode, setMode] = useState('standard')
  const [autonomy, setAutonomy] = useState('L1')
  const [maxCost, setMaxCost] = useState(5)
  const [maxIterations, setMaxIterations] = useState(3)
  const [selectedModel, setSelectedModel] = useState('opencode_free/nemotron-3-ultra-free')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    apiGet<Provider[]>('/api/providers').then(setProviders).catch(() => setProviders([]))
    apiGet<any>('/api/settings').then((data) => {
      if (data.mode) setMode(data.mode)
      if (data.autonomy) setAutonomy(data.autonomy)
      if (data.max_cost != null) setMaxCost(data.max_cost)
      if (data.max_iterations != null) setMaxIterations(data.max_iterations)
      if (data.default_model) setSelectedModel(data.default_model)
    }).catch(() => {})
  }, [])

  async function save() {
    await apiPost('/api/settings', { mode, autonomy, max_cost: maxCost, max_iterations: maxIterations, default_model: selectedModel })
    setSaved(true)
    window.setTimeout(() => setSaved(false), 2200)
  }

  return (
    <AppShell title="Settings" description="Choose how Providence responds, researches, and connects to models." action={<button className="primary-button" onClick={save}>{saved ? <><Check size={15} /> Saved</> : 'Save changes'}</button>}>
      <div className="page-stack">
        <section className="settings-section">
          <div className="section-title"><SlidersHorizontal size={18} /><div><h2>Research defaults</h2><p>These defaults apply when the composer does not override them.</p></div></div>
          <div className="settings-grid">
            <label>Research depth<select value={mode} onChange={(e) => setMode(e.target.value)}><option value="quick">Quick</option><option value="standard">Standard</option><option value="deep">Deep</option><option value="academic">Academic</option><option value="recency">Recency</option><option value="compare">Compare</option></select></label>
            <label>Autonomy<select value={autonomy} onChange={(e) => setAutonomy(e.target.value)}><option value="L1">Automatic</option><option value="L2">Review plan first</option><option value="L3">Hard budget cap</option></select></label>
            <label>Maximum budget<input type="number" min="0" step="0.5" value={maxCost} onChange={(e) => setMaxCost(Number(e.target.value))} /></label>
            <label>Maximum iterations<input type="number" min="1" max="10" value={maxIterations} onChange={(e) => setMaxIterations(Number(e.target.value))} /></label>
          </div>
          <label className="wide-field">Default model<input value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} placeholder="Select a model from the chat composer" /><small>The chat composer is the primary model picker. Models are loaded from the runtime provider catalog.</small></label>
        </section>

        <ProviderSettings providers={providers} onProvidersChange={setProviders} />
      </div>
    </AppShell>
  )
}
