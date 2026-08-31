'use client'

import { useEffect, useState } from 'react'
import { ChevronDown, KeyRound, Plus, ShieldCheck } from 'lucide-react'
import { apiGet, apiPost } from '@/lib/api'

interface Provider {
  id?: string
  name: string
  base_url: string
  has_auth: boolean
  models: string[]
}

interface ProviderPreset { id: string; name: string; base_url: string; protocol: string; env_key: string; free: boolean; requires_key: boolean }

interface ProviderSettingsProps {
  providers: Provider[]
  onProvidersChange: (providers: Provider[]) => void
}

export function ProviderSettings({ providers, onProvidersChange }: ProviderSettingsProps) {
  const [open, setOpen] = useState(false)
  const [showForm, setShowForm] = useState(false)
  const [selectedProvider, setSelectedProvider] = useState('')
  const [name, setName] = useState('')
  const [url, setUrl] = useState('')
  const [key, setKey] = useState('')
  const [models, setModels] = useState('')
  const [status, setStatus] = useState('')
  const [discovered, setDiscovered] = useState<Record<string, { id: string; free?: boolean; status?: string }[]>>({})
  const [catalogProviders, setCatalogProviders] = useState<{ id: string; name: string; base_url: string; authenticated: boolean; free: boolean; models: { id: string; free?: boolean }[] }[]>([])
  const [presets, setPresets] = useState<ProviderPreset[]>([])
  // The connected list is driven exclusively by the runtime catalog (free
  // models only). Paid-only providers like OpenAI never appear here.
  const availableProviders = catalogProviders

  async function loadPresets() {
    try {
      const data = await apiGet<{ providers?: ProviderPreset[] }>('/api/providers/presets')
      setPresets(data.providers || [])
    } catch { setPresets([]) }
  }

  async function refreshModels() {
    try {
      const data = await apiGet<{ providers?: { id: string; name: string; base_url: string; authenticated: boolean; free: boolean; models: { id: string; free?: boolean }[] }[] }>('/api/providers/catalog?discover=true')
      const next: Record<string, { id: string; free?: boolean; status?: string }[]> = {}
      setCatalogProviders((data.providers || []).filter((provider) => provider.models.length > 0))
      for (const group of data.providers || []) next[group.id] = group.models
      setDiscovered(next)
      setStatus('Connected provider catalogs refreshed.')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Unable to refresh model catalogs')
    }
  }

  useEffect(() => { loadPresets(); refreshModels() }, [])

  async function addProvider(event: React.FormEvent) {
    event.preventDefault()
    if (!name.trim() || !url.trim()) return
    try {
      await apiPost('/api/providers', {
        name: name.trim(),
        base_url: url.trim(),
        api_key: key,
        models: models.split(',').map((model) => model.trim()).filter(Boolean),
      })
      onProvidersChange(await apiGet<Provider[]>('/api/providers'))
      setName(''); setUrl(''); setKey(''); setModels(''); setSelectedProvider(''); setShowForm(false)
      setStatus('Provider connected. Its models are now available in the composer.')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Unable to save provider')
    }
  }

  return (
    <section className="settings-panel">
      <button type="button" className="settings-panel-header" onClick={() => setOpen(!open)}>
        <span className="flex items-center gap-3">
          <span className="settings-icon"><KeyRound size={17} /></span>
          <span><strong>Bring your own provider</strong><small>Connect an API-compatible provider and choose its models</small></span>
        </span>
        <ChevronDown size={18} className={`transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      {open && (
        <div className="settings-panel-body space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg bg-blue-50 p-3 text-xs text-blue-800 dark:bg-blue-950/40 dark:text-blue-200">
            <span className="flex items-start gap-2"><ShieldCheck size={16} className="mt-0.5 shrink-0" />
            <span>Keys are used server-side. Models are discovered from each provider or its configured catalog.</span></span>
            <button type="button" className="secondary-button !px-2 !py-1 !text-xs" onClick={refreshModels}>Refresh catalogs</button>
          </div>
          {availableProviders.map((provider) => (
            <div key={provider.id} className="provider-row">
              <div className="min-w-0"><strong>{provider.name}</strong><small>{provider.base_url}</small></div>
              <div className="flex flex-wrap justify-end gap-1">
                {(discovered[provider.id] || provider.models).map((model) => <span key={model.id} className="model-chip">{model.id}{model.free ? ' · free' : ''}</span>)}
              </div>
              <span className="status-dot active" title="Connected" />
            </div>
          ))}
          {!availableProviders.length && <div className="empty-state"><KeyRound size={18} /><strong>No connected providers</strong><span>Add a provider below to enable BYOK models.</span></div>}
          <button type="button" className="secondary-button" onClick={() => setShowForm(!showForm)}><Plus size={15} /> Add BYOK provider</button>
          {showForm && (
            <form onSubmit={addProvider} className="provider-form">
              <select value={selectedProvider} onChange={(e) => { setSelectedProvider(e.target.value); const provider = presets.find((item) => item.id === e.target.value); if (provider) { setName(provider.name); setUrl(provider.base_url); setModels('') } }}>
                <option value="">Select a provider preset or use custom</option>
                {presets.filter((provider) => provider.requires_key).map((provider) => <option key={provider.id} value={provider.id}>{provider.name}</option>)}
                <option value="custom">Custom OpenAI-compatible provider</option>
              </select>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Provider name" required />
              <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://api.example.com/v1" required />
              <input value={key} onChange={(e) => setKey(e.target.value)} placeholder="API key" type="password" required />
              <input value={models} onChange={(e) => setModels(e.target.value)} placeholder="Model IDs, comma-separated" />
              <button type="submit" className="primary-button">Connect provider</button>
            </form>
          )}
          {status && <p className="text-xs text-gray-500">{status}</p>}
        </div>
      )}
    </section>
  )
}
