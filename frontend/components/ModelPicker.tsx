'use client'

import { useEffect, useState, useMemo } from 'react'
import { apiGet, apiPost } from '@/lib/api'

type ModelRow = {
  id: string
  label: string
  free: boolean
  source: string
  status: 'ok' | 'fail' | 'unknown'
  latency_s?: number
  error?: string
  reply?: string
}

type ProviderGroup = {
  provider: string
  provider_name: string
  free: boolean
  has_key: boolean
  env_key: string
  base_url: string
  default?: boolean
  models: ModelRow[]
}

interface ModelPickerProps {
  selected?: string // "provider/model"
  onSelect?: (provider: string, model: string) => void
}

export function ModelPicker({ selected, onSelect }: ModelPickerProps) {
  const [groups, setGroups] = useState<ProviderGroup[]>([])
  const [loading, setLoading] = useState(true)
  const [probing, setProbing] = useState(false)
  const [error, setError] = useState('')
  const [note, setNote] = useState('')
  const [expanded, setExpanded] = useState<string>('opencode_free')
  const freeGroups = useMemo(() => groups.filter((g) => g.models.length > 0), [groups])

  async function load(probeZen = false) {
    setLoading(true)
    setError('')
    try {
      const q = probeZen ? '/api/models?discover=true&probe=true' : '/api/models?discover=true'
      const data = await apiGet<{
        groups: ProviderGroup[]
        note?: string
        default_provider?: string
        default_model?: string
      }>(q)
      setGroups(data.groups || [])
      setNote(data.note || '')
      if (!selected && data.default_provider && data.default_model && onSelect) {
        onSelect(data.default_provider, data.default_model)
      }
    } catch (e: any) {
      setError(e.message || 'Failed to load models')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load(false)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function probeZen() {
    setProbing(true)
    setError('')
    try {
      await load(true)
    } finally {
      setProbing(false)
    }
  }

  async function probeProvider(provider: string) {
    setProbing(true)
    setError('')
    try {
      const res = await apiPost<{ results: any[] }>('/api/models/probe', { provider })
      const map: Record<string, any> = {}
      for (const r of res.results || []) {
        map[`${r.provider}/${r.model}`] = r
      }
      setGroups((prev) =>
        prev.map((g) => {
          if (g.provider !== provider) return g
          return {
            ...g,
            models: g.models.map((m) => {
              const p = map[`${provider}/${m.id}`]
              if (!p) return m
              return {
                ...m,
                status: p.ok ? 'ok' : 'fail',
                latency_s: p.latency_s,
                error: p.error || '',
                reply: p.reply || '',
              }
            }),
          }
        })
      )
    } catch (e: any) {
      setError(e.message || 'Probe failed')
    } finally {
      setProbing(false)
    }
  }

  if (loading && !groups.length) {
    return <div className="text-sm text-gray-500 py-4">Loading model catalog…</div>
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2 items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold">Model picker</h2>
          <p className="text-xs text-gray-500 mt-1">
            Showing free models only. OpenCode Zen free endpoints need no API key.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => load(false)}
            className="px-3 py-1.5 text-sm rounded border hover:bg-gray-50 dark:hover:bg-gray-700"
          >
            Refresh list
          </button>
          <button
            type="button"
            disabled={probing}
            onClick={probeZen}
            className="px-3 py-1.5 text-sm rounded bg-green-600 text-white hover:bg-green-700 disabled:opacity-50"
          >
            {probing ? 'Probing…' : 'Test Zen free models'}
          </button>
        </div>
      </div>

      {note && (
        <p className="text-xs text-blue-700 dark:text-blue-300 bg-blue-50 dark:bg-blue-900/20 p-3 rounded">
          {note}
        </p>
      )}
      {error && (
        <p className="text-xs text-red-600 bg-red-50 dark:bg-red-900/20 p-3 rounded">{error}</p>
      )}

      <div className="space-y-3">
        {freeGroups.map((g) => {
          const open = expanded === g.provider
          const okCount = g.models.filter((m) => m.status === 'ok').length
          const failCount = g.models.filter((m) => m.status === 'fail').length
          return (
            <div
              key={g.provider}
              className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden bg-white dark:bg-gray-800"
            >
              <button
                type="button"
                className="w-full flex items-center justify-between px-4 py-3 text-left hover:bg-gray-50 dark:hover:bg-gray-700/50"
                onClick={() => setExpanded(open ? '' : g.provider)}
              >
                <div>
                  <div className="font-medium flex items-center gap-2">
                    {g.provider_name}
                    {g.provider === 'opencode_free' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300">
                        DEFAULT · FREE
                      </span>
                    )}
                    {!g.has_key && g.provider !== 'opencode_free' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 dark:bg-amber-900/40">
                        needs {g.env_key || 'API key'}
                      </span>
                    )}
                    {g.has_key && g.provider !== 'opencode_free' && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 text-blue-800 dark:bg-blue-900/40">
                        key set
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-500 font-mono mt-0.5">{g.base_url}</div>
                </div>
                <div className="text-xs text-gray-500 text-right">
                  <div>{g.models.length} models</div>
                  {(okCount > 0 || failCount > 0) && (
                    <div>
                      {okCount > 0 && <span className="text-green-600">{okCount} ok </span>}
                      {failCount > 0 && <span className="text-red-600">{failCount} fail</span>}
                    </div>
                  )}
                  <div>{open ? '▲' : '▼'}</div>
                </div>
              </button>

              {open && (
                <div className="border-t border-gray-200 dark:border-gray-700 p-3 space-y-2">
                  <div className="flex justify-end">
                    <button
                      type="button"
                      disabled={probing || (!g.has_key && g.provider !== 'opencode_free')}
                      onClick={() => probeProvider(g.provider)}
                      className="text-xs px-2 py-1 rounded bg-gray-800 text-white dark:bg-gray-200 dark:text-gray-900 disabled:opacity-40"
                    >
                      Test all free {g.provider_name} models
                    </button>
                  </div>
                  <div className="divide-y divide-gray-100 dark:divide-gray-700">
                    {g.models.map((m) => {
                      const value = `${g.provider}/${m.id}`
                      const isSel = selected === value
                      return (
                        <div
                          key={m.id}
                          className={`flex items-start justify-between gap-3 py-2 px-2 rounded ${
                            isSel ? 'bg-blue-50 dark:bg-blue-900/20' : ''
                          }`}
                        >
                          <button
                            type="button"
                            className="text-left flex-1 min-w-0"
                            onClick={() => onSelect?.(g.provider, m.id)}
                          >
                            <div className="font-mono text-sm truncate">{m.id}</div>
                            <div className="text-[11px] text-gray-500 flex flex-wrap gap-2 mt-0.5">
                              {m.free && <span className="text-green-600">free</span>}
                              <span>{m.source}</span>
                              {m.status === 'ok' && (
                                <span className="text-green-600">
                                  ✓ {m.latency_s != null ? `${m.latency_s}s` : 'ok'}
                                  {m.reply ? ` · “${m.reply.slice(0, 40)}”` : ''}
                                </span>
                              )}
                              {m.status === 'fail' && (
                                <span className="text-red-600">✗ {m.error?.slice(0, 80)}</span>
                              )}
                              {m.status === 'unknown' && <span>not tested</span>}
                            </div>
                          </button>
                          {isSel && (
                            <span className="text-xs font-semibold text-blue-600 shrink-0">Selected</span>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
