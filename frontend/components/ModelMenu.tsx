'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import { Check, ChevronDown, Search } from 'lucide-react'

export interface CatalogModel {
  id: string
  label?: string
  free: boolean
  source?: string
  status?: string
}

export interface CatalogProvider {
  id: string
  name: string
  base_url: string
  free: boolean
  models: CatalogModel[]
}

interface ModelMenuProps {
  selected: string // "provider/model"
  onSelect: (provider: string, model: string) => void
  providers: CatalogProvider[]
  loading: boolean
  disabled?: boolean
}

export function ModelMenu({ selected, onSelect, providers, loading, disabled }: ModelMenuProps) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const rootRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)

  const flat = useMemo(() => {
    const rows: { value: string; label: string; model: string; provider: string; providerId: string; free: boolean }[] = []
    for (const provider of providers) {
      for (const model of provider.models) {
        if (!model.free) continue
        rows.push({
          value: `${provider.id}/${model.id}`,
          label: model.label || model.id,
          model: model.id,
          provider: provider.name,
          providerId: provider.id,
          free: model.free,
        })
      }
    }
    return rows
  }, [providers])

  const q = query.trim().toLowerCase()
  const filtered = q
    ? flat.filter((row) => row.label.toLowerCase().includes(q) || row.provider.toLowerCase().includes(q) || row.value.toLowerCase().includes(q))
    : flat

  const current = flat.find((row) => row.value === selected)

  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false)
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  useEffect(() => {
    if (open) searchRef.current?.focus()
  }, [open])

  const toggle = () => {
    if (disabled) return
    setOpen((prev) => !prev)
    setQuery('')
  }

  return (
    <div ref={rootRef} className="model-menu relative">
      <button
        type="button"
        onClick={toggle}
        disabled={disabled || loading}
        className="model-menu-trigger"
        title="Choose a model"
      >
        <span className="model-menu-current">{loading ? 'Loading…' : current ? current.label : 'No free models'}</span>
        <span className="model-menu-provider">{current ? current.provider : ''}</span>
        <ChevronDown size={14} className="model-menu-chevron" />
      </button>

      {open && (
        <div className="model-menu-popover">
          <div className="model-menu-search">
            <Search size={14} />
            <input ref={searchRef} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search models" />
          </div>
          <div className="model-menu-list">
            {filtered.length === 0 && <div className="model-menu-empty">No models match “{query}”</div>}
            {providers.map((provider) => {
              const rows = filtered.filter((row) => row.providerId === provider.id)
              if (rows.length === 0) return null
              return (
                <div key={provider.id} className="model-menu-group">
                  <div className="model-menu-group-header">
                    {provider.name}
                    {provider.free && <span className="model-free-badge">free</span>}
                  </div>
                  {rows.map((row) => (
                    <button
                      key={row.value}
                      type="button"
                      className="model-menu-item"
                      onClick={() => {
                        onSelect(row.providerId, row.model)
                        setOpen(false)
                      }}
                    >
                      <span className="model-menu-item-label">{row.label}</span>
                      <span className="model-menu-item-sub">{row.provider}</span>
                      {row.value === selected && <Check size={15} className="model-menu-check" />}
                    </button>
                  ))}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
