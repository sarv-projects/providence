'use client'

import { useState } from 'react'
import { apiGet, safeHttpUrl } from '@/lib/api'
import { AppShell } from '@/components'
import { Search, ExternalLink } from 'lucide-react'

interface VaultResult {
  url?: string
  title?: string
  content?: string
  query?: string
}

export default function VaultPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [results, setResults] = useState<VaultResult[]>([])
  const [loading, setLoading] = useState(false)
  const [searched, setSearched] = useState(false)
  const [error, setError] = useState('')

  async function handleSearch(e: React.FormEvent) {
    e.preventDefault()
    if (!searchQuery.trim()) return

    setLoading(true)
    setSearched(true)
    setError('')
    try {
      const data = await apiGet<{ results?: VaultResult[] }>(
        `/api/vault/search?query=${encodeURIComponent(searchQuery)}&limit=10`
      )
      setResults(data.results || [])
    } catch (err) {
      setResults([])
      setError(err instanceof Error ? err.message : 'Vault search failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AppShell title="Research vault" description="Search persistent sources, factoids, and findings across runs.">
      <div className="page-stack">
        <form onSubmit={handleSearch} className="search-field search-field-large"><Search size={18} /><input type="text" placeholder="Search sources and findings" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} /><button type="submit" disabled={loading} className="primary-button">{loading ? 'Searching…' : 'Search'}</button></form>

        {/* Results List */}
        <div className="space-y-4">
          {error && <div className="notice notice-error">{error}</div>}
          {results.map((r, i) => (
            <div key={i} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow border border-gray-200 dark:border-gray-700">
              <h3 className="font-semibold text-lg text-gray-900 dark:text-gray-100 mb-1">
                {r.title || r.url || `Vault Document ${i+1}`}
              </h3>
              {safeHttpUrl(r.url) && (
                <a href={safeHttpUrl(r.url) || undefined} target="_blank" rel="noopener noreferrer" className="text-xs font-mono text-gray-400 hover:underline block mb-3">
                  <ExternalLink size={12} className="inline" /> {safeHttpUrl(r.url)}
                </a>
              )}
              <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap leading-relaxed">
                {r.content ? r.content.slice(0, 500) : 'No preview available'}
                {r.content && r.content.length > 500 ? '...' : ''}
              </p>
            </div>
          ))}
        </div>

        {searched && !loading && results.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            No vault documents found matching your query.
          </div>
        )}
      </div>
    </AppShell>
  )
}
