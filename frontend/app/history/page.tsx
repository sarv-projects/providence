'use client'

import { useState, useEffect } from 'react'
import { Search, X, History as HistoryIcon, FileText as FileIcon } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { apiGet, safeHttpUrl } from '@/lib/api'
import { AppShell } from '@/components'

interface HistoryItem {
  query: string
  search_queries?: string[]
  report_path?: string
  findings_count?: number
  timestamp: string
}

interface ReportItem {
  name: string
  path: string
  format: string
  size_bytes: number
  modified: string
}

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [reports, setReports] = useState<ReportItem[]>([])
  const [preview, setPreview] = useState<{ name: string; content: string } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState('')

  useEffect(() => {
    async function load() {
      try {
        const [h, r] = await Promise.all([
          apiGet<HistoryItem[]>('/api/history?limit=30'),
          apiGet<{ reports: ReportItem[] }>('/api/reports?limit=40'),
        ])
        setHistory(Array.isArray(h) ? h : [])
        setReports(r.reports || [])
      } catch (err: any) {
        setError(err.message || 'Failed to load history')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [])

  async function openReport(name: string) {
    try {
      const data = await apiGet<{ name: string; content: string }>(
        `/api/reports/${encodeURIComponent(name)}`
      )
      setPreview({ name: data.name, content: data.content })
    } catch (err: any) {
      setError(err.message || 'Failed to open report')
    }
  }

  useEffect(() => {
    if (!preview) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setPreview(null)
    }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [preview])

  const filteredHistory = history.filter((item) => item.query.toLowerCase().includes(filter.toLowerCase()))
  const filteredReports = reports.filter((item) => item.name.toLowerCase().includes(filter.toLowerCase()))

  return (
    <AppShell title="History & reports" description="Find previous research runs and open generated reports." action={<a href="/" className="primary-button">New research</a>}>
      <div className="page-stack">
        <div className="search-field"><Search size={17} /><input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Search history and reports" />{filter && <button onClick={() => setFilter('')} aria-label="Clear search"><X size={15} /></button>}</div>

        {error && <div className="notice notice-error">{error}</div>}

        {loading ? (
          <div className="text-center py-12 text-gray-500">Loading...</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div>
              <h2 className="text-xl font-semibold mb-4">Search history</h2>
              <div className="space-y-3">
                                {filteredHistory.map((item, idx) => (
                  <div
                    key={idx}
                    className="card-hover rounded-xl border p-4 shadow-sm"
                    style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
                  >
                    <h3 className="font-semibold">{item.query}</h3>
                    <div className="mt-2 flex justify-between text-xs" style={{ color: 'var(--muted)' }}>
                      <span>{item.timestamp}</span>
                      <span>{item.findings_count ?? 0} findings</span>
                    </div>
                    {item.report_path && (
                      <p className="mt-1 truncate font-mono text-xs opacity-50">
                        {item.report_path}
                      </p>
                    )}
                  </div>
                ))}
                {!filteredHistory.length && <div className="empty-state"><HistoryIcon /><strong>{filter ? 'No matching runs' : 'No research yet'}</strong><span>{filter ? 'Try a different search.' : 'Your completed research runs will appear here.'}</span></div>}
              </div>
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4">Reports on disk</h2>
              <div className="space-y-2 max-h-[70vh] overflow-y-auto">
                                {filteredReports.map((r) => (
                  <button
                    key={r.name}
                    onClick={() => openReport(r.name)}
                    className="card-hover w-full rounded-xl border p-3 text-left text-sm shadow-sm transition"
                    style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
                  >
                    <div className="truncate font-medium">{r.name}</div>
                    <div className="mt-1 flex justify-between text-xs" style={{ color: 'var(--muted)' }}>
                      <span>{r.format}</span>
                      <span>{r.modified}</span>
                      <span>{Math.round(r.size_bytes / 1024)} KB</span>
                    </div>
                  </button>
                ))}
                {!filteredReports.length && <div className="empty-state"><FileIcon /><strong>{filter ? 'No matching reports' : 'No reports yet'}</strong><span>Generated reports will appear here.</span></div>}
              </div>
            </div>
          </div>
        )}

        {preview && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-sm"
            onClick={() => setPreview(null)}
            role="dialog"
            aria-modal="true"
            aria-label={`Preview ${preview.name}`}
          >
            <div
              className="flex max-h-[85vh] w-full max-w-4xl flex-col overflow-hidden rounded-2xl border shadow-2xl"
              style={{ borderColor: 'var(--border)', background: 'var(--surface)', color: 'var(--foreground)' }}
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between border-b px-6 py-4" style={{ borderColor: 'var(--border)' }}>
                <h3 className="truncate pr-4 font-semibold">{preview.name}</h3>
                <button onClick={() => setPreview(null)} className="secondary-button !px-3 !py-1 !text-xs">
                  Close
                </button>
              </div>
              <div className="markdown-body flex-1 overflow-y-auto p-6">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    a: ({ href, children }) => {
                      const safe = safeHttpUrl(typeof href === 'string' ? href : undefined)
                      if (!safe) return <span>{children}</span>
                      return (
                        <a href={safe} target="_blank" rel="noopener noreferrer">
                          {children}
                        </a>
                      )
                    },
                  }}
                >
                  {preview.content.slice(0, 100000)}
                </ReactMarkdown>
              </div>
            </div>
          </div>
        )}
      </div>
    </AppShell>
  )
}
