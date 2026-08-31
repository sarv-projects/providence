'use client'

import { useState, useEffect } from 'react'
import { CheckCircle, XCircle, Pencil } from 'lucide-react'

export type ResearchPlanPayload = {
  plan_id: string
  query: string
  mode?: string
  autonomy?: string
  status?: string
  plan?: {
    topic?: string
    subtopics?: string[]
    outline?: { title?: string; queries?: string[] }[]
    search_queries?: string[]
    rationale?: string
    assumptions?: string[]
    refined_query_hint?: string
  }
  outline?: { title?: string; order?: number }[]
  search_queries?: string[]
  clarifying_questions?: string[]
  needs_clarification?: boolean
  job_id?: string
}

interface PlanEditorProps {
  plan: ResearchPlanPayload
  onApprove: (edited: {
    outline: { title: string }[]
    search_queries: string[]
    clarifications?: Record<string, string>
  }) => void
  onCancel: () => void
  busy?: boolean
}

export function PlanEditor({ plan, onApprove, onCancel, busy }: PlanEditorProps) {
  const initialOutline =
    plan.outline?.map((o) => o.title || '') ||
    plan.plan?.outline?.map((o) => o.title || '') ||
    []
  const initialQueries =
    plan.search_queries || plan.plan?.search_queries || []

  const [outlineText, setOutlineText] = useState(initialOutline.join('\n'))
  const [queriesText, setQueriesText] = useState(initialQueries.join('\n'))
  const [answers, setAnswers] = useState<Record<string, string>>({})

  useEffect(() => {
    const outline =
      plan.outline?.map((o) => o.title || '') ||
      plan.plan?.outline?.map((o) => o.title || '') ||
      []
    const queries = plan.search_queries || plan.plan?.search_queries || []
    setOutlineText(outline.join('\n'))
    setQueriesText(queries.join('\n'))
  }, [plan])

  const questions = plan.clarifying_questions || []

  const handleApprove = () => {
    const outline = outlineText
      .split('\n')
      .map((t) => t.trim())
      .filter(Boolean)
      .map((title) => ({ title }))
    const search_queries = queriesText
      .split('\n')
      .map((t) => t.trim())
      .filter(Boolean)
    const clarifications =
      Object.keys(answers).length > 0 ? answers : undefined
    onApprove({ outline, search_queries, clarifications })
  }

  return (
    <div className="plan-editor">
      <div className="flex items-center gap-2 text-sm font-semibold" style={{ color: '#b45309' }}>
        <Pencil className="h-4 w-4" />
        Editable research plan
        <span className="text-xs font-normal opacity-70">
          ({plan.plan_id} · {plan.status || 'draft'})
        </span>
      </div>

      <div className="text-xs opacity-80">
        <strong>Topic:</strong> {plan.plan?.topic || plan.query}
      </div>
      {plan.plan?.rationale && <div className="text-xs opacity-80">{plan.plan.rationale}</div>}

      {plan.needs_clarification && questions.length > 0 && (
        <div className="space-y-2 border-t pt-3" style={{ borderColor: 'color-mix(in srgb, #f59e0b 30%, transparent)' }}>
          <div className="text-xs font-medium uppercase tracking-wide opacity-70">
            Clarifying questions
          </div>
          {questions.map((q, i) => (
            <label key={i} className="block text-xs">
              <span className="mb-1 block opacity-90">{q}</span>
              <input
                value={answers[q] || ''}
                onChange={(e) => setAnswers((prev) => ({ ...prev, [q]: e.target.value }))}
                placeholder="Optional — leave blank to let the agent assume defaults"
                className="w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors"
                style={{ borderColor: 'var(--border)', background: 'var(--surface-2)', color: 'var(--foreground)' }}
              />
            </label>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 border-t pt-3 sm:grid-cols-2" style={{ borderColor: 'color-mix(in srgb, #f59e0b 30%, transparent)' }}>
        <div>
          <span className="plan-field-label">Outline (one section per line)</span>
          <textarea
            value={outlineText}
            onChange={(e) => setOutlineText(e.target.value)}
            rows={Math.max(4, outlineText.split('\n').length + 1)}
          />
        </div>
        <div>
          <span className="plan-field-label">Search queries (one per line)</span>
          <textarea
            value={queriesText}
            onChange={(e) => setQueriesText(e.target.value)}
            rows={Math.max(4, queriesText.split('\n').length + 1)}
          />
        </div>
      </div>

      <div className="flex items-center justify-end gap-2 border-t pt-3" style={{ borderColor: 'color-mix(in srgb, #f59e0b 30%, transparent)' }}>
        <button type="button" onClick={onCancel} disabled={busy} className="ghost-button">
          <XCircle size={14} /> Cancel
        </button>
        <button type="button" onClick={handleApprove} disabled={busy} className="primary-button">
          <CheckCircle size={14} /> {busy ? 'Starting…' : 'Approve & research'}
        </button>
      </div>
    </div>
  )
}
