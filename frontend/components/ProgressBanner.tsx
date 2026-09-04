'use client'

import { useEffect, useState } from 'react'
import {
  BookOpen,
  Brain,
  CircleDot,
  Compass,
  Eye,
  FileText,
  Globe,
  Layers,
  Lightbulb,
  Loader2,
  TriangleAlert,
} from 'lucide-react'

export interface ProgressBannerProps {
  status: string
  visible?: boolean
  learned?: string[]
  gaps?: string[]
  nextAction?: string
  thoughts?: { kind?: string; text?: string }[]
  pagesScanned?: number
  sourcesCount?: number
  findingsCount?: number
  offTopic?: boolean
  stage?: string
  perspectives?: string[]
}

const KIND_STYLE: Record<string, { icon: typeof Brain; className: string }> = {
  learned: { icon: Lightbulb, className: 'text-emerald-600 dark:text-emerald-400' },
  gap: { icon: TriangleAlert, className: 'text-amber-600 dark:text-amber-400' },
  next: { icon: Compass, className: 'text-indigo-600 dark:text-indigo-400' },
  tool: { icon: Globe, className: 'text-cyan-600 dark:text-cyan-400' },
  stage: { icon: Layers, className: 'text-violet-600 dark:text-violet-400' },
}

const STAGE_ICON: Record<string, typeof Brain> = {
  scouting: Eye,
  planning: Compass,
  researching: Globe,
  searching: Globe,
  gathering: BookOpen,
  analyzing: Brain,
  synthesizing: FileText,
  writing: FileText,
  compiling: Layers,
}

function StagePill({ stage }: { stage?: string }) {
  const Icon = (stage && STAGE_ICON[stage]) || CircleDot
  return (
    <span className="progress-chip inline-flex items-center gap-1.5">
      <Loader2 size={11} className="animate-spin" />
      <Icon size={11} />
      {stage || 'running'}
    </span>
  )
}

function StatPill({ icon: Icon, label, value }: { icon: typeof FileText; label: string; value?: number }) {
  if (typeof value !== 'number' || value <= 0) return null
  return (
    <span className="inline-flex items-center gap-1 opacity-80">
      <Icon size={11} /> {label} {value}
    </span>
  )
}

export function ProgressBanner({
  status,
  visible = true,
  learned = [],
  gaps = [],
  nextAction = '',
  thoughts = [],
  pagesScanned,
  sourcesCount,
  findingsCount,
  offTopic,
  stage,
  perspectives = [],
}: ProgressBannerProps) {
  const [expandedThoughts, setExpandedThoughts] = useState(true)
  // Auto-collapse the stream after it grows long so it never dominates the chat.
  useEffect(() => {
    if (thoughts.length > 14) setExpandedThoughts(false)
  }, [thoughts.length])

  if (!visible || !status) return null

  const recentThoughts = thoughts.slice(-8)

  return (
    <div className="progress-banner" role="status" aria-live="polite">
      {/* Row 1: status + stage + stats */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="progress-label">Research in flight</span>
        <span className="truncate">{status}</span>
        {stage && <StagePill stage={stage} />}
        <StatPill icon={Lightbulb} label="findings" value={findingsCount} />
        <StatPill icon={Globe} label="sources" value={sourcesCount} />
        <StatPill icon={Eye} label="pages" value={pagesScanned} />
        {offTopic && (
          <span
            className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:text-amber-300"
            style={{ background: 'color-mix(in srgb, #f59e0b 14%, transparent)' }}
          >
            <TriangleAlert size={11} /> off-topic recovery
          </span>
        )}
      </div>

      {/* Row 2: perspective lenses (STORM steal) */}
      {perspectives.length > 0 && (
        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 border-t pt-1" style={{ borderColor: 'var(--border)' }}>
          <span className="thinking-col-title mb-0 flex items-center gap-1">
            <Eye size={11} /> Perspectives
          </span>
          {perspectives.map((p, i) => (
            <span key={i} className="progress-chip">{p}</span>
          ))}
        </div>
      )}

      {/* Row 3: next action / learned / gaps */}
      {(nextAction || learned.length > 0 || gaps.length > 0) && (
        <div className="thinking-grid">
          <div>
            <div className="thinking-col-title">Next action</div>
            <div className="text-[11px] leading-snug opacity-90">{nextAction || '—'}</div>
          </div>
          <div>
            <div className="thinking-col-title flex items-center gap-1">
              <Lightbulb size={11} /> Learned
            </div>
            <ul className="list-disc space-y-0.5 pl-4 text-[11px] leading-snug opacity-90">
              {(learned.slice(-4).length ? learned.slice(-4) : ['—']).map((item, i) => (
                <li key={i}>{typeof item === 'string' ? item : '—'}</li>
              ))}
            </ul>
          </div>
          <div>
            <div className="thinking-col-title flex items-center gap-1">
              <TriangleAlert size={11} /> Gaps
            </div>
            <ul className="list-disc space-y-0.5 pl-4 text-[11px] leading-snug opacity-90">
              {(gaps.slice(-4).length ? gaps.slice(-4) : ['—']).map((item, i) => (
                <li key={i}>{typeof item === 'string' ? item : '—'}</li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {/* Row 4: live thinking stream */}
      {recentThoughts.length > 0 && (
        <div className="border-t pt-1.5" style={{ borderColor: 'var(--border)' }}>
          <button
            type="button"
            onClick={() => setExpandedThoughts((v) => !v)}
            className="thinking-col-title flex w-full items-center gap-1 uppercase"
            aria-expanded={expandedThoughts}
          >
            <Brain size={11} /> Thinking stream ({thoughts.length})
            <span className="ml-auto normal-case opacity-60">{expandedThoughts ? 'hide' : 'show'}</span>
          </button>
          {expandedThoughts && (
            <div className="mt-1 flex max-h-24 flex-col gap-1 overflow-y-auto pr-1">
              {recentThoughts.map((t, i) => {
                const meta = KIND_STYLE[t.kind || ''] || { icon: CircleDot, className: 'opacity-70' }
                const Icon = meta.icon
                const text = t.text || ''
                return (
                  <div key={i} className="flex items-start gap-1.5 text-[11px] opacity-90">
                    <Icon size={11} className={`mt-0.5 shrink-0 ${meta.className}`} />
                    <span className="min-w-0 break-words" title={text}>
                      <span className="font-medium opacity-60">[{t.kind || 'note'}]</span> {text}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
