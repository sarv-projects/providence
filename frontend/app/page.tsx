'use client'

import { useState, useRef, useEffect } from 'react'
import { X } from 'lucide-react'
import 'katex/dist/katex.min.css'
import {
  Sidebar,
  ProgressBanner,
  MessageBubble,
  ApprovalBanner,
  LoadingDots,
  PlanEditor,
  ChatComposer,
  EMPTY_LENSES,
} from '@/components'
import type { ChatMessage, ApprovalRequest, ResearchPlanPayload, ResearchDepth, ResearchLenses } from '@/components'
import { apiFetch, apiGet, apiPost, apiPut, type ProgressSnapshot } from '@/lib/api'

// Legacy depth modes resolve to standard + an implied lens (mirrors
// src/engine/modes.py LEGACY_LENS_MODES).
const LEGACY_DEPTH: Record<string, { depth: ResearchDepth; lens: keyof ResearchLenses | null }> = {
  recency: { depth: 'standard', lens: 'recency' },
  academic: { depth: 'deep', lens: 'academic' },
  compare: { depth: 'standard', lens: 'compare' },
  quick: { depth: 'standard', lens: null },
  'ultra-long': { depth: 'deep', lens: null },
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content:
        "Hello! I'm **Providence**, your deep-research engine. I can help with:\n\n- **Chat**: multi-turn conversation with memory (streaming)\n- **Research**: multi-agent cited reports with live progress\n- **Evidence**: every claim verified against this run's sources\n\nHow can I help you today?",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [mode, setMode] = useState<'chat' | 'research'>('chat')
  const [researchDepth, setResearchDepth] = useState<ResearchDepth>('standard')
  const [lenses, setLenses] = useState<ResearchLenses>({ ...EMPTY_LENSES })
  const [autonomy, setAutonomy] = useState('L1')
  const [selectedModel, setSelectedModel] = useState('')
  const [maxCost, setMaxCost] = useState(5)
  const [maxIterations, setMaxIterations] = useState(3)
  const [planFirst, setPlanFirst] = useState(false)
  const [pendingPlan, setPendingPlan] = useState<ResearchPlanPayload | null>(null)
  const [planBusy, setPlanBusy] = useState(false)
  const [activeJobId, setActiveJobId] = useState('')
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([])
  const [progressStatus, setProgressStatus] = useState('')
  const [thinking, setThinking] = useState<{
    learned: string[]
    gaps: string[]
    nextAction: string
    thoughts: { kind?: string; text?: string }[]
    pagesScanned?: number
    sourcesCount?: number
    findingsCount?: number
    offTopic?: boolean
    stage?: string
    perspectives?: string[]
  }>({ learned: [], gaps: [], nextAction: '', thoughts: [] })
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const pollAbortRef = useRef<AbortController | null>(null)
  const activeJobIdRef = useRef('')

  async function pollResearchJob(jobId: string, userText: string, signal: AbortSignal) {
    activeJobIdRef.current = jobId
    setActiveJobId(jobId)
    let finished = false
    let jobError = ''
    const sleep = (ms: number) =>
      new Promise<void>((resolve, reject) => {
        if (signal.aborted) {
          reject(new DOMException('Aborted', 'AbortError'))
          return
        }
        const timer = setTimeout(() => {
          signal.removeEventListener('abort', onAbort)
          resolve()
        }, ms)
        const onAbort = () => {
          clearTimeout(timer)
          reject(new DOMException('Aborted', 'AbortError'))
        }
        signal.addEventListener('abort', onAbort, { once: true })
      })
    for (let i = 0; i < 900 && !finished; i++) {
      await sleep(1000)
      let snap: ProgressSnapshot = {}
      try {
        if (jobId) {
          try {
            const job = await apiGet<ProgressSnapshot & { status?: string }>(
              `/api/jobs/${jobId}`
            )
            // L2 plan pause
            if (job.status === 'awaiting_plan') {
              setProgressStatus('Awaiting plan approval…')
              // try to load plan from thoughts or list
              const planId = (job as ProgressSnapshot & { plan_id?: string }).plan_id
              const match = planId
                ? await apiGet<ResearchPlanPayload>(`/api/research/plans/${planId}`, { signal })
                : null
              if (match) {
                setPendingPlan(match)
                setIsLoading(false)
                return
              }
            }
            snap = {
              ...job,
              finished:
                job.finished ||
                ['complete', 'error', 'aborted'].includes(job.status || ''),
              status: job.status || job.stage,
            }
          } catch {
            snap = await apiGet(`/api/research/progress?job_id=${encodeURIComponent(jobId)}`, { signal })
          }
        } else {
          snap = await apiGet('/api/research/progress', { signal })
        }
        const label = snap.status || snap.stage || 'running'
        const secs = snap.section_progress || ''
        setProgressStatus(
          `${label}${secs ? ` · sections ${secs}` : ''}${
            snap.findings_count ? ` · findings ${snap.findings_count}` : ''
          } · ${snap.elapsed_s || 0}s`
        )
        setThinking({
          learned: snap.learned || [],
          gaps: snap.gaps || [],
          nextAction: snap.next_action || '',
          thoughts: snap.thoughts || [],
          pagesScanned: snap.pages_scanned,
          sourcesCount: snap.sources_count,
          findingsCount: snap.findings_count,
          offTopic: snap.off_topic,
          stage: snap.stage,
          perspectives: snap.perspectives || [],
        })
        if (snap.error) {
          // Job failed — capture the message and stop polling
          jobError = snap.error
          finished = true
        } else {
          finished = !!snap.finished
        }
      } catch {
        // Transient poll failure (backend restart etc.) — keep polling
      }
    }
    if (!finished) throw new Error('Research timed out before completion')

    let finalSnap: ProgressSnapshot = {}
    if (jobId) {
      finalSnap = await apiGet(`/api/jobs/${jobId}`, { signal }).catch(() => ({}))
    }
    if (!finalSnap.report) {
      finalSnap = await apiGet(`/api/research/progress?job_id=${encodeURIComponent(jobId)}`, { signal }).catch(() => ({}))
    }
    // Surface real failures instead of a misleading "Research complete" message
    const errorText = jobError || finalSnap.error
    if (errorText) {
      throw new Error(errorText)
    }
    if (!finalSnap.report) throw new Error('Research ended without a report')
    let lastReport = finalSnap.report
    if (finalSnap.markdown_path) {
      lastReport += `\n\n---\n_Saved: \`${finalSnap.markdown_path}\`_`
    }

    setMessages((prev) => [
      ...prev,
      { role: 'assistant', content: lastReport, timestamp: new Date() },
    ])
    setProgressStatus('')
    setThinking({ learned: [], gaps: [], nextAction: '', thoughts: [], perspectives: [] })
    activeJobIdRef.current = ''
    setActiveJobId('')
  }

  async function handlePlanApprove(edited: {
    outline: { title: string }[]
    search_queries: string[]
    clarifications?: Record<string, string>
  }) {
    if (!pendingPlan) return
    setPlanBusy(true)
    try {
      await apiPut(`/api/research/plans/${pendingPlan.plan_id}`, {
        outline: edited.outline,
        search_queries: edited.search_queries,
        clarifications: edited.clarifications,
        plan: {
          ...(pendingPlan.plan || {}),
          outline: edited.outline,
          search_queries: edited.search_queries,
        },
      })
      const runRes = await apiPost<{ job_id?: string; status?: string }>(
        `/api/research/plans/${pendingPlan.plan_id}/run`,
        {
          background: true,
          clarifications: edited.clarifications,
          model: selectedModel || undefined,
          max_cost_usd: maxCost,
          max_iterations: maxIterations,
          recency: lenses.recency,
          academic: lenses.academic,
          compare: lenses.compare,
        }
      )
      setPendingPlan(null)
      setIsLoading(true)
      setProgressStatus('Plan approved — researching…')
      setThinking({
        learned: [],
        gaps: [],
        nextAction: 'Gathering sources with approved plan',
        thoughts: [],
        perspectives: [],
      })
      pollAbortRef.current = new AbortController()
      await pollResearchJob(runRes?.job_id || '', pendingPlan.query, pollAbortRef.current.signal)
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Plan run failed: ${err instanceof Error ? err.message : 'unknown'}`,
          timestamp: new Date(),
        },
      ])
    } finally {
      setPlanBusy(false)
      setIsLoading(false)
      setProgressStatus('')
      activeJobIdRef.current = ''
      setActiveJobId('')
    }
  }

  useEffect(() => {
    apiGet<{ mode?: string; autonomy?: string; default_model?: string; max_cost?: number; max_iterations?: number; lens_recency?: boolean; lens_academic?: boolean; lens_compare?: boolean }>('/api/settings')
      .then((data) => {
        // Legacy stored modes (recency/academic/compare/...) migrate to
        // depth + implied lens so old settings keep working.
        let baseLenses = {
          recency: !!data?.lens_recency,
          academic: !!data?.lens_academic,
          compare: !!data?.lens_compare,
        }
        if (data?.mode) {
          const legacy = LEGACY_DEPTH[data.mode]
          if (legacy) {
            setResearchDepth(legacy.depth)
            if (legacy.lens) baseLenses = { ...baseLenses, [legacy.lens]: true }
          } else if (data.mode === 'deep' || data.mode === 'standard') {
            setResearchDepth(data.mode)
          }
        }
        setLenses(baseLenses)
        if (data?.autonomy) setAutonomy(data.autonomy)
        if (data?.default_model) setSelectedModel(data.default_model)
        if (data?.max_cost != null) setMaxCost(data.max_cost)
        if (data?.max_iterations != null) setMaxIterations(data.max_iterations)
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, progressStatus])

  useEffect(() => {
    async function checkApprovals() {
      try {
        const data = await apiGet<{ approvals?: ApprovalRequest[] }>('/api/approvals')
        setApprovals(data.approvals || [])
      } catch {
        /* quiet */
      }
    }
    checkApprovals()
    const interval = setInterval(checkApprovals, 10000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => () => pollAbortRef.current?.abort(), [])

  async function cancelActiveResearch() {
    const jobId = activeJobIdRef.current
    if (jobId) await apiPost(`/api/jobs/${jobId}/cancel`, {}).catch(() => {})
    pollAbortRef.current?.abort()
    pollAbortRef.current = null
    activeJobIdRef.current = ''
    setActiveJobId('')
    setIsLoading(false)
    setProgressStatus('Research cancelled')
  }

  async function handleApprovalResponse(approvalId: string, approved: boolean) {
    try {
      await apiPost(`/api/approvals/${approvalId}/respond`, {
        approved,
        comments: approved ? 'Approved by user' : 'Rejected by user',
      })
      setApprovals((prev) => prev.filter((a) => a.approval_id !== approvalId))
    } catch (err) {
      console.error('Failed to submit approval:', err)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userText = input
    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userText, timestamp: new Date() },
    ])
    setInput('')
    setIsLoading(true)
    setProgressStatus('')

    try {
      if (mode === 'chat') {
        const controller = new AbortController()
        pollAbortRef.current = controller
        const response = await apiFetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          signal: controller.signal,
          timeoutMs: 120000,
          body: JSON.stringify({
            message: userText,
            mode: 'fast',
            model: selectedModel || undefined,
            session_id: 'default',
            stream: true,
            escalate: true,
          }),
        })

        const contentType = response.headers.get('content-type') || ''
        if (contentType.includes('text/event-stream') && response.body) {
          const reader = response.body.getReader()
          const decoder = new TextDecoder()
          let acc = ''
          setMessages((prev) => [
            ...prev,
            { role: 'assistant', content: '', timestamp: new Date() },
          ])
          let buffer = ''
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })
            const parts = buffer.split('\n\n')
            buffer = parts.pop() || ''
            for (const part of parts) {
              const line = part.trim()
              // Tolerate both "data: {...}" and "data:{...}" framings.
              if (!line.startsWith('data:')) continue
              try {
                const payload = line.slice(5).trim()
                const evt = JSON.parse(payload)
                if (evt.type === 'token') {
                  acc += evt.text || ''
                  setMessages((prev) => {
                    const copy = [...prev]
                    copy[copy.length - 1] = {
                      role: 'assistant',
                      content: acc,
                      timestamp: new Date(),
                    }
                    return copy
                  })
                } else if (evt.type === 'done' && evt.text) {
                  acc = evt.text
                  setMessages((prev) => {
                    const copy = [...prev]
                    copy[copy.length - 1] = {
                      role: 'assistant',
                      content: acc,
                      timestamp: new Date(),
                    }
                    return copy
                  })
                } else if (evt.type === 'error') {
                  throw new Error(evt.error || 'stream error')
                }
              } catch (parseErr) {
                if (parseErr instanceof SyntaxError) continue
                throw parseErr
              }
            }
          }
        } else {
          const data = await response.json()
          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              content: data.response || 'Error: No response received',
              timestamp: new Date(),
            },
          ])
        }
      } else {
        setProgressStatus('Starting research...')
        setThinking({ learned: [], gaps: [], nextAction: 'Planning…', thoughts: [], perspectives: [] })
        setPendingPlan(null)

        // L2 required plan review; L1 optional via planFirst toggle
        const wantPlan = planFirst || autonomy === 'L2'
        if (wantPlan) {
          setProgressStatus('Generating editable research plan…')
          const planRes = await apiPost<ResearchPlanPayload>('/api/research/plans', {
            query: userText,
            mode: researchDepth,
            autonomy,
            model: selectedModel || undefined,
            max_cost_usd: maxCost,
            max_iterations: maxIterations,
            recency: lenses.recency,
            academic: lenses.academic,
            compare: lenses.compare,
          })
          setPendingPlan(planRes)
          setMessages((prev) => [
            ...prev,
            {
              role: 'assistant',
              content:
                `## Research plan ready\n\n` +
                `**Plan ID:** \`${planRes.plan_id}\`\n\n` +
                `Review the outline and search queries below, then click **Approve & research**.` +
                (planRes.needs_clarification
                  ? `\n\n_Query looks ambiguous — answer clarifying questions if you can._`
                  : ''),
              timestamp: new Date(),
            },
          ])
          setProgressStatus('')
          setIsLoading(false)
          return
        }

        const startRes = await apiPost<{ job_id?: string }>('/api/research', {
          query: userText,
          mode: researchDepth,
          autonomy,
          background: true,
          skip_clarify: true,
          model: selectedModel || undefined,
          max_cost_usd: maxCost,
          max_iterations: maxIterations,
          recency: lenses.recency,
          academic: lenses.academic,
          compare: lenses.compare,
        })
        const jobId = startRes?.job_id || ''
        pollAbortRef.current = new AbortController()
        await pollResearchJob(jobId, userText, pollAbortRef.current.signal)
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
          timestamp: new Date(),
        },
      ])
      setProgressStatus('')
    } finally {
      setIsLoading(false)
      activeJobIdRef.current = ''
      setActiveJobId('')
    }
  }

  return (
    <div className="flex h-screen chat-root">
      <Sidebar mode={mode} onModeChange={setMode} />

      <div className="flex flex-1 flex-col">
        <ApprovalBanner approvals={approvals} onRespond={handleApprovalResponse} />
        <ProgressBanner
          status={progressStatus}
          visible={!!progressStatus}
          learned={thinking.learned}
          gaps={thinking.gaps}
          nextAction={thinking.nextAction}
          thoughts={thinking.thoughts}
          pagesScanned={thinking.pagesScanned}
          sourcesCount={thinking.sourcesCount}
          findingsCount={thinking.findingsCount}
          offTopic={thinking.offTopic}
          stage={thinking.stage}
          perspectives={thinking.perspectives}
        />

        <div className="chat-scroll flex-1 overflow-y-auto p-4">
          <div className="chat-inner">
            {messages.map((message, index) => (
              <MessageBubble key={index} message={message} />
            ))}
            {pendingPlan && (
              <PlanEditor
                plan={pendingPlan}
                busy={planBusy}
                onCancel={() => {
                  setPendingPlan(null)
                  setMessages((prev) => [
                    ...prev,
                    {
                      role: 'assistant',
                      content: 'Plan cancelled. Submit a new research query when ready.',
                      timestamp: new Date(),
                    },
                  ])
                }}
                onApprove={handlePlanApprove}
              />
            )}
            {isLoading && (
              <div>
                <LoadingDots label={mode === 'research' ? progressStatus || 'Executing multi-agent research graph...' : 'Streaming response...'} />
                {mode === 'research' && activeJobId && (
                  <button type="button" onClick={cancelActiveResearch} className="danger-button mt-2 !px-3 !py-1.5 !text-xs">
                    <X size={12} /> Cancel research
                  </button>
                )}
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        <ChatComposer
          input={input}
          onInputChange={setInput}
          onSubmit={handleSubmit}
          disabled={isLoading}
          mode={mode}
          onModeChange={setMode}
          researchDepth={researchDepth}
          onResearchDepthChange={setResearchDepth}
          lenses={lenses}
          onLensesChange={setLenses}
          autonomy={autonomy}
          onAutonomyChange={setAutonomy}
          planFirst={planFirst}
          onPlanFirstChange={setPlanFirst}
          selectedModel={selectedModel || 'opencode_free/nemotron-3-ultra-free'}
          onModelChange={setSelectedModel}
        />
      </div>
    </div>
  )
}
