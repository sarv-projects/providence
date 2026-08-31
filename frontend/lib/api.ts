/** API helpers — all paths go through Next.js rewrites to the FastAPI backend. */

export const API_BASE = '' // relative → next.config rewrites /api/* → backend

type ApiOptions = RequestInit & { timeoutMs?: number }

async function request(path: string, options: ApiOptions = {}): Promise<Response> {
  const { timeoutMs, signal, ...init } = options
  let timer: ReturnType<typeof setTimeout> | undefined
  let requestSignal = signal
  if (timeoutMs && typeof AbortController !== 'undefined') {
    const controller = new AbortController()
    if (signal?.aborted) controller.abort()
    else if (signal) signal.addEventListener('abort', () => controller.abort(), { once: true })
    timer = setTimeout(() => controller.abort(), timeoutMs)
    requestSignal = controller.signal
  }
  try {
    const res = await fetch(`${API_BASE}${path}`, { ...init, signal: requestSignal })
    if (!res.ok) {
      const err = await res.json().catch(() => ({}))
      throw new Error(err.detail || err.message || res.statusText || `Request ${path} failed`)
    }
    return res
  } finally {
    if (timer) clearTimeout(timer)
  }
}

export async function apiFetch(path: string, options: ApiOptions = {}): Promise<Response> {
  return request(path, options)
}

export async function apiGet<T = any>(path: string, options: ApiOptions = {}): Promise<T> {
  const res = await request(path, options)
  return res.json()
}

export async function apiPost<T = any>(path: string, body: unknown, options: ApiOptions = {}): Promise<T> {
  const res = await request(path, {
    ...options,
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  // Stream responses leave body for caller
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('text/event-stream')) {
    return res as unknown as T
  }
  return res.json()
}

export async function apiPut<T = any>(path: string, body: unknown, options: ApiOptions = {}): Promise<T> {
  const res = await request(path, {
    ...options,
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return res.json()
}

export function safeHttpUrl(raw: string | undefined): string | null {
  try {
    const url = new URL(raw || '')
    return url.protocol === 'http:' || url.protocol === 'https:' ? url.href : null
  } catch {
    return null
  }
}

export type ProgressSnapshot = {
  stage?: string
  status?: string
  finished?: boolean
  error?: string
  findings_count?: number
  factoids_count?: number
  sources_count?: number
  pages_scanned?: number
  elapsed_s?: number
  section_progress?: string
  sections?: { title: string; chars: number }[]
  report?: string
  markdown_path?: string
  query?: string
  job_id?: string
  mode?: string
  // Thinking panel (Deep Research style)
  learned?: string[]
  gaps?: string[]
  next_action?: string
  thoughts?: { ts?: number; kind?: string; text?: string }[]
  off_topic?: boolean
  plan?: Record<string, unknown>
}
