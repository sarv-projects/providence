'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import { Check, Copy, Sparkles, User } from 'lucide-react'
import { useEffect, useState } from 'react'

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
}

interface MessageBubbleProps {
  message: ChatMessage
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const [time, setTime] = useState('')
  const [copied, setCopied] = useState(false)
  // Long markdown assistant messages are "reports" → show a copy affordance.
  const isReport = !isUser && message.content.length > 800

  // Date formatting is locale/time-zone dependent. Render the same placeholder
  // on SSR and the first client render, then format only after hydration.
  useEffect(() => {
    setTime(message.timestamp.toLocaleTimeString())
  }, [message.timestamp])

  async function copyReport() {
    try {
      await navigator.clipboard.writeText(message.content)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1800)
    } catch {
      /* clipboard unavailable — ignore */
    }
  }

  return (
    <div className={`flex items-start gap-2.5 ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      <div
        className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full text-white shadow-sm"
        style={
          isUser
            ? { background: 'var(--surface-3)', color: 'var(--muted)' }
            : { background: 'linear-gradient(135deg, var(--accent), var(--accent-2))' }
        }
        aria-hidden
      >
        {isUser ? <User size={13} /> : <Sparkles size={13} />}
      </div>

      <div className={`chat-bubble ${isUser ? 'user' : 'assistant'}`}>
        {message.role === 'assistant' ? (
          <>
            {isReport && (
              <div className="mb-1.5 flex items-center justify-between border-b pb-1.5" style={{ borderColor: 'var(--border)' }}>
                <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--accent)' }}>
                  Research report
                </span>
                <button
                  type="button"
                  onClick={copyReport}
                  className="flex items-center gap-1 text-[10px] text-muted transition-colors hover:text-[var(--foreground)]"
                  aria-label="Copy report"
                >
                  {copied ? <Check size={11} className="text-emerald-500" /> : <Copy size={11} />}
                  {copied ? 'Copied' : 'Copy'}
                </button>
              </div>
            )}
            <div className="markdown-body">
              <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
                {message.content}
              </ReactMarkdown>
            </div>
          </>
        ) : (
          <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        )}
        <div className="msg-time">{time || '\u00a0'}</div>
      </div>
    </div>
  )
}
