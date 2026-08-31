'use client'

import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
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

  // Date formatting is locale/time-zone dependent. Render the same placeholder
  // on SSR and the first client render, then format only after hydration.
  useEffect(() => {
    setTime(message.timestamp.toLocaleTimeString())
  }, [message.timestamp])

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div
        className={`max-w-[85%] rounded-lg px-5 py-4 ${
          isUser
            ? 'bg-blue-600 text-white shadow'
            : 'bg-white dark:bg-gray-800 shadow border border-gray-200 dark:border-gray-700'
        }`}
      >
        {message.role === 'assistant' ? (
          <div className="prose dark:prose-invert max-w-none text-sm leading-relaxed">
            <ReactMarkdown
              remarkPlugins={[remarkGfm, remarkMath]}
              rehypePlugins={[rehypeKatex]}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        ) : (
          <p className="whitespace-pre-wrap text-sm">{message.content}</p>
        )}
        <div className="text-[10px] mt-2 opacity-60 text-right">
          {time || '\u00a0'}
        </div>
      </div>
    </div>
  )
}
