'use client'

interface LoadingDotsProps {
  label?: string
}

export function LoadingDots({ label = 'Loading...' }: LoadingDotsProps) {
  return (
    <div className="loading-card flex justify-start">
      <div
        className="flex items-center gap-3 rounded-2xl border px-4 py-3 shadow-sm"
        style={{ borderColor: 'var(--border)', background: 'var(--surface)' }}
      >
        <div className="flex space-x-1.5">
          <div className="loading-dot" style={{ animationDelay: '0ms' }} />
          <div className="loading-dot" style={{ animationDelay: '150ms' }} />
          <div className="loading-dot" style={{ animationDelay: '300ms' }} />
        </div>
        <span className="text-xs" style={{ color: 'var(--muted)' }}>{label}</span>
      </div>
    </div>
  )
}
