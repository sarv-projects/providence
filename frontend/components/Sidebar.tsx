'use client'

import { Sparkles, FileText, Settings, History, Database, Sun, Moon, Menu, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { applyTheme, initTheme, resolveTheme, type Theme } from '@/lib/theme'

type Mode = 'chat' | 'research'

interface SidebarProps {
  mode: Mode
  onModeChange: (m: Mode) => void
}

export function Sidebar({ mode, onModeChange }: SidebarProps) {
  const [dark, setDark] = useState(false)
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    initTheme()
    setDark(resolveTheme() === 'dark')
  }, [])

  function toggleTheme() {
    const next: Theme = dark ? 'light' : 'dark'
    setDark(next === 'dark')
    applyTheme(next)
  }

  const modeButton = (m: Mode, label: string, Icon: typeof Sparkles) => (
    <button
      onClick={() => { onModeChange(m); setMobileOpen(false) }}
      className={`flex w-full items-center gap-2.5 rounded-xl px-3.5 py-2.5 text-left text-sm font-medium transition-all duration-200 ${
        mode === m
          ? 'text-white shadow-md'
          : 'text-muted hover:bg-[var(--surface-2)] hover:text-[var(--foreground)]'
      }`}
      style={mode === m ? { background: 'linear-gradient(135deg, var(--accent), var(--accent-2))', boxShadow: '0 6px 16px -8px rgba(99,102,241,.7)' } : undefined}
    >
      <Icon size={16} />
      {label}
    </button>
  )

  const navLink = (href: string, label: string, Icon: typeof Settings, color?: string) => (
    <a
      href={href}
      onClick={() => setMobileOpen(false)}
      className="flex items-center gap-2.5 rounded-xl px-3.5 py-2 text-sm text-muted transition-all duration-200 hover:bg-[var(--surface-2)] hover:text-[var(--foreground)]"
    >
      <Icon size={15} style={color ? { color } : undefined} />
      {label}
    </a>
  )

  return (
    <>
      <button type="button" className="mobile-menu-button" onClick={() => setMobileOpen(!mobileOpen)} aria-label="Toggle navigation">
        {mobileOpen ? <X size={19} /> : <Menu size={19} />}
      </button>
      {mobileOpen && <button type="button" className="mobile-backdrop" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}
      <aside className={`sidebar flex h-screen w-64 shrink-0 flex-col border-r p-4 ${mobileOpen ? 'mobile-open' : ''}`}
        style={{ background: 'color-mix(in srgb, var(--surface) 82%, transparent)', backdropFilter: 'blur(16px)' }}
      >
        <div className="app-brand">
          <span className="brand-mark"><Sparkles size={16} /></span>
          <strong>Providence</strong>
        </div>

        <div className="space-y-1.5">
          {modeButton('chat', 'Chat', Sparkles)}
          {modeButton('research', 'Deep Research', FileText)}
        </div>

        <div className="mt-auto space-y-1 border-t pt-4" style={{ borderColor: 'var(--border)' }}>
          <button
            type="button"
            onClick={toggleTheme}
            className="flex w-full items-center justify-between rounded-xl px-3.5 py-2 text-sm text-muted transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--foreground)]"
            aria-label="Toggle theme"
          >
            <span className="flex items-center gap-2.5">{dark ? <Moon size={15} /> : <Sun size={15} />} Theme</span>
            <span className="text-xs opacity-60">{dark ? 'Dark' : 'Light'}</span>
          </button>
          {navLink('/vault', 'Research Vault', Database, '#a855f7')}
          {navLink('/history', 'History', History, '#f59e0b')}
          {navLink('/settings', 'Settings', Settings)}
        </div>
      </aside>
    </>
  )
}
