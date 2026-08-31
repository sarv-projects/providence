'use client'

import { ReactNode, useEffect, useState } from 'react'
import { Database, FileText, History, Menu, Moon, Settings, Sparkles, Sun, X } from 'lucide-react'
import { applyTheme, initTheme, resolveTheme, type Theme } from '@/lib/theme'

interface AppShellProps {
  children: ReactNode
  title: string
  description?: string
  action?: ReactNode
}

const links = [
  { href: '/', label: 'Chat', icon: Sparkles },
  { href: '/history', label: 'History', icon: History },
  { href: '/vault', label: 'Vault', icon: Database },
  { href: '/settings', label: 'Settings', icon: Settings },
]

export function AppShell({ children, title, description, action }: AppShellProps) {
  const [dark, setDark] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    initTheme()
    setDark(resolveTheme() === 'dark')
  }, [])

  function toggleTheme() {
    const next: Theme = dark ? 'light' : 'dark'
    setDark(next === 'dark')
    applyTheme(next)
  }

  return (
    <div className="app-shell">
      <button className="mobile-menu-button" onClick={() => setOpen(!open)} aria-label="Open navigation">
        {open ? <X size={19} /> : <Menu size={19} />}
      </button>
      {open && <button className="mobile-backdrop" onClick={() => setOpen(false)} aria-label="Close navigation" />}
      <aside className={`app-nav ${open ? 'mobile-open' : ''}`}>
        <div className="app-brand"><span className="brand-mark"><Sparkles size={16} /></span><strong>Providence</strong></div>
        <nav>{links.map(({ href, label, icon: Icon }) => <a key={href} href={href} onClick={() => setOpen(false)}><Icon size={16} />{label}</a>)}</nav>
        <button className="theme-button" onClick={toggleTheme}>{dark ? <Moon size={16} /> : <Sun size={16} />}<span>{dark ? 'Dark mode' : 'Light mode'}</span></button>
      </aside>
      <section className="app-main">
        <header className="page-header"><div><p className="eyebrow">Providence</p><h1>{title}</h1>{description && <p>{description}</p>}</div>{action}</header>
        {children}
      </section>
    </div>
  )
}
