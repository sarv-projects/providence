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

  return (
    <>
      <button type="button" className="mobile-menu-button" onClick={() => setMobileOpen(!mobileOpen)} aria-label="Toggle navigation">
        {mobileOpen ? <X size={19} /> : <Menu size={19} />}
      </button>
      {mobileOpen && <button type="button" className="mobile-backdrop" onClick={() => setMobileOpen(false)} aria-label="Close navigation" />}
    <div className={`sidebar w-64 bg-gray-100 dark:bg-gray-800 p-4 flex flex-col border-r border-gray-200 dark:border-gray-700 ${mobileOpen ? 'mobile-open' : ''}`}>
      <div className="flex items-center gap-2 mb-6">
        <Sparkles className="w-6 h-6 text-blue-600" />
        <h1 className="font-bold text-lg">Providence</h1>
      </div>

      <div className="space-y-2 mb-6">
        <button
          onClick={() => { onModeChange('chat'); setMobileOpen(false) }}
          className={`w-full flex items-center gap-2 px-4 py-2.5 rounded-lg text-left transition-colors font-medium text-sm ${
            mode === 'chat'
              ? 'bg-blue-600 text-white shadow'
              : 'hover:bg-gray-200 dark:hover:bg-gray-700'
          }`}
        >
          <Sparkles className="w-4 h-4" />
          Chat
        </button>
        <button
          onClick={() => { onModeChange('research'); setMobileOpen(false) }}
          className={`w-full flex items-center gap-2 px-4 py-2.5 rounded-lg text-left transition-colors font-medium text-sm ${
            mode === 'research'
              ? 'bg-blue-600 text-white shadow'
              : 'hover:bg-gray-200 dark:hover:bg-gray-700'
          }`}
        >
          <FileText className="w-4 h-4" />
          Deep Research
        </button>
      </div>

      <div className="mt-auto space-y-2 pt-4 border-t border-gray-200 dark:border-gray-700">
        <button
          type="button"
          onClick={toggleTheme}
          className="w-full flex items-center justify-between rounded-lg px-4 py-2 text-sm text-gray-600 hover:bg-gray-200 dark:text-gray-300 dark:hover:bg-gray-700"
          aria-label="Toggle theme"
        >
          <span className="flex items-center gap-2">{dark ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />} Theme</span>
          <span className="text-xs text-gray-400">{dark ? 'Dark' : 'Light'}</span>
        </button>
        <a
          href="/vault" onClick={() => setMobileOpen(false)}
          className="w-full flex items-center gap-2 px-4 py-2 rounded-lg text-sm hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          <Database className="w-4 h-4 text-purple-500" />
          Research Vault
        </a>
        <a
          href="/history" onClick={() => setMobileOpen(false)}
          className="w-full flex items-center gap-2 px-4 py-2 rounded-lg text-sm hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          <History className="w-4 h-4 text-amber-500" />
          History
        </a>
        <a
          href="/settings" onClick={() => setMobileOpen(false)}
          className="w-full flex items-center gap-2 px-4 py-2 rounded-lg text-sm hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          <Settings className="w-4 h-4 text-gray-500" />
          Settings
        </a>
      </div>
    </div>
    </>
  )
}
