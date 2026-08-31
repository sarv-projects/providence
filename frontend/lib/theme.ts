'use client'

export type Theme = 'light' | 'dark'

export function getStoredTheme(): Theme | null {
  try {
    const value = localStorage.getItem('theme')
    return value === 'dark' || value === 'light' ? value : null
  } catch {
    return null
  }
}

export function applyTheme(theme: Theme) {
  document.documentElement.classList.toggle('dark', theme === 'dark')
  try {
    localStorage.setItem('theme', theme)
  } catch {
    /* ignore */
  }
}

export function resolveTheme(): Theme {
  const stored = getStoredTheme()
  if (stored) return stored
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

export function initTheme() {
  applyTheme(resolveTheme())
}
