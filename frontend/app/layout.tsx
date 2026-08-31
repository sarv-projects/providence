import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Providence — Deep Research Engine',
  description:
    'Autonomous multi-agent deep research with verified citations, adversarial critique, and a strict compiler ship-gate. Zero API keys required.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    // suppressHydrationWarning avoids theme flash; html data-theme mirrors the
    // class toggled in lib/theme (localStorage + prefers-color-scheme).
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: `(function(){try{var t=localStorage.getItem('theme');var d=t?t==='dark':window.matchMedia('(prefers-color-scheme: dark)').matches;document.documentElement.classList.toggle('dark',d);document.documentElement.setAttribute('data-theme',d?'dark':'light');}catch(e){}})()`,
          }}
        />
      </head>
      <body className="font-sans antialiased">{children}</body>
    </html>
  )
}
