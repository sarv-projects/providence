'use client'

import { CheckCircle, XCircle } from 'lucide-react'

export interface ApprovalRequest {
  approval_id: string
  gate_type: string
  data?: any
  status?: string
}

interface ApprovalBannerProps {
  approvals: ApprovalRequest[]
  onRespond: (id: string, approved: boolean) => void
}

export function ApprovalBanner({ approvals, onRespond }: ApprovalBannerProps) {
  if (!approvals.length) return null
  const a = approvals[0]
  return (
    <div className="approval-banner">
      <div className="flex items-center gap-2 font-medium">
        ⏳ Pending Workflow Approval ({approvals.length}): {a.gate_type.toUpperCase()} gate
        requires review.
      </div>
      <div className="flex gap-2">
        <button
          onClick={() => onRespond(a.approval_id, true)}
          className="flex items-center gap-1 rounded-lg bg-emerald-600 px-3 py-1 text-xs font-semibold text-white transition-colors hover:bg-emerald-500"
        >
          <CheckCircle className="h-3.5 w-3.5" /> Approve
        </button>
        <button
          onClick={() => onRespond(a.approval_id, false)}
          className="flex items-center gap-1 rounded-lg bg-red-700 px-3 py-1 text-xs font-semibold text-white transition-colors hover:bg-red-600"
        >
          <XCircle className="h-3.5 w-3.5" /> Reject
        </button>
      </div>
    </div>
  )
}
