import { useRef } from 'react'

/** 行内重命名输入框：Enter 提交，Esc 取消，失焦提交 */
export function InlineRename({
  value,
  onChange,
  onSubmit,
  onCancel,
}: {
  value: string
  onChange: (v: string) => void
  onSubmit: () => void
  onCancel: () => void
}) {
  // Enter 提交后输入框会卸载并触发一次 blur，用 ref 防止重复提交
  const submitted = useRef(false)
  const submit = () => {
    if (submitted.current) return
    submitted.current = true
    onSubmit()
  }

  return (
    <input
      className="flex-1 min-w-0 bg-white/[0.06] border border-brand/40 rounded-md px-1.5 py-0.5 text-xs text-gray-100 outline-none focus:border-brand"
      value={value}
      autoFocus
      onChange={(e) => onChange(e.target.value)}
      onClick={(e) => e.stopPropagation()}
      onBlur={submit}
      onKeyDown={(e) => {
        if (e.key === 'Enter') submit()
        if (e.key === 'Escape') {
          submitted.current = true
          onCancel()
        }
      }}
    />
  )
}
