import { Component, type ReactNode } from 'react'

interface Props {
  children: ReactNode
}

interface State {
  error: Error | null
}

/** 调试面板的渲染异常不应该把整个应用打成白屏。 */
export class DebugErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 backdrop-blur-sm p-6">
          <div className="max-w-md rounded-2xl border border-red-500/30 bg-surface-1 p-5 space-y-3">
            <div className="text-sm text-red-300">调试面板渲染出错</div>
            <div className="text-[11px] text-gray-500 break-words">{this.state.error.message}</div>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => this.setState({ error: null })}
                className="btn-primary text-[11px] py-1.5 px-3"
              >
                重试
              </button>
              <button
                type="button"
                onClick={() => window.location.reload()}
                className="h-7 px-3 rounded-lg border border-white/[0.08] text-[11px] text-gray-400"
              >
                刷新页面
              </button>
            </div>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}
