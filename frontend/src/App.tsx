import { useEffect } from 'react'
import { Sidebar, MainContent } from './components/layout/Sidebar'

export default function App() {
  // 跟随鼠标的环境光：把坐标写进 CSS 变量，由 .cursor-glow 呈现
  useEffect(() => {
    if (window.matchMedia('(hover: none)').matches) return
    let raf = 0
    const onMove = (e: PointerEvent) => {
      if (raf) return
      raf = requestAnimationFrame(() => {
        raf = 0
        const root = document.documentElement
        root.style.setProperty('--mx', `${e.clientX}px`)
        root.style.setProperty('--my', `${e.clientY}px`)
      })
    }
    window.addEventListener('pointermove', onMove, { passive: true })
    return () => {
      window.removeEventListener('pointermove', onMove)
      if (raf) cancelAnimationFrame(raf)
    }
  }, [])

  return (
    <div className="relative flex h-screen overflow-hidden">
      {/* 环境光：缓慢漂移的几团光斑 */}
      <div className="ambient-light" aria-hidden="true">
        <span className="ambient-orb ambient-orb--key" />
        <span className="ambient-orb ambient-orb--violet" />
        <span className="ambient-orb ambient-orb--blue" />
        <span className="ambient-orb ambient-orb--cyan" />
      </div>
      {/* 鼠标位置的柔光 */}
      <div className="cursor-glow" aria-hidden="true" />
      {/* 顶部灯带 + 颗粒质感 */}
      <div className="top-light" aria-hidden="true" />
      <div className="app-grain" aria-hidden="true" />

      <Sidebar />
      <MainContent />
    </div>
  )
}
