import { Sidebar, MainContent } from './components/layout/Sidebar'

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden bg-gray-950">
      <Sidebar />
      <MainContent />
    </div>
  )
}
