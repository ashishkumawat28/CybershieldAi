
import { useEffect, useState } from 'react'
import {
  ShieldCheck,
  Activity,
  Network,
  AlertTriangle,
  Server,
  RefreshCw,
  CheckCircle2,
} from 'lucide-react'

function App() {
  const [backendStatus, setBackendStatus] = useState('Checking...')
  const [loading, setLoading] = useState(false)

  const checkBackend = async () => {
    try {
      setLoading(true)

      const response = await fetch('/api/health')

      if (!response.ok) {
        throw new Error('Backend unavailable')
      }

      const data = await response.json()
      setBackendStatus(data.status)
    } catch (error) {
      setBackendStatus('Disconnected')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    checkBackend()
  }, [])

  const stats = [
    {
      title: 'Total Network Traffic',
      value: '0',
      icon: Network,
    },
    {
      title: 'Threats Detected',
      value: '0',
      icon: AlertTriangle,
    },
    {
      title: 'Active Monitoring',
      value: 'Offline',
      icon: Activity,
    },
    {
      title: 'Security Status',
      value: 'Protected',
      icon: ShieldCheck,
    },
  ]

  return (
    <div className="min-h-screen bg-[#080d18] text-gray-100">

      {/* Navbar */}
      <nav className="border-b border-slate-800 px-6 py-4">
        <div className="mx-auto flex max-w-7xl items-center justify-between">

          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-cyan-500/10 p-2">
              <ShieldCheck className="text-cyan-400" size={28} />
            </div>

            <div>
              <h1 className="text-xl font-bold tracking-wide">
                CyberShield <span className="text-cyan-400">AI</span>
              </h1>
              <p className="text-xs text-slate-400">
                Intelligent Threat Detection System
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-slate-700 px-4 py-2">
            <span className="h-2 w-2 rounded-full bg-green-400" />
            <span className="text-sm text-slate-300">
              System Dashboard
            </span>
          </div>

        </div>
      </nav>

      {/* Main Content */}
      <main className="mx-auto max-w-7xl px-6 py-10">

        <div className="mb-8">
          <p className="mb-2 text-sm font-medium text-cyan-400">
            SECURITY OVERVIEW
          </p>

          <h2 className="text-3xl font-bold">
            Welcome to CyberShield AI
          </h2>

          <p className="mt-2 text-slate-400">
            Monitor network activity and detect potential
            cybersecurity threats using artificial intelligence.
          </p>
        </div>

        {/* Backend Connection */}
        <div className="mb-8 flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-[#111827] p-5">

          <div className="flex items-center gap-3">
            <Server className="text-cyan-400" size={24} />

            <div>
              <p className="font-semibold">Backend Connection</p>

              <p className="text-sm text-slate-400">
                Status: {backendStatus}
              </p>
            </div>
          </div>

          <button
            onClick={checkBackend}
            disabled={loading}
            className="flex items-center gap-2 rounded-lg bg-cyan-500 px-4 py-2 font-semibold text-slate-950 hover:bg-cyan-400 disabled:opacity-50"
          >
            <RefreshCw size={16} />
            {loading ? 'Checking...' : 'Check Connection'}
          </button>

        </div>

        {/* Statistics */}
        <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-4">

          {stats.map((item) => {
            const Icon = item.icon

            return (
              <div
                key={item.title}
                className="rounded-xl border border-slate-800 bg-[#111827] p-6 transition hover:border-cyan-500/50"
              >
                <div className="mb-5 flex items-center justify-between">
                  <p className="text-sm text-slate-400">
                    {item.title}
                  </p>

                  <Icon className="text-cyan-400" size={22} />
                </div>

                <h3 className="text-2xl font-bold">
                  {item.value}
                </h3>
              </div>
            )
          })}

        </div>

        {/* Initial Status */}
        <div className="mt-8 rounded-xl border border-slate-800 bg-[#111827] p-6">

          <div className="mb-4 flex items-center gap-3">
            <CheckCircle2 className="text-green-400" size={24} />

            <h3 className="text-lg font-semibold">
              Project Initialization
            </h3>
          </div>

          <p className="text-sm leading-7 text-slate-400">
            CyberShield AI dashboard has been initialized.
            Network traffic analysis, machine learning
            classification, anomaly detection and threat
            intelligence modules will be integrated in
            subsequent development phases.
          </p>

        </div>

        <footer className="mt-12 border-t border-slate-800 pt-6 text-center text-sm text-slate-500">
          CyberShield AI | AI-Powered Cybersecurity Platform
        </footer>

      </main>
    </div>
  )
}

export default App
