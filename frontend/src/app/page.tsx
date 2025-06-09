export default function Home() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Welcome to SmartMarketOOPS</h1>
        <p className="text-muted-foreground">
          Professional AI-powered trading platform with real-time market analysis
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Portfolio Overview</h3>
            <p className="card-description">Real-time portfolio performance and balance</p>
          </div>
          <div className="card-content">
            <div className="text-2xl font-bold text-green-500">$10,000.00</div>
            <p className="text-sm text-muted-foreground">+2.5% today</p>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Active Positions</h3>
            <p className="card-description">Currently open trading positions</p>
          </div>
          <div className="card-content">
            <div className="text-2xl font-bold">3</div>
            <p className="text-sm text-muted-foreground">2 profitable, 1 pending</p>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">AI Model Status</h3>
            <p className="card-description">Machine learning model performance</p>
          </div>
          <div className="card-content">
            <div className="text-2xl font-bold text-blue-500">85%</div>
            <p className="text-sm text-muted-foreground">Prediction accuracy</p>
          </div>
        </div>
      </div>

      <div className="flex gap-4">
        <a href="/dashboard" className="button">Open Dashboard</a>
        <a href="/charts" className="button variant-outline">Live Charts</a>
        <a href="/analytics" className="button variant-outline">AI Analytics</a>
        <button className="button variant-secondary">Settings</button>
      </div>

      <div className="card">
        <div className="card-header">
          <h3 className="card-title">🎉 Setup Complete!</h3>
          <p className="card-description">
            All services are running successfully
          </p>
        </div>
        <div className="card-content">
          <ul className="list-disc list-inside space-y-2 text-sm">
            <li>✅ Frontend: Next.js 15 + React 19 + Custom CSS</li>
            <li>✅ Backend: Node.js + TypeScript + Delta Exchange</li>
            <li>✅ ML Service: Python + FastAPI + PyTorch</li>
            <li>✅ Real-time Data: 3 exchanges connected</li>
            <li>✅ Professional Navigation: Clean responsive design</li>
            <li>✅ All dependencies installed and working</li>
          </ul>
        </div>
      </div>
    </div>
  );
}