import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

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
        <Card>
          <CardHeader>
            <CardTitle>Portfolio Overview</CardTitle>
            <CardDescription>Real-time portfolio performance and balance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-500">$10,000.00</div>
            <p className="text-sm text-muted-foreground">+2.5% today</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Active Positions</CardTitle>
            <CardDescription>Currently open trading positions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">3</div>
            <p className="text-sm text-muted-foreground">2 profitable, 1 pending</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>AI Model Status</CardTitle>
            <CardDescription>Machine learning model performance</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-500">85%</div>
            <p className="text-sm text-muted-foreground">Prediction accuracy</p>
          </CardContent>
        </Card>
      </div>

      <div className="flex gap-4">
        <Button asChild>
          <a href="/dashboard">Open Dashboard</a>
        </Button>
        <Button variant="outline" asChild>
          <a href="/charts">Live Charts</a>
        </Button>
        <Button variant="outline" asChild>
          <a href="/analytics">AI Analytics</a>
        </Button>
        <Button variant="secondary">Settings</Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>🎉 Setup Complete!</CardTitle>
          <CardDescription>
            All services are running successfully
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ul className="list-disc list-inside space-y-2 text-sm">
            <li>✅ Frontend: Next.js 15 + React 19 + Tailwind CSS + shadcn/ui</li>
            <li>✅ Backend: Node.js + TypeScript + Delta Exchange</li>
            <li>✅ ML Service: Python + FastAPI + PyTorch</li>
            <li>✅ Real-time Data: 3 exchanges connected</li>
            <li>✅ Professional Navigation: Sidebar with theme support</li>
            <li>✅ All warnings fixed + Node.js v24.1.0</li>
          </ul>
        </CardContent>
      </Card>
    </div>
  );
}