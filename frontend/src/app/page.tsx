import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

export default function Home() {
  return (
    <div className="min-h-screen bg-background p-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-8">
          <h1 className="text-4xl font-bold text-foreground mb-2">
            SmartMarketOOPS
          </h1>
          <p className="text-xl text-muted-foreground">
            Professional Trading Dashboard
          </p>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Overview</CardTitle>
              <CardDescription>
                Real-time portfolio performance and balance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-500">$10,000.00</div>
              <p className="text-sm text-muted-foreground">+2.5% today</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Active Positions</CardTitle>
              <CardDescription>
                Currently open trading positions
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">3</div>
              <p className="text-sm text-muted-foreground">2 profitable, 1 pending</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>AI Model Status</CardTitle>
              <CardDescription>
                Machine learning model performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-500">85%</div>
              <p className="text-sm text-muted-foreground">Prediction accuracy</p>
            </CardContent>
          </Card>
        </div>

        <div className="mt-8 flex gap-4">
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

        <div className="mt-8">
          <Card>
            <CardHeader>
              <CardTitle>Setup Complete!</CardTitle>
              <CardDescription>
                Next.js 15, React 19, Tailwind CSS, and shadcn/ui are now configured
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="list-disc list-inside space-y-2 text-sm">
                <li>✅ Next.js 15 with TypeScript</li>
                <li>✅ Tailwind CSS with custom theme</li>
                <li>✅ shadcn/ui components</li>
                <li>✅ Dark/Light theme support</li>
                <li>✅ Trading-specific dependencies installed</li>
              </ul>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}