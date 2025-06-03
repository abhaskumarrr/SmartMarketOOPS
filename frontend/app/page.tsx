/**
 * SmartMarketOOPS Home Page
 * Redirects to the main dashboard
 */

'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function HomePage() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to dashboard with proper error handling
    const redirectToDashboard = () => {
      try {
        router.push('/dashboard');
      } catch (error) {
        console.error('Navigation error:', error);
        // Fallback to window.location if router fails
        if (typeof window !== 'undefined') {
          window.location.href = '/dashboard';
        }
      }
    };

    // Small delay to ensure router is ready
    const timer = setTimeout(redirectToDashboard, 100);
    return () => clearTimeout(timer);
  }, [router]);

  return (
    <div className="min-h-screen bg-slate-950 text-white flex items-center justify-center">
      <div className="text-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
        <h1 className="text-2xl font-bold text-white mb-2">SmartMarketOOPS</h1>
        <p className="text-slate-300">Loading Real-Time Trading Dashboard...</p>
        <div className="mt-4">
          <a
            href="/dashboard"
            className="text-blue-400 hover:text-blue-300 underline"
          >
            Click here if not redirected automatically
          </a>
        </div>
      </div>
    </div>
  );
}
