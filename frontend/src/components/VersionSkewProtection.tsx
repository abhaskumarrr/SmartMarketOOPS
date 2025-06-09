'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';

export function VersionSkewProtection({ children }: { children: React.ReactNode }) {
  const [error, setError] = useState<Error | null>(null);
  const router = useRouter();
  const pathname = usePathname();
  const deploymentId = process.env.NEXT_PUBLIC_DEPLOYMENT_ID;

  // Error boundary to catch webpack errors
  useEffect(() => {
    const originalError = console.error;
    
    // Override console.error to catch specific webpack errors
    console.error = (...args) => {
      const errorMessage = args[0]?.toString() || '';
      
      // Check for the specific error pattern
      if (
        errorMessage.includes('TypeError: Cannot read properties of undefined (reading \'call\')') ||
        errorMessage.includes('ChunkLoadError')
      ) {
        // Force a full page refresh on version skew errors
        setError(new Error('Version skew detected'));
        
        // Perform a full page refresh after a short delay
        setTimeout(() => {
          window.location.href = pathname;
        }, 100);
        
        // Don't output the error to console to avoid noise
        return;
      }
      
      // Call the original console.error for other errors
      originalError.apply(console, args);
    };
    
    // Keep track of the deployment ID in localStorage
    const storedDeploymentId = localStorage.getItem('deployment-id');
    
    if (deploymentId && storedDeploymentId && deploymentId !== storedDeploymentId) {
      console.log('Deployment ID changed, refreshing page');
      localStorage.setItem('deployment-id', deploymentId);
      window.location.href = pathname;
    } else if (deploymentId) {
      localStorage.setItem('deployment-id', deploymentId);
    }
    
    // Restore the original console.error on cleanup
    return () => {
      console.error = originalError;
    };
  }, [pathname, deploymentId]);
  
  // Show a simple error message if a version skew is detected
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-100">
        <div className="p-6 max-w-sm mx-auto bg-white rounded-xl shadow-md flex items-center space-x-4">
          <div>
            <div className="text-xl font-medium text-black">Updating application...</div>
            <p className="text-gray-500">Please wait while we refresh the page.</p>
          </div>
        </div>
      </div>
    );
  }
  
  return <>{children}</>;
} 