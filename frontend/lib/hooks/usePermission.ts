import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';

/**
 * A hook to check if the current user has specific permissions
 * @param permissions - A single permission or array of permissions to check
 * @param requireAll - If true, user must have all permissions, otherwise any one permission is sufficient
 * @returns Object containing loading state, whether the user has the requested permission(s), and any error
 */
export const usePermission = (
  permissions: string | string[], 
  requireAll: boolean = true
): {
  hasPermission: boolean;
  loading: boolean;
  error: string | null;
} => {
  const { user, token, isAuthenticated } = useAuth();
  const [permissionResults, setPermissionResults] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const permissionsArray = Array.isArray(permissions) ? permissions : [permissions];

  useEffect(() => {
    const checkPermissions = async () => {
      // Reset state at the beginning of each check
      setError(null);
      
      if (!isAuthenticated || !token || !user) {
        setPermissionResults({});
        setLoading(false);
        return;
      }

      setLoading(true);
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';
      const results: Record<string, boolean> = {};

      try {
        // Check each permission in parallel
        const promises = permissionsArray.map(async (permission) => {
          try {
            const response = await fetch(`${API_URL}/api/roles/check-permission/${permission}`, {
              headers: {
                Authorization: `Bearer ${token}`
              }
            });

            if (response.ok) {
              const data = await response.json();
              results[permission] = data.data.hasPermission;
            } else {
              const errorData = await response.json().catch(() => ({ message: 'Unknown error' }));
              console.error(`Permission check failed for ${permission}:`, errorData);
              results[permission] = false;
              
              // Only set error if we don't already have one
              if (!error) {
                setError(`Error checking permission: ${errorData.message || response.statusText}`);
              }
            }
          } catch (error) {
            console.error(`Error checking permission ${permission}:`, error);
            results[permission] = false;
            setError(`Network error while checking permissions: ${(error as Error).message}`);
          }
        });

        await Promise.all(promises);
        setPermissionResults(results);
      } catch (error) {
        console.error('Error checking permissions:', error);
        setError(`Error checking permissions: ${(error as Error).message}`);
      } finally {
        setLoading(false);
      }
    };

    checkPermissions();
  }, [isAuthenticated, token, user, JSON.stringify(permissionsArray)]);

  // Calculate the final permission result based on requireAll flag
  const hasPermission = !loading && Object.keys(permissionResults).length > 0
    ? requireAll
      ? permissionsArray.every(p => permissionResults[p] === true)
      : permissionsArray.some(p => permissionResults[p] === true)
    : false;

  return { hasPermission, loading, error };
};

export default usePermission; 