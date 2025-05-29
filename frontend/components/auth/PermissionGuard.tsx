import React from 'react';
import usePermission from '../../lib/hooks/usePermission';
import { CircularProgress, Alert } from '@mui/material';

interface PermissionGuardProps {
  permission: string | string[];
  children: React.ReactNode;
  fallback?: React.ReactNode;
  requireAll?: boolean;
  showLoading?: boolean;
  showError?: boolean;
}

/**
 * A component that conditionally renders content based on user permissions
 * @param permission - The permission(s) to check for
 * @param children - Content to show if user has permission
 * @param fallback - Optional content to show if user doesn't have permission
 * @param requireAll - If true (default), user must have all permissions; if false, any one is sufficient
 * @param showLoading - Whether to show a loading indicator (defaults to true)
 * @param showError - Whether to show error messages (defaults to true)
 */
const PermissionGuard: React.FC<PermissionGuardProps> = ({
  permission,
  children,
  fallback = null,
  requireAll = true,
  showLoading = true,
  showError = true
}) => {
  const { hasPermission, loading, error } = usePermission(permission, requireAll);

  if (loading && showLoading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', padding: '20px' }}>
        <CircularProgress size={24} />
      </div>
    );
  }

  if (error && showError) {
    return (
      <Alert severity="error" sx={{ my: 2 }}>
        {error}
      </Alert>
    );
  }

  return hasPermission ? <>{children}</> : <>{fallback}</>;
};

export default PermissionGuard; 