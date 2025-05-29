import React, { useState } from 'react';
import { useAuth } from '../../lib/contexts/AuthContext';
import SessionList from '../auth/SessionList';
import {
  Box,
  Paper,
  Typography,
  Divider,
  Switch,
  FormGroup,
  FormControlLabel,
  Button,
  TextField,
  Alert,
  CircularProgress,
  Stack,
} from '@mui/material';
import LockIcon from '@mui/icons-material/Lock';
import SecurityIcon from '@mui/icons-material/Security';

const AccountSecurity: React.FC = () => {
  const { user, token, updateUser } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  
  // Password change form
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPasswordForm, setShowPasswordForm] = useState(false);
  
  // Security preferences (these would be fetched from user settings in a real app)
  const [preferences, setPreferences] = useState({
    twoFactorEnabled: false,
    loginNotifications: true,
    sessionTimeout: true,
    rememberDevices: true,
  });

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

  const handlePreferenceChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setPreferences({
      ...preferences,
      [event.target.name]: event.target.checked,
    });
    
    // In a real app, you would save these preferences to the backend
    // This is simplified for demonstration
    setSuccess('Preferences updated successfully');
    setTimeout(() => setSuccess(''), 3000);
  };

  const handlePasswordChange = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Validate passwords
    if (newPassword !== confirmPassword) {
      setError('New passwords do not match');
      return;
    }
    
    if (newPassword.length < 8) {
      setError('New password must be at least 8 characters long');
      return;
    }
    
    setLoading(true);
    setError('');
    setSuccess('');
    
    try {
      // This would be implemented in a real app to change the password
      // Here's a placeholder implementation for demonstration
      const response = await fetch(`${API_URL}/api/users/change-password`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify({
          currentPassword,
          newPassword,
        }),
      });
      
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.message || 'Failed to change password');
      }
      
      setSuccess('Password changed successfully');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setShowPasswordForm(false);
    } catch (err) {
      console.error('Password change error:', err);
      setError((err as Error).message || 'Failed to change password');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center' }}>
        <SecurityIcon sx={{ mr: 1 }} />
        Account Security
      </Typography>
      
      {/* Security Preferences */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Security Preferences
        </Typography>
        
        {success && <Alert severity="success" sx={{ mb: 2 }}>{success}</Alert>}
        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
        
        <FormGroup>
          <FormControlLabel
            control={
              <Switch
                checked={preferences.twoFactorEnabled}
                onChange={handlePreferenceChange}
                name="twoFactorEnabled"
              />
            }
            label="Two-factor authentication"
          />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, ml: 4 }}>
            Add an extra layer of security to your account by requiring a verification code in addition to your password.
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={preferences.loginNotifications}
                onChange={handlePreferenceChange}
                name="loginNotifications"
              />
            }
            label="Login notifications"
          />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, ml: 4 }}>
            Receive email notifications when your account is accessed from a new device or location.
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={preferences.sessionTimeout}
                onChange={handlePreferenceChange}
                name="sessionTimeout"
              />
            }
            label="Session timeout"
          />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, ml: 4 }}>
            Automatically log out after 30 minutes of inactivity for enhanced security.
          </Typography>
          
          <FormControlLabel
            control={
              <Switch
                checked={preferences.rememberDevices}
                onChange={handlePreferenceChange}
                name="rememberDevices"
              />
            }
            label="Remember trusted devices"
          />
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2, ml: 4 }}>
            Skip additional verification on devices you've previously marked as trusted.
          </Typography>
        </FormGroup>
      </Paper>
      
      {/* Password Change */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
          <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center' }}>
            <LockIcon sx={{ mr: 1 }} />
            Password
          </Typography>
          
          {!showPasswordForm && (
            <Button 
              variant="outlined" 
              onClick={() => setShowPasswordForm(true)}
            >
              Change Password
            </Button>
          )}
        </Box>
        
        {showPasswordForm ? (
          <Box component="form" onSubmit={handlePasswordChange} sx={{ mt: 2 }}>
            <Stack spacing={2}>
              <TextField
                label="Current Password"
                type="password"
                fullWidth
                value={currentPassword}
                onChange={(e) => setCurrentPassword(e.target.value)}
                required
                disabled={loading}
              />
              
              <TextField
                label="New Password"
                type="password"
                fullWidth
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                required
                disabled={loading}
                helperText="Must be at least 8 characters"
              />
              
              <TextField
                label="Confirm New Password"
                type="password"
                fullWidth
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                required
                disabled={loading}
              />
              
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2, mt: 2 }}>
                <Button 
                  variant="outlined" 
                  onClick={() => {
                    setShowPasswordForm(false);
                    setCurrentPassword('');
                    setNewPassword('');
                    setConfirmPassword('');
                    setError('');
                  }}
                  disabled={loading}
                >
                  Cancel
                </Button>
                
                <Button 
                  type="submit" 
                  variant="contained" 
                  disabled={loading}
                >
                  {loading ? <CircularProgress size={24} /> : 'Update Password'}
                </Button>
              </Box>
            </Stack>
          </Box>
        ) : (
          <Typography variant="body2" color="text.secondary">
            Last login: {user?.lastLoginAt ? new Date(user.lastLoginAt).toLocaleDateString() : 'Never'}.
            We recommend updating your password regularly for security purposes.
          </Typography>
        )}
      </Paper>
      
      {/* Active Sessions */}
      <SessionList />
    </Box>
  );
};

export default AccountSecurity; 