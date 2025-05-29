import React, { useState, useEffect } from 'react';
import { useAuth } from '../../lib/contexts/AuthContext';
import {
  Box,
  Button,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Paper,
  Divider,
  Alert,
  CircularProgress,
  Tooltip,
} from '@mui/material';
import DevicesIcon from '@mui/icons-material/Devices';
import LogoutIcon from '@mui/icons-material/Logout';
import DeleteIcon from '@mui/icons-material/Delete';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import AccessTimeIcon from '@mui/icons-material/AccessTime';

interface Session {
  id: string;
  device: string;
  ipAddress: string;
  lastActive: string;
  createdAt: string;
  isCurrentSession: boolean;
}

const SessionList: React.FC = () => {
  const { token } = useAuth();
  const [sessions, setSessions] = useState<Session[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [revoking, setRevoking] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3001';

  useEffect(() => {
    fetchSessions();
  }, [token]);

  const fetchSessions = async () => {
    if (!token) return;

    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/sessions`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch sessions');
      }

      const data = await response.json();
      setSessions(data.data);
    } catch (err) {
      console.error('Error fetching sessions:', err);
      setError('Unable to load sessions. Please try again later.');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const getTimeSince = (dateString: string) => {
    const now = new Date();
    const date = new Date(dateString);
    const diffMs = now.getTime() - date.getTime();
    
    // Convert to minutes, hours, days
    const diffMins = Math.floor(diffMs / 60000);
    if (diffMins < 60) {
      return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
    }
    
    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) {
      return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
    }
    
    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  };

  const revokeSession = async (sessionId: string) => {
    if (!token) return;
    
    try {
      setRevoking(sessionId);
      const response = await fetch(`${API_URL}/api/sessions/${sessionId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to revoke session');
      }

      // Update the sessions list after successful revocation
      setSessions(sessions.filter(session => session.id !== sessionId));
    } catch (err) {
      console.error('Error revoking session:', err);
      setError('Failed to revoke session. Please try again.');
    } finally {
      setRevoking(null);
    }
  };

  const revokeAllOtherSessions = async () => {
    if (!token) return;
    
    try {
      setRevoking('all');
      const response = await fetch(`${API_URL}/api/sessions`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to revoke all sessions');
      }

      // Fetch the updated sessions list
      fetchSessions();
    } catch (err) {
      console.error('Error revoking all sessions:', err);
      setError('Failed to revoke sessions. Please try again.');
    } finally {
      setRevoking(null);
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 3, mt: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <DevicesIcon sx={{ mr: 1 }} />
        <Typography variant="h6">Active Sessions</Typography>
      </Box>
      
      {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}
      
      {sessions.length === 0 ? (
        <Alert severity="info">No active sessions found.</Alert>
      ) : (
        <>
          <List>
            {sessions.map((session) => (
              <React.Fragment key={session.id}>
                <ListItem alignItems="flex-start">
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography variant="subtitle1">
                          {session.device}
                          {session.isCurrentSession && (
                            <Tooltip title="Current session">
                              <CheckCircleIcon color="success" sx={{ ml: 1, fontSize: 16 }} />
                            </Tooltip>
                          )}
                        </Typography>
                      </Box>
                    }
                    secondary={
                      <>
                        <Typography variant="body2" color="text.secondary">
                          IP: {session.ipAddress}
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                          <AccessTimeIcon sx={{ fontSize: 16, mr: 0.5, color: 'text.secondary' }} />
                          <Typography variant="body2" color="text.secondary">
                            Last active: {getTimeSince(session.lastActive)}
                          </Typography>
                        </Box>
                      </>
                    }
                  />
                  <ListItemSecondaryAction>
                    {!session.isCurrentSession && (
                      <Tooltip title="Revoke session">
                        <IconButton 
                          edge="end" 
                          color="error"
                          onClick={() => revokeSession(session.id)}
                          disabled={revoking === session.id || revoking === 'all'}
                        >
                          {revoking === session.id ? (
                            <CircularProgress size={24} />
                          ) : (
                            <DeleteIcon />
                          )}
                        </IconButton>
                      </Tooltip>
                    )}
                  </ListItemSecondaryAction>
                </ListItem>
                <Divider variant="inset" component="li" />
              </React.Fragment>
            ))}
          </List>
          
          {sessions.filter(s => !s.isCurrentSession).length > 0 && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Button
                variant="outlined"
                color="error"
                startIcon={revoking === 'all' ? <CircularProgress size={24} /> : <LogoutIcon />}
                onClick={revokeAllOtherSessions}
                disabled={revoking !== null}
              >
                Revoke All Other Sessions
              </Button>
            </Box>
          )}
        </>
      )}
    </Paper>
  );
};

export default SessionList; 