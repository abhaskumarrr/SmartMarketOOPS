import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Button, 
  Card, 
  CardContent, 
  CardHeader, 
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider, 
   
  IconButton, 
  Snackbar, 
  Tab, 
  Tabs, 
  TextField, 
  Typography,
  Alert,
  Chip,
  Switch,
  FormControlLabel,
  CircularProgress,
  Tooltip
} from '@mui/material';
import { 
  Add as AddIcon, 
  Refresh as RefreshIcon,
  Delete as DeleteIcon,
  Star as StarIcon,
  StarBorder as StarBorderIcon,
  ContentCopy as CopyIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon
} from '@mui/icons-material';
import { useAuth } from '../../lib/contexts/AuthContext';

// API client for handling API key operations
const apiClient = {
  getApiKeys: async () => {
    const response = await fetch('/api/trading/api-keys', {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to fetch API keys');
    return data.data;
  },
  
  createApiKey: async (keyData: any) => {
    const response = await fetch('/api/trading/api-keys', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify(keyData)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to create API key');
    return data.data;
  },
  
  updateApiKey: async (id: string, updates: any) => {
    const response = await fetch(`/api/trading/api-keys/${id}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify(updates)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to update API key');
    return data.data;
  },
  
  revokeApiKey: async (id: string, reason: string) => {
    const response = await fetch(`/api/trading/api-keys/${id}`, {
      method: 'DELETE',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      },
      body: JSON.stringify({ reason })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to revoke API key');
    return data.data;
  },
  
  setDefaultApiKey: async (id: string) => {
    const response = await fetch(`/api/trading/api-keys/${id}/default`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Failed to set default API key');
    return data.data;
  }
};

// Environment label component
const EnvironmentLabel: React.FC<{ environment: string }> = ({ environment }) => {
  const isTestnet = environment === 'testnet';
  return (
    <Chip 
      label={isTestnet ? 'Testnet' : 'Mainnet'} 
      color={isTestnet ? 'info' : 'error'} 
      size="small"
      sx={{ ml: 1 }}
    />
  );
};

// API key status component
const ApiKeyStatus: React.FC<{ isExpired: boolean, isRevoked: boolean, expiry: string }> = ({ 
  isExpired, 
  isRevoked, 
  expiry 
}) => {
  if (isRevoked) {
    return <Chip label="Revoked" color="error" size="small" />;
  }
  if (isExpired) {
    return <Chip label="Expired" color="warning" size="small" />;
  }
  
  // Show days remaining
  const daysRemaining = Math.ceil((new Date(expiry).getTime() - Date.now()) / (1000 * 60 * 60 * 24));
  const color = daysRemaining <= 7 ? 'warning' : 'success';
  
  return (
    <Chip 
      label={`Active (${daysRemaining} days)`} 
      color={color} 
      size="small" 
    />
  );
};

// API key creation dialog
const AddApiKeyDialog: React.FC<{
  open: boolean;
  onClose: () => void;
  onAdd: (keyData: any) => void;
}> = ({ open, onClose, onAdd }) => {
  const [name, setName] = useState('');
  const [key, setKey] = useState('');
  const [secret, setSecret] = useState('');
  const [environment, setEnvironment] = useState('testnet');
  const [expiryDays, setExpiryDays] = useState(30);
  const [isDefault, setIsDefault] = useState(false);
  const [showSecret, setShowSecret] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async () => {
    try {
      setLoading(true);
      setError('');
      
      // Calculate expiry date
      const expiry = new Date();
      expiry.setDate(expiry.getDate() + expiryDays);
      
      const keyData = {
        name,
        key,
        secret,
        environment,
        expiry: expiry.toISOString(),
        isDefault,
        scopes: ['read', 'trade']
      };
      
      await onAdd(keyData);
      onClose();
    } catch (err: any) {
      setError(err.message || 'Failed to add API key');
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setName('');
    setKey('');
    setSecret('');
    setEnvironment('testnet');
    setExpiryDays(30);
    setIsDefault(false);
    setShowSecret(false);
    setError('');
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>Add New API Key</DialogTitle>
      <DialogContent>
        <DialogContentText sx={{ mb: 2 }}>
          Add your Delta Exchange API key details. For security, all keys are encrypted and securely stored.
        </DialogContentText>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <TextField
          label="Name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        
        <TextField
          label="API Key"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          fullWidth
          margin="normal"
          required
        />
        
        <TextField
          label="API Secret"
          value={secret}
          onChange={(e) => setSecret(e.target.value)}
          type={showSecret ? 'text' : 'password'}
          fullWidth
          margin="normal"
          required
          InputProps={{
            endAdornment: (
              <IconButton
                onClick={() => setShowSecret(!showSecret)}
                edge="end"
              >
                {showSecret ? <VisibilityOffIcon /> : <VisibilityIcon />}
              </IconButton>
            )
          }}
        />
        
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Environment
          </Typography>
          <Tabs
            value={environment}
            onChange={(e, newValue) => setEnvironment(newValue)}
            variant="fullWidth"
          >
            <Tab label="Testnet" value="testnet" />
            <Tab label="Mainnet" value="mainnet" />
          </Tabs>
        </Box>
        
        <TextField
          label="Expiry (Days)"
          type="number"
          value={expiryDays}
          onChange={(e) => setExpiryDays(parseInt(e.target.value, 10))}
          fullWidth
          margin="normal"
          InputProps={{ inputProps: { min: 1, max: 365 } }}
        />
        
        <FormControlLabel
          control={
            <Switch
              checked={isDefault}
              onChange={(e) => setIsDefault(e.target.checked)}
            />
          }
          label="Set as default key"
          sx={{ mt: 1 }}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={handleClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          color="primary"
          disabled={!name || !key || !secret || loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Add API Key'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

// API key revocation dialog
const RevokeApiKeyDialog: React.FC<{
  open: boolean;
  keyId: string;
  keyName: string;
  onClose: () => void;
  onRevoke: (id: string, reason: string) => void;
}> = ({ open, keyId, keyName, onClose, onRevoke }) => {
  const [reason, setReason] = useState('');
  const [loading, setLoading] = useState(false);
  
  const handleRevoke = async () => {
    try {
      setLoading(true);
      await onRevoke(keyId, reason);
      onClose();
    } catch (error) {
      console.error('Failed to revoke key:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <Dialog open={open} onClose={onClose}>
      <DialogTitle>Revoke API Key</DialogTitle>
      <DialogContent>
        <DialogContentText>
          Are you sure you want to revoke the API key "{keyName}"? This action cannot be undone.
        </DialogContentText>
        <TextField
          label="Reason (Optional)"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
          fullWidth
          margin="normal"
          multiline
          rows={2}
        />
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={handleRevoke}
          color="error"
          variant="contained"
          disabled={loading}
        >
          {loading ? <CircularProgress size={24} /> : 'Revoke Key'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

// Main API key management component
const ApiKeyManagement: React.FC = () => {
  const { user } = useAuth();
  const [apiKeys, setApiKeys] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [addDialogOpen, setAddDialogOpen] = useState(false);
  const [revokeDialogOpen, setRevokeDialogOpen] = useState(false);
  const [selectedKeyId, setSelectedKeyId] = useState('');
  const [selectedKeyName, setSelectedKeyName] = useState('');
  const [activeTab, setActiveTab] = useState('active');
  
  // Fetch API keys
  const fetchApiKeys = async () => {
    try {
      setLoading(true);
      const keys = await apiClient.getApiKeys();
      setApiKeys(keys);
      setError('');
    } catch (err: any) {
      setError(err.message || 'Failed to fetch API keys');
    } finally {
      setLoading(false);
    }
  };
  
  // Add API key
  const handleAddApiKey = async (keyData: any) => {
    try {
      const newKey = await apiClient.createApiKey(keyData);
      setApiKeys([...apiKeys, newKey]);
      setSuccessMessage('API key added successfully');
    } catch (err: any) {
      setError(err.message || 'Failed to add API key');
      throw err;
    }
  };
  
  // Set default API key
  const handleSetDefault = async (keyId: string) => {
    try {
      await apiClient.setDefaultApiKey(keyId);
      
      // Update local state
      setApiKeys(apiKeys.map(key => ({
        ...key,
        isDefault: key.id === keyId
      })));
      
      setSuccessMessage('Default API key updated');
    } catch (err: any) {
      setError(err.message || 'Failed to set default API key');
    }
  };
  
  // Revoke API key
  const handleRevokeApiKey = async (keyId: string, reason: string) => {
    try {
      await apiClient.revokeApiKey(keyId, reason);
      
      // Update local state
      setApiKeys(apiKeys.map(key => 
        key.id === keyId ? { ...key, isRevoked: true } : key
      ));
      
      setSuccessMessage('API key revoked successfully');
    } catch (err: any) {
      setError(err.message || 'Failed to revoke API key');
    }
  };
  
  // Open revoke dialog
  const openRevokeDialog = (keyId: string, keyName: string) => {
    setSelectedKeyId(keyId);
    setSelectedKeyName(keyName);
    setRevokeDialogOpen(true);
  };
  
  // Filter keys based on active tab
  const filteredKeys = apiKeys.filter(key => {
    if (activeTab === 'active') {
      return !key.isRevoked && !key.isExpired;
    } else if (activeTab === 'testnet') {
      return key.environment === 'testnet' && !key.isRevoked;
    } else if (activeTab === 'mainnet') {
      return key.environment === 'mainnet' && !key.isRevoked;
    } else if (activeTab === 'revoked') {
      return key.isRevoked;
    } else if (activeTab === 'expired') {
      return key.isExpired && !key.isRevoked;
    }
    return true;
  });
  
  // Load API keys on component mount
  useEffect(() => {
    fetchApiKeys();
  }, []);
  
  // Clear success message after 5 seconds
  useEffect(() => {
    if (successMessage) {
      const timer = setTimeout(() => {
        setSuccessMessage('');
      }, 5000);
      return () => clearTimeout(timer);
    }
  }, [successMessage]);
  
  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setSuccessMessage('Copied to clipboard');
  };
  
  return (
    <Box>
      <Card>
        <CardHeader
          title="API Key Management"
          subheader="Manage your Delta Exchange API keys"
          action={
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setAddDialogOpen(true)}
            >
              Add New Key
            </Button>
          }
        />
        <Divider />
        <CardContent>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          
          <Tabs
            value={activeTab}
            onChange={(e, newValue) => setActiveTab(newValue)}
            sx={{ mb: 2 }}
          >
            <Tab label="Active" value="active" />
            <Tab label="Testnet" value="testnet" />
            <Tab label="Mainnet" value="mainnet" />
            <Tab label="Revoked" value="revoked" />
            <Tab label="Expired" value="expired" />
          </Tabs>
          
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
              <CircularProgress />
            </Box>
          ) : filteredKeys.length === 0 ? (
            <Typography variant="body2" color="textSecondary" sx={{ p: 2, textAlign: 'center' }}>
              No API keys found in this category
            </Typography>
          ) : (
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 2 }}>
              {filteredKeys.map((key) => (
                <Box>
                  <Card variant="outlined">
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="h6">
                            {key.name}
                          </Typography>
                          <EnvironmentLabel environment={key.environment} />
                          {key.isDefault && (
                            <Tooltip title="Default API Key">
                              <StarIcon sx={{ ml: 1, color: 'gold' }} />
                            </Tooltip>
                          )}
                        </Box>
                        <Box>
                          <ApiKeyStatus 
                            isExpired={key.isExpired} 
                            isRevoked={key.isRevoked} 
                            expiry={key.expiry} 
                          />
                        </Box>
                      </Box>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" color="textSecondary">
                          Key: {key.maskedKey}
                        </Typography>
                        <IconButton 
                          size="small" 
                          onClick={() => copyToClipboard(key.maskedKey)}
                          sx={{ ml: 1 }}
                        >
                          <CopyIcon fontSize="small" />
                        </IconButton>
                      </Box>
                      
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                        <Typography variant="body2" color="textSecondary" component="span">
                          Scopes: 
                        </Typography>
                        {key.scopes.map((scope: string) => (
                          <Chip 
                            key={scope} 
                            label={scope} 
                            size="small" 
                            sx={{ ml: 1 }} 
                          />
                        ))}
                      </Box>
                      
                      <Typography variant="body2" color="textSecondary">
                        Created: {new Date(key.createdAt).toLocaleDateString()}
                        {key.lastUsedAt && ` | Last used: ${new Date(key.lastUsedAt).toLocaleDateString()}`}
                        {` | Usage count: ${key.usageCount || 0}`}
                      </Typography>
                      
                      {!key.isRevoked && (
                        <Box sx={{ mt: 2, display: 'flex', gap: 1 }}>
                          {!key.isDefault && (
                            <Button
                              size="small"
                              startIcon={<StarBorderIcon />}
                              onClick={() => handleSetDefault(key.id)}
                              variant="outlined"
                            >
                              Set Default
                            </Button>
                          )}
                          <Button
                            size="small"
                            startIcon={<DeleteIcon />}
                            onClick={() => openRevokeDialog(key.id, key.name)}
                            color="error"
                            variant="outlined"
                          >
                            Revoke
                          </Button>
                        </Box>
                      )}
                    </CardContent>
                  </Card>
                </Box>
              ))}
            </Box>
          )}
        </CardContent>
      </Card>
      
      {/* Add API Key Dialog */}
      <AddApiKeyDialog
        open={addDialogOpen}
        onClose={() => setAddDialogOpen(false)}
        onAdd={handleAddApiKey}
      />
      
      {/* Revoke API Key Dialog */}
      <RevokeApiKeyDialog
        open={revokeDialogOpen}
        keyId={selectedKeyId}
        keyName={selectedKeyName}
        onClose={() => setRevokeDialogOpen(false)}
        onRevoke={handleRevokeApiKey}
      />
      
      {/* Success Snackbar */}
      <Snackbar
        open={Boolean(successMessage)}
        autoHideDuration={5000}
        onClose={() => setSuccessMessage('')}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={() => setSuccessMessage('')} 
          severity="success" 
          sx={{ width: '100%' }}
        >
          {successMessage}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ApiKeyManagement; 