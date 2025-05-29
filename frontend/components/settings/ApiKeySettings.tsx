import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Card,
  CardContent,
  CardHeader,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogContentText,
  DialogTitle,
  Divider,
  FormControlLabel,
  IconButton,
  Paper,
  Switch,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
  useTheme
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import ErrorIcon from '@mui/icons-material/Error';
import { useSnackbar } from 'notistack';
import axios from 'axios';
import { format } from 'date-fns';

interface ApiKey {
  id: string;
  label: string;
  key: string;
  testnet: boolean;
  scopes: string;
  expiry: string;
  createdAt: string;
}

const ApiKeySettings: React.FC = () => {
  const theme = useTheme();
  const { enqueueSnackbar } = useSnackbar();
  const [loading, setLoading] = useState(false);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [openAddDialog, setOpenAddDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [selectedKey, setSelectedKey] = useState<ApiKey | null>(null);
  const [formData, setFormData] = useState({
    apiKey: '',
    apiSecret: '',
    label: '',
    testnet: false
  });
  const [validating, setValidating] = useState(false);
  const [validationStatus, setValidationStatus] = useState<'success' | 'error' | null>(null);

  // Fetch API keys on component mount
  useEffect(() => {
    fetchApiKeys();
  }, []);

  const fetchApiKeys = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/keys');
      setApiKeys(response.data.data);
    } catch (error) {
      console.error('Error fetching API keys:', error);
      enqueueSnackbar('Failed to load API keys', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, checked } = e.target;
    setFormData({
      ...formData,
      [name]: name === 'testnet' ? checked : value
    });
    
    // Reset validation status when inputs change
    if (name === 'apiKey' || name === 'apiSecret') {
      setValidationStatus(null);
    }
  };

  const validateApiKey = async () => {
    if (!formData.apiKey || !formData.apiSecret) {
      enqueueSnackbar('API key and secret are required', { variant: 'warning' });
      return;
    }

    try {
      setValidating(true);
      const response = await axios.post('/api/keys/validate', {
        apiKey: formData.apiKey,
        apiSecret: formData.apiSecret,
        testnet: formData.testnet
      });
      
      if (response.data.success) {
        setValidationStatus('success');
        enqueueSnackbar('API key validated successfully', { variant: 'success' });
      } else {
        setValidationStatus('error');
        enqueueSnackbar('API key validation failed', { variant: 'error' });
      }
    } catch (error) {
      console.error('API key validation error:', error);
      setValidationStatus('error');
      enqueueSnackbar('API key validation failed', { variant: 'error' });
    } finally {
      setValidating(false);
    }
  };

  const handleAddKey = async () => {
    if (!formData.apiKey || !formData.apiSecret || !formData.label) {
      enqueueSnackbar('All fields are required', { variant: 'warning' });
      return;
    }

    try {
      setLoading(true);
      const response = await axios.post('/api/keys', formData);
      
      if (response.data.success) {
        enqueueSnackbar('API key added successfully', { variant: 'success' });
        setOpenAddDialog(false);
        setFormData({
          apiKey: '',
          apiSecret: '',
          label: '',
          testnet: false
        });
        fetchApiKeys();
      }
    } catch (error) {
      console.error('Error adding API key:', error);
      enqueueSnackbar('Failed to add API key', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteKey = async () => {
    if (!selectedKey) return;
    
    try {
      setLoading(true);
      const response = await axios.delete(`/api/keys/${selectedKey.id}`);
      
      if (response.data.success) {
        enqueueSnackbar('API key deleted successfully', { variant: 'success' });
        setOpenDeleteDialog(false);
        setSelectedKey(null);
        fetchApiKeys();
      }
    } catch (error) {
      console.error('Error deleting API key:', error);
      enqueueSnackbar('Failed to delete API key', { variant: 'error' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card sx={{ mb: 3 }}>
      <CardHeader 
        title="API Key Management" 
        subheader="Manage your Delta Exchange API keys"
        action={
          <Button
            variant="contained"
            color="primary"
            startIcon={<AddIcon />}
            onClick={() => setOpenAddDialog(true)}
          >
            Add New Key
          </Button>
        }
      />
      <Divider />
      <CardContent>
        <Typography variant="body2" sx={{ mb: 2 }}>
          API keys are used to connect securely to your Delta Exchange account for automated trading.
          Your keys are stored encrypted and never shared with third parties.
        </Typography>
        
        {loading && !openAddDialog && !openDeleteDialog ? (
          <Box display="flex" justifyContent="center" my={3}>
            <CircularProgress />
          </Box>
        ) : apiKeys.length === 0 ? (
          <Paper sx={{ p: 3, textAlign: 'center', backgroundColor: theme.palette.background.default }}>
            <Typography variant="body1">
              You don't have any API keys yet. Add one to get started.
            </Typography>
          </Paper>
        ) : (
          <TableContainer component={Paper} sx={{ mt: 2 }}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Label</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Expires</TableCell>
                  <TableCell align="right">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {apiKeys.map((key) => (
                  <TableRow key={key.id}>
                    <TableCell>{key.label}</TableCell>
                    <TableCell>
                      {key.testnet ? 'Testnet' : 'Mainnet'}
                    </TableCell>
                    <TableCell>
                      {format(new Date(key.createdAt), 'MMM d, yyyy')}
                    </TableCell>
                    <TableCell>
                      {format(new Date(key.expiry), 'MMM d, yyyy')}
                    </TableCell>
                    <TableCell align="right">
                      <Tooltip title="Delete API Key">
                        <IconButton 
                          color="error" 
                          onClick={() => {
                            setSelectedKey(key);
                            setOpenDeleteDialog(true);
                          }}
                        >
                          <DeleteIcon />
                        </IconButton>
                      </Tooltip>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}

        {/* Add API Key Dialog */}
        <Dialog open={openAddDialog} onClose={() => setOpenAddDialog(false)} maxWidth="sm" fullWidth>
          <DialogTitle>Add Delta Exchange API Key</DialogTitle>
          <DialogContent>
            <DialogContentText sx={{ mb: 2 }}>
              Enter your Delta Exchange API credentials. These will be encrypted and stored securely.
            </DialogContentText>
            <TextField
              autoFocus
              margin="dense"
              name="label"
              label="Key Label"
              type="text"
              fullWidth
              variant="outlined"
              value={formData.label}
              onChange={handleInputChange}
              sx={{ mb: 2 }}
              placeholder="e.g., Trading Bot 1"
            />
            <TextField
              margin="dense"
              name="apiKey"
              label="API Key"
              type="text"
              fullWidth
              variant="outlined"
              value={formData.apiKey}
              onChange={handleInputChange}
              sx={{ mb: 2 }}
              InputProps={{
                endAdornment: validationStatus === 'success' ? (
                  <CheckCircleIcon color="success" />
                ) : validationStatus === 'error' ? (
                  <ErrorIcon color="error" />
                ) : null
              }}
            />
            <TextField
              margin="dense"
              name="apiSecret"
              label="API Secret"
              type="password"
              fullWidth
              variant="outlined"
              value={formData.apiSecret}
              onChange={handleInputChange}
              sx={{ mb: 2 }}
            />
            <FormControlLabel
              control={
                <Switch
                  checked={formData.testnet}
                  onChange={handleInputChange}
                  name="testnet"
                  color="primary"
                />
              }
              label="Use Testnet (for testing)"
            />
          </DialogContent>
          <DialogActions sx={{ px: 3, pb: 3 }}>
            <Button onClick={() => setOpenAddDialog(false)}>Cancel</Button>
            <Button 
              onClick={validateApiKey} 
              color="info" 
              disabled={validating || !formData.apiKey || !formData.apiSecret}
            >
              {validating ? <CircularProgress size={24} /> : 'Validate'}
            </Button>
            <Button 
              onClick={handleAddKey} 
              variant="contained" 
              color="primary"
              disabled={loading || !formData.apiKey || !formData.apiSecret || !formData.label}
            >
              {loading ? <CircularProgress size={24} /> : 'Add Key'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Delete Confirmation Dialog */}
        <Dialog
          open={openDeleteDialog}
          onClose={() => setOpenDeleteDialog(false)}
        >
          <DialogTitle>Delete API Key</DialogTitle>
          <DialogContent>
            <DialogContentText>
              Are you sure you want to delete the API key labeled "{selectedKey?.label}"? 
              This action cannot be undone.
            </DialogContentText>
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setOpenDeleteDialog(false)}>Cancel</Button>
            <Button onClick={handleDeleteKey} color="error" disabled={loading}>
              {loading ? <CircularProgress size={24} /> : 'Delete'}
            </Button>
          </DialogActions>
        </Dialog>
      </CardContent>
    </Card>
  );
};

export default ApiKeySettings; 