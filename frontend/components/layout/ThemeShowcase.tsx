import React from 'react';
import {
  Box,
  Paper,
  Typography,
  
  Button,
  TextField,
  Switch,
  Chip,
  Card,
  CardContent,
  CardActions,
  Divider,
  Alert,
  useTheme as useMuiTheme
} from '@mui/material';
import { lightTheme, darkTheme } from '../../lib/theme';

interface ThemeShowcaseProps {
  expanded?: boolean;
}

const ThemeShowcase: React.FC<ThemeShowcaseProps> = ({ expanded = false }) => {
  const theme = useMuiTheme();
  const darkMode = theme.palette.mode === 'dark';

  // Color palette showcase
  const colorPalette = [
    { name: 'Primary', color: theme.palette.primary.main, contrastText: theme.palette.primary.contrastText },
    { name: 'Secondary', color: theme.palette.secondary.main, contrastText: theme.palette.secondary.contrastText },
    { name: 'Error', color: theme.palette.error.main, contrastText: '#ffffff' },
    { name: 'Warning', color: theme.palette.warning.main, contrastText: '#000000' },
    { name: 'Info', color: theme.palette.info.main, contrastText: '#ffffff' },
    { name: 'Success', color: theme.palette.success.main, contrastText: '#ffffff' },
    { name: 'Text Primary', color: theme.palette.text.primary, contrastText: theme.palette.background.default },
    { name: 'Text Secondary', color: theme.palette.text.secondary, contrastText: theme.palette.background.default },
    { name: 'Background Default', color: theme.palette.background.default, contrastText: theme.palette.text.primary },
    { name: 'Background Paper', color: theme.palette.background.paper, contrastText: theme.palette.text.primary },
  ];

  // Component variants
  const buttons = [
    { variant: 'contained', color: 'primary', label: 'Primary' },
    { variant: 'contained', color: 'secondary', label: 'Secondary' },
    { variant: 'outlined', color: 'primary', label: 'Outlined' },
    { variant: 'text', color: 'primary', label: 'Text' },
  ];

  // Alerts
  const alerts = [
    { severity: 'error', message: 'Error Alert' },
    { severity: 'warning', message: 'Warning Alert' },
    { severity: 'info', message: 'Info Alert' },
    { severity: 'success', message: 'Success Alert' },
  ];

  return (
    <Box sx={{ py: 2 }}>
      <Typography variant="h5" gutterBottom>Theme Showcase</Typography>
      <Typography variant="body2" paragraph>
        Current theme: <strong>{darkMode ? 'Dark Mode' : 'Light Mode'}</strong>
      </Typography>
      
      <Typography variant="h6" gutterBottom sx={{ mt: 3 }}>Color Palette</Typography>
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "repeat(2, 1fr)", md: "repeat(3, 1fr)" }, gap: 2 }}>
        {colorPalette.map((item) => (
          <Box>
            <Paper
              sx={{
                bgcolor: item.color,
                color: item.contrastText,
                p: 2,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                justifyContent: 'space-between',
                borderRadius: 1,
                transition: 'transform 0.2s ease-in-out',
                '&:hover': {
                  transform: 'scale(1.05)',
                },
              }}
              elevation={3}
            >
              <Typography variant="subtitle2">{item.name}</Typography>
              <Typography variant="caption">{item.color}</Typography>
            </Paper>
          </Box>
        ))}
      </Box>
      
      {expanded && (
        <>
          <Typography variant="h6" gutterBottom sx={{ mt: 4 }}>Components</Typography>
          
          <Box sx={{ mb: 4 }}>
            <Typography variant="subtitle1" gutterBottom>Buttons</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
              {buttons.map((btn, index) => (
                <Button 
                  key={index}
                  variant={btn.variant as any}
                  color={btn.color as any}
                >
                  {btn.label}
                </Button>
              ))}
              <Button disabled>Disabled</Button>
              <Button variant="contained" color="success">Success</Button>
              <Button variant="contained" color="error">Error</Button>
            </Box>
          </Box>
          
          <Box sx={{ mb: 4 }}>
            <Typography variant="subtitle1" gutterBottom>Text Fields</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
              <TextField label="Standard" />
              <TextField label="Outlined" variant="outlined" />
              <TextField label="Filled" variant="filled" />
              <TextField label="Disabled" disabled />
              <TextField label="Error" error helperText="Error message" />
            </Box>
          </Box>
          
          <Box sx={{ mb: 4 }}>
            <Typography variant="subtitle1" gutterBottom>Chips</Typography>
            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap', mb: 2 }}>
              <Chip label="Primary" color="primary" />
              <Chip label="Secondary" color="secondary" />
              <Chip label="Success" color="success" />
              <Chip label="Error" color="error" />
              <Chip label="Warning" color="warning" />
              <Chip label="Info" color="info" />
              <Chip label="Default" />
              <Chip label="Disabled" disabled />
              <Chip label="Clickable" onClick={() => {}} />
              <Chip label="Deletable" onDelete={() => {}} />
            </Box>
          </Box>
          
          <Box sx={{ mb: 4 }}>
            <Typography variant="subtitle1" gutterBottom>Card</Typography>
            <Card sx={{ maxWidth: 345 }}>
              <CardContent>
                <Typography gutterBottom variant="h5" component="div">
                  Card Title
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  This is an example card component with some sample content to demonstrate
                  how cards appear in the current theme.
                </Typography>
              </CardContent>
              <Divider />
              <CardActions>
                <Button size="small">Action 1</Button>
                <Button size="small">Action 2</Button>
              </CardActions>
            </Card>
          </Box>
          
          <Box sx={{ mb: 4 }}>
            <Typography variant="subtitle1" gutterBottom>Alerts</Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              {alerts.map((alert, index) => (
                <Alert key={index} severity={alert.severity as any}>
                  {alert.message}
                </Alert>
              ))}
            </Box>
          </Box>
          
          <Box sx={{ mb: 4 }}>
            <Typography variant="subtitle1" gutterBottom>Switch</Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <Switch defaultChecked />
              <Switch />
              <Switch disabled defaultChecked />
              <Switch disabled />
            </Box>
          </Box>
        </>
      )}
    </Box>
  );
};

export default ThemeShowcase; 