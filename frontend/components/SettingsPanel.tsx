import React, { useState } from 'react';
import {
  Box,
  Button,
  Divider,
  Drawer,
  IconButton,
  Tab,
  Tabs,
  Typography,
  useTheme,
  useMediaQuery,
} from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';
import SaveIcon from '@mui/icons-material/Save';
import UndoIcon from '@mui/icons-material/Undo';
import GetAppIcon from '@mui/icons-material/GetApp';
import PublishIcon from '@mui/icons-material/Publish';

import { usePreferences } from '../lib/contexts/PreferencesContext';
import ChartSettings from './settings/ChartSettings';
import NotificationSettings from './settings/NotificationSettings';
import LayoutSettings from './settings/LayoutSettings';
import DataSettings from './settings/DataSettings';
import ThemeSettings from './settings/ThemeSettings';
import ApiKeyManagement from './settings/ApiKeyManagement';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
      style={{ height: '100%', overflowY: 'auto' }}
    >
      {value === index && (
        <Box sx={{ p: 3, height: '100%' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function a11yProps(index: number) {
  return {
    id: `settings-tab-${index}`,
    'aria-controls': `settings-tabpanel-${index}`,
  };
}

interface SettingsPanelProps {
  open: boolean;
  onClose: () => void;
}

const SettingsPanel: React.FC<SettingsPanelProps> = ({ open, onClose }) => {
  const [tabValue, setTabValue] = useState(0);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);
  const [resetConfirmOpen, setResetConfirmOpen] = useState(false);
  
  const { preferences, resetAllPreferences, exportPreferencesToJSON, importPreferencesFromJSON } = usePreferences();
  const theme = useTheme();
  const isSmallScreen = useMediaQuery(theme.breakpoints.down('sm'));
  
  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };
  
  const handleReset = () => {
    resetAllPreferences();
    setResetConfirmOpen(false);
  };
  
  const handleExport = () => {
    const json = exportPreferencesToJSON();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = 'smartmarket-preferences.json';
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    URL.revokeObjectURL(url);
    document.body.removeChild(a);
    setExportDialogOpen(false);
  };
  
  const handleImport = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      const contents = e.target?.result as string;
      if (contents) {
        const success = importPreferencesFromJSON(contents);
        if (!success) {
          // Handle error - could add a snackbar here
          console.error('Failed to import preferences');
        }
      }
    };
    reader.readAsText(file);
    setImportDialogOpen(false);
  };
  
  return (
    <Drawer
      anchor="right"
      open={open}
      onClose={onClose}
      sx={{
        '& .MuiDrawer-paper': {
          width: isSmallScreen ? '100%' : '500px',
          boxSizing: 'border-box',
        },
      }}
    >
      <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        {/* Header */}
        <Box sx={{ display: 'flex', alignItems: 'center', p: 2, borderBottom: 1, borderColor: 'divider' }}>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Settings
          </Typography>
          <IconButton onClick={onClose} edge="end" aria-label="close">
            <CloseIcon />
          </IconButton>
        </Box>
        
        {/* Tabs */}
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs
            value={tabValue}
            onChange={handleTabChange}
            aria-label="settings tabs"
            variant={isSmallScreen ? "scrollable" : "fullWidth"}
            scrollButtons="auto"
          >
            <Tab label="Chart" {...a11yProps(0)} />
            <Tab label="Notifications" {...a11yProps(1)} />
            <Tab label="Layout" {...a11yProps(2)} />
            <Tab label="Data" {...a11yProps(3)} />
            <Tab label="Theme" {...a11yProps(4)} />
            <Tab label="API Keys" {...a11yProps(5)} />
          </Tabs>
        </Box>
        
        {/* Tab Panels */}
        <Box sx={{ flexGrow: 1, overflow: 'auto' }}>
          <TabPanel value={tabValue} index={0}>
            <ChartSettings />
          </TabPanel>
          <TabPanel value={tabValue} index={1}>
            <NotificationSettings />
          </TabPanel>
          <TabPanel value={tabValue} index={2}>
            <LayoutSettings />
          </TabPanel>
          <TabPanel value={tabValue} index={3}>
            <DataSettings />
          </TabPanel>
          <TabPanel value={tabValue} index={4}>
            <ThemeSettings />
          </TabPanel>
          <TabPanel value={tabValue} index={5}>
            <ApiKeyManagement />
          </TabPanel>
        </Box>
        
        {/* Actions */}
        <Box sx={{ p: 2, borderTop: 1, borderColor: 'divider', display: 'flex', justifyContent: 'space-between' }}>
          <Button
            variant="outlined"
            color="error"
            startIcon={<UndoIcon />}
            onClick={() => setResetConfirmOpen(true)}
          >
            Reset All
          </Button>
          
          <Box>
            <Button
              variant="outlined"
              startIcon={<PublishIcon />}
              sx={{ mr: 1 }}
              component="label"
            >
              Import
              <input
                type="file"
                accept=".json"
                hidden
                onChange={handleImport}
              />
            </Button>
            <Button
              variant="outlined"
              startIcon={<GetAppIcon />}
              onClick={handleExport}
              sx={{ mr: 1 }}
            >
              Export
            </Button>
            <Button
              variant="contained"
              startIcon={<SaveIcon />}
              onClick={onClose}
              color="primary"
            >
              Done
            </Button>
          </Box>
        </Box>
      </Box>
    </Drawer>
  );
};

export default SettingsPanel; 