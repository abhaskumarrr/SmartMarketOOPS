import React, { createContext, useContext, useMemo, useState } from 'react';
import { ThemeProvider as MuiThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import { PaletteMode } from '@mui/material';
import { createAppTheme } from '../theme';

// Define context type
interface ThemeContextType {
  mode: PaletteMode;
  toggleColorMode: () => void;
}

// Create context with default values
const ThemeContext = createContext<ThemeContextType>({
  mode: 'dark',
  toggleColorMode: () => {},
});

// Custom hook to use the theme context
export const useThemeContext = () => useContext(ThemeContext);

// Theme provider component
export const ThemeProvider: React.FC<{children: React.ReactNode}> = ({ children }) => {
  // Default to dark mode
  const [mode, setMode] = useState<PaletteMode>('dark');

  // Toggle between light and dark modes
  const toggleColorMode = () => {
    setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
  };

  // Create theme based on current mode
  const theme = useMemo(() => createAppTheme(mode), [mode]);

  // Context value
  const contextValue = useMemo(() => ({
    mode,
    toggleColorMode,
  }), [mode]);

  return (
    <ThemeContext.Provider value={contextValue}>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </MuiThemeProvider>
    </ThemeContext.Provider>
  );
};

export default ThemeContext; 