import { createTheme, Theme, alpha } from '@mui/material/styles';
import { PaletteMode } from '@mui/material';

// Create theme based on mode (light/dark)
export const createAppTheme = (mode: PaletteMode): Theme => {
  // Common palette values
  const primaryColor = '#3861FB'; // Crypto blue
  const secondaryColor = '#14B8A6'; // Teal green
  const errorColor = '#F43F5E'; // Red
  const successColor = '#10B981'; // Green
  const warningColor = '#F59E0B'; // Amber

  return createTheme({
    palette: {
      mode,
      primary: {
        main: primaryColor,
        light: mode === 'light' ? '#5E7EFC' : '#5E7EFC',
        dark: mode === 'light' ? '#2043D6' : '#2043D6',
        contrastText: '#FFFFFF',
      },
      secondary: {
        main: secondaryColor,
        light: mode === 'light' ? '#43C6B8' : '#43C6B8',
        dark: mode === 'light' ? '#0E8A7D' : '#0E8A7D',
        contrastText: '#FFFFFF',
      },
      error: {
        main: errorColor,
        light: mode === 'light' ? '#F76583' : '#F76583',
        dark: mode === 'light' ? '#D11A38' : '#D11A38',
      },
      warning: {
        main: warningColor,
        light: mode === 'light' ? '#FBBF4D' : '#FBBF4D',
        dark: mode === 'light' ? '#B45309' : '#B45309',
      },
      success: {
        main: successColor,
        light: mode === 'light' ? '#34D399' : '#34D399',
        dark: mode === 'light' ? '#059669' : '#059669',
      },
      background: {
        default: mode === 'light' ? '#F9FAFB' : '#0F172A', // Light gray or deep blue-black
        paper: mode === 'light' ? '#FFFFFF' : '#1E293B', // White or dark blue-gray
      },
      text: {
        primary: mode === 'light' ? '#111827' : '#F3F4F6',
        secondary: mode === 'light' ? '#4B5563' : '#9CA3AF',
        disabled: mode === 'light' ? '#9CA3AF' : '#4B5563',
      },
      divider: mode === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.08)',
      action: {
        active: mode === 'light' ? 'rgba(0, 0, 0, 0.54)' : 'rgba(255, 255, 255, 0.7)',
        hover: mode === 'light' ? 'rgba(0, 0, 0, 0.04)' : 'rgba(255, 255, 255, 0.08)',
        selected: mode === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.16)',
        disabled: mode === 'light' ? 'rgba(0, 0, 0, 0.26)' : 'rgba(255, 255, 255, 0.3)',
        disabledBackground: mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)',
      },
    },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: { fontWeight: 600 },
      h2: { fontWeight: 600 },
      h3: { fontWeight: 600 },
      h4: { fontWeight: 500 },
      h5: { fontWeight: 500 },
      h6: { fontWeight: 500 },
      button: { fontWeight: 500, textTransform: 'none' },
    },
    shape: {
      borderRadius: 8,
    },
    components: {
      MuiCssBaseline: {
        styleOverrides: {
          body: {
            scrollbarWidth: 'thin',
            '&::-webkit-scrollbar': {
              width: '8px',
              height: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: mode === 'light' ? '#F1F5F9' : '#1E293B',
            },
            '&::-webkit-scrollbar-thumb': {
              backgroundColor: mode === 'light' ? '#CBD5E1' : '#475569',
              borderRadius: '4px',
            },
            '&::-webkit-scrollbar-thumb:hover': {
              backgroundColor: mode === 'light' ? '#94A3B8' : '#64748B',
            },
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            boxShadow: mode === 'light' 
              ? '0px 2px 4px rgba(0, 0, 0, 0.05)' 
              : '0px 2px 4px rgba(0, 0, 0, 0.2)',
            borderRadius: 12,
          },
        },
      },
      MuiButton: {
        styleOverrides: {
          root: {
            textTransform: 'none',
            fontWeight: 500,
            boxShadow: 'none',
            ':hover': {
              boxShadow: 'none',
            },
          },
          contained: {
            ':hover': {
              boxShadow: 'none',
            },
          },
        },
      },
      MuiTableCell: {
        styleOverrides: {
          root: {
            borderBottom: mode === 'light' 
              ? '1px solid rgba(0, 0, 0, 0.08)' 
              : '1px solid rgba(255, 255, 255, 0.08)',
          },
        },
      },
      MuiTableRow: {
        styleOverrides: {
          root: {
            '&.MuiTableRow-hover:hover': {
              backgroundColor: mode === 'light' 
                ? 'rgba(0, 0, 0, 0.04)' 
                : 'rgba(255, 255, 255, 0.08)',
            },
          },
        },
      },
      MuiListItemButton: {
        styleOverrides: {
          root: {
            borderRadius: 6,
            margin: '2px 8px',
            padding: '8px 12px',
            '&.Mui-selected': {
              backgroundColor: mode === 'light' 
                ? alpha(primaryColor, 0.1) 
                : alpha(primaryColor, 0.2),
              '&:hover': {
                backgroundColor: mode === 'light' 
                  ? alpha(primaryColor, 0.15) 
                  : alpha(primaryColor, 0.25),
              },
            },
          },
        },
      },
      MuiMenuItem: {
        styleOverrides: {
          root: {
            borderRadius: 4,
            marginTop: 2,
            marginBottom: 2,
          },
        },
      },
      MuiChip: {
        styleOverrides: {
          root: {
            borderRadius: 6,
          },
        },
      },
      MuiSwitch: {
        styleOverrides: {
          root: {
            width: 46,
            height: 24,
            padding: 0,
          },
          switchBase: {
            padding: 2,
            '&.Mui-checked': {
              transform: 'translateX(22px)',
            },
          },
          thumb: {
            width: 20,
            height: 20,
          },
          track: {
            borderRadius: 12,
          },
        },
      },
      MuiPaper: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
      MuiInputBase: {
        styleOverrides: {
          root: {
            borderRadius: 8,
          },
        },
      },
      MuiPopover: {
        styleOverrides: {
          paper: {
            borderRadius: 8,
          },
        },
      },
      MuiDrawer: {
        styleOverrides: {
          paper: {
            borderRight: mode === 'light' 
              ? '1px solid rgba(0, 0, 0, 0.08)' 
              : '1px solid rgba(255, 255, 255, 0.08)',
          },
        },
      },
      MuiAppBar: {
        styleOverrides: {
          root: {
            backgroundImage: 'none',
          },
        },
      },
    },
  });
};

// Export both light and dark themes
export const lightTheme = createAppTheme('light');
export const darkTheme = createAppTheme('dark');

export default createAppTheme; 