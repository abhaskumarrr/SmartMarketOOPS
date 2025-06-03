import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import {
  UserPreferences,
  defaultPreferences,
  loadPreferences,
  savePreferences,
  updatePreferences,
  resetPreferences,
  exportPreferences,
  importPreferences,
} from '../userPreferences';

interface PreferencesContextType {
  preferences: UserPreferences;
  updatePreference: <K extends keyof Omit<UserPreferences, 'version' | 'lastUpdated'>>(
    section: K,
    values: Partial<UserPreferences[K]>
  ) => void;
  resetAllPreferences: () => void;
  exportPreferencesToJSON: () => string;
  importPreferencesFromJSON: (json: string) => boolean;
  isLoading: boolean;
}

const PreferencesContext = createContext<PreferencesContextType | undefined>(undefined);

interface PreferencesProviderProps {
  children: ReactNode;
}

export const PreferencesProvider: React.FC<PreferencesProviderProps> = ({ children }) => {
  const [preferences, setPreferences] = useState<UserPreferences>(defaultPreferences);
  const [isLoading, setIsLoading] = useState(true);

  // Load preferences on mount
  useEffect(() => {
    // Use setTimeout to ensure this doesn't block rendering
    const timer = setTimeout(() => {
      try {
        const loadedPreferences = loadPreferences();
        setPreferences(loadedPreferences);
      } catch (error) {
        console.error('Error loading preferences:', error);
        // Fall back to defaults if there's an error
        setPreferences(defaultPreferences);
      } finally {
        setIsLoading(false);
      }
    }, 0);

    return () => clearTimeout(timer);
  }, []);

  // Update a specific preference section
  const updatePreference = <K extends keyof Omit<UserPreferences, 'version' | 'lastUpdated'>>(
    section: K,
    values: Partial<UserPreferences[K]>
  ) => {
    const updatedPreferences = updatePreferences(preferences, section, values);
    setPreferences(updatedPreferences);
  };

  // Reset all preferences to defaults
  const resetAllPreferences = () => {
    const defaultPrefs = resetPreferences();
    setPreferences(defaultPrefs);
  };

  // Export preferences as JSON
  const exportPreferencesToJSON = () => {
    return exportPreferences(preferences);
  };

  // Import preferences from JSON
  const importPreferencesFromJSON = (json: string) => {
    try {
      const importedPrefs = importPreferences(json);
      
      if (importedPrefs) {
        setPreferences(importedPrefs);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Error importing preferences:', error);
      return false;
    }
  };

  const contextValue: PreferencesContextType = {
    preferences,
    updatePreference,
    resetAllPreferences,
    exportPreferencesToJSON,
    importPreferencesFromJSON,
    isLoading,
  };

  return (
    <PreferencesContext.Provider value={contextValue}>
      {children}
    </PreferencesContext.Provider>
  );
};

// Custom hook to use the preferences context
export const usePreferences = (): PreferencesContextType => {
  const context = useContext(PreferencesContext);
  
  if (context === undefined) {
    throw new Error('usePreferences must be used within a PreferencesProvider');
  }
  
  return context;
}; 