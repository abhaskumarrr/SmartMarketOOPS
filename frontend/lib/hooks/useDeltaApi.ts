import { useState, useEffect, useCallback } from 'react';
import { useSnackbar } from 'notistack';
import deltaExchangeApi from '../api/deltaExchangeApi';

interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  timestamp: number | null;
}

interface Options {
  refreshInterval?: number;
  errorMessage?: string;
  onSuccess?: (data: any) => void;
  onError?: (error: any) => void;
}

/**
 * Hook to fetch data from the Delta Exchange API with caching and auto-refresh
 * @param method - Method name from deltaExchangeApi
 * @param args - Arguments to pass to the method
 * @param options - Additional options
 */
function useDeltaApi<T>(
  method: keyof typeof deltaExchangeApi,
  args: any[] = [],
  options: Options = {}
): [T | null, boolean, string | null, () => Promise<void>] {
  const {
    refreshInterval,
    errorMessage = 'Error fetching data',
    onSuccess,
    onError
  } = options;
  
  const { enqueueSnackbar } = useSnackbar();
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: true,
    error: null,
    timestamp: null
  });
  
  // Memoize the fetchData function
  const fetchData = useCallback(async () => {
    try {
      setState(prev => ({ ...prev, loading: true, error: null }));
      
      // Call the method from deltaExchangeApi with the provided args
      const response = await (deltaExchangeApi[method] as (...args: any[]) => Promise<any>)(...args);
      
      if (response?.success) {
        setState({
          data: response.data,
          loading: false,
          error: null,
          timestamp: Date.now()
        });
        
        if (onSuccess) {
          onSuccess(response.data);
        }
      } else {
        throw new Error(response?.message || 'Unknown error');
      }
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : String(error);
      console.error(`Error in ${method}:`, error);
      
      setState(prev => ({
        ...prev,
        loading: false,
        error: errorMsg
      }));
      
      if (onError) {
        onError(error);
      } else {
        enqueueSnackbar(errorMessage, { variant: 'error' });
      }
    }
  }, [method, args, errorMessage, onSuccess, onError, enqueueSnackbar]);
  
  // Initial fetch and refresh interval
  useEffect(() => {
    fetchData();
    
    // Set up refresh interval if specified
    if (refreshInterval && refreshInterval > 0) {
      const intervalId = setInterval(fetchData, refreshInterval);
      return () => clearInterval(intervalId);
    }
    
    return undefined;
  }, [fetchData, refreshInterval]);
  
  return [state.data, state.loading, state.error, fetchData];
}

export default useDeltaApi; 