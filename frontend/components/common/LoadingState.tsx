/**
 * Loading State Component
 * Comprehensive loading states with progress indicators and user feedback
 */

'use client';

import React from 'react';
import {
  Box,
  CircularProgress,
  LinearProgress,
  Typography,
  Skeleton,
  Card,
  CardContent,
  Stack,
  Fade,
  Backdrop,
  Paper,
} from '@mui/material';
import { TrendingUp, Assessment, AccountBalance } from '@mui/icons-material';

export type LoadingType = 
  | 'spinner' 
  | 'linear' 
  | 'skeleton' 
  | 'overlay' 
  | 'inline' 
  | 'card'
  | 'table'
  | 'chart';

export interface LoadingStateProps {
  type?: LoadingType;
  message?: string;
  progress?: number;
  size?: 'small' | 'medium' | 'large';
  color?: 'primary' | 'secondary' | 'inherit';
  fullScreen?: boolean;
  overlay?: boolean;
  rows?: number;
  height?: number | string;
  width?: number | string;
  variant?: 'text' | 'rectangular' | 'circular';
  animation?: 'pulse' | 'wave' | false;
  children?: React.ReactNode;
}

const LoadingState: React.FC<LoadingStateProps> = ({
  type = 'spinner',
  message = 'Loading...',
  progress,
  size = 'medium',
  color = 'primary',
  fullScreen = false,
  overlay = false,
  rows = 3,
  height = 40,
  width = '100%',
  variant = 'text',
  animation = 'pulse',
  children,
}) => {
  const getSizeValue = () => {
    switch (size) {
      case 'small': return 24;
      case 'large': return 64;
      default: return 40;
    }
  };

  const renderSpinner = () => (
    <Box
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      gap={2}
      p={3}
    >
      <CircularProgress
        size={getSizeValue()}
        color={color}
        variant={progress !== undefined ? 'determinate' : 'indeterminate'}
        value={progress}
      />
      {message && (
        <Typography variant="body2" color="text.secondary" textAlign="center">
          {message}
          {progress !== undefined && ` (${Math.round(progress)}%)`}
        </Typography>
      )}
    </Box>
  );

  const renderLinear = () => (
    <Box width="100%" p={2}>
      {message && (
        <Typography variant="body2" color="text.secondary" mb={1}>
          {message}
        </Typography>
      )}
      <LinearProgress
        color={color}
        variant={progress !== undefined ? 'determinate' : 'indeterminate'}
        value={progress}
      />
      {progress !== undefined && (
        <Typography variant="caption" color="text.secondary" mt={1}>
          {Math.round(progress)}% complete
        </Typography>
      )}
    </Box>
  );

  const renderSkeleton = () => (
    <Stack spacing={1} p={2}>
      {Array.from({ length: rows }).map((_, index) => (
        <Skeleton
          key={index}
          variant={variant}
          height={height}
          width={width}
          animation={animation}
        />
      ))}
    </Stack>
  );

  const renderCard = () => (
    <Card>
      <CardContent>
        <Stack spacing={2}>
          <Skeleton variant="text" width="60%" height={24} animation={animation} />
          <Skeleton variant="rectangular" height={120} animation={animation} />
          <Stack direction="row" spacing={1}>
            <Skeleton variant="circular" width={40} height={40} animation={animation} />
            <Stack spacing={1} flex={1}>
              <Skeleton variant="text" width="80%" animation={animation} />
              <Skeleton variant="text" width="60%" animation={animation} />
            </Stack>
          </Stack>
        </Stack>
      </CardContent>
    </Card>
  );

  const renderTable = () => (
    <Box>
      {/* Table header */}
      <Stack direction="row" spacing={2} p={2} borderBottom={1} borderColor="divider">
        {Array.from({ length: 4 }).map((_, index) => (
          <Skeleton
            key={index}
            variant="text"
            width={`${20 + Math.random() * 30}%`}
            height={20}
            animation={animation}
          />
        ))}
      </Stack>
      
      {/* Table rows */}
      {Array.from({ length: rows }).map((_, rowIndex) => (
        <Stack
          key={rowIndex}
          direction="row"
          spacing={2}
          p={2}
          borderBottom={1}
          borderColor="divider"
        >
          {Array.from({ length: 4 }).map((_, colIndex) => (
            <Skeleton
              key={colIndex}
              variant="text"
              width={`${15 + Math.random() * 25}%`}
              height={16}
              animation={animation}
            />
          ))}
        </Stack>
      ))}
    </Box>
  );

  const renderChart = () => (
    <Box p={2}>
      <Stack spacing={2}>
        {/* Chart title */}
        <Skeleton variant="text" width="40%" height={24} animation={animation} />
        
        {/* Chart area */}
        <Box position="relative" height={200}>
          <Skeleton
            variant="rectangular"
            width="100%"
            height="100%"
            animation={animation}
          />
          
          {/* Simulate chart elements */}
          <Box
            position="absolute"
            top="50%"
            left="50%"
            sx={{ transform: 'translate(-50%, -50%)' }}
          >
            <TrendingUp sx={{ fontSize: 48, color: 'action.disabled' }} />
          </Box>
        </Box>
        
        {/* Chart legend */}
        <Stack direction="row" spacing={2} justifyContent="center">
          {Array.from({ length: 3 }).map((_, index) => (
            <Stack key={index} direction="row" spacing={1} alignItems="center">
              <Skeleton variant="circular" width={12} height={12} animation={animation} />
              <Skeleton variant="text" width={60} height={16} animation={animation} />
            </Stack>
          ))}
        </Stack>
      </Stack>
    </Box>
  );

  const renderInline = () => (
    <Box display="inline-flex" alignItems="center" gap={1}>
      <CircularProgress size={16} color={color} />
      {message && (
        <Typography variant="body2" color="text.secondary">
          {message}
        </Typography>
      )}
    </Box>
  );

  const renderContent = () => {
    switch (type) {
      case 'linear':
        return renderLinear();
      case 'skeleton':
        return renderSkeleton();
      case 'card':
        return renderCard();
      case 'table':
        return renderTable();
      case 'chart':
        return renderChart();
      case 'inline':
        return renderInline();
      case 'overlay':
      case 'spinner':
      default:
        return renderSpinner();
    }
  };

  const content = (
    <Fade in timeout={300}>
      <Box>
        {renderContent()}
        {children}
      </Box>
    </Fade>
  );

  if (fullScreen) {
    return (
      <Backdrop
        open
        sx={{
          color: '#fff',
          zIndex: (theme) => theme.zIndex.drawer + 1,
          backgroundColor: 'rgba(0, 0, 0, 0.7)',
        }}
      >
        <Paper
          elevation={3}
          sx={{
            p: 4,
            borderRadius: 2,
            backgroundColor: 'background.paper',
            color: 'text.primary',
          }}
        >
          {renderSpinner()}
        </Paper>
      </Backdrop>
    );
  }

  if (overlay) {
    return (
      <Box
        position="absolute"
        top={0}
        left={0}
        right={0}
        bottom={0}
        display="flex"
        alignItems="center"
        justifyContent="center"
        bgcolor="rgba(255, 255, 255, 0.8)"
        zIndex={1}
      >
        {content}
      </Box>
    );
  }

  return content;
};

// Specialized loading components
export const TableLoading: React.FC<{ rows?: number }> = ({ rows = 5 }) => (
  <LoadingState type="table" rows={rows} />
);

export const ChartLoading: React.FC = () => (
  <LoadingState type="chart" />
);

export const CardLoading: React.FC = () => (
  <LoadingState type="card" />
);

export const InlineLoading: React.FC<{ message?: string }> = ({ message }) => (
  <LoadingState type="inline" message={message} />
);

export const FullScreenLoading: React.FC<{ message?: string }> = ({ message }) => (
  <LoadingState type="spinner" message={message} fullScreen />
);

// Trading-specific loading states
export const TradingLoadingStates = {
  BotStatus: () => (
    <Card>
      <CardContent>
        <Stack spacing={2}>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Skeleton variant="circular" width={40} height={40} />
            <Stack spacing={1} flex={1}>
              <Skeleton variant="text" width="60%" height={20} />
              <Skeleton variant="text" width="40%" height={16} />
            </Stack>
            <Skeleton variant="rectangular" width={80} height={32} />
          </Stack>
          <Skeleton variant="rectangular" height={100} />
        </Stack>
      </CardContent>
    </Card>
  ),

  Portfolio: () => (
    <Stack spacing={2}>
      {/* Portfolio summary */}
      <Card>
        <CardContent>
          <Stack direction="row" spacing={4}>
            {Array.from({ length: 3 }).map((_, index) => (
              <Stack key={index} spacing={1} alignItems="center" flex={1}>
                <Skeleton variant="text" width="80%" height={16} />
                <Skeleton variant="text" width="60%" height={24} />
              </Stack>
            ))}
          </Stack>
        </CardContent>
      </Card>
      
      {/* Portfolio positions */}
      <TableLoading rows={4} />
    </Stack>
  ),

  MarketData: () => (
    <Stack spacing={2}>
      {/* Price ticker */}
      <Stack direction="row" spacing={2} alignItems="center">
        <Skeleton variant="text" width={100} height={24} />
        <Skeleton variant="text" width={120} height={32} />
        <Skeleton variant="rectangular" width={80} height={24} />
      </Stack>
      
      {/* Chart */}
      <ChartLoading />
    </Stack>
  ),
};

export default LoadingState;
