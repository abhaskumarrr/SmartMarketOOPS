import React, { useState, useEffect } from 'react';
import {
  Box,
  Paper,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  TableSortLabel,
  Chip,
  TextField,
  InputAdornment,
  IconButton,
  Button,
  Menu,
  MenuItem,
  Tooltip,
  CircularProgress,
} from '@mui/material';
import {
  Search as SearchIcon,
  FilterList as FilterIcon,
  GetApp as DownloadIcon,
  VisibilityOutlined as ViewIcon,
} from '@mui/icons-material';

// Trade type definition aligned with API
export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit';
  price: number;
  quantity: number;
  timestamp: number;
  status: 'pending' | 'completed' | 'cancelled';
  pnl?: number;
  fee: number;
  userId: string;
  strategyId?: string;
  botId?: string;
  signalId?: string;
}

// Props definition
interface TradeHistoryProps {
  darkMode: boolean;
  onViewTrade?: (trade: Trade) => void;
}

// Table header definition
interface HeadCell {
  id: keyof Trade;
  label: string;
  numeric: boolean;
  sortable: boolean;
}

const headCells: HeadCell[] = [
  { id: 'timestamp', label: 'Date & Time', numeric: false, sortable: true },
  { id: 'symbol', label: 'Symbol', numeric: false, sortable: true },
  { id: 'side', label: 'Side', numeric: false, sortable: true },
  { id: 'type', label: 'Type', numeric: false, sortable: true },
  { id: 'price', label: 'Price', numeric: true, sortable: true },
  { id: 'quantity', label: 'Quantity', numeric: true, sortable: true },
  { id: 'fee', label: 'Fee', numeric: true, sortable: true },
  { id: 'pnl', label: 'P/L', numeric: true, sortable: true },
  { id: 'status', label: 'Status', numeric: false, sortable: true },
];

export const TradeHistory: React.FC<TradeHistoryProps> = ({ darkMode, onViewTrade }) => {
  // State
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [searchQuery, setSearchQuery] = useState('');
  const [orderBy, setOrderBy] = useState<keyof Trade>('timestamp');
  const [order, setOrder] = useState<'asc' | 'desc'>('desc');
  const [filterMenuAnchorEl, setFilterMenuAnchorEl] = useState<null | HTMLElement>(null);
  const [currentFilter, setCurrentFilter] = useState<'all' | 'buy' | 'sell' | 'profit' | 'loss'>('all');
  const [trades, setTrades] = useState<Trade[]>([]);
  const [filteredData, setFilteredData] = useState<Trade[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch trade data from API
  useEffect(() => {
    const fetchTrades = async () => {
      try {
        setIsLoading(true);
        const response = await fetch('http://localhost:3333/api/trades/public');
        const data = await response.json();
        
        if (data.success) {
          setTrades(data.data || []);
          setFilteredData(data.data || []);
        } else {
          setError(data.message || 'Failed to fetch trade data');
        }
      } catch (error) {
        console.error('Error fetching trades:', error);
        setError('Failed to connect to the server');
      } finally {
        setIsLoading(false);
      }
    };
    
    fetchTrades();
  }, []);

  // Update filtered data when search, filter, or sort changes
  useEffect(() => {
    let filtered = [...trades];
    
    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((trade) => 
        trade.symbol.toLowerCase().includes(query) ||
        trade.side.toLowerCase().includes(query) ||
        trade.type.toLowerCase().includes(query) ||
        trade.status.toLowerCase().includes(query) ||
        trade.price.toString().includes(query) ||
        trade.quantity.toString().includes(query)
      );
    }
    
    // Apply type filter
    if (currentFilter === 'buy') {
      filtered = filtered.filter(trade => trade.side === 'buy');
    } else if (currentFilter === 'sell') {
      filtered = filtered.filter(trade => trade.side === 'sell');
    } else if (currentFilter === 'profit') {
      filtered = filtered.filter(trade => (trade.pnl || 0) > 0);
    } else if (currentFilter === 'loss') {
      filtered = filtered.filter(trade => (trade.pnl || 0) < 0);
    }
    
    // Apply sorting
    filtered.sort((a, b) => {
      const valueA = a[orderBy];
      const valueB = b[orderBy];
      
      if (typeof valueA === 'number' && typeof valueB === 'number') {
        return order === 'asc' ? valueA - valueB : valueB - valueA;
      } else if (valueA !== undefined && valueB !== undefined) {
        return order === 'asc' 
          ? String(valueA).localeCompare(String(valueB))
          : String(valueB).localeCompare(String(valueA));
      }
      
      return 0;
    });
    
    setFilteredData(filtered);
  }, [searchQuery, currentFilter, orderBy, order, trades]);

  const handleRequestSort = (property: keyof Trade) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  const handleChangePage = (_: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleFilterMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setFilterMenuAnchorEl(event.currentTarget);
  };

  const handleFilterMenuClose = () => {
    setFilterMenuAnchorEl(null);
  };

  const handleFilterSelect = (filter: 'all' | 'buy' | 'sell' | 'profit' | 'loss') => {
    setCurrentFilter(filter);
    handleFilterMenuClose();
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  const exportToCSV = () => {
    // Create CSV content
    const headers = headCells.map(cell => cell.label).join(',');
    const rows = filteredData.map(trade => {
      return [
        formatDate(trade.timestamp),
        trade.symbol,
        trade.side,
        trade.type,
        trade.price,
        trade.quantity,
        trade.fee,
        trade.pnl || 0,
        trade.status
      ].join(',');
    }).join('\n');
    
    const csvContent = `${headers}\n${rows}`;
    
    // Create download link
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', `trade_history_${new Date().toISOString().split('T')[0]}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getStyles = () => {
    return {
      paper: {
        backgroundColor: darkMode ? '#1E1E2D' : '#FFFFFF',
        boxShadow: darkMode ? '0 4px 20px rgba(0,0,0,0.25)' : '0 4px 20px rgba(0,0,0,0.1)',
      },
      title: {
        color: darkMode ? '#E0E0E0' : '#121212',
      },
      text: {
        color: darkMode ? '#E0E0E0' : '#121212',
      },
      subtitle: {
        color: darkMode ? '#B0B0C0' : '#5F5F7A',
      },
      tableHeader: {
        backgroundColor: darkMode ? '#2A2A3C' : '#F5F5F5',
      },
      tableHeaderText: {
        color: darkMode ? '#E0E0E0' : '#121212',
        fontWeight: 600,
      },
      tableRow: {
        backgroundColor: darkMode ? '#2A2A3C' : '#FFFFFF',
        '&:nth-of-type(odd)': {
          backgroundColor: darkMode ? '#252536' : '#F9F9F9',
        },
        '&:hover': {
          backgroundColor: darkMode ? '#333345' : '#F0F0F0',
        },
      },
      chip: {
        buy: {
          backgroundColor: darkMode ? '#1B5E20' : '#E8F5E9',
          color: darkMode ? '#FFFFFF' : '#1B5E20',
        },
        sell: {
          backgroundColor: darkMode ? '#B71C1C' : '#FFEBEE',
          color: darkMode ? '#FFFFFF' : '#B71C1C',
        },
        market: {
          backgroundColor: darkMode ? '#0D47A1' : '#E3F2FD',
          color: darkMode ? '#FFFFFF' : '#0D47A1',
        },
        limit: {
          backgroundColor: darkMode ? '#4A148C' : '#F3E5F5',
          color: darkMode ? '#FFFFFF' : '#4A148C',
        },
        completed: {
          backgroundColor: darkMode ? '#1A237E' : '#E8EAF6',
          color: darkMode ? '#FFFFFF' : '#1A237E',
        },
        pending: {
          backgroundColor: darkMode ? '#FF6F00' : '#FFF8E1',
          color: darkMode ? '#FFFFFF' : '#FF6F00',
        },
        cancelled: {
          backgroundColor: darkMode ? '#424242' : '#EEEEEE',
          color: darkMode ? '#FFFFFF' : '#424242',
        },
      },
      profit: {
        color: darkMode ? '#4CAF50' : '#1E8E3E',
      },
      loss: {
        color: darkMode ? '#FF5252' : '#D32F2F',
      },
      actionButton: {
        color: darkMode ? '#90CAF9' : '#1976D2',
      },
    };
  };

  const styles = getStyles();

  return (
    <Paper elevation={3} style={styles.paper} sx={{ borderRadius: '8px', overflow: 'hidden' }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" sx={{ p: 3, pb: 2 }}>
        <Typography variant="h6" style={styles.title}>
          Trade History
        </Typography>
        
        <Box display="flex" alignItems="center">
          {/* Search Field */}
          <TextField
            size="small"
            placeholder="Search trades..."
            variant="outlined"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            sx={{ mr: 2, width: '240px' }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
            }}
          />
          
          {/* Filter Button */}
          <Button 
            variant="outlined" 
            size="small" 
            startIcon={<FilterIcon />}
            onClick={handleFilterMenuClick}
            sx={{ mr: 2 }}
          >
            {currentFilter === 'all' ? 'All Trades' : 
             currentFilter === 'buy' ? 'Buy Orders' :
             currentFilter === 'sell' ? 'Sell Orders' :
             currentFilter === 'profit' ? 'Profitable' : 'Loss-making'}
          </Button>
          
          {/* Filter Menu */}
          <Menu
            anchorEl={filterMenuAnchorEl}
            open={Boolean(filterMenuAnchorEl)}
            onClose={handleFilterMenuClose}
          >
            <MenuItem onClick={() => handleFilterSelect('all')}>All Trades</MenuItem>
            <MenuItem onClick={() => handleFilterSelect('buy')}>Buy Orders</MenuItem>
            <MenuItem onClick={() => handleFilterSelect('sell')}>Sell Orders</MenuItem>
            <MenuItem onClick={() => handleFilterSelect('profit')}>Profitable</MenuItem>
            <MenuItem onClick={() => handleFilterSelect('loss')}>Loss-making</MenuItem>
          </Menu>
          
          {/* Export Button */}
          <Button 
            variant="outlined" 
            size="small" 
            startIcon={<DownloadIcon />}
            onClick={exportToCSV}
          >
            Export
          </Button>
        </Box>
      </Box>
      
      {isLoading ? (
        <Box display="flex" justifyContent="center" alignItems="center" sx={{ p: 5 }}>
          <CircularProgress />
        </Box>
      ) : error ? (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography color="error">{error}</Typography>
          <Button 
            variant="contained" 
            sx={{ mt: 2 }} 
            onClick={() => window.location.reload()}
          >
            Retry
          </Button>
        </Box>
      ) : filteredData.length === 0 ? (
        <Box sx={{ p: 3, textAlign: 'center' }}>
          <Typography style={styles.text}>No trades found.</Typography>
        </Box>
      ) : (
        <>
          <TableContainer sx={{ maxHeight: 440 }}>
            <Table stickyHeader>
              <TableHead>
                <TableRow>
                  {headCells.map((headCell) => (
                    <TableCell
                      key={headCell.id}
                      align={headCell.numeric ? 'right' : 'left'}
                      style={{ ...styles.tableHeader, ...styles.tableHeaderText }}
                      sortDirection={orderBy === headCell.id ? order : false}
                    >
                      {headCell.sortable ? (
                        <TableSortLabel
                          active={orderBy === headCell.id}
                          direction={orderBy === headCell.id ? order : 'asc'}
                          onClick={() => handleRequestSort(headCell.id)}
                        >
                          {headCell.label}
                        </TableSortLabel>
                      ) : (
                        headCell.label
                      )}
                    </TableCell>
                  ))}
                  <TableCell style={styles.tableHeader}></TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredData
                  .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                  .map((trade) => (
                    <TableRow key={trade.id} hover style={styles.tableRow}>
                      <TableCell style={styles.text}>{formatDate(trade.timestamp)}</TableCell>
                      <TableCell style={styles.text}>{trade.symbol}</TableCell>
                      <TableCell>
                        <Chip 
                          label={trade.side.toUpperCase()} 
                          size="small"
                          style={
                            trade.side === 'buy' 
                              ? styles.chip.buy 
                              : styles.chip.sell
                          }
                        />
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={trade.type.toUpperCase()} 
                          size="small"
                          style={
                            trade.type === 'market' 
                              ? styles.chip.market 
                              : styles.chip.limit
                          }
                        />
                      </TableCell>
                      <TableCell align="right" style={styles.text}>
                        ${trade.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </TableCell>
                      <TableCell align="right" style={styles.text}>
                        {trade.quantity.toLocaleString(undefined, { minimumFractionDigits: 6, maximumFractionDigits: 6 })}
                      </TableCell>
                      <TableCell align="right" style={styles.text}>
                        ${trade.fee.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                      </TableCell>
                      <TableCell align="right" style={
                        trade.pnl === undefined
                          ? styles.text
                          : trade.pnl >= 0 
                            ? styles.profit 
                            : styles.loss
                      }>
                        {trade.pnl === undefined 
                          ? '—'
                          : `${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
                        }
                      </TableCell>
                      <TableCell>
                        <Chip 
                          label={trade.status.toUpperCase()} 
                          size="small"
                          style={styles.chip[trade.status as keyof typeof styles.chip] || styles.chip.completed}
                        />
                      </TableCell>
                      <TableCell>
                        {onViewTrade && (
                          <Tooltip title="View Details">
                            <IconButton 
                              size="small" 
                              onClick={() => onViewTrade(trade)}
                              style={styles.actionButton}
                            >
                              <ViewIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
          <TablePagination
            rowsPerPageOptions={[5, 10, 25]}
            component="div"
            count={filteredData.length}
            rowsPerPage={rowsPerPage}
            page={page}
            onPageChange={handleChangePage}
            onRowsPerPageChange={handleChangeRowsPerPage}
            style={styles.text}
          />
        </>
      )}
    </Paper>
  );
};

export default TradeHistory; 