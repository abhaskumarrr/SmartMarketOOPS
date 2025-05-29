import React, { useState } from 'react';

export default function Home() {
  const [trades, setTrades] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchTrades = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch('http://localhost:3333/api/trades/public');
      const data = await response.json();
      
      if (data.success) {
        setTrades(data.data || []);
      } else {
        setError(data.message || 'Failed to fetch trade data');
      }
    } catch (err) {
      console.error('Error fetching trades:', err);
      setError('Failed to connect to the server');
    } finally {
      setIsLoading(false);
    }
  };

  // Format date from timestamp
  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleString();
  };

  // Simple CSS styles
  const styles = {
    container: {
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '20px',
      fontFamily: 'Arial, sans-serif',
    },
    header: {
      fontSize: '28px',
      marginBottom: '20px',
    },
    card: {
      backgroundColor: '#1e1e1e',
      borderRadius: '8px',
      padding: '20px',
      marginBottom: '20px',
      color: 'white',
    },
    button: {
      backgroundColor: '#3f51b5',
      color: 'white',
      border: 'none',
      padding: '10px 15px',
      borderRadius: '4px',
      cursor: 'pointer',
      fontWeight: 'bold',
    },
    table: {
      width: '100%',
      borderCollapse: 'collapse',
    },
    th: {
      textAlign: 'left',
      padding: '12px 15px',
      borderBottom: '1px solid #ddd',
    },
    td: {
      padding: '12px 15px',
      borderBottom: '1px solid #ddd',
    },
    error: {
      backgroundColor: '#f44336',
      color: 'white',
      padding: '15px',
      borderRadius: '4px',
      marginBottom: '20px',
    },
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.header}>SmartMarketOOPS Trading Dashboard</h1>
      
      <div style={styles.card}>
        <h2>Welcome to your trading platform</h2>
        <p>This dashboard connects to your backend API to display trading data.</p>
        <button 
          style={styles.button} 
          onClick={fetchTrades}
          disabled={isLoading}
        >
          {isLoading ? "Loading..." : "Load Trade Data"}
        </button>
      </div>

      {error && (
        <div style={styles.error}>
          <p>{error}</p>
        </div>
      )}

      {trades.length > 0 && (
        <div style={styles.card}>
          <h2>Trade History:</h2>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Date & Time</th>
                <th style={styles.th}>Symbol</th>
                <th style={styles.th}>Side</th>
                <th style={styles.th}>Type</th>
                <th style={styles.th}>Price</th>
                <th style={styles.th}>Quantity</th>
                <th style={styles.th}>Fee</th>
                <th style={styles.th}>Total Value</th>
                <th style={styles.th}>Status</th>
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => (
                <tr key={trade.id}>
                  <td style={styles.td}>{formatDate(trade.timestamp)}</td>
                  <td style={styles.td}>{trade.symbol}</td>
                  <td style={styles.td}>{trade.side}</td>
                  <td style={styles.td}>{trade.type}</td>
                  <td style={styles.td}>${trade.price.toFixed(2)}</td>
                  <td style={styles.td}>{trade.quantity}</td>
                  <td style={styles.td}>${trade.fee.toFixed(2)}</td>
                  <td style={styles.td}>${trade.totalValue.toFixed(2)}</td>
                  <td style={styles.td}>{trade.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
