import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import WebSocketStatus from '../../../components/layout/WebSocketStatus';
import WebSocketService, { ConnectionStatus } from '../../../lib/websocketService';

// Get the mock from jest.setup.js
const mockWebSocketService = WebSocketService.getInstance() as any;

describe('WebSocketStatus component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Reset the mock service state
    mockWebSocketService.mockChangeStatus(ConnectionStatus.DISCONNECTED);
  });

  test('renders disconnected status initially', () => {
    render(<WebSocketStatus />);
    
    // In disconnected state, the tooltip should show "Disconnected"
    const statusElement = screen.getByTitle('Disconnected');
    expect(statusElement).toBeInTheDocument();
  });

  test('updates when connection status changes', () => {
    const { rerender } = render(<WebSocketStatus />);
    
    // Change status to CONNECTING
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTING);
    rerender(<WebSocketStatus />);
    
    // In connecting state, the tooltip should show "Connecting..."
    const connectingElement = screen.getByTitle('Connecting...');
    expect(connectingElement).toBeInTheDocument();
    
    // Change status to CONNECTED
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    rerender(<WebSocketStatus />);
    
    // In connected state, the tooltip should show "WebSocket Connected"
    const connectedElement = screen.getByTitle('WebSocket Connected');
    expect(connectedElement).toBeInTheDocument();
    
    // Change status to ERROR
    mockWebSocketService.mockChangeStatus(ConnectionStatus.ERROR);
    rerender(<WebSocketStatus />);
    
    // In error state, the tooltip should show "Connection Error"
    const errorElement = screen.getByTitle('Connection Error');
    expect(errorElement).toBeInTheDocument();
  });

  test('clicking on disconnected status tries to reconnect', () => {
    const connectSpy = jest.spyOn(mockWebSocketService, 'connect');
    
    render(<WebSocketStatus />);
    
    // Find and click on the disconnected status icon
    const statusElement = screen.getByTitle('Disconnected');
    fireEvent.click(statusElement);
    
    // Should try to reconnect
    expect(connectSpy).toHaveBeenCalled();
  });

  test('clicking on error status tries to reconnect', () => {
    const connectSpy = jest.spyOn(mockWebSocketService, 'connect');
    
    render(<WebSocketStatus />);
    
    // Set status to ERROR
    mockWebSocketService.mockChangeStatus(ConnectionStatus.ERROR);
    
    // Find and click on the error status icon
    const statusElement = screen.getByTitle('Connection Error');
    fireEvent.click(statusElement);
    
    // Should try to reconnect
    expect(connectSpy).toHaveBeenCalled();
  });

  test('clicking on connected status does nothing', () => {
    const connectSpy = jest.spyOn(mockWebSocketService, 'connect');
    
    render(<WebSocketStatus />);
    
    // Set status to CONNECTED
    mockWebSocketService.mockChangeStatus(ConnectionStatus.CONNECTED);
    
    // Find and click on the connected status icon
    const statusElement = screen.getByTitle('WebSocket Connected');
    fireEvent.click(statusElement);
    
    // Should not try to reconnect
    expect(connectSpy).not.toHaveBeenCalled();
  });

  test('does not render tooltip when showTooltip is false', () => {
    render(<WebSocketStatus showTooltip={false} />);
    
    // Without tooltip, the title attribute should not be present
    const statusElement = screen.queryByTitle('Disconnected');
    expect(statusElement).not.toBeInTheDocument();
  });

  test('uses different icon size based on size prop', () => {
    const { rerender } = render(<WebSocketStatus size="small" />);
    
    // The small size should be used by default
    let icon = screen.getByRole('img', { hidden: true });
    expect(icon).toHaveAttribute('font-size', 'small');
    
    // Change to medium size
    rerender(<WebSocketStatus size="medium" />);
    
    icon = screen.getByRole('img', { hidden: true });
    expect(icon).toHaveAttribute('font-size', 'medium');
    
    // Change to large size
    rerender(<WebSocketStatus size="large" />);
    
    icon = screen.getByRole('img', { hidden: true });
    expect(icon).toHaveAttribute('font-size', 'large');
  });
}); 