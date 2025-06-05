/**
 * Notification Service
 * Centralized notification management with multiple notification types
 */

import { enqueueSnackbar, closeSnackbar, SnackbarKey } from 'notistack';

export type NotificationType = 'success' | 'error' | 'warning' | 'info';

export interface NotificationOptions {
  persist?: boolean;
  autoHideDuration?: number;
  action?: React.ReactNode;
  preventDuplicate?: boolean;
  anchorOrigin?: {
    vertical: 'top' | 'bottom';
    horizontal: 'left' | 'center' | 'right';
  };
  dense?: boolean;
  hideIconVariant?: boolean;
}

export interface NotificationData {
  id: string;
  type: NotificationType;
  message: string;
  timestamp: number;
  read: boolean;
  persistent?: boolean;
  metadata?: Record<string, any>;
}

class NotificationService {
  private notifications: NotificationData[] = [];
  private activeSnackbars: Map<string, SnackbarKey> = new Map();
  private maxNotifications = 100;

  /**
   * Show a success notification
   */
  success(message: string, options?: NotificationOptions): string {
    return this.show(message, 'success', options);
  }

  /**
   * Show an error notification
   */
  error(message: string, options?: NotificationOptions): string {
    return this.show(message, 'error', {
      persist: true,
      ...options,
    });
  }

  /**
   * Show a warning notification
   */
  warning(message: string, options?: NotificationOptions): string {
    return this.show(message, 'warning', options);
  }

  /**
   * Show an info notification
   */
  info(message: string, options?: NotificationOptions): string {
    return this.show(message, 'info', options);
  }

  /**
   * Show a notification with custom type
   */
  show(
    message: string, 
    type: NotificationType = 'info', 
    options: NotificationOptions = {}
  ): string {
    const id = this.generateId();
    
    // Check for duplicates if preventDuplicate is enabled
    if (options.preventDuplicate) {
      const duplicate = this.notifications.find(
        n => n.message === message && n.type === type && !n.read
      );
      if (duplicate) {
        return duplicate.id;
      }
    }

    // Create notification data
    const notification: NotificationData = {
      id,
      type,
      message,
      timestamp: Date.now(),
      read: false,
      persistent: options.persist,
      metadata: {},
    };

    // Add to notifications list
    this.addNotification(notification);

    // Show snackbar
    const snackbarKey = enqueueSnackbar(message, {
      variant: type,
      persist: options.persist,
      autoHideDuration: options.autoHideDuration,
      action: options.action,
      preventDuplicate: options.preventDuplicate,
      anchorOrigin: options.anchorOrigin || { vertical: 'top', horizontal: 'right' },
      dense: options.dense,
      hideIconVariant: options.hideIconVariant,
      onClose: () => {
        this.markAsRead(id);
        this.activeSnackbars.delete(id);
      },
    });

    this.activeSnackbars.set(id, snackbarKey);

    return id;
  }

  /**
   * Close a specific notification
   */
  close(id: string): void {
    const snackbarKey = this.activeSnackbars.get(id);
    if (snackbarKey) {
      closeSnackbar(snackbarKey);
      this.activeSnackbars.delete(id);
    }
    this.markAsRead(id);
  }

  /**
   * Close all notifications
   */
  closeAll(): void {
    this.activeSnackbars.forEach((snackbarKey) => {
      closeSnackbar(snackbarKey);
    });
    this.activeSnackbars.clear();
    this.markAllAsRead();
  }

  /**
   * Get all notifications
   */
  getAll(): NotificationData[] {
    return [...this.notifications];
  }

  /**
   * Get unread notifications
   */
  getUnread(): NotificationData[] {
    return this.notifications.filter(n => !n.read);
  }

  /**
   * Get notifications by type
   */
  getByType(type: NotificationType): NotificationData[] {
    return this.notifications.filter(n => n.type === type);
  }

  /**
   * Mark notification as read
   */
  markAsRead(id: string): void {
    const notification = this.notifications.find(n => n.id === id);
    if (notification) {
      notification.read = true;
    }
  }

  /**
   * Mark all notifications as read
   */
  markAllAsRead(): void {
    this.notifications.forEach(n => n.read = true);
  }

  /**
   * Clear all notifications
   */
  clear(): void {
    this.notifications = [];
    this.activeSnackbars.clear();
  }

  /**
   * Clear old notifications (older than specified time)
   */
  clearOld(olderThanMs: number = 24 * 60 * 60 * 1000): void {
    const cutoff = Date.now() - olderThanMs;
    this.notifications = this.notifications.filter(n => 
      n.timestamp > cutoff || n.persistent
    );
  }

  /**
   * Get notification statistics
   */
  getStats(): {
    total: number;
    unread: number;
    byType: Record<NotificationType, number>;
  } {
    const stats = {
      total: this.notifications.length,
      unread: this.getUnread().length,
      byType: {
        success: 0,
        error: 0,
        warning: 0,
        info: 0,
      } as Record<NotificationType, number>,
    };

    this.notifications.forEach(n => {
      stats.byType[n.type]++;
    });

    return stats;
  }

  /**
   * Show API error notification with enhanced details
   */
  apiError(error: any, context?: string): string {
    let message = 'An unexpected error occurred';
    
    if (error?.response?.data?.message) {
      message = error.response.data.message;
    } else if (error?.message) {
      message = error.message;
    }

    if (context) {
      message = `${context}: ${message}`;
    }

    return this.error(message, {
      persist: true,
      action: (
        <button
          onClick={() => this.showErrorDetails(error)}
          style={{
            background: 'none',
            border: 'none',
            color: 'inherit',
            textDecoration: 'underline',
            cursor: 'pointer',
          }}
        >
          Details
        </button>
      ),
    });
  }

  /**
   * Show network error notification
   */
  networkError(): string {
    return this.error('Network connection failed. Please check your internet connection.', {
      persist: true,
    });
  }

  /**
   * Show loading notification
   */
  loading(message: string = 'Loading...'): string {
    return this.info(message, {
      persist: true,
      hideIconVariant: true,
    });
  }

  /**
   * Show progress notification
   */
  progress(message: string, progress: number): string {
    const progressMessage = `${message} (${Math.round(progress)}%)`;
    return this.info(progressMessage, {
      persist: true,
      preventDuplicate: true,
    });
  }

  /**
   * Show trading-specific notifications
   */
  trading = {
    orderPlaced: (symbol: string, side: string, amount: number) => {
      return this.success(`${side.toUpperCase()} order placed: ${amount} ${symbol}`);
    },

    orderFilled: (symbol: string, side: string, amount: number, price: number) => {
      return this.success(`Order filled: ${side.toUpperCase()} ${amount} ${symbol} at $${price}`);
    },

    orderCancelled: (symbol: string) => {
      return this.warning(`Order cancelled for ${symbol}`);
    },

    orderFailed: (symbol: string, reason: string) => {
      return this.error(`Order failed for ${symbol}: ${reason}`);
    },

    botStarted: (botName: string) => {
      return this.success(`Trading bot "${botName}" started successfully`);
    },

    botStopped: (botName: string) => {
      return this.info(`Trading bot "${botName}" stopped`);
    },

    botError: (botName: string, error: string) => {
      return this.error(`Trading bot "${botName}" error: ${error}`);
    },

    positionOpened: (symbol: string, side: string, size: number) => {
      return this.info(`Position opened: ${side.toUpperCase()} ${size} ${symbol}`);
    },

    positionClosed: (symbol: string, pnl: number) => {
      const pnlText = pnl >= 0 ? `+$${pnl.toFixed(2)}` : `-$${Math.abs(pnl).toFixed(2)}`;
      return this.success(`Position closed for ${symbol}: ${pnlText}`);
    },
  };

  private generateId(): string {
    return `notification_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private addNotification(notification: NotificationData): void {
    this.notifications.unshift(notification);
    
    // Keep only the most recent notifications
    if (this.notifications.length > this.maxNotifications) {
      this.notifications = this.notifications.slice(0, this.maxNotifications);
    }
  }

  private showErrorDetails(error: any): void {
    const details = {
      message: error?.message || 'Unknown error',
      status: error?.response?.status,
      statusText: error?.response?.statusText,
      data: error?.response?.data,
      stack: error?.stack,
    };

    console.error('Error Details:', details);
    
    // In a real application, you might show a modal with error details
    // or send the user to an error details page
  }
}

// Singleton instance
export const notificationService = new NotificationService();

// Convenience functions
export const notify = {
  success: (message: string, options?: NotificationOptions) => 
    notificationService.success(message, options),
  error: (message: string, options?: NotificationOptions) => 
    notificationService.error(message, options),
  warning: (message: string, options?: NotificationOptions) => 
    notificationService.warning(message, options),
  info: (message: string, options?: NotificationOptions) => 
    notificationService.info(message, options),
  apiError: (error: any, context?: string) => 
    notificationService.apiError(error, context),
  networkError: () => 
    notificationService.networkError(),
  loading: (message?: string) => 
    notificationService.loading(message),
  trading: notificationService.trading,
};

export default notificationService;
