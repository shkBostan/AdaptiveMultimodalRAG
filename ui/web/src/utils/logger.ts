/**
 * Production-grade logging utility for frontend applications.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */

export enum LogLevel {
  TRACE = 0,
  DEBUG = 1,
  INFO = 2,
  WARN = 3,
  ERROR = 4,
  NONE = 5
}

interface LogContext {
  correlationId?: string;
  userId?: string;
  requestId?: string;
  [key: string]: any;
}

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
  context?: LogContext;
  error?: Error;
  stack?: string;
}

class Logger {
  private level: LogLevel;
  private context: LogContext = {};
  private logBuffer: LogEntry[] = [];
  private maxBufferSize: number = 100;
  private enableRemoteLogging: boolean = false;
  private remoteLoggingEndpoint: string = '/api/logs';
  private sentryEnabled: boolean = false;

  constructor() {
    // Determine log level from environment
    const env = import.meta.env.MODE || 'development';
    this.level = env === 'production' ? LogLevel.INFO : LogLevel.DEBUG;
    
    // Enable remote logging in production
    this.enableRemoteLogging = env === 'production';
    
    // Initialize Sentry if available
    this.initializeSentry();
  }

  private initializeSentry(): void {
    // Check if Sentry is available
    if (typeof window !== 'undefined' && (window as any).Sentry) {
      this.sentryEnabled = true;
    }
  }

  /**
   * Set the log level.
   */
  setLevel(level: LogLevel): void {
    this.level = level;
  }

  /**
   * Set context that will be included in all logs.
   */
  setContext(context: LogContext): void {
    this.context = { ...this.context, ...context };
  }

  /**
   * Clear context.
   */
  clearContext(): void {
    this.context = {};
  }

  /**
   * Generate a correlation ID.
   */
  generateCorrelationId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Log a message.
   */
  private log(level: LogLevel, levelName: string, message: string, error?: Error, extra?: LogContext): void {
    if (level < this.level) {
      return;
    }

    const logEntry: LogEntry = {
      timestamp: new Date().toISOString(),
      level: levelName,
      message,
      context: { ...this.context, ...extra },
      error: error,
      stack: error?.stack
    };

    // Console logging (always enabled in development)
    const consoleMethod = this.getConsoleMethod(levelName);
    if (consoleMethod) {
      const logMessage = `[${levelName}] ${message}`;
      if (error) {
        consoleMethod(logMessage, error, logEntry.context);
      } else {
        consoleMethod(logMessage, logEntry.context);
      }
    }

    // Add to buffer
    this.logBuffer.push(logEntry);
    if (this.logBuffer.length > this.maxBufferSize) {
      this.logBuffer.shift();
    }

    // Send to remote logging service
    if (this.enableRemoteLogging && level >= LogLevel.INFO) {
      this.sendToRemote(logEntry);
    }

    // Send errors to Sentry
    if (this.sentryEnabled && level === LogLevel.ERROR && error) {
      this.sendToSentry(error, logEntry);
    }
  }

  private getConsoleMethod(level: string): ((...args: any[]) => void) | null {
    switch (level) {
      case 'TRACE':
      case 'DEBUG':
        return console.debug;
      case 'INFO':
        return console.info;
      case 'WARN':
        return console.warn;
      case 'ERROR':
        return console.error;
      default:
        return null;
    }
  }

  private async sendToRemote(logEntry: LogEntry): Promise<void> {
    try {
      await fetch(this.remoteLoggingEndpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(logEntry),
      });
    } catch (error) {
      // Silently fail - don't log logging errors
      console.error('Failed to send log to remote service', error);
    }
  }

  private sendToSentry(error: Error, logEntry: LogEntry): void {
    try {
      const Sentry = (window as any).Sentry;
      if (Sentry) {
        Sentry.withScope((scope: any) => {
          scope.setContext('logEntry', logEntry.context);
          scope.setLevel(LogLevel.ERROR);
          Sentry.captureException(error);
        });
      }
    } catch (err) {
      // Silently fail
    }
  }

  /**
   * Get all buffered logs.
   */
  getLogs(): LogEntry[] {
    return [...this.logBuffer];
  }

  /**
   * Clear log buffer.
   */
  clearLogs(): void {
    this.logBuffer = [];
  }

  /**
   * Export logs as JSON.
   */
  exportLogs(): string {
    return JSON.stringify(this.logBuffer, null, 2);
  }

  // Public logging methods
  trace(message: string, extra?: LogContext): void {
    this.log(LogLevel.TRACE, 'TRACE', message, undefined, extra);
  }

  debug(message: string, extra?: LogContext): void {
    this.log(LogLevel.DEBUG, 'DEBUG', message, undefined, extra);
  }

  info(message: string, extra?: LogContext): void {
    this.log(LogLevel.INFO, 'INFO', message, undefined, extra);
  }

  warn(message: string, extra?: LogContext): void {
    this.log(LogLevel.WARN, 'WARN', message, undefined, extra);
  }

  error(message: string, error?: Error, extra?: LogContext): void {
    this.log(LogLevel.ERROR, 'ERROR', message, error, extra);
  }
}

// Export singleton instance
export const logger = new Logger();

// Export logger class for testing
export { Logger };

