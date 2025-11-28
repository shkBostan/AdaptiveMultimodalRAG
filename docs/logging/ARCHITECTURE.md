# Logging Architecture

## Overview

This document describes the production-grade logging system for AdaptiveMultimodalRAG. The system provides structured, scalable, and maintainable logging across Java, Python, and Frontend components.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Application Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  Java (SLF4J + Log4j2)  │  Python (logging)  │  Frontend (JS)  │
└──────────────┬───────────┴──────────┬─────────┴────────┬────────┘
               │                      │                   │
               ▼                      ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Logging Facade Layer                          │
│  - Correlation ID Management                                      │
│  - Structured Logging (JSON)                                     │
│  - Environment-based Configuration                               │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Log Handlers                                  │
│  - Console Handler (Dev)                                         │
│  - File Handler (Rotating)                                       │
│  - JSON Handler (Production)                                     │
│  - Error Handler (Separate)                                      │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Log Storage                                   │
│  - Local Files (logs/)                                           │
│  - External Services (Sentry, LogRocket, ELK)                    │
└─────────────────────────────────────────────────────────────────┘
```

## Event Flow

### Request Flow with Correlation ID

```
1. Request arrives → Generate Correlation ID
2. Correlation ID stored in ThreadContext/MDC
3. All logs include Correlation ID
4. Async operations propagate Correlation ID
5. Response includes Correlation ID header
6. Logs aggregated by Correlation ID
```

### Log Levels

- **TRACE**: Very detailed information, typically only of interest when diagnosing problems
- **DEBUG**: Detailed information for debugging
- **INFO**: General informational messages about application flow
- **WARN**: Warning messages for potentially harmful situations
- **ERROR**: Error events that might still allow the application to continue

## Components

### 1. Java Logging (SLF4J + Log4j2)

**Location**: `java/src/main/resources/log4j2-spring.xml`

**Features**:
- SLF4J facade for abstraction
- Log4j2 implementation
- ThreadContext for correlation IDs
- JSON appender for production
- Rolling file appender
- Async logging support

### 2. Python Logging

**Location**: `python/logging_config.yaml`

**Features**:
- Standard logging module
- YAML configuration
- Correlation ID middleware
- JSON formatter for production
- Rotating file handlers
- FastAPI integration

### 3. Frontend Logging

**Location**: `ui/web/src/utils/logger.ts`

**Features**:
- Client-side logging utility
- Error boundary integration
- API request/response logging
- Production log level control
- External service integration (Sentry)

## Correlation ID Management

### Java
```java
ThreadContext.put("correlationId", correlationId);
// Automatically included in all logs
ThreadContext.clearAll(); // Clean up after request
```

### Python
```python
import contextvars
correlation_id = contextvars.ContextVar('correlation_id')
# Included in all log records
```

### Frontend
```typescript
const correlationId = generateCorrelationId();
logger.setContext({ correlationId });
```

## Environment Configuration

### Development
- Console logging with colors
- DEBUG level
- Human-readable format
- File logging to `logs/dev/`

### Production
- JSON structured logging
- INFO level (WARN/ERROR for critical)
- File logging to `logs/prod/`
- External service integration
- Log rotation and retention

## Security Considerations

### Sensitive Data Filtering

**Never log**:
- Passwords
- API keys
- Tokens
- Credit card numbers
- Social security numbers
- Personal identifiable information (PII)

**Sanitization**:
- Use log sanitizers before logging
- Mask sensitive fields
- Redact full objects containing sensitive data

## Best Practices

1. **Use appropriate log levels**
   - TRACE: Detailed execution flow
   - DEBUG: Development debugging
   - INFO: Important business events
   - WARN: Recoverable issues
   - ERROR: Failures requiring attention

2. **Include context**
   - Correlation IDs
   - User IDs (if applicable)
   - Request IDs
   - Operation names

3. **Structured logging**
   - Use key-value pairs
   - JSON format in production
   - Searchable fields

4. **Performance**
   - Use async logging where possible
   - Avoid logging in tight loops
   - Use appropriate log levels

5. **Error logging**
   - Include stack traces
   - Context information
   - Recovery actions taken

## Log File Structure

```
logs/
├── dev/
│   ├── app.log
│   ├── app-error.log
│   └── app-YYYY-MM-DD.log
├── prod/
│   ├── app.log
│   ├── app-error.log
│   ├── app-YYYY-MM-DD.log
│   └── app-YYYY-MM-DD.json
└── test/
    └── test.log
```

## Monitoring and Alerting

- **Error Rate**: Monitor ERROR log frequency
- **Response Times**: Log request durations
- **Correlation**: Track requests by correlation ID
- **Anomalies**: Alert on unusual patterns

## Integration Points

- **Sentry**: Error tracking and monitoring
- **LogRocket**: Frontend session replay
- **ELK Stack**: Centralized log aggregation
- **CloudWatch/DataDog**: Cloud logging services

