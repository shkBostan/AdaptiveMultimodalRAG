# Logging System Documentation

## Overview

This project implements a production-grade, multi-language logging system with structured logging, correlation IDs, and environment-based configuration.

## Quick Start

### Java

1. **Dependencies**: Already configured in `pom.xml`
2. **Configuration**: `java/src/main/resources/log4j2-spring.xml`
3. **Usage**:
```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.adaptiveRAG.utils.LoggingUtils;

private static final Logger logger = LoggerFactory.getLogger(MyClass.class);

// Set correlation ID
LoggingUtils.setCorrelationId("abc-123");

// Log
logger.info("Processing request", kv("requestId", requestId));
```

### Python

1. **Setup**: Initialize logging at application startup
```python
from src.logging import setup_logging, get_logger

# Initialize logging
setup_logging(env="dev")

# Get logger
logger = get_logger(__name__)
logger.info("Application started")
```

2. **FastAPI Integration**: Add middleware
```python
from src.logging import CorrelationIDMiddleware

app.add_middleware(CorrelationIDMiddleware)
```

### Frontend

1. **Import logger**:
```typescript
import { logger } from './utils/logger';

logger.info('Component loaded', { componentName: 'Home' });
```

2. **Use Error Boundary**:
```tsx
import { ErrorBoundary } from './utils/errorBoundary';

<ErrorBoundary>
  <YourComponent />
</ErrorBoundary>
```

## Features

- ✅ Structured logging (JSON in production)
- ✅ Correlation ID tracking
- ✅ Environment-based configuration
- ✅ Rotating log files
- ✅ Sensitive data sanitization
- ✅ Async logging support
- ✅ Multi-language support (Java, Python, TypeScript)

## Configuration

### Environment Variables

- `ENV`: Set to `dev`, `prod`, or `test`
- `LOG_LEVEL`: Override default log level
- `LOG_DIR`: Custom log directory

### Log Levels

- **TRACE**: Very detailed debugging
- **DEBUG**: Development debugging
- **INFO**: General information
- **WARN**: Warnings
- **ERROR**: Errors

## File Structure

```
logs/
├── app.log              # Main application log
├── app-error.log        # Error-only log
├── app.json             # JSON structured log (prod)
└── archive/             # Archived logs
    ├── app-2025-11-15-1.log.gz
    └── app-2025-11-15-1.json.gz
```

## Best Practices

See [BEST_PRACTICES.md](./BEST_PRACTICES.md) for detailed guidelines.

### Key Points:

1. **Always include context**: correlation IDs, user IDs, request IDs
2. **Sanitize sensitive data**: Never log passwords, API keys, tokens
3. **Use appropriate levels**: DEBUG for development, INFO for production
4. **Include stack traces**: Always log exceptions with `exc_info=True` (Python) or exception object (Java)
5. **Structured logging**: Use key-value pairs, not string concatenation

## Integration

### Sentry (Error Tracking)

**Frontend**:
```typescript
// Already integrated in logger.ts
// Just include Sentry SDK in your app
```

**Python**:
```python
import sentry_sdk
sentry_sdk.init(dsn="your-dsn")
```

**Java**:
```xml
<dependency>
    <groupId>io.sentry</groupId>
    <artifactId>sentry-log4j2</artifactId>
</dependency>
```

### ELK Stack

Logs are in JSON format and can be directly ingested into Elasticsearch.

## Troubleshooting

### Logs not appearing

1. Check log level configuration
2. Verify log directory exists and is writable
3. Check environment variable `ENV`

### Correlation IDs missing

1. Ensure middleware/filter is registered
2. Check ThreadContext/MDC is properly set
3. Verify context propagation in async operations

### Performance issues

1. Enable async logging (already configured)
2. Reduce log level in production
3. Check log file rotation settings



