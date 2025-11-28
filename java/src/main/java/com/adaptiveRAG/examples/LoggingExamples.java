package com.adaptiveRAG.examples;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import com.adaptiveRAG.utils.LoggingUtils;
import com.adaptiveRAG.utils.LogSanitizer;

import java.util.Map;

/**
 * Examples demonstrating proper logging practices in Java.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */
public class LoggingExamples {
    private static final Logger logger = LoggerFactory.getLogger(LoggingExamples.class);
    
    /**
     * Example: Service method with proper logging.
     */
    public void processDocument(String documentId, String userId) {
        // Entry log with context
        logger.info("Processing document",
            kv("documentId", documentId),
            kv("userId", userId),
            kv("action", "processDocument")
        );
        
        long startTime = System.currentTimeMillis();
        
        try {
            // Debug log for detailed information
            logger.debug("Retrieving document from repository",
                kv("documentId", documentId)
            );
            
            // Simulate business logic
            String content = retrieveDocument(documentId);
            
            // Info log for important business events
            logger.info("Document retrieved successfully",
                kv("documentId", documentId),
                kv("contentLength", content.length())
            );
            
            // Process document
            String processed = processContent(content);
            
            long duration = System.currentTimeMillis() - startTime;
            
            // Exit log with metrics
            logger.info("Document processing completed",
                kv("documentId", documentId),
                kv("durationMs", duration),
                kv("status", "success")
            );
            
        } catch (DocumentNotFoundException e) {
            logger.warn("Document not found",
                kv("documentId", documentId),
                kv("error", e.getMessage()),
                e
            );
            throw e;
            
        } catch (Exception e) {
            long duration = System.currentTimeMillis() - startTime;
            
            // Error log with full context
            logger.error("Document processing failed",
                kv("documentId", documentId),
                kv("userId", userId),
                kv("durationMs", duration),
                kv("error", e.getMessage()),
                kv("errorType", e.getClass().getSimpleName()),
                e  // Include exception for stack trace
            );
            throw new ProcessingException("Failed to process document", e);
        }
    }
    
    /**
     * Example: API endpoint logging (would be in a controller).
     */
    public Map<String, Object> handleApiRequest(String requestId, String userId, String query) {
        // Initialize logging context
        String correlationId = LoggingUtils.initializeContext(null, userId, requestId);
        
        logger.info("API request received",
            kv("endpoint", "/api/query"),
            kv("method", "POST"),
            kv("userId", userId),
            kv("queryLength", query.length())
        );
        
        try {
            // Process request
            Map<String, Object> result = processQuery(query);
            
            logger.info("API request completed successfully",
                kv("requestId", requestId),
                kv("resultSize", result.size())
            );
            
            return result;
            
        } catch (Exception e) {
            logger.error("API request failed",
                kv("requestId", requestId),
                kv("userId", userId),
                kv("error", e.getMessage()),
                e
            );
            throw e;
            
        } finally {
            // Clean up context
            LoggingUtils.clearContext();
        }
    }
    
    /**
     * Example: Logging with sensitive data sanitization.
     */
    public void authenticateUser(String username, String password, String apiKey) {
        // NEVER log passwords
        // BAD: logger.info("Login attempt", kv("password", password));
        
        // GOOD: Log only safe information
        logger.info("User login attempt",
            kv("username", username),
            kv("apiKey", LogSanitizer.sanitizeApiKey(apiKey))
        );
        
        // Simulate authentication
        boolean authenticated = performAuthentication(username, password);
        
        if (authenticated) {
            logger.info("User authenticated successfully",
                kv("username", username)
            );
        } else {
            logger.warn("Authentication failed",
                kv("username", username)
            );
        }
    }
    
    /**
     * Example: Async operation with correlation ID propagation.
     */
    public void processAsyncOperation(String taskId) {
        // Get current correlation ID
        String correlationId = LoggingUtils.getCorrelationId();
        
        logger.info("Starting async operation",
            kv("taskId", taskId),
            kv("correlationId", correlationId)
        );
        
        // In async context, propagate correlation ID
        new Thread(() -> {
            // Set correlation ID in new thread
            LoggingUtils.setCorrelationId(correlationId);
            
            try {
                logger.debug("Async operation executing",
                    kv("taskId", taskId)
                );
                
                // Perform async work
                performAsyncWork(taskId);
                
                logger.info("Async operation completed",
                    kv("taskId", taskId)
                );
                
            } catch (Exception e) {
                logger.error("Async operation failed",
                    kv("taskId", taskId),
                    e
                );
            } finally {
                LoggingUtils.clearContext();
            }
        }).start();
    }
    
    /**
     * Example: Database operation logging.
     */
    public void saveDocument(String documentId, String content) {
        logger.debug("Saving document to database",
            kv("documentId", documentId),
            kv("contentLength", content.length())
        );
        
        try {
            // Database operation
            repository.save(documentId, content);
            
            logger.info("Document saved successfully",
                kv("documentId", documentId)
            );
            
        } catch (DataAccessException e) {
            logger.error("Database error while saving document",
                kv("documentId", documentId),
                kv("error", e.getMessage()),
                kv("sqlState", e.getSQLState()),
                e
            );
            throw new ServiceException("Failed to save document", e);
        }
    }
    
    /**
     * Example: Performance logging.
     */
    public void processBatch(String[] items) {
        if (logger.isDebugEnabled()) {
            logger.debug("Processing batch",
                kv("batchSize", items.length)
            );
        }
        
        long startTime = System.currentTimeMillis();
        int processed = 0;
        
        for (String item : items) {
            try {
                processItem(item);
                processed++;
            } catch (Exception e) {
                logger.warn("Failed to process item",
                    kv("item", item),
                    e
                );
            }
        }
        
        long duration = System.currentTimeMillis() - startTime;
        
        logger.info("Batch processing completed",
            kv("totalItems", items.length),
            kv("processedItems", processed),
            kv("failedItems", items.length - processed),
            kv("durationMs", duration),
            kv("itemsPerSecond", items.length * 1000.0 / duration)
        );
    }
    
    // Helper method for structured logging (placeholder)
    private Object kv(String key, Object value) {
        return key + "=" + value;
    }
    
    // Placeholder methods
    private String retrieveDocument(String id) { return ""; }
    private String processContent(String content) { return content; }
    private Map<String, Object> processQuery(String query) { return Map.of(); }
    private boolean performAuthentication(String u, String p) { return true; }
    private void performAsyncWork(String taskId) {}
    private void processItem(String item) {}
    
    // Placeholder classes
    private static class DocumentNotFoundException extends Exception {}
    private static class ProcessingException extends RuntimeException {
        ProcessingException(String msg, Throwable cause) { super(msg, cause); }
    }
    private static class DataAccessException extends Exception {
        String getSQLState() { return ""; }
    }
    private static class ServiceException extends RuntimeException {
        ServiceException(String msg, Throwable cause) { super(msg, cause); }
    }
    private static class Repository {
        void save(String id, String content) throws DataAccessException {}
    }
    private Repository repository = new Repository();
}

