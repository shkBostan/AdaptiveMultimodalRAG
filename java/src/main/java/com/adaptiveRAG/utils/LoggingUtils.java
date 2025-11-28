package com.adaptiveRAG.utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.apache.logging.log4j.ThreadContext;

import java.util.UUID;

/**
 * Utility class for logging operations including correlation ID management.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */
public class LoggingUtils {
    private static final Logger logger = LoggerFactory.getLogger(LoggingUtils.class);
    private static final String CORRELATION_ID_KEY = "correlationId";
    private static final String USER_ID_KEY = "userId";
    private static final String REQUEST_ID_KEY = "requestId";
    
    /**
     * Generate a new correlation ID.
     * 
     * @return A new correlation ID as a UUID string
     */
    public static String generateCorrelationId() {
        return UUID.randomUUID().toString();
    }
    
    /**
     * Set correlation ID in ThreadContext (MDC).
     * 
     * @param correlationId The correlation ID to set
     */
    public static void setCorrelationId(String correlationId) {
        ThreadContext.put(CORRELATION_ID_KEY, correlationId);
        MDC.put(CORRELATION_ID_KEY, correlationId);
    }
    
    /**
     * Get the current correlation ID from ThreadContext.
     * 
     * @return The current correlation ID, or null if not set
     */
    public static String getCorrelationId() {
        return ThreadContext.get(CORRELATION_ID_KEY);
    }
    
    /**
     * Set user ID in ThreadContext.
     * 
     * @param userId The user ID to set
     */
    public static void setUserId(String userId) {
        ThreadContext.put(USER_ID_KEY, userId);
        MDC.put(USER_ID_KEY, userId);
    }
    
    /**
     * Set request ID in ThreadContext.
     * 
     * @param requestId The request ID to set
     */
    public static void setRequestId(String requestId) {
        ThreadContext.put(REQUEST_ID_KEY, requestId);
        MDC.put(REQUEST_ID_KEY, requestId);
    }
    
    /**
     * Clear all ThreadContext data (should be called at end of request).
     */
    public static void clearContext() {
        ThreadContext.clearAll();
        MDC.clear();
    }
    
    /**
     * Initialize logging context for a new request.
     * 
     * @param correlationId Optional correlation ID (generates new one if null)
     * @param userId Optional user ID
     * @param requestId Optional request ID
     * @return The correlation ID used
     */
    public static String initializeContext(String correlationId, String userId, String requestId) {
        if (correlationId == null || correlationId.isEmpty()) {
            correlationId = generateCorrelationId();
        }
        setCorrelationId(correlationId);
        
        if (userId != null && !userId.isEmpty()) {
            setUserId(userId);
        }
        
        if (requestId != null && !requestId.isEmpty()) {
            setRequestId(requestId);
        }
        
        logger.debug("Logging context initialized", 
            kv("correlationId", correlationId),
            kv("userId", userId),
            kv("requestId", requestId)
        );
        
        return correlationId;
    }
    
    /**
     * Helper method for structured logging key-value pairs.
     * Note: This is a placeholder - actual implementation depends on your logging framework.
     * For Log4j2, use StructuredDataMessage or custom layout.
     */
    private static Object kv(String key, Object value) {
        // This would be used with a structured logging framework
        // For now, return a simple string representation
        return key + "=" + value;
    }
    
    /**
     * Sanitize sensitive data before logging.
     * 
     * @param data The data to sanitize
     * @return Sanitized data
     */
    public static String sanitizeApiKey(String apiKey) {
        if (apiKey == null || apiKey.length() < 8) {
            return "***";
        }
        return apiKey.substring(0, 4) + "***" + apiKey.substring(apiKey.length() - 4);
    }
    
    /**
     * Sanitize email address for logging.
     * 
     * @param email The email to sanitize
     * @return Sanitized email
     */
    public static String sanitizeEmail(String email) {
        if (email == null || !email.contains("@")) {
            return "***";
        }
        int atIndex = email.indexOf('@');
        String local = email.substring(0, atIndex);
        String domain = email.substring(atIndex + 1);
        return local.substring(0, Math.min(2, local.length())) + "***@" + domain;
    }
}

