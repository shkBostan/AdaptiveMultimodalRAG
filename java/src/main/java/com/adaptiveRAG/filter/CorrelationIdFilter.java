package com.adaptiveRAG.filter;

import com.adaptiveRAG.utils.LoggingUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jakarta.servlet.Filter;
import jakarta.servlet.FilterChain;
import jakarta.servlet.FilterConfig;
import jakarta.servlet.ServletException;
import jakarta.servlet.ServletRequest;
import jakarta.servlet.ServletResponse;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

import java.io.IOException;

/**
 * Servlet filter to handle correlation IDs for logging.
 * Extracts correlation ID from request header or generates a new one.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */
public class CorrelationIdFilter implements Filter {
    private static final Logger logger = LoggerFactory.getLogger(CorrelationIdFilter.class);
    private static final String CORRELATION_ID_HEADER = "X-Correlation-ID";
    
    @Override
    public void init(FilterConfig filterConfig) throws ServletException {
        logger.info("CorrelationIdFilter initialized");
    }
    
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain)
            throws IOException, ServletException {
        
        HttpServletRequest httpRequest = (HttpServletRequest) request;
        HttpServletResponse httpResponse = (HttpServletResponse) response;
        
        // Get correlation ID from header or generate new one
        String correlationId = httpRequest.getHeader(CORRELATION_ID_HEADER);
        if (correlationId == null || correlationId.isEmpty()) {
            correlationId = LoggingUtils.generateCorrelationId();
        }
        
        // Set correlation ID in ThreadContext
        LoggingUtils.setCorrelationId(correlationId);
        
        // Add correlation ID to response header
        httpResponse.setHeader(CORRELATION_ID_HEADER, correlationId);
        
        logger.debug("Request received with correlation ID",
            kv("correlationId", correlationId),
            kv("method", httpRequest.getMethod()),
            kv("uri", httpRequest.getRequestURI())
        );
        
        try {
            chain.doFilter(request, response);
        } finally {
            // Clean up ThreadContext after request
            LoggingUtils.clearContext();
        }
    }
    
    @Override
    public void destroy() {
        logger.info("CorrelationIdFilter destroyed");
    }
    
    private Object kv(String key, Object value) {
        return key + "=" + value;
    }
}

