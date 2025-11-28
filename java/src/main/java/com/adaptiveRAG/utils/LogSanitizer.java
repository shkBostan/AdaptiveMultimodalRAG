package com.adaptiveRAG.utils;

/**
 * Utility class for sanitizing sensitive data before logging.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */
public class LogSanitizer {
    
    private static final String MASK = "***";
    private static final int VISIBLE_CHARS = 4;
    
    /**
     * Sanitize API key - shows first 4 and last 4 characters.
     * 
     * @param apiKey The API key to sanitize
     * @return Sanitized API key
     */
    public static String sanitizeApiKey(String apiKey) {
        if (apiKey == null || apiKey.isEmpty()) {
            return MASK;
        }
        if (apiKey.length() <= VISIBLE_CHARS * 2) {
            return MASK;
        }
        return apiKey.substring(0, VISIBLE_CHARS) + MASK + 
               apiKey.substring(apiKey.length() - VISIBLE_CHARS);
    }
    
    /**
     * Sanitize token - shows first 4 characters only.
     * 
     * @param token The token to sanitize
     * @return Sanitized token
     */
    public static String sanitizeToken(String token) {
        if (token == null || token.isEmpty()) {
            return MASK;
        }
        if (token.length() <= VISIBLE_CHARS) {
            return MASK;
        }
        return token.substring(0, VISIBLE_CHARS) + MASK;
    }
    
    /**
     * Sanitize email address - shows first 2 characters of local part.
     * 
     * @param email The email to sanitize
     * @return Sanitized email
     */
    public static String sanitizeEmail(String email) {
        if (email == null || !email.contains("@")) {
            return MASK;
        }
        int atIndex = email.indexOf('@');
        String local = email.substring(0, atIndex);
        String domain = email.substring(atIndex + 1);
        int visibleChars = Math.min(2, local.length());
        return local.substring(0, visibleChars) + MASK + "@" + domain;
    }
    
    /**
     * Sanitize credit card number - shows last 4 digits only.
     * 
     * @param cardNumber The card number to sanitize
     * @return Sanitized card number
     */
    public static String sanitizeCardNumber(String cardNumber) {
        if (cardNumber == null || cardNumber.isEmpty()) {
            return MASK;
        }
        // Remove spaces and dashes
        String cleaned = cardNumber.replaceAll("[\\s-]", "");
        if (cleaned.length() < 4) {
            return MASK;
        }
        return MASK + cleaned.substring(cleaned.length() - 4);
    }
    
    /**
     * Sanitize SSN - shows last 4 digits only.
     * 
     * @param ssn The SSN to sanitize
     * @return Sanitized SSN
     */
    public static String sanitizeSSN(String ssn) {
        if (ssn == null || ssn.isEmpty()) {
            return MASK;
        }
        String cleaned = ssn.replaceAll("[\\s-]", "");
        if (cleaned.length() < 4) {
            return MASK;
        }
        return MASK + cleaned.substring(cleaned.length() - 4);
    }
    
    /**
     * Sanitize password - always returns mask.
     * 
     * @param password The password (will be ignored)
     * @return Always returns mask
     */
    public static String sanitizePassword(String password) {
        return MASK;
    }
    
    /**
     * Sanitize phone number - shows last 4 digits.
     * 
     * @param phoneNumber The phone number to sanitize
     * @return Sanitized phone number
     */
    public static String sanitizePhoneNumber(String phoneNumber) {
        if (phoneNumber == null || phoneNumber.isEmpty()) {
            return MASK;
        }
        String cleaned = phoneNumber.replaceAll("[\\s()-]", "");
        if (cleaned.length() < 4) {
            return MASK;
        }
        return MASK + cleaned.substring(cleaned.length() - 4);
    }
    
    /**
     * Check if a string might contain sensitive data.
     * 
     * @param value The value to check
     * @return True if value might be sensitive
     */
    public static boolean mightBeSensitive(String value) {
        if (value == null || value.isEmpty()) {
            return false;
        }
        String lower = value.toLowerCase();
        return lower.contains("password") || 
               lower.contains("secret") || 
               lower.contains("key") || 
               lower.contains("token") || 
               lower.contains("api") ||
               lower.contains("auth");
    }
}

