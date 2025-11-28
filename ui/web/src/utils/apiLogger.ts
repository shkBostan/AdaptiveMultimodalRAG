/**
 * API request/response logging interceptor.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */

import { logger } from './logger';

interface ApiRequest {
  url: string;
  method: string;
  headers?: HeadersInit;
  body?: any;
}

interface ApiResponse {
  status: number;
  statusText: string;
  headers: Headers;
  body?: any;
}

/**
 * Log API request.
 */
export function logApiRequest(request: ApiRequest): void {
  logger.debug('API request', {
    url: request.url,
    method: request.method,
    headers: sanitizeHeaders(request.headers),
    hasBody: !!request.body
  });
}

/**
 * Log API response.
 */
export function logApiResponse(
  request: ApiRequest,
  response: ApiResponse,
  duration: number
): void {
  const level = response.status >= 400 ? 'error' : 'info';
  const logData = {
    url: request.url,
    method: request.method,
    status: response.status,
    statusText: response.statusText,
    duration_ms: duration,
    headers: sanitizeHeaders(response.headers)
  };

  if (level === 'error') {
    logger.error('API request failed', undefined, logData);
  } else {
    logger.info('API request completed', logData);
  }
}

/**
 * Log API error.
 */
export function logApiError(
  request: ApiRequest,
  error: Error,
  duration?: number
): void {
  logger.error('API request error', error, {
    url: request.url,
    method: request.method,
    duration_ms: duration
  });
}

/**
 * Sanitize headers to remove sensitive information.
 */
function sanitizeHeaders(headers?: HeadersInit): Record<string, string> {
  if (!headers) return {};

  const sanitized: Record<string, string> = {};
  const sensitiveKeys = ['authorization', 'cookie', 'x-api-key', 'x-auth-token'];

  if (headers instanceof Headers) {
    headers.forEach((value, key) => {
      const lowerKey = key.toLowerCase();
      if (sensitiveKeys.some(sk => lowerKey.includes(sk))) {
        sanitized[key] = '***';
      } else {
        sanitized[key] = value;
      }
    });
  } else if (Array.isArray(headers)) {
    headers.forEach(([key, value]) => {
      const lowerKey = key.toLowerCase();
      if (sensitiveKeys.some(sk => lowerKey.includes(sk))) {
        sanitized[key] = '***';
      } else {
        sanitized[key] = value;
      }
    });
  } else {
    Object.entries(headers).forEach(([key, value]) => {
      const lowerKey = key.toLowerCase();
      if (sensitiveKeys.some(sk => lowerKey.includes(sk))) {
        sanitized[key] = '***';
      } else {
        sanitized[key] = String(value);
      }
    });
  }

  return sanitized;
}

/**
 * Create a fetch wrapper with logging.
 */
export function createLoggedFetch(originalFetch: typeof fetch): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request: ApiRequest = {
      url: typeof input === 'string' ? input : input.toString(),
      method: init?.method || 'GET',
      headers: init?.headers,
      body: init?.body
    };

    logApiRequest(request);

    const startTime = Date.now();

    try {
      const response = await originalFetch(input, init);
      const duration = Date.now() - startTime;

      // Clone response to read body without consuming it
      const clonedResponse = response.clone();
      let body: any = null;

      try {
        const contentType = response.headers.get('content-type');
        if (contentType?.includes('application/json')) {
          body = await clonedResponse.json();
        } else {
          body = await clonedResponse.text();
        }
      } catch (e) {
        // Ignore body parsing errors
      }

      logApiResponse(request, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
        body
      }, duration);

      return response;
    } catch (error) {
      const duration = Date.now() - startTime;
      logApiError(request, error as Error, duration);
      throw error;
    }
  };
}

