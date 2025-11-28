/**
 * React hook for logging with component context.
 * 
 * Author: s Bostan
 * Created on: Nov, 2025
 */

import { useEffect, useRef } from 'react';
import { logger } from '../utils/logger';

interface UseLoggerOptions {
  componentName: string;
  correlationId?: string;
  userId?: string;
}

export function useLogger(options: UseLoggerOptions) {
  const { componentName, correlationId, userId } = options;
  const mountedRef = useRef(false);

  useEffect(() => {
    // Set context when component mounts
    logger.setContext({
      component: componentName,
      correlationId,
      userId
    });

    logger.debug(`${componentName} mounted`);

    mountedRef.current = true;

    return () => {
      // Clean up context when component unmounts
      logger.debug(`${componentName} unmounted`);
      logger.clearContext();
      mountedRef.current = false;
    };
  }, [componentName, correlationId, userId]);

  return {
    trace: (message: string, extra?: any) => logger.trace(message, { component: componentName, ...extra }),
    debug: (message: string, extra?: any) => logger.debug(message, { component: componentName, ...extra }),
    info: (message: string, extra?: any) => logger.info(message, { component: componentName, ...extra }),
    warn: (message: string, extra?: any) => logger.warn(message, { component: componentName, ...extra }),
    error: (message: string, error?: Error, extra?: any) => logger.error(message, error, { component: componentName, ...extra })
  };
}

