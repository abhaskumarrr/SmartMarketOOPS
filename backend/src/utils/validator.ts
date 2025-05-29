/**
 * Request Validation Utilities
 * Functions for validating API requests
 */

import { Request } from 'express';
import { ValidationChain, validationResult } from 'express-validator';

/**
 * Validate a request against a set of validation rules
 * @param req Express request object
 * @param rules Array of validation rules
 * @returns Array of validation errors or undefined if valid
 */
export const validateRequest = async (
  req: Request,
  rules: ValidationChain[]
): Promise<{ msg: string; param: string }[] | undefined> => {
  // Apply validation rules
  await Promise.all(rules.map(rule => rule.run(req)));
  
  // Check for validation errors
  const errors = validationResult(req);
  
  if (errors.isEmpty()) {
    return undefined;
  }
  
  // Format errors
  return errors.array().map(error => ({
    msg: error.msg,
    param: error.param
  }));
}; 