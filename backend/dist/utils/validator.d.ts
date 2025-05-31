/**
 * Request Validation Utilities
 * Functions for validating API requests
 */
import { Request } from 'express';
import { ValidationChain } from 'express-validator';
/**
 * Validate a request against a set of validation rules
 * @param req Express request object
 * @param rules Array of validation rules
 * @returns Array of validation errors or undefined if valid
 */
export declare const validateRequest: (req: Request, rules: ValidationChain[]) => Promise<{
    msg: string;
    param: string;
}[] | undefined>;
