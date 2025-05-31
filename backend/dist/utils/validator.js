"use strict";
/**
 * Request Validation Utilities
 * Functions for validating API requests
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.validateRequest = void 0;
const express_validator_1 = require("express-validator");
/**
 * Validate a request against a set of validation rules
 * @param req Express request object
 * @param rules Array of validation rules
 * @returns Array of validation errors or undefined if valid
 */
const validateRequest = async (req, rules) => {
    // Apply validation rules
    await Promise.all(rules.map(rule => rule.run(req)));
    // Check for validation errors
    const errors = (0, express_validator_1.validationResult)(req);
    if (errors.isEmpty()) {
        return undefined;
    }
    // Format errors
    return errors.array().map(error => ({
        msg: error.msg,
        param: error.param
    }));
};
exports.validateRequest = validateRequest;
//# sourceMappingURL=validator.js.map