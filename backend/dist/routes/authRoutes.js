"use strict";
/**
 * Authentication Routes
 * Handles user registration, login, token management, email verification, and password reset
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const authController_1 = require("../controllers/authController");
const auth_1 = require("../middleware/auth");
const router = express_1.default.Router();
// CSRF token endpoint - provides token for frontend forms
router.get('/csrf-token', auth_1.csrfProtection, (req, res) => {
    res.json({ csrfToken: req.csrfToken() });
});
// Register new user - apply rate limiting to prevent spam
router.post('/register', auth_1.authRateLimiter, auth_1.csrfProtection, authController_1.register);
// Login user - apply rate limiting to prevent brute force attacks
router.post('/login', auth_1.authRateLimiter, auth_1.csrfProtection, authController_1.login);
// Refresh token - no CSRF/auth required as it uses the refresh token itself
router.post('/refresh-token', authController_1.refreshToken);
// Verify email
router.get('/verify-email/:token', authController_1.verifyEmail);
// Forgot password - request reset - apply rate limiting
router.post('/forgot-password', auth_1.authRateLimiter, auth_1.csrfProtection, authController_1.forgotPassword);
// Reset password with token - apply rate limiting
router.post('/reset-password', auth_1.authRateLimiter, auth_1.csrfProtection, authController_1.resetPassword);
// OAuth login/signup
router.post('/oauth', auth_1.csrfProtection, authController_1.oauthLogin);
// Logout - requires authentication
router.post('/logout', auth_1.protect, auth_1.csrfProtection, authController_1.logout);
exports.default = router;
//# sourceMappingURL=authRoutes.js.map