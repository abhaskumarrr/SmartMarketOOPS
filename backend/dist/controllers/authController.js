"use strict";
/**
 * Authentication Controller
 * Handles user registration, login, token management, password reset, and OAuth
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.logout = exports.oauthLogin = exports.resetPassword = exports.forgotPassword = exports.refreshToken = exports.login = exports.verifyEmail = exports.register = void 0;
const prismaClient_1 = __importDefault(require("../utils/prismaClient"));
const bcryptjs_1 = __importDefault(require("bcryptjs"));
const crypto_1 = __importDefault(require("crypto"));
const jwt_1 = require("../utils/jwt");
const email_1 = require("../utils/email");
const env_1 = __importDefault(require("../utils/env"));
const sessionManager_1 = __importDefault(require("../utils/sessionManager"));
const sessionMiddleware_1 = require("../middleware/sessionMiddleware");
const logger_1 = require("../utils/logger");
// Environment variables
const { JWT_SECRET = 'your-secret-key', JWT_EXPIRY = '1d' } = process.env;
// Create logger
const logger = (0, logger_1.createLogger)('AuthController');
/**
 * Register a new user
 * @route POST /api/auth/register
 * @access Public
 */
const register = async (req, res) => {
    try {
        const { name, email, password } = req.body;
        // Validate input
        if (!name || !email || !password) {
            res.status(400).json({
                success: false,
                message: 'Please provide all required fields'
            });
            return;
        }
        // Check if user already exists
        const userExists = await prismaClient_1.default.user.findUnique({
            where: { email }
        });
        if (userExists) {
            res.status(400).json({
                success: false,
                message: 'User already exists'
            });
            return;
        }
        // Hash password
        const salt = await bcryptjs_1.default.genSalt(10);
        const hashedPassword = await bcryptjs_1.default.hash(password, salt);
        // Generate email verification token
        const verificationToken = crypto_1.default.randomBytes(32).toString('hex');
        const verificationTokenExpiry = new Date(Date.now() + 24 * 60 * 60 * 1000); // 24 hours
        // Create user
        const user = await prismaClient_1.default.user.create({
            data: {
                name,
                email,
                password: hashedPassword,
                verificationToken,
                verificationTokenExpiry
            }
        });
        if (user) {
            // Generate tokens
            const token = (0, jwt_1.generateToken)(user.id);
            const refreshToken = (0, jwt_1.generateRefreshToken)(user.id);
            // Send verification email
            const verificationUrl = `${env_1.default.CLIENT_URL}/verify-email?token=${verificationToken}`;
            try {
                await (0, email_1.sendEmail)({
                    to: email,
                    subject: 'Please verify your email',
                    text: `Please verify your email by clicking on the following link: ${verificationUrl}`,
                    html: `<p>Please verify your email by clicking on the following link: <a href="${verificationUrl}">${verificationUrl}</a></p>`
                });
            }
            catch (emailError) {
                console.error('Failed to send verification email:', emailError);
                // Continue with registration even if email fails
            }
            res.status(201).json({
                success: true,
                data: {
                    id: user.id,
                    name: user.name,
                    email: user.email,
                    token,
                    refreshToken,
                    isVerified: false,
                    message: 'Registration successful. Please check your email for verification.'
                }
            });
        }
        else {
            res.status(500).json({
                success: false,
                message: 'Failed to create user'
            });
        }
    }
    catch (error) {
        console.error('Registration error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during registration',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.register = register;
/**
 * Verify user email
 * @route GET /api/auth/verify-email/:token
 * @access Public
 */
const verifyEmail = async (req, res) => {
    try {
        const { token } = req.params;
        // Find user with the verification token
        const user = await prismaClient_1.default.user.findFirst({
            where: {
                verificationToken: token,
                verificationTokenExpiry: {
                    gte: new Date()
                }
            }
        });
        if (!user) {
            res.status(400).json({
                success: false,
                message: 'Invalid or expired verification token'
            });
            return;
        }
        // Update user as verified and clear token
        await prismaClient_1.default.user.update({
            where: { id: user.id },
            data: {
                isVerified: true,
                verificationToken: null,
                verificationTokenExpiry: null
            }
        });
        res.status(200).json({
            success: true,
            message: 'Email successfully verified'
        });
    }
    catch (error) {
        console.error('Email verification error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during email verification',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.verifyEmail = verifyEmail;
/**
 * Authenticate user and get token
 * @route POST /api/auth/login
 * @access Public
 */
const login = async (req, res) => {
    try {
        const { email, password, rememberMe = false } = req.body;
        // Validate input
        if (!email || !password) {
            res.status(400).json({
                success: false,
                message: 'Please provide email and password'
            });
            return;
        }
        // Check if user exists
        const user = await prismaClient_1.default.user.findUnique({
            where: { email }
        });
        if (!user) {
            res.status(401).json({
                success: false,
                message: 'Invalid credentials'
            });
            return;
        }
        // Check if password matches
        const isMatch = await bcryptjs_1.default.compare(password, user.password);
        if (!isMatch) {
            res.status(401).json({
                success: false,
                message: 'Invalid credentials'
            });
            return;
        }
        // Create a new session with device tracking
        const { token, refreshToken, session } = await sessionManager_1.default.createSession(user.id, req, rememberMe);
        // Update last login time
        await prismaClient_1.default.user.update({
            where: { id: user.id },
            data: { lastLoginAt: new Date() }
        });
        // Set remember me cookie if requested
        (0, sessionMiddleware_1.setRememberMeCookie)(req, res, rememberMe);
        res.status(200).json({
            success: true,
            data: {
                id: user.id,
                name: user.name,
                email: user.email,
                role: user.role,
                isVerified: user.isVerified,
                token,
                refreshToken,
                sessionId: session.id
            }
        });
    }
    catch (error) {
        console.error('Login error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during login',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.login = login;
/**
 * Refresh access token using refresh token
 * @route POST /api/auth/refresh-token
 * @access Public
 */
const refreshToken = async (req, res) => {
    try {
        const { refreshToken: requestRefreshToken } = req.body;
        if (!requestRefreshToken) {
            res.status(400).json({
                success: false,
                message: 'Refresh token is required'
            });
            return;
        }
        // Refresh the session
        const result = await sessionManager_1.default.refreshSession(requestRefreshToken, req);
        if (!result) {
            res.status(401).json({
                success: false,
                message: 'Invalid or expired refresh token'
            });
            return;
        }
        // Return new tokens
        res.status(200).json({
            success: true,
            data: {
                token: result.token,
                refreshToken: result.refreshToken,
                sessionId: result.session.id
            }
        });
    }
    catch (error) {
        console.error('Refresh token error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during token refresh',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.refreshToken = refreshToken;
/**
 * Forgot password - send reset email
 * @route POST /api/auth/forgot-password
 * @access Public
 */
const forgotPassword = async (req, res) => {
    try {
        const { email } = req.body;
        if (!email) {
            res.status(400).json({
                success: false,
                message: 'Please provide your email address'
            });
            return;
        }
        // Find user by email
        const user = await prismaClient_1.default.user.findUnique({
            where: { email }
        });
        if (!user) {
            // Don't reveal that the user doesn't exist for security
            res.status(200).json({
                success: true,
                message: 'If a user with that email exists, a password reset link has been sent'
            });
            return;
        }
        // Generate reset token
        const resetToken = crypto_1.default.randomBytes(32).toString('hex');
        const resetTokenExpiry = new Date(Date.now() + 60 * 60 * 1000); // 1 hour
        // Save reset token and expiry
        await prismaClient_1.default.user.update({
            where: { id: user.id },
            data: {
                resetToken,
                resetTokenExpiry
            }
        });
        // Send reset email
        const resetUrl = `${env_1.default.CLIENT_URL}/reset-password?token=${resetToken}`;
        try {
            await (0, email_1.sendEmail)({
                to: email,
                subject: 'Password Reset Request',
                text: `You requested a password reset. Please use the following link to reset your password: ${resetUrl}`,
                html: `<p>You requested a password reset. Please use the following link to reset your password: <a href="${resetUrl}">${resetUrl}</a></p>`
            });
        }
        catch (emailError) {
            console.error('Failed to send reset email:', emailError);
            res.status(500).json({
                success: false,
                message: 'Failed to send password reset email'
            });
            return;
        }
        res.status(200).json({
            success: true,
            message: 'If a user with that email exists, a password reset link has been sent'
        });
    }
    catch (error) {
        console.error('Forgot password error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during password reset request',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.forgotPassword = forgotPassword;
/**
 * Reset password with token
 * @route POST /api/auth/reset-password
 * @access Public
 */
const resetPassword = async (req, res) => {
    try {
        const { token, password } = req.body;
        if (!token || !password) {
            res.status(400).json({
                success: false,
                message: 'Please provide reset token and new password'
            });
            return;
        }
        // Find user with reset token
        const user = await prismaClient_1.default.user.findFirst({
            where: {
                resetToken: token,
                resetTokenExpiry: {
                    gte: new Date()
                }
            }
        });
        if (!user) {
            res.status(400).json({
                success: false,
                message: 'Invalid or expired reset token'
            });
            return;
        }
        // Hash new password
        const salt = await bcryptjs_1.default.genSalt(10);
        const hashedPassword = await bcryptjs_1.default.hash(password, salt);
        // Update user's password and clear reset token
        await prismaClient_1.default.user.update({
            where: { id: user.id },
            data: {
                password: hashedPassword,
                resetToken: null,
                resetTokenExpiry: null
            }
        });
        res.status(200).json({
            success: true,
            message: 'Password has been reset successfully'
        });
    }
    catch (error) {
        console.error('Reset password error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during password reset',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.resetPassword = resetPassword;
/**
 * OAuth login/signup
 * @route POST /api/auth/oauth
 * @access Public
 */
const oauthLogin = async (req, res) => {
    try {
        const { provider, token, email, name } = req.body;
        if (!provider || !token) {
            res.status(400).json({
                success: false,
                message: 'Please provide OAuth provider and token'
            });
            return;
        }
        // Here you would verify the OAuth token with the provider
        // This is a simplified version - in a real implementation you would
        // use the OAuth provider's API to verify the token and get user info
        // For demonstration purposes, we're assuming the token is valid and
        // we have the user's email and name
        if (!email) {
            res.status(400).json({
                success: false,
                message: 'Email is required for OAuth login'
            });
            return;
        }
        // Check if user exists
        let user = await prismaClient_1.default.user.findUnique({
            where: { email }
        });
        if (!user) {
            // Create new user if doesn't exist
            const password = crypto_1.default.randomBytes(16).toString('hex');
            const salt = await bcryptjs_1.default.genSalt(10);
            const hashedPassword = await bcryptjs_1.default.hash(password, salt);
            user = await prismaClient_1.default.user.create({
                data: {
                    name: name || email.split('@')[0],
                    email,
                    password: hashedPassword,
                    isVerified: true, // OAuth users are already verified
                    oauthProvider: provider,
                    oauthId: token // In a real app, this would be the ID from the provider
                }
            });
        }
        else {
            // Update existing user's OAuth info
            user = await prismaClient_1.default.user.update({
                where: { id: user.id },
                data: {
                    oauthProvider: provider,
                    oauthId: token,
                    lastLoginAt: new Date()
                }
            });
        }
        // Generate tokens
        const jwtToken = (0, jwt_1.generateToken)(user.id);
        const refreshToken = (0, jwt_1.generateRefreshToken)(user.id);
        res.status(200).json({
            success: true,
            data: {
                id: user.id,
                name: user.name,
                email: user.email,
                isVerified: user.isVerified,
                token: jwtToken,
                refreshToken
            }
        });
    }
    catch (error) {
        console.error('OAuth login error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during OAuth login',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.oauthLogin = oauthLogin;
/**
 * Logout user
 * @route POST /api/auth/logout
 * @access Private
 */
const logout = async (req, res) => {
    try {
        // Get token from header
        const authHeader = req.headers.authorization;
        if (authHeader && authHeader.startsWith('Bearer ')) {
            const token = authHeader.split(' ')[1];
            // Invalidate the session
            await sessionManager_1.default.invalidateSession(token);
        }
        // Clear cookies
        res.clearCookie('rememberMe');
        // Note: Don't clear deviceId cookie as it's used for security tracking
        res.status(200).json({
            success: true,
            message: 'Logged out successfully'
        });
    }
    catch (error) {
        console.error('Logout error:', error);
        res.status(500).json({
            success: false,
            message: 'Server error during logout',
            error: env_1.default.NODE_ENV === 'development' ? error.message : undefined
        });
    }
};
exports.logout = logout;
exports.default = {
    register: exports.register,
    verifyEmail: exports.verifyEmail,
    login: exports.login,
    refreshToken: exports.refreshToken,
    forgotPassword: exports.forgotPassword,
    resetPassword: exports.resetPassword,
    oauthLogin: exports.oauthLogin,
    logout: exports.logout
};
//# sourceMappingURL=authController.js.map