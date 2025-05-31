"use strict";
/**
 * Email Service
 * Handles sending emails via nodemailer
 */
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.sendEmail = void 0;
const nodemailer_1 = __importDefault(require("nodemailer"));
const env_1 = __importDefault(require("./env"));
/**
 * Configure nodemailer transport
 * Uses different settings based on environment
 */
const getTransporter = () => {
    // In development, use a test account
    if (env_1.default.NODE_ENV === 'development') {
        return nodemailer_1.default.createTransport({
            host: 'smtp.ethereal.email',
            port: 587,
            secure: false,
            auth: {
                user: env_1.default.EMAIL_USER || '',
                pass: env_1.default.EMAIL_PASSWORD || ''
            }
        });
    }
    // In production, use configured SMTP service
    return nodemailer_1.default.createTransport({
        host: env_1.default.EMAIL_HOST,
        port: parseInt(env_1.default.EMAIL_PORT || '587'),
        secure: env_1.default.EMAIL_PORT === '465', // true for 465, false for other ports
        auth: {
            user: env_1.default.EMAIL_USER,
            pass: env_1.default.EMAIL_PASSWORD
        }
    });
};
/**
 * Send an email
 * @param {EmailOptions} options - Email configuration options
 * @returns {Promise<any>} Send result
 */
const sendEmail = async (options) => {
    try {
        const transporter = getTransporter();
        const mailOptions = {
            from: `"${env_1.default.EMAIL_FROM_NAME}" <${env_1.default.EMAIL_FROM}>`,
            to: options.to,
            subject: options.subject,
            text: options.text,
            html: options.html
        };
        const info = await transporter.sendMail(mailOptions);
        // Log email URL for dev environment (ethereal)
        if (env_1.default.NODE_ENV === 'development') {
            console.log('Preview URL: %s', nodemailer_1.default.getTestMessageUrl(info));
        }
        return info;
    }
    catch (error) {
        console.error('Email sending failed:', error);
        throw error;
    }
};
exports.sendEmail = sendEmail;
exports.default = {
    sendEmail: exports.sendEmail
};
//# sourceMappingURL=email.js.map