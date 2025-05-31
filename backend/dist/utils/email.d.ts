/**
 * Email Service
 * Handles sending emails via nodemailer
 */
interface EmailOptions {
    to: string;
    subject: string;
    text: string;
    html?: string;
}
/**
 * Send an email
 * @param {EmailOptions} options - Email configuration options
 * @returns {Promise<any>} Send result
 */
export declare const sendEmail: (options: EmailOptions) => Promise<any>;
declare const _default: {
    sendEmail: (options: EmailOptions) => Promise<any>;
};
export default _default;
