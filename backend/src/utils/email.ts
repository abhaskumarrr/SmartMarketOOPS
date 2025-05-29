/**
 * Email Service
 * Handles sending emails via nodemailer
 */

import nodemailer from 'nodemailer';
import env from './env';

interface EmailOptions {
  to: string;
  subject: string;
  text: string;
  html?: string;
}

/**
 * Configure nodemailer transport
 * Uses different settings based on environment
 */
const getTransporter = () => {
  // In development, use a test account
  if (env.NODE_ENV === 'development') {
    return nodemailer.createTransport({
      host: 'smtp.ethereal.email',
      port: 587,
      secure: false,
      auth: {
        user: env.EMAIL_USER || '',
        pass: env.EMAIL_PASSWORD || ''
      }
    });
  }

  // In production, use configured SMTP service
  return nodemailer.createTransport({
    host: env.EMAIL_HOST,
    port: parseInt(env.EMAIL_PORT || '587'),
    secure: env.EMAIL_PORT === '465', // true for 465, false for other ports
    auth: {
      user: env.EMAIL_USER,
      pass: env.EMAIL_PASSWORD
    }
  });
};

/**
 * Send an email
 * @param {EmailOptions} options - Email configuration options
 * @returns {Promise<any>} Send result
 */
export const sendEmail = async (options: EmailOptions): Promise<any> => {
  try {
    const transporter = getTransporter();

    const mailOptions = {
      from: `"${env.EMAIL_FROM_NAME}" <${env.EMAIL_FROM}>`,
      to: options.to,
      subject: options.subject,
      text: options.text,
      html: options.html
    };

    const info = await transporter.sendMail(mailOptions);
    
    // Log email URL for dev environment (ethereal)
    if (env.NODE_ENV === 'development') {
      console.log('Preview URL: %s', nodemailer.getTestMessageUrl(info));
    }

    return info;
  } catch (error) {
    console.error('Email sending failed:', error);
    throw error;
  }
};

export default {
  sendEmail
}; 