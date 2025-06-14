import Joi from 'joi';

/**
 * Validation schema for order parameters
 */
export const validateOrderParams = (orderData: any) => {
  const orderSchema = Joi.object({
    // Required parameters
    product_id: Joi.number().required().messages({
      'number.base': 'Product ID must be a number',
      'any.required': 'Product ID is required'
    }),
    
    side: Joi.string().valid('buy', 'sell').required().messages({
      'string.base': 'Side must be a string',
      'any.only': 'Side must be either "buy" or "sell"',
      'any.required': 'Side is required'
    }),
    
    order_type: Joi.string().valid('market_order', 'limit_order').required().messages({
      'string.base': 'Order type must be a string',
      'any.only': 'Order type must be either "market_order" or "limit_order"',
      'any.required': 'Order type is required'
    }),
    
    size: Joi.number().positive().required().messages({
      'number.base': 'Size must be a number',
      'number.positive': 'Size must be positive',
      'any.required': 'Size is required'
    }),
    
    // Optional parameters
    limit_price: Joi.number().positive().when('order_type', {
      is: 'limit_order',
      then: Joi.required(),
      otherwise: Joi.optional()
    }).messages({
      'number.base': 'Limit price must be a number',
      'number.positive': 'Limit price must be positive',
      'any.required': 'Limit price is required for limit orders'
    }),
    
    leverage: Joi.number().min(1).max(25).messages({
      'number.base': 'Leverage must be a number',
      'number.min': 'Leverage must be at least 1',
      'number.max': 'Leverage cannot exceed 25'
    }),
    
    reduce_only: Joi.boolean().messages({
      'boolean.base': 'Reduce only must be a boolean'
    }),
    
    post_only: Joi.boolean().messages({
      'boolean.base': 'Post only must be a boolean'
    }),
    
    client_order_id: Joi.string().max(40).messages({
      'string.base': 'Client order ID must be a string',
      'string.max': 'Client order ID cannot exceed 40 characters'
    }),
    
    // Add additional parameters as needed
    time_in_force: Joi.string().valid('gtc', 'ioc', 'fok').messages({
      'string.base': 'Time in force must be a string',
      'any.only': 'Time in force must be one of "gtc", "ioc", or "fok"'
    })
  });
  
  return orderSchema.validate(orderData, { abortEarly: false });
};

/**
 * Validation schema for order cancellation parameters
 */
export const validateCancelOrderParams = (cancelData: any) => {
  const cancelSchema = Joi.object({
    order_id: Joi.string().required().messages({
      'string.base': 'Order ID must be a string',
      'any.required': 'Order ID is required'
    })
  });
  
  return cancelSchema.validate(cancelData, { abortEarly: false });
};

/**
 * Validation schema for market lookup parameters
 */
export const validateMarketLookupParams = (lookupData: any) => {
  const lookupSchema = Joi.object({
    symbol: Joi.string().required().messages({
      'string.base': 'Symbol must be a string',
      'any.required': 'Symbol is required'
    })
  });
  
  return lookupSchema.validate(lookupData, { abortEarly: false });
}; 