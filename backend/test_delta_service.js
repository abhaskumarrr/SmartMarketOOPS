// Quick test of Delta Exchange service
const { DeltaExchangeService } = require('./dist/services/deltaExchangeService.js');

console.log('Testing Delta Exchange Service...');
console.log('DeltaExchangeService:', typeof DeltaExchangeService);

if (DeltaExchangeService) {
  try {
    const credentials = {
      apiKey: 'test',
      apiSecret: 'test',
      testnet: true
    };
    
    const service = new DeltaExchangeService(credentials);
    console.log('Service created:', typeof service);
    console.log('Service methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(service)));
    console.log('isReady method:', typeof service.isReady);
    
    setTimeout(() => {
      console.log('Service ready:', service.isReady());
    }, 2000);
    
  } catch (error) {
    console.error('Error creating service:', error.message);
  }
} else {
  console.error('DeltaExchangeService not found');
}
