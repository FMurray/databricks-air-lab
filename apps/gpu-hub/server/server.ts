import { createApp, analytics, server } from '@databricks/appkit';
import { broker } from '../plugins/broker/index.js';

createApp({
  plugins: [
    analytics(),
    server(),
    broker(),
  ],
}).catch(console.error);
