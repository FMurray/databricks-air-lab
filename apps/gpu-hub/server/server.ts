import { createApp, analytics, server } from '@databricks/appkit';
import { broker } from '../plugins/broker/index.js';

// Safety net for a hosted, long-lived hub: a stray async rejection (e.g. a dropped Lakebase
// connection in background work) must not exit the process and take the whole app down.
// Route handlers already return 500 on failure (Broker.safeRoute); this catches anything
// outside a request. We log and stay up rather than crash-loop.
process.on('unhandledRejection', (reason) => {
  console.error('[gpu-hub] unhandledRejection:', reason);
});
process.on('uncaughtException', (err) => {
  console.error('[gpu-hub] uncaughtException:', err);
});

createApp({
  plugins: [
    analytics(),
    server(),
    broker(),
  ],
}).catch(console.error);
