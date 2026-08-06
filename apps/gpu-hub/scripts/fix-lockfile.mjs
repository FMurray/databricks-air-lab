// Rewrite corp-proxy resolved URLs to the public registry: local installs go through
// npm-proxy.cloud.databricks.com (corp .npmrc), but the hosted Apps builder 404s on
// packages absent from the proxy allowlist — while it CAN fetch registry.npmjs.org.
// Run after any npm install that changes the lockfile (wired as postinstall).
import fs from "node:fs";
const p = new URL("../package-lock.json", import.meta.url);
const lock = JSON.parse(fs.readFileSync(p, "utf8"));
let n = 0;
for (const meta of Object.values(lock.packages ?? {})) {
  if (meta && typeof meta === "object" && typeof meta.resolved === "string" &&
      meta.resolved.includes("npm-proxy.cloud.databricks.com")) {
    meta.resolved = meta.resolved.replace(
      "https://npm-proxy.cloud.databricks.com", "https://registry.npmjs.org");
    n++;
  }
}
fs.writeFileSync(p, JSON.stringify(lock, null, 2));
console.log(`fix-lockfile: rewrote ${n} proxy URLs to registry.npmjs.org`);
