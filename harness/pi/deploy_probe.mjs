// Exercise Pi's real TypeScript extension loader without creating a model session.
import { fileURLToPath } from 'node:url';
import { loadExtensions } from './node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js';

const directory = fileURLToPath(new URL('.', import.meta.url));
const loaded = await loadExtensions([directory + 'world.ts'], directory);
if (loaded.errors.length) throw new Error(JSON.stringify(loaded.errors));
const registered = loaded.extensions.flatMap(extension => [...extension.tools.values()]);
const world = registered.find(tool => tool.definition.name === 'world');
if (!world) throw new Error('world tool was not registered');
console.log('PASS: actual Pi extension loader registered world');
try {
  await world.definition.execute('deployment-probe', {
    operation: 'status', args: { db: '/dev/null/pi-deploy-missing.sqlite' },
  }, new AbortController().signal);
  throw new Error('Expected missing-world rejection');
} catch (error) {
  if (!String(error).includes('World must exist; fork an existing world first')) throw error;
  console.log('PASS: world tool spawned remote Python and reached deterministic dispatch');
}
