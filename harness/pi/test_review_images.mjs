// Verify image evidence survives the actual Pi tool boundary, without model calls.
import { fileURLToPath } from 'node:url';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import assert from 'node:assert/strict';
import { loadExtensions } from './node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js';

const directory=fileURLToPath(new URL('.',import.meta.url));
const root=resolve(directory,'../..');
const loaded=await loadExtensions([join(directory,'world.ts')],directory);
assert.equal(loaded.errors.length,0,JSON.stringify(loaded.errors));
const world=loaded.extensions.flatMap(e=>[...e.tools.values()]).find(t=>t.definition.name==='world');
assert.ok(world);
const output=await mkdtemp(join(tmpdir(),'pi-review-images-'));
try {
  const result=await world.definition.execute('review-image-test',{
    operation:'inspect_video',args:{
      plan:join(root,'outputs/posture_live_20261010/segment_00/plan'),
      video:join(root,'outputs/posture_live_20261010/segment_00/render/controlled.mp4'),
      output:join(output,'evidence')
    }
  },new AbortController().signal);
  const images=result.content.filter(c=>c.type==='image');
  assert.equal(images.length,3);
  assert.ok(images.every(c=>Buffer.from(c.data,'base64').length>1000));
  assert.equal(images[0].mimeType,'image/jpeg');
  assert.ok(result.details.review_contract.actor_assessment_ids.length>0);
  console.log('PASS: Pi tool returned contact sheet, opening and ending images, plus explicit review contract.');
} finally {
  await rm(output,{recursive:true,force:true});
}
