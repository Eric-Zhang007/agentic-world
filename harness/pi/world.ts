import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { readFile } from 'node:fs/promises';
import { Type } from '@earendil-works/pi-ai';
import type { ExtensionAPI } from '@earendil-works/pi-coding-agent';

export default function(pi: ExtensionAPI) {
  pi.registerTool({
    name: 'world', label: 'Persistent world',
    description: 'World tools: geometry_propose, preview, initialize, fork, status, extend, plan, compile, memory_capture, memory_query, inspect_video, render, commit. Read harness/pi/world_instructions.md for arguments. No internal language-model calls.',
    parameters: Type.Object({operation: Type.Union(['geometry_propose','preview','initialize','fork','status','extend','plan','compile','memory_capture','memory_query','inspect_video','render','commit'].map(x=>Type.Literal(x))), args: Type.Record(Type.String(), Type.Unknown())}),
    async execute(_id, params, signal) {
      const output = await new Promise<string>((resolve, reject) => {
        const child = spawn(process.env.WORLD_PYTHON || 'python', ['-X','utf8',fileURLToPath(new URL('./world_tools.py', import.meta.url))], {signal});
        let stdout = '', stderr = '';
        child.stdout.on('data', x => stdout += x);
        child.stderr.on('data', x => stderr += x);
        child.on('error', reject);
        child.on('close', code => code === 0 ? resolve(stdout) : reject(new Error(stderr || `World tool exit ${code}`)));
        child.stdin.end(JSON.stringify(params));
      });
      const details = JSON.parse(output);
      const content: any[] = [{type:'text', text:output}];
      for (const path of details.review_images || []) {
        content.push({type:'image', data:(await readFile(path)).toString('base64'), mimeType:path.endsWith('.jpg')?'image/jpeg':'image/png'});
      }
      return {content, details};
    }
  });
}
