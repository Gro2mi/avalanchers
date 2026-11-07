// Compiles every .wgsl file in the given directories with Tint, the WGSL
// compiler of Chrome, through the Dawn bindings in the `webgpu` package. Tint
// enforces rules that naga does not, such as uniformity analysis, so a shader
// can run fine natively and in Firefox and still fail in Chrome.
//
// usage: node check_shaders.mjs <dir>...
import fs from 'node:fs';
import path from 'node:path';
import { create, globals } from 'webgpu';

Object.assign(globalThis, globals);

const files = process.argv.slice(2).flatMap((dir) =>
  fs.readdirSync(dir)
    .filter((name) => name.endsWith('.wgsl'))
    .map((name) => path.join(dir, name)));
if (files.length === 0) {
  console.error('no shaders found');
  process.exit(1);
}

// Shader compilation needs no GPU, so the null backend runs anywhere. The
// instance has to stay referenced, the bindings crash if it gets collected.
const gpu = create(['backend=null']);
const adapter = await gpu.requestAdapter();
if (!adapter) {
  console.error('no WebGPU adapter available');
  process.exit(1);
}
const device = await adapter.requestDevice();

let failed = 0;
for (const file of files) {
  const module = device.createShaderModule({ code: fs.readFileSync(file, 'utf8') });
  const { messages } = await module.getCompilationInfo();
  const errors = messages.filter((message) => message.type === 'error');
  if (errors.length > 0) {
    failed += 1;
    console.error(file);
    for (const { lineNum, linePos, message } of errors) {
      console.error(`  ${lineNum}:${linePos} ${message}`);
    }
  }
}

console.log(`${files.length} shaders checked, ${failed} failed`);
process.exit(failed > 0 ? 1 : 0);
