var simData, releasePoints;
var device;

var simTimer = new Timer("Avalanche Simulation");

class ShaderNode {
  constructor(name, runNode) {
    this.name = name;
    this.code = null;
    this.module = null;
    this.pipeline = null;
    this.bindGroup = null;
    this.computePass = null;
    this.runNode = runNode;
  }

  async compile(lineOffset = 0) {
    if (!this.code) {
      throw new Error("Shader code is not set for " + this.name);
    }
    this.module = device.createShaderModule({ code: this.code });
    const info = await this.module.getCompilationInfo();
    if (info.messages.length > 0) {
      console.group(this.name + " Shader Compilation Messages:");
      for (const msg of info.messages) {
        const type = msg.type.toUpperCase();
        console.log(`${type} [${msg.lineNum - lineOffset}:${msg.linePos}] ${msg.message}`);
      }
      console.groupEnd();

      const hadError = info.messages.some(m => m.type === "error");
      if (hadError) {
        throw new Error(this.name + " Shader Compilation failed. See log for details.");
      }
    } else {
      console.log(this.name + " Shader compiled successfully.");
    }
    return this.module;
  }

  createPipeline(compute = { module: this.module, entryPoint: this.name, }, layout = 'auto') {
    if (!this.module) {
      throw new Error("Shader module is not created for " + this.name);
    }
    this.pipeline = device.createComputePipeline({
      label: this.name + " Compute Pipeline",
      layout: 'auto',
      compute: compute,
    });
    return this.pipeline;
  }

  createBindGroup(entries) {
    if (!this.pipeline) {
      throw new Error("Pipeline is not created for " + this.name);
    }
    this.bindGroup = device.createBindGroup({
      label: this.name + " BindGroup",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: entries,
    });
    return this.bindGroup;
  }

  createComputePass(commandEncoder, workgroupCountX = Math.ceil(dem.width / 16), workgroupCountY = Math.ceil(dem.height / 16)) {
    if (!this.pipeline) {
      throw new Error("Pipeline is not created for " + this.name);
    }
    if (!this.bindGroup) {
      throw new Error("BindGroup is not created for " + this.name);
    }
    this.computePass = commandEncoder.beginComputePass({ label: this.name + " Compute Pass" });
    this.computePass.setPipeline(this.pipeline);
    this.computePass.setBindGroup(0, this.bindGroup);
    this.computePass.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
    this.computePass.end();
  }
}


class Shaders {
  constructor() {
    this.shaderImports = new ShaderNode("Imports");
    this.linesImported = 0;

    this.computeNormals = new ShaderNode("computeNormals", true);
    this.computeRoughness = new ShaderNode("computeRoughness", true);
    this.computeReleasePoints = new ShaderNode("computeReleasePoints", false);
    this.loadReleasePoints = new ShaderNode("loadReleasePoints", false);
    this.initializeParticles = new ShaderNode("initializeParticles", false);
    this.computeParticles = new ShaderNode("computeParticles", false);
    this.resetMaxVelocity = new ShaderNode("resetMaxVelocity", false);

    this.trajectory = new ShaderNode("computeTrajectories", false);
  }

  async fetch() {
    // TODO: implement include functionality for WGSL
    this.shaderImports.code = await loadAndConcatShaders([
      "wgsl/util.wgsl",
      "wgsl/random.wgsl",
    ]);
    await this.shaderImports.compile();
    this.linesImported = countLines(this.shaderImports.code);

    this.computeNormals.code = await loadAndConcatShaders(["wgsl/util.wgsl", "wgsl/random.wgsl", 'wgsl/computeNormals.wgsl']);
    this.computeRoughness.code = await loadAndConcatShaders(["wgsl/util.wgsl", "wgsl/random.wgsl", 'wgsl/computeRoughness.wgsl']);
    this.computeReleasePoints.code = await loadAndConcatShaders(["wgsl/util.wgsl", "wgsl/random.wgsl", 'wgsl/computeReleasePoints.wgsl']);
    this.loadReleasePoints.code = await loadAndConcatShaders(['wgsl/loadReleasePoints.wgsl']);
    this.initializeParticles.code = await loadAndConcatShaders(["wgsl/util.wgsl", "wgsl/random.wgsl", 'wgsl/initializeParticles.wgsl']);
    this.computeParticles.code = 
      (await loadAndConcatShaders(["wgsl/util.wgsl", "wgsl/random.wgsl", 'wgsl/computeParticles.wgsl']))
      .replace(/WORKGROUP_SIZE/g, maxWorkgroupX.toString());
    this.resetMaxVelocity.code = await loadAndConcatShaders(["wgsl/util.wgsl", 'wgsl/resetMaxVelocity.wgsl']);
    // this.trajectory.code = await loadAndConcatShaders(['wgsl/computeTrajectories.wgsl']);
  }

  async compile() {
    // await this.decodeDem.compile();
    await this.computeNormals.compile(this.linesImported);
    await this.computeRoughness.compile(this.linesImported);
    await this.computeReleasePoints.compile(this.linesImported);
    await this.loadReleasePoints.compile();
    await this.initializeParticles.compile(this.linesImported);
    await this.computeParticles.compile(this.linesImported);
    await this.resetMaxVelocity.compile();
    // await this.trajectory.compile();
  }

  createPipelines() {
    this.computeNormals.createPipeline();
    this.computeRoughness.createPipeline();
    this.computeReleasePoints.createPipeline();
    this.loadReleasePoints.createPipeline();
    this.initializeParticles.createPipeline();
    this.computeParticles.createPipeline();
    this.resetMaxVelocity.createPipeline();
    // this.trajectory.createPipeline();
  }

  static async fetchAndConcat(urls) {
    const codes = await Promise.all(
      urls.map(url => fetch(url).then(res => res.text()))
    );
    return codes.join('\n') + '\n';
  }

  topologicalSort(nodes) {
    const sorted = [];
    const visited = new Set();

    function visit(nodeName) {
      if (visited.has(nodeName)) return;
      visited.add(nodeName);

      const node = nodes[nodeName];
      for (const depGroup of node.dependencies) {
        if (depGroup.some(dep => visited.has(dep))) {
          continue; // Skip if any dependency in the group is ready
        }
        depGroup.forEach(visit); // Visit all dependencies in the group
      }
      sorted.push(nodeName);
    }

    Object.keys(nodes).forEach(visit);
    return sorted.reverse();
  }

}


function createRgba16floatTexture() {
  return device.createTexture({
    size: [dem.width, dem.height],
    format: "rgba16float",
    usage: GPUTextureUsage.STORAGE_BINDING
      | GPUTextureUsage.TEXTURE_BINDING
      | GPUTextureUsage.COPY_SRC
      | GPUTextureUsage.COPY_DST,
  });
}

async function run(simSettings, dem, release_point, predefinedReleasePoints) {

  var shaders = new Shaders();
  const numberGpuTimestamps = 5;
  // TODO: currently only works with 3 which is enough for test cases
  const trackedTrajectories = 3;
  const debugBufferSize = 100 * 4;

  const width = dem.width;
  const height = dem.height;
  const bytesPerRowUnpadded = width * 4;
  const bytesPerRow = align(bytesPerRowUnpadded, 256);

  const simSettingsBuffer = createInputBuffer(device, SimSettings.byteSize);
  device.queue.writeBuffer(simSettingsBuffer, 0, simSettings.createBuffer());

  simTimer = new Timer("Avalanche Simulation")
  // only load shaders if not already loaded or in debug mode
  if (debug || shaders.computeNormals.code == null) {
    await shaders.fetch();
    simTimer.checkpoint("shader fetching");
    await shaders.compile();
    simTimer.checkpoint("shader compilation");
    shaders.createPipelines();
  }

  const sampler = device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
  });

  const boundsBuffer = createInputBuffer(device, 4);
  device.queue.writeBuffer(boundsBuffer, 0, new Float32Array([dem.world_resolution]));

  const demTexture = createDemTextureAndBuffer(device, dem.data1d);

  // Create output texture for normals
  const normalsTexture = createRgba16floatTexture();
  const slopeTexture = createRgba16floatTexture();
  const roughnessTexture = createRgba16floatTexture();
  const releasePointsTexture = createRgba16floatTexture();


  const windTexture = device.createTexture({
    size: [width, height, 1],
    format: 'rgba16float',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  const landcoverTexture = device.createTexture({
    size: [width, height, 1],
    format: 'rgba8uint',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });


  // Release points compute shader

  const outNumberReleaseCells = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(outNumberReleaseCells, 0, new Uint32Array([0]));



  const outDebugRelease = createStorageBuffer(device, debugBufferSize);
  const readbackDebugRelease = createReadbackBuffer(device, debugBufferSize);


  const pixelData = await loadPNG("avaframe/" + simSettings.casename + "releaseTexture.png");

  function align(value, alignment) {
    return Math.ceil(value / alignment) * alignment;
  }

  const paddedData = new Uint8Array(bytesPerRow * height);
  const rgbaData = flipFlatArrayInY(pixelData.rgba, pixelData.width, pixelData.height);
  for (let row = 0; row < height; row++) {
    const srcOffset = row * bytesPerRowUnpadded;
    const dstOffset = row * bytesPerRow;

    // Copy one row from original data to padded buffer
    paddedData.set(rgbaData.subarray(srcOffset, srcOffset + bytesPerRowUnpadded), dstOffset);
    // The remaining bytes in the row are already zero by default
  }
  const releasePointsIn = device.createTexture({
    size: [width, height, 1],
    format: 'rgba8uint',
    usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
  });
  device.queue.writeTexture(
    { texture: releasePointsIn },
    paddedData,
    {
      offset: 0,
      bytesPerRow: bytesPerRow,
      rowsPerImage: height,
    },
    {
      width: width,
      height: height,
      depthOrArrayLayers: 1,
    }
  );


  const unpaddedBytesPerRow = dem.width * 8;
  const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256;
  const readReleaseTextureBuffer = device.createBuffer({
    size: paddedBytesPerRow * dem.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const readSlopeTextureBuffer = device.createBuffer({
    size: paddedBytesPerRow * dem.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const readRoughnessTextureBuffer = device.createBuffer({
    size: paddedBytesPerRow * dem.height,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const atomicCounter = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: "atomicCounter",
  });
  device.queue.writeBuffer(atomicCounter, 0, new Uint32Array([0]));

  const maxVelocityAtomicBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    label: "maxVelocityAtomicBuffer",
  });
  device.queue.writeBuffer(maxVelocityAtomicBuffer, 0, new Uint32Array([0]));


  const outDebugNormals = createStorageBuffer(device, debugBufferSize);
  const readbackDebugNormals = createReadbackBuffer(device, debugBufferSize);

  shaders.computeNormals.createBindGroup([
    { binding: 0, resource: { buffer: simSettingsBuffer } },
    { binding: 1, resource: demTexture.createView() },
    { binding: 2, resource: windTexture.createView() },
    { binding: 3, resource: normalsTexture.createView() },
    { binding: 4, resource: slopeTexture.createView() },
    { binding: 5, resource: { buffer: outDebugNormals } },
  ],
  );

  shaders.computeRoughness.createBindGroup([
    { binding: 0, resource: { buffer: simSettingsBuffer } },
    { binding: 1, resource: normalsTexture.createView() },
    { binding: 2, resource: landcoverTexture.createView() },
    { binding: 3, resource: roughnessTexture.createView() },
  ],
  );

  shaders.computeReleasePoints.createBindGroup([
    { binding: 0, resource: { buffer: simSettingsBuffer } },
    { binding: 1, resource: demTexture.createView() },
    { binding: 2, resource: slopeTexture.createView() },
    { binding: 3, resource: roughnessTexture.createView() },
    { binding: 6, resource: releasePointsTexture.createView() },
    { binding: 7, resource: { buffer: outDebugRelease } },
    { binding: 8, resource: { buffer: outNumberReleaseCells } },
  ],
  );
  shaders.loadReleasePoints.createBindGroup([
    { binding: 0, resource: releasePointsIn.createView() },
    { binding: 1, resource: releasePointsTexture.createView() },
    { binding: 2, resource: { buffer: outDebugRelease } },
    { binding: 3, resource: { buffer: outNumberReleaseCells } },
  ],
  );



  const inputPointData = new Float32Array([release_point[0], release_point[1]]);

  const inputPointBuffer = createInputBuffer(device, inputPointData.byteLength);

  device.queue.writeBuffer(inputPointBuffer, 0, inputPointData);


  const outputTextureSize = 4 * dem.width * dem.height;
  const outputTextureBuffer = createStorageBuffer(device, outputTextureSize);
  const outputVelocityTextureBuffer = createStorageBuffer(device, outputTextureSize);

  // Create all output buffers
  const simDataBufferSize = trackedTrajectories * SimData.timeStepByteSize * simSettings.maxSteps;
  const simInfoBuffer = device.createBuffer({
    size: SimInfo.byteSize,
    usage: GPUBufferUsage.STORAGE
      | GPUBufferUsage.COPY_SRC
      | GPUBufferUsage.COPY_DST,
  });
  const outBuffer = createStorageBuffer(device, simDataBufferSize);
  const outDebugTrajectories = createStorageBuffer(device, debugBufferSize);
  const outTrajectoryAtomicBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });

  const readbackSimInfo = createReadbackBuffer(device, SimInfo.byteSize);
  const readbackSimData = createReadbackBuffer(device, simDataBufferSize);
  const readbackDebugTrajectories = createReadbackBuffer(device, debugBufferSize);
  const readbackOutputTexture = createReadbackBuffer(device, outputTextureSize);
  const readbackVelocityTexture = createReadbackBuffer(device, outputTextureSize);
  const readbackAtomicBuffer = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
  });


  let bufferTimestamps;
  if (debug) {
    bufferTimestamps = device.createBuffer({
      size: numberGpuTimestamps * 8, // 2 timestamps * 8 bytes each
      usage: GPUBufferUsage.QUERY_RESOLVE
        | GPUBufferUsage.STORAGE
        | GPUBufferUsage.COPY_SRC
        | GPUBufferUsage.COPY_DST,
    });
    shaders.timestampQuerySet = device.createQuerySet({ type: 'timestamp', count: numberGpuTimestamps });
  }

  // Encode commands
  const commandEncoderPreperation = device.createCommandEncoder();
  // if (debug) { commandEncoder.writeTimestamp(timestampQuerySet, 0) };
  shaders.computeNormals.createComputePass(commandEncoderPreperation);
  shaders.computeRoughness.createComputePass(commandEncoderPreperation);
  // if (debug) { commandEncoder.writeTimestamp(timestampQuerySet, 1) };
  if (predefinedReleasePoints) {
    shaders.loadReleasePoints.createComputePass(commandEncoderPreperation);
  } else {
    shaders.computeReleasePoints.createComputePass(commandEncoderPreperation);
  }

  copyTexture(commandEncoderPreperation, releasePointsTexture, readReleaseTextureBuffer, paddedBytesPerRow);
  copyTexture(commandEncoderPreperation, slopeTexture, readSlopeTextureBuffer, paddedBytesPerRow);
  copyTexture(commandEncoderPreperation, roughnessTexture, readRoughnessTextureBuffer, paddedBytesPerRow);
  device.queue.submit([commandEncoderPreperation.finish()]);
  await device.queue.onSubmittedWorkDone();

  simTimer.checkpoint("preparation");

  const numberParticles = (await copyAndReadBuffer(outNumberReleaseCells, Uint32Array))[0] * simSettings.releasedParticlesPerCell;

  const commandEncoderInitialization = device.createCommandEncoder();

  const particles = device.createBuffer({
    size: numberParticles * Particle.byteSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  shaders.initializeParticles.createBindGroup([
    { binding: 0, resource: { buffer: simSettingsBuffer } },
    { binding: 1, resource: { buffer: simInfoBuffer } },
    { binding: 2, resource: demTexture.createView() },
    { binding: 3, resource: releasePointsTexture.createView() },
    { binding: 4, resource: sampler },
    { binding: 5, resource: { buffer: particles } },
    { binding: 6, resource: { buffer: atomicCounter } },
    { binding: 7, resource: { buffer: maxVelocityAtomicBuffer } },
  ],
  );
  // if (debug) { commandEncoder.writeTimestamp(timestampQuerySet, 2) };
  shaders.initializeParticles.createComputePass(commandEncoderInitialization);
  device.queue.submit([commandEncoderInitialization.finish()]);
  await device.queue.onSubmittedWorkDone();
  const startParticles = await copyAndReadBuffer(particles, Float32Array);


  shaders.computeParticles.createBindGroup([
    { binding: 0, resource: { buffer: simSettingsBuffer } },
    { binding: 1, resource: { buffer: simInfoBuffer } },
    { binding: 2, resource: demTexture.createView() },
    { binding: 3, resource: normalsTexture.createView() },
    { binding: 4, resource: { buffer: particles } },
    { binding: 5, resource: sampler },

    { binding: 6, resource: { buffer: maxVelocityAtomicBuffer } },
    { binding: 7, resource: { buffer: outBuffer } },
    { binding: 8, resource: { buffer: outDebugTrajectories } },
    { binding: 9, resource: { buffer: outputTextureBuffer } },
    { binding: 10, resource: { buffer: outputVelocityTextureBuffer } },
    // { binding: 11, resource: { buffer: outTrajectoryAtomicBuffer } },
  ],
  );
  
  shaders.resetMaxVelocity.createBindGroup([
    { binding: 1, resource: { buffer: simInfoBuffer } },
    { binding: 2, resource: { buffer: maxVelocityAtomicBuffer } },
    // { binding: 11, resource: { buffer: outTrajectoryAtomicBuffer } },
  ],
  );

  device.queue.writeBuffer(simInfoBuffer, 0, new Uint32Array([0, numberParticles]));
  // for (let i = 0; i < simSettings.maxSteps; i++) {
  const commandEncoderCompute = device.createCommandEncoder();
  const computePass = commandEncoderCompute.beginComputePass();
  for (let i = 0; i < simSettings.maxSteps; i++) {
    // device.queue.writeBuffer(simInfoBuffer, 0, new Uint32Array([i]));
    // shaders.computeParticles.createComputePass(commandEncoderCompute, Math.ceil(numberParticles / 64), 1);

    computePass.setPipeline(shaders.computeParticles.pipeline);
    computePass.setBindGroup(0, shaders.computeParticles.bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numberParticles / maxWorkgroupX));
    computePass.setPipeline(shaders.resetMaxVelocity.pipeline);
    computePass.setBindGroup(0, shaders.resetMaxVelocity.bindGroup);
    computePass.dispatchWorkgroups(1);
  }
  computePass.end()
  device.queue.submit([commandEncoderCompute.finish()]);
  await device.queue.onSubmittedWorkDone();
  // shaders.trajectory.createComputePass(commandEncoder, n = simSettings.numberTrajectories);
  // if (debug) { commandEncoder.writeTimestamp(timestampQuerySet, 3) };

  const commandEncoder = device.createCommandEncoder();
  // Wait for copy to finish
  // Copy outputs to readback buffers
  commandEncoder.copyBufferToBuffer(outBuffer, 0, readbackSimData, 0, simDataBufferSize);
  commandEncoder.copyBufferToBuffer(outputTextureBuffer, 0, readbackOutputTexture, 0, outputTextureSize);
  commandEncoder.copyBufferToBuffer(outputVelocityTextureBuffer, 0, readbackVelocityTexture, 0, outputTextureSize);
  commandEncoder.copyBufferToBuffer(outTrajectoryAtomicBuffer, 0, readbackAtomicBuffer, 0, 4);

  commandEncoder.copyBufferToBuffer(outDebugTrajectories, 0, readbackDebugTrajectories, 0, debugBufferSize);
  commandEncoder.copyBufferToBuffer(outDebugNormals, 0, readbackDebugNormals, 0, debugBufferSize);
  commandEncoder.copyBufferToBuffer(outDebugRelease, 0, readbackDebugRelease, 0, debugBufferSize);
  // if (debug) {
  //   commandEncoder.writeTimestamp(timestampQuerySet, 4)
  //   commandEncoder.resolveQuerySet(timestampQuerySet, 0, numberGpuTimestamps, bufferTimestamps, 0);
  // }

  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  simTimer.checkpoint("shader execution");

  // Read results
  const totalTimesteps = await readBuffer(readbackAtomicBuffer, Uint32Array);
  console.log("Max timesteps: ", totalTimesteps[0]);
  // TODO read simData
  const bufferSimData = await readBuffer(readbackSimData, Float32Array, trackedTrajectories * SimData.timeStepByteSize * simSettings.maxSteps);
  const bufferOutputTexture = await readBuffer(readbackOutputTexture, Uint32Array);
  const bufferVelocityTexture = await readBuffer(readbackVelocityTexture, Uint32Array);

  const bufferDebugNormals = await readBuffer(readbackDebugNormals, Float32Array);
  const bufferDebugRelease = await readBuffer(readbackDebugRelease, Float32Array);
  const bufferDebugTrajectories = await readBuffer(readbackDebugTrajectories, Float32Array);

  const particlesData = await copyAndReadBuffer(particles, Float32Array);
  console.log("Start Particles: ", startParticles);
  console.log("Particles: ", particlesData);
  console.log("NumberParticles: ", numberParticles);
  simTimer.checkpoint("readback buffer");

  if (debug) {
    // const gpuTimestampsNs = await copyAndReadBuffer(commandEncoder, bufferTimestamps, BigInt64Array);
    // // loses precision for numbers > 2^53 -1
    // const gpuTimestamps = {
    //   normals: Number(gpuTimestampsNs[1] - gpuTimestampsNs[0]) / 1e6,
    //   releasePoints: Number(gpuTimestampsNs[2] - gpuTimestampsNs[1]) / 1e6,
    //   trajectories: Number(gpuTimestampsNs[3] - gpuTimestampsNs[2]) / 1e6,
    //   copy: Number(gpuTimestampsNs[4] - gpuTimestampsNs[3]) / 1e6,
    // }
    // console.log("GPU timestamps: ", gpuTimestamps);

    console.log("Debug info normals: ", debugBufferLine(bufferDebugNormals,
      ["nx", "ny", "nz", "resolution", "dzdx", "dzdy", "dxx", "dyy", "dxy", "curvature"]
    ));
    console.log("Debug info release: ", debugBufferLine(bufferDebugRelease,
      ["r"]
    ));
    console.log("Debug info trajectories: ", debugBufferLine(bufferDebugTrajectories,
      ["normalx", "normaly", "u", "v", "elevation", "elevation_threshold", "", "xmin", "ymin", "xmax", "ymax"]
    ));
  }

  const { r: releasePoints, g: gpxArea, b: predictor, a } = await readAndParseRGBA16FloatBuffer(readReleaseTextureBuffer, dem.width, dem.height);
  const { r: slopeAngle, g: slopeAspect, b: windShelterIndex, a2 } = await readAndParseRGBA16FloatBuffer(readSlopeTextureBuffer, dem.width, dem.height);
  const { r: roughness, g: forest, b: b, a3 } = await readAndParseRGBA16FloatBuffer(readRoughnessTextureBuffer, dem.width, dem.height);
  const lastTimestep = new Uint32Array(particlesData.buffer, 12 * 4, 1)[0];
  simTimer.checkpoint("readback textures");
  simData = new SimData(simSettings.cellSize);
  simData.parse(bufferSimData, lastTimestep, trackedTrajectories);

  console.log([...bufferSimData.slice(0, SimData.timeStepByteSize / 4)]);
  simTimer.printSummary();
  simData.parseSlopeTexture(slopeAngle, slopeAspect, windShelterIndex);
  simData.parseReleaseTexture(releasePoints, gpxArea, predictor)
  simData.parseRoughnessTexture(roughness, forest)
  simData.parseVelocityTexture([...bufferVelocityTexture]);
  simData.parseCellCountTexture([...bufferOutputTexture]);
  simTimer.checkpoint("parse results");
}

function copyTexture(commandEncoder, texture, buffer, paddedBytesPerRow) {
  commandEncoder.copyTextureToBuffer(
    {
      texture: texture,
      mipLevel: 0,
      origin: { x: 0, y: 0, z: 0 },
    },
    {
      buffer: buffer,
      bytesPerRow: paddedBytesPerRow,
    },
    {
      width: dem.width,
      height: dem.height,
      depthOrArrayLayers: 1,
    }
  );
}
async function readBuffer(buffer, ctor, size = 0) {
  await buffer.mapAsync(GPUMapMode.READ);
  const arrayBuffer = buffer.getMappedRange();
  const result = new ctor(arrayBuffer.slice(0, size == 0 ? buffer.size : size));
  buffer.unmap();
  return result;
}

async function copyAndReadBuffer(buffer, ctor = Float32Array) {
  const size = buffer.size;
  const gpuReadBuffer = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToBuffer(buffer, 0, gpuReadBuffer, 0, size);
  device.queue.submit([copyEncoder.finish()]);
  await gpuReadBuffer.mapAsync(GPUMapMode.READ);
  return new ctor(gpuReadBuffer.getMappedRange());
}

function debugBufferLine(bufferDebug, descriptions = [], debugBufferSize = 100) {
  var line = "";
  for (let i = 0; i < debugBufferSize; i++) {
    const desc = descriptions[i] || "";
    if (bufferDebug[i] == 0 && desc == "") continue; // Skip empty descriptions
    line += `(${i}) ${desc}: ${bufferDebug[i].toFixed(2)}, `;
  }
  return line;
}

// Create readback buffers (for CPU reading)
function createReadbackBuffer(device, size) {
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
}

function createStorageBuffer(device, bufferSize) {
  return device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
}

function createInputBuffer(device, bufferSize) {
  return device.createBuffer({
    size: bufferSize,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
}

async function loadAndConcatShaders(urls) {
  const codes = await Promise.all(
    urls.map(url => fetch(url).then(res => res.text()))
  );
  return codes.join('\n');
}

function createDemTextureAndBuffer(device, data, format = 'r32float', ctor = Float32Array) {
  switch (format) {
    case 'rgba8uint':
    case 'rgba8unorm':
      var bytesPerPixel = 4;
      break;
    case 'rgba16float':
      var bytesPerPixel = 8;
      break;
    case 'r32float':
      var bytesPerPixel = 4;
      break;
    default:
      throw new Error("Unsupported texture format: " + format + ". Add the format in createDemTextureAndBuffer function.");
  }
  const bytesPerRow = Math.ceil(dem.width * bytesPerPixel / 256) * 256; // 4 bytes per float32 pixel, must be aligned to 256 bytes

  const texture = device.createTexture({
    size: [dem.width, dem.height, 1],
    format: format,
    usage: GPUTextureUsage.TEXTURE_BINDING
      | GPUTextureUsage.COPY_DST,
  });

  const textureBuffer = device.createBuffer({
    size: bytesPerRow * dem.height,
    usage: GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true,
  });

  padInputTextureData(textureBuffer, data, bytesPerPixel, ctor);
  const copyEncoder = device.createCommandEncoder();
  copyEncoder.copyBufferToTexture(
    {
      buffer: textureBuffer,
      bytesPerRow: bytesPerRow,
    },
    {
      texture: texture,
    },
    [dem.width, dem.height, 1]
  );
  device.queue.submit([copyEncoder.finish()]);

  return texture;
}

function padInputTextureData(buffer, data, bytesPerPixel, ctor) {
  const bytesPerRow = Math.ceil(dem.width * bytesPerPixel / 256) * 256;
  const mappedRange = buffer.getMappedRange();
  const dst = new ctor(mappedRange);

  for (let row = 0; row < dem.height; row++) {
    const srcOffset = row * dem.width;
    const dstOffset = (bytesPerRow / bytesPerPixel) * row;
    dst.set(data.subarray(srcOffset, srcOffset + dem.width), dstOffset);
  }
  buffer.unmap();
}

function exportReleasePointsToPNG(releasePoints, width, height, canvasId = "releaseCanvas") {
  // Create or select a canvas
  let canvas = document.getElementById(canvasId);
  if (!canvas) {
    canvas = document.createElement('canvas');
    canvas.id = canvasId;
    document.body.appendChild(canvas);
  }
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  function flipImageDataVertically(src, width, height) {
    const rowSize = width * 4; // 4 bytes per pixel (RGBA)
    const flipped = new Uint8ClampedArray(src.length);

    for (let y = 0; y < height; y++) {
      const srcOffset = y * rowSize;
      const dstOffset = (height - 1 - y) * rowSize;
      flipped.set(src.subarray(srcOffset, srcOffset + rowSize), dstOffset);
    }
    return flipped;
  }
}

async function readAndParseRGBA16FloatBuffer(buffer) {
  await buffer.mapAsync(GPUMapMode.READ);
  const mappedBuffer = new Uint16Array(buffer.getMappedRange());
  const { r, g, b, a } = processRGBA16FloatBuffer(mappedBuffer, dem.width, dem.height);
  buffer.unmap();
  return { r, g, b, a };
}

function processRGBA16FloatBuffer(mapped, width, height) {
  const bytesPerPixel = 8; // 4 channels × 2 bytes (float16)
  const unpaddedBytesPerRow = width * bytesPerPixel;
  const paddedBytesPerRow = Math.ceil(unpaddedBytesPerRow / 256) * 256; const r = Array.from({ length: height }, () => new Float32Array(width));
  const g = Array.from({ length: height }, () => new Float32Array(width));
  const b = Array.from({ length: height }, () => new Float32Array(width));
  const a = Array.from({ length: height }, () => new Float32Array(width));

  const paddedUint16sPerRow = paddedBytesPerRow / 2; // 2 bytes per Uint16
  const rowPixelUint16s = width * 4;

  for (let y = 0; y < height; y++) {
    const rowOffset = y * paddedUint16sPerRow;

    for (let x = 0; x < width; x++) {
      const index = rowOffset + x * 4;

      r[y][x] = decodeFloat16(mapped[index + 0]);
      g[y][x] = decodeFloat16(mapped[index + 1]);
      b[y][x] = decodeFloat16(mapped[index + 2]);
      a[y][x] = decodeFloat16(mapped[index + 3]);
    }
  }

  return { r, g, b, a };
}
