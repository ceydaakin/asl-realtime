// On-device ASL demo: MediaPipe Holistic landmarks -> preprocess.js -> the exported TFLite model.
import {FilesetResolver, HolisticLandmarker} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/+esm';
import {Tensor, loadAndCompile, loadLiteRt} from 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm';

import {INPUT_SIZE, N_COLS, SignSegmenter, holisticFrame, toWindow} from './preprocess.js';

const MEDIAPIPE_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm';
const LITERT_WASM = 'https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/';
const LANDMARKER = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task';
const BENCHMARK_RUNS = 200;
const FILE_FPS = 30;

const $ = (id) => document.getElementById(id);
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

async function loadClassifier() {
  await loadLiteRt(LITERT_WASM);
  const model = await loadAndCompile('model/asl.tflite', {accelerator: 'wasm'});
  const signs = await (await fetch('model/signs.json')).json();
  // The converter does not keep input names, so tell xy and mask apart by rank.
  const xyFirst = model.getInputDetails()[0].shape.length === 4;

  return async ({xy, mask}) => {
    const inputs = [new Tensor(xy, [1, INPUT_SIZE, N_COLS, 2]), new Tensor(mask, [1, INPUT_SIZE])];
    const start = performance.now();
    const outputs = await model.run(xyFirst ? inputs : inputs.reverse());
    const probs = Array.from(outputs[0].toTypedArray());
    const ms = performance.now() - start;
    [...inputs, ...outputs].forEach((t) => t.delete());
    const top = probs.map((p, i) => [signs[i], p]).sort((a, b) => b[1] - a[1]).slice(0, 3);
    return {top, ms};
  };
}

async function loadLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  return HolisticLandmarker.createFromOptions(fileset, {
    baseOptions: {modelAssetPath: LANDMARKER, delegate: 'GPU'},
    runningMode: 'VIDEO',
  });
}

function show(top) {
  $('sign').textContent = top.length ? top[0][0] : 'no hand detected';
  $('others').textContent = top.map(([sign, p]) => `${sign} ${Math.round(p * 100)}%`).join(' · ');
}

async function main() {
  const video = $('video');
  const [classify, landmarker] = await Promise.all([loadClassifier(), loadLandmarker()]);
  const landmarkMs = [], modelMs = [];
  let segmenter = new SignSegmenter();
  let live = false;
  let lastTime = -1;
  let clock = 0;  // landmarker timestamps must keep increasing across camera and file runs

  function landmarks() {
    const start = performance.now();
    clock = Math.max(clock + 1, start);
    const frame = holisticFrame(landmarker.detectForVideo(video, clock));
    landmarkMs.push(performance.now() - start);
    $('timing').textContent = `landmarks ${median(landmarkMs.slice(-60)).toFixed(1)} ms/frame`
      + (modelMs.length ? ` · model ${median(modelMs).toFixed(2)} ms` : '');
    return frame;
  }

  async function recognise(frames) {
    const window = toWindow(frames);
    if (!window.length) return show([]);
    const {top, ms} = await classify(window);
    modelMs.push(ms);
    show(top);
    // Read by the page test and handy in the console.
    globalThis.lastPrediction = {top, hand_frames: window.length};
  }

  function onCameraFrame() {
    if (live && video.readyState >= 2 && video.currentTime !== lastTime) {
      lastTime = video.currentTime;
      const sign = segmenter.push(landmarks());
      if (sign) recognise(sign);
    }
    video.requestVideoFrameCallback(onCameraFrame);
  }

  $('camera').onclick = async () => {
    segmenter = new SignSegmenter();
    video.srcObject = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user'}});
    video.classList.add('mirrored');
    await video.play();
    live = true;
    $('status').textContent = 'Camera on';
  };

  // A file is one sign. Step through it by seeking, so slow devices don't skip frames.
  $('file').onchange = async (event) => {
    live = false;
    video.srcObject = null;
    video.classList.remove('mirrored');
    video.src = URL.createObjectURL(event.target.files[0]);
    await new Promise((resolve) => { video.onloadeddata = resolve; });
    const frames = [];
    for (let t = 0; t < video.duration; t += 1 / FILE_FPS) {
      video.currentTime = t;
      await new Promise((resolve) => { video.onseeked = resolve; });
      frames.push(landmarks());
      $('status').textContent = `Reading video… ${Math.round(100 * t / video.duration)}%`;
    }
    await recognise(frames);
    $('status').textContent = `Video done (${frames.length} frames)`;
  };

  $('benchmark').onclick = async () => {
    $('status').textContent = 'Benchmarking…';
    const window = {xy: new Float32Array(INPUT_SIZE * N_COLS * 2).fill(0.5), mask: new Float32Array(INPUT_SIZE).fill(1)};
    const times = [];
    for (let i = 0; i < BENCHMARK_RUNS; i++) times.push((await classify(window)).ms);
    const sorted = [...times].sort((a, b) => a - b);
    const result = {median_ms: median(times), p95_ms: sorted[Math.floor(times.length * 0.95)], runs: times.length};
    globalThis.lastBenchmark = result;
    $('status').textContent = `Model: ${result.median_ms.toFixed(2)} ms median, ${result.p95_ms.toFixed(2)} ms p95 `
      + `over ${result.runs} runs (full 64-frame window, CPU)`;
  };

  video.requestVideoFrameCallback(onCameraFrame);
  ['camera', 'file', 'benchmark'].forEach((id) => { $(id).disabled = false; });
  $('status').textContent = 'Ready';
}

main().catch((error) => {
  $('status').textContent = `Failed to start: ${error.message}`;
  throw error;
});
