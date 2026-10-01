// node --test app/web/preprocess.test.mjs
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

import {INPUT_SIZE, N_COLS, SignSegmenter, holisticFrame, toWindow} from './preprocess.js';

const LEFT_HAND = 468, RIGHT_HAND = 522;

function frame({hand = 'left', value = 0.5} = {}) {
  const f = new Float32Array(543 * 2).fill(value);
  const missing = hand === 'left' ? [RIGHT_HAND] : hand === 'right' ? [LEFT_HAND] : [LEFT_HAND, RIGHT_HAND];
  for (const start of missing) f.fill(NaN, start * 2, (start + 21) * 2);
  return f;
}

test('toWindow matches the Python preprocessing on the shared fixture', () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/window.json', import.meta.url)));
  for (const {name, frames, xy, mask} of fixture) {
    const out = toWindow(frames.map((f) => Float32Array.from(f.flat(), (v) => v ?? NaN)));
    assert.deepEqual(Array.from(out.mask), mask, name);
    const expected = xy.flat(2);
    assert.equal(out.xy.length, expected.length, name);
    out.xy.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-6, `${name}: value ${i}: ${v} vs ${expected[i]}`));
  }
});

test('a long clip is averaged down to the window size', () => {
  const frames = Array.from({length: INPUT_SIZE * 2}, (_, i) => {
    const f = frame();
    f[LEFT_HAND * 2] = i;
    return f;
  });

  const {xy, mask, length} = toWindow(frames);

  assert.equal(length, INPUT_SIZE);
  assert.ok(mask.every((m) => m === 1));
  for (let t = 0; t < INPUT_SIZE; t++) assert.equal(xy[t * N_COLS * 2 + 40 * 2], t * 2 + 0.5);
});

test('holisticFrame fills detected parts and leaves the rest NaN', () => {
  const points = (n, v) => Array.from({length: n}, () => ({x: v, y: v + 0.1, z: 0}));
  const f = holisticFrame({faceLandmarks: [points(478, 0.1)], poseLandmarks: [points(33, 0.3)],
                           leftHandLandmarks: [], rightHandLandmarks: [points(21, 0.4)]});

  assert.ok(Math.abs(f[467 * 2] - 0.1) < 1e-6);
  assert.ok(Number.isNaN(f[LEFT_HAND * 2]));
  assert.ok(Math.abs(f[489 * 2] - 0.3) < 1e-6);
  assert.ok(Math.abs(f[RIGHT_HAND * 2 + 1] - 0.5) < 1e-6);
});

test('segmenter emits a sign once the hand has left, and ignores blips', () => {
  const seg = new SignSegmenter({gap: 3, minFrames: 2});
  const idle = () => frame({hand: null});

  assert.equal(seg.push(idle()), null);
  for (let i = 0; i < 5; i++) assert.equal(seg.push(frame()), null);
  assert.equal(seg.push(idle()), null);
  assert.equal(seg.push(idle()), null);
  const sign = seg.push(idle());
  assert.equal(sign.length, 8);
  assert.equal(toWindow(sign).length, 5);

  assert.equal(seg.push(frame()), null);
  assert.deepEqual([seg.push(idle()), seg.push(idle()), seg.push(idle())], [null, null, null]);
});
