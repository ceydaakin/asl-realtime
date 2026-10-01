// Browser port of src/asl_realtime/live.py. Keep the two in step: fixtures/window.json
// is written by the Python version and checked by both test suites.
//
// A frame is a Float32Array of 543 * 2 values (x, y per Holistic landmark), NaN = not detected.

export const INPUT_SIZE = 64;
export const N_COLS = 66;
const N_HOLISTIC = 543;

const range = (start, stop) => Array.from({length: stop - start}, (_, i) => start + i);
const LIPS = [
  61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
  291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
  78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
  95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
];
const LEFT_HAND = range(468, 489);
const RIGHT_HAND = range(522, 543);
const LEFT_DOMINANT = [...LIPS, ...LEFT_HAND, 502, 504, 506, 508, 510];
const RIGHT_DOMINANT = [...LIPS, ...RIGHT_HAND, 503, 505, 507, 509, 511];
// Holistic order: face, left hand, pose, right hand.
const PARTS = [['faceLandmarks', 0, 468], ['leftHandLandmarks', 468, 21],
               ['poseLandmarks', 489, 33], ['rightHandLandmarks', 522, 21]];

const detected = (frame, idxs) => idxs.some((i) => !Number.isNaN(frame[i * 2]));

/** MediaPipe HolisticLandmarkerResult -> frame. */
export function holisticFrame(result) {
  const frame = new Float32Array(N_HOLISTIC * 2).fill(NaN);
  for (const [field, start, count] of PARTS) {
    const points = result[field]?.[0];
    if (!points?.length) continue;
    for (let i = 0; i < count; i++) {
      frame[(start + i) * 2] = points[i].x;
      frame[(start + i) * 2 + 1] = points[i].y;
    }
  }
  return frame;
}

/** np.array_split: the first (n % parts) chunks are one longer. */
function chunks(n, parts) {
  const base = Math.floor(n / parts), extra = n % parts;
  let start = 0;
  return range(0, parts).map((i) => {
    const size = base + (i < extra ? 1 : 0);
    start += size;
    return [start - size, start];
  });
}

/** frames -> {xy: Float32Array(64 * 66 * 2), mask: Float32Array(64), length}. */
export function toWindow(frames) {
  const left = frames.filter((f) => detected(f, LEFT_HAND));
  const right = frames.filter((f) => detected(f, RIGHT_HAND));
  const leftDominant = left.length >= right.length;
  const idxs = leftDominant ? LEFT_DOMINANT : RIGHT_DOMINANT;
  const clip = (leftDominant ? left : right).map((frame) => {
    const out = new Float32Array(N_COLS * 2);
    idxs.forEach((src, col) => {
      const x = frame[src * 2];
      // Right-dominant clips are mirrored: hand and arm, not the lips.
      out[col * 2] = leftDominant || col < LIPS.length ? x : 1 - x;
      out[col * 2 + 1] = frame[src * 2 + 1];
    });
    return out;
  });

  const spans = clip.length > INPUT_SIZE ? chunks(clip.length, INPUT_SIZE) : clip.map((_, i) => [i, i + 1]);
  const xy = new Float32Array(INPUT_SIZE * N_COLS * 2);
  spans.forEach(([start, stop], t) => {
    for (let v = 0; v < N_COLS * 2; v++) {
      let sum = 0, count = 0;
      for (let i = start; i < stop; i++) {
        if (!Number.isNaN(clip[i][v])) { sum += clip[i][v]; count++; }
      }
      xy[t * N_COLS * 2 + v] = count ? sum / count : 0;
    }
  });
  const mask = new Float32Array(INPUT_SIZE);
  mask.fill(1, 0, spans.length);
  return {xy, mask, length: spans.length};
}

/** A sign is a run of frames with a hand in view; push() returns its frames once the hand is gone. */
export class SignSegmenter {
  constructor({gap = 8, minFrames = 4, maxFrames = 4 * INPUT_SIZE} = {}) {
    Object.assign(this, {gap, minFrames, maxFrames});
    this.reset();
  }

  reset() {
    this.frames = [];
    this.handFrames = 0;
    this.missing = 0;
  }

  push(frame) {
    const hand = detected(frame, LEFT_HAND) || detected(frame, RIGHT_HAND);
    if (!hand && !this.frames.length) return null;
    this.frames = [...this.frames, frame];
    this.handFrames += hand ? 1 : 0;
    this.missing = hand ? 0 : this.missing + 1;
    if (this.missing < this.gap && this.frames.length < this.maxFrames) return null;
    const {frames, handFrames} = this;
    this.reset();
    return handFrames >= this.minFrames ? frames : null;
  }
}
