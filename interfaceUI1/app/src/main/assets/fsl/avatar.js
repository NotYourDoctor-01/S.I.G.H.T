/*
 * avatar.js - draws and animates the signing character from the motion pack. No dependencies.
 *
 *   const player = new FSLAvatar.Player(canvas, FSL_PACK, FSL_MOTION);
 *   player.play(playlist, {onSign, onTick, onEnd, onError});   // playlist from FSL.translate(...)
 *
 * Binary layout per sign: all poses, then all hands, then face (block order, not interleaved).
 * Coordinates are "shoulder widths": origin = shoulder centre, x right, y DOWN (same orientation as
 * the original footage). Motion is stored per sign as base64: for each frame 13 body points (x,y) and
 * 2 x 21 hand points (x,y) as int16 (value / scale), then 10 face-expression bytes (/255).
 */
(function (root) {
  "use strict";
  const N_POSE = 13, N_HAND = 21, FACE_N = 10;
  const P = { NOSE: 0, LEYE: 1, REYE: 2, LEAR: 3, REAR: 4, LSH: 5, RSH: 6, LEL: 7, REL: 8, LWR: 9, RWR: 10, LHIP: 11, RHIP: 12 };
  const F = { BROW_DOWN: 0, BROW_INNER: 1, BROW_OUTER: 2, BLINK: 3, WIDE: 4, JAW: 5, SMILE: 6, FROWN: 7, FUNNEL: 8, PUCKER: 9 };
  const POSE_LEN = N_POSE * 2, HANDS_LEN = 2 * N_HAND * 2, FRAME_LEN = POSE_LEN + HANDS_LEN;

  const TONES = ["#f2cba5", "#dba679", "#b97c50", "#8c5a3a", "#5d3a26"];
  const STYLE = { shirt: "#101c40", shirtLine: "#1d2d62", hair: "#1b1a22", eye: "#1b1a22", lip: "#9a3f3a",
                  skinDark: (c) => shade(c, -0.22) };

  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const f = (v) => Math.max(0, Math.min(255, Math.round(v + 255 * amt)));
    return "rgb(" + f(n >> 16) + "," + f((n >> 8) & 255) + "," + f(n & 255) + ")";
  }

  // ------------------------------------------------------------------ decoding
  function decode(b64, frames, scale) {
    const bin = atob(b64), n = bin.length, u8 = new Uint8Array(n);
    for (let i = 0; i < n; i++) u8[i] = bin.charCodeAt(i);
    // layout: [all poses (T x 26 int16)] [all hands (T x 84 int16)] [face (T x 10 bytes)]
    const ints = new Int16Array(u8.buffer, 0, frames * FRAME_LEN);       // little-endian hosts
    const pose = new Float32Array(frames * POSE_LEN), hands = new Float32Array(frames * HANDS_LEN);
    for (let i = 0; i < frames * POSE_LEN; i++) pose[i] = ints[i] / scale;
    for (let i = 0; i < frames * HANDS_LEN; i++) hands[i] = ints[frames * POSE_LEN + i] / scale;
    const face = new Float32Array(frames * FACE_N), off = frames * FRAME_LEN * 2;
    for (let i = 0; i < frames * FACE_N; i++) face[i] = u8[off + i] / 255;
    return { frames, pose, hands, face };
  }

  // ------------------------------------------------------------------ sequencing
  /** Join the signs of a playlist into one frame stream, blending joints between signs. */
  function buildSequence(playlist, pack, motion, blend, cache) {
    const items = [];
    for (const it of playlist) {
      const meta = pack.signs[it.label];
      if (!meta || !motion[it.label]) { items.push({ it, err: true }); continue; }
      let d = cache[it.label];
      if (!d) d = cache[it.label] = decode(motion[it.label], meta.frames, pack.avatar.scale);
      items.push({ it, d });
    }
    const ok = items.filter(x => !x.err);
    let T = 0;
    ok.forEach((x, k) => { T += x.d.frames + (k ? blend : 0); });
    const pose = new Float32Array(T * POSE_LEN), hands = new Float32Array(T * HANDS_LEN), face = new Float32Array(T * FACE_N);
    const marks = [];
    let t = 0, prev = null;
    ok.forEach((x, k) => {
      const d = x.d;
      if (prev) {
        for (let b = 1; b <= blend; b++, t++) {
          const a = b / (blend + 1);
          lerpInto(pose, t * POSE_LEN, prev.pose, (prev.frames - 1) * POSE_LEN, d.pose, 0, POSE_LEN, a);
          lerpInto(hands, t * HANDS_LEN, prev.hands, (prev.frames - 1) * HANDS_LEN, d.hands, 0, HANDS_LEN, a);
          lerpInto(face, t * FACE_N, prev.face, (prev.frames - 1) * FACE_N, d.face, 0, FACE_N, a);
        }
      }
      pose.set(d.pose, t * POSE_LEN); hands.set(d.hands, t * HANDS_LEN); face.set(d.face, t * FACE_N);
      marks.push({ item: x.it, start: t, end: t + d.frames });
      t += d.frames; prev = d;
    });
    const skipped = items.filter(x => x.err).map(x => x.it);
    return { T, pose, hands, face, marks, skipped };
  }
  function lerpInto(out, o, a, ao, b, bo, n, w) { for (let i = 0; i < n; i++) out[o + i] = a[ao + i] * (1 - w) + b[bo + i] * w; }

  // ------------------------------------------------------------------ drawing
  const VIEW_TOP = -1.2, VIEW_H = 2.75;                  // shoulder widths shown vertically

  function drawFrame(ctx, W, H, fr, skin) {
    const S = H / VIEW_H, cx = W / 2;
    const X = (x) => cx + x * S, Y = (y) => (y - VIEW_TOP) * S;
    const pt = (i) => [fr.pose[i * 2], fr.pose[i * 2 + 1]];
    const dark = STYLE.skinDark(skin);
    ctx.clearRect(0, 0, W, H);
    ctx.lineCap = "round"; ctx.lineJoin = "round";

    const sL = pt(P.LSH), sR = pt(P.RSH), eL = pt(P.LEL), eR = pt(P.REL), wL = pt(P.LWR), wR = pt(P.RWR);
    const limb = (a, b, w, col) => {
      ctx.strokeStyle = col; ctx.lineWidth = w * S;
      ctx.beginPath(); ctx.moveTo(X(a[0]), Y(a[1])); ctx.lineTo(X(b[0]), Y(b[1])); ctx.stroke();
    };

    // neck, torso
    const shMidX = (sL[0] + sR[0]) / 2, shMidY = (sL[1] + sR[1]) / 2;
    ctx.fillStyle = dark; ctx.strokeStyle = dark;
    limb([shMidX, shMidY - 0.45], [shMidX, shMidY - 0.05], 0.2, dark);
    limb([shMidX, shMidY - 0.45], [shMidX, shMidY - 0.05], 0.17, skin);
    ctx.fillStyle = STYLE.shirt;
    ctx.beginPath();
    ctx.moveTo(X(sR[0] - 0.08), Y(sR[1] + 0.08));
    ctx.quadraticCurveTo(X(sR[0] + 0.1), Y(sR[1] - 0.13), X(shMidX - 0.17), Y(shMidY - 0.18));
    ctx.lineTo(X(shMidX + 0.17), Y(shMidY - 0.18));
    ctx.quadraticCurveTo(X(sL[0] - 0.1), Y(sL[1] - 0.13), X(sL[0] + 0.08), Y(sL[1] + 0.08));
    ctx.lineTo(X(sL[0] + 0.03), Y(VIEW_TOP + VIEW_H + 0.2));
    ctx.lineTo(X(sR[0] - 0.03), Y(VIEW_TOP + VIEW_H + 0.2));
    ctx.closePath(); ctx.fill();
    ctx.fillStyle = skin;                                              // neckline
    ctx.beginPath(); ctx.ellipse(X(shMidX), Y(shMidY - 0.18), 0.17 * S, 0.1 * S, 0, 0, Math.PI); ctx.fill();

    // arms: sleeve (upper arm) then bare forearm
    for (const [s, e, w] of [[sR, eR, wR], [sL, eL, wL]]) {
      limb(s, e, 0.3, STYLE.shirt);
      limb(e, w, 0.2, dark);
      limb(e, w, 0.17, skin);
    }

    drawHead(ctx, S, X, Y, fr, skin, dark);
    drawHand(ctx, S, X, Y, fr.hands, 1, skin, dark);                   // person's right = viewer's left
    drawHand(ctx, S, X, Y, fr.hands, 0, skin, dark);
  }

  function drawHead(ctx, S, X, Y, fr, skin, dark) {
    const g = (i) => [fr.pose[i * 2], fr.pose[i * 2 + 1]];
    const nose = g(P.NOSE), eL = g(P.LEYE), eR = g(P.REYE), aL = g(P.LEAR), aR = g(P.REAR);
    const earDist = Math.max(0.2, Math.hypot(aL[0] - aR[0], aL[1] - aR[1]));
    const c = [(aL[0] + aR[0]) / 2, (aL[1] + aR[1]) / 2 - 0.01];
    const roll = Math.atan2(eL[1] - eR[1], eL[0] - eR[0]);
    const yaw = clamp(((nose[0] - (aL[0] + aR[0]) / 2) / earDist) * 1.6, -0.7, 0.7);
    const pitch = clamp(((nose[1] - (eL[1] + eR[1]) / 2) / earDist - 0.28) * 1.6, -0.5, 0.5);
    const rx = 0.27 * S, ry = 0.335 * S, f = fr.face;
    ctx.save();
    ctx.translate(X(c[0]), Y(c[1])); ctx.rotate(roll);
    // hair (behind), ears, face
    ctx.fillStyle = STYLE.hair;
    ctx.beginPath(); ctx.ellipse(0, -0.08 * ry, rx * 1.1, ry * 1.08, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = skin; ctx.strokeStyle = dark; ctx.lineWidth = 0.012 * S;
    for (const sx of [-1, 1]) { ctx.beginPath(); ctx.ellipse(sx * rx * 0.98, 0.05 * ry, rx * 0.12, ry * 0.2, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); }
    ctx.beginPath(); ctx.ellipse(0, 0.07 * ry, rx, ry, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = STYLE.hair;                                          // fringe
    ctx.beginPath(); ctx.ellipse(0, -0.55 * ry, rx * 1.02, ry * 0.5, 0, Math.PI, Math.PI * 2); ctx.fill();

    // features move with the head turn (yaw) and nod (pitch)
    ctx.translate(yaw * rx * 0.55, pitch * ry * 0.45);
    const ink = STYLE.eye;
    const blink = f[F.BLINK], wide = f[F.WIDE];
    const browLift = f[F.BROW_INNER] * 0.9 + f[F.BROW_OUTER] * 0.5, browDrop = f[F.BROW_DOWN];
    for (const sx of [-1, 1]) {
      const ex = sx * rx * 0.42, ey = -0.04 * ry;
      ctx.fillStyle = "#fff";
      const eh = Math.max(0.025, (0.1 + 0.05 * wide) * (1 - 0.92 * blink)) * ry;
      ctx.beginPath(); ctx.ellipse(ex, ey, rx * 0.17, eh, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = ink;
      ctx.beginPath(); ctx.ellipse(ex + yaw * rx * 0.06, ey, rx * 0.075, Math.min(eh, ry * 0.08), 0, 0, Math.PI * 2); ctx.fill();
      // brow: lifted by inner/outer-up; brow-down pulls the INNER end down (furrow)
      const by = -0.30 * ry - browLift * 0.16 * ry;
      const innerY = by + browDrop * 0.09 * ry - f[F.BROW_INNER] * 0.04 * ry;
      const outerY = by - browDrop * 0.02 * ry - f[F.BROW_OUTER] * 0.05 * ry;
      ctx.strokeStyle = STYLE.hair; ctx.lineWidth = 0.035 * S;
      ctx.beginPath();
      ctx.moveTo(ex - sx * rx * 0.2, innerY);
      ctx.lineTo(ex + sx * rx * 0.2, outerY);
      ctx.stroke();
    }
    ctx.strokeStyle = dark; ctx.lineWidth = 0.014 * S;                   // nose
    ctx.beginPath(); ctx.moveTo(0, 0.04 * ry); ctx.quadraticCurveTo(rx * 0.07, 0.24 * ry, -rx * 0.02, 0.27 * ry); ctx.stroke();
    // mouth
    const jaw = f[F.JAW], round = Math.max(f[F.FUNNEL], f[F.PUCKER]), smile = f[F.SMILE], frown = f[F.FROWN];
    const mw = rx * (0.36 - 0.14 * round), my = 0.5 * ry, mh = ry * (0.03 + 0.34 * jaw + 0.12 * round);
    ctx.strokeStyle = STYLE.lip; ctx.fillStyle = "#4a1a1e"; ctx.lineWidth = 0.022 * S;
    ctx.beginPath();
    if (jaw > 0.12 || round > 0.35) {
      ctx.ellipse(0, my + mh * 0.3, mw, mh, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    } else {
      const lift = (smile * 0.18 - frown * 0.12) * ry;
      ctx.moveTo(-mw, my - lift * 0.4); ctx.quadraticCurveTo(0, my + lift, mw, my - lift * 0.4); ctx.stroke();
    }
    ctx.restore();
  }

  const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
  function drawHand(ctx, S, X, Y, hands, slot, skin, dark) {
    const base = slot * N_HAND * 2;
    const p = (k) => [X(hands[base + k * 2]), Y(hands[base + k * 2 + 1])];
    const palmW = Math.hypot(hands[base + 10] - hands[base + 34], hands[base + 11] - hands[base + 35]);   // pts 5 and 17
    const fw = clamp(palmW * 0.3, 0.05, 0.12) * S;
    const palm = [0, 1, 5, 9, 13, 17].map(p);
    const path = (pts) => { ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); };
    // outline pass, then fill pass (so overlapping fingers stay distinguishable)
    for (const [col, extra] of [[dark, 0.022 * S], [skin, 0]]) {
      ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = fw + extra;
      path(palm); ctx.closePath(); ctx.fill(); ctx.lineWidth = (extra ? 0.04 : 0.025) * S; ctx.stroke();
      ctx.lineWidth = fw + extra;
      for (const f of FINGERS) { path(f.map(p)); ctx.stroke(); }
    }
  }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  // ------------------------------------------------------------------ player
  class Player {
    constructor(canvas, pack, motion) {
      this.canvas = canvas; this.ctx = canvas.getContext("2d");
      this.pack = pack; this.motion = motion; this.fps = pack.avatar.fps;
      this.cache = {}; this.token = 0; this.rate = 1; this.skin = TONES[1];
      this.blend = Math.max(2, Math.round(pack.fade_ms / 1000 * this.fps));
      this.tmp = { pose: new Float32Array(POSE_LEN), hands: new Float32Array(HANDS_LEN), face: new Float32Array(FACE_N) };
      this.last = null;
      this._fit();
      if (typeof ResizeObserver !== "undefined") new ResizeObserver(() => { this._fit(); this.redraw(); }).observe(canvas);
    }
    _fit() {
      const dpr = Math.min(2, root.devicePixelRatio || 1), r = this.canvas.getBoundingClientRect();
      this.canvas.width = Math.max(2, Math.round(r.width * dpr)); this.canvas.height = Math.max(2, Math.round(r.height * dpr));
    }
    setRate(r) { this.rate = r; }
    setSkin(c) { this.skin = c; this.redraw(); }
    redraw() { if (this.last) drawFrame(this.ctx, this.canvas.width, this.canvas.height, this.last, this.skin); }
    stop() { this.token++; }

    /** Draw the resting pose (first frame of the first sign) until something is played. */
    showRest() {
      const g = Object.keys(this.pack.signs)[0];
      if (!g) return;
      const d = this.cache[g] || (this.cache[g] = decode(this.motion[g], this.pack.signs[g].frames, this.pack.avatar.scale));
      this.last = { pose: d.pose.subarray(0, POSE_LEN), hands: d.hands.subarray(0, HANDS_LEN), face: d.face.subarray(0, FACE_N) };
      this.redraw();
    }

    /** Stop and draw one frame of a playlist (frame index, fractions allowed). Returns the frame count. */
    seek(playlist, f) {
      this.stop();
      const seq = buildSequence(playlist, this.pack, this.motion, this.blend, this.cache);
      if (!seq.T) return 0;
      f = clamp(f, 0, seq.T - 1);
      const i = Math.floor(f), a = f - i, j = Math.min(seq.T - 1, i + 1), tmp = this.tmp;
      lerpInto(tmp.pose, 0, seq.pose, i * POSE_LEN, seq.pose, j * POSE_LEN, POSE_LEN, a);
      lerpInto(tmp.hands, 0, seq.hands, i * HANDS_LEN, seq.hands, j * HANDS_LEN, HANDS_LEN, a);
      lerpInto(tmp.face, 0, seq.face, i * FACE_N, seq.face, j * FACE_N, FACE_N, a);
      this.last = tmp;
      drawFrame(this.ctx, this.canvas.width, this.canvas.height, tmp, this.skin);
      return seq.T;
    }

    play(playlist, cb) {
      cb = cb || {};
      this.stop();
      const token = this.token;
      const seq = buildSequence(playlist, this.pack, this.motion, this.blend, this.cache);
      seq.skipped.forEach(it => cb.onError && cb.onError(it));
      if (!seq.T) { if (cb.onEnd) cb.onEnd(); return; }
      const fps = this.fps, tmp = this.tmp;
      let cur = -1, t0 = null;
      const step = (now) => {
        if (token !== this.token) return;
        if (t0 === null) t0 = now;
        const f = Math.min(seq.T - 1, ((now - t0) / 1000) * fps * this.rate);
        const i = Math.floor(f), a = f - i, j = Math.min(seq.T - 1, i + 1);
        lerpInto(tmp.pose, 0, seq.pose, i * POSE_LEN, seq.pose, j * POSE_LEN, POSE_LEN, a);
        lerpInto(tmp.hands, 0, seq.hands, i * HANDS_LEN, seq.hands, j * HANDS_LEN, HANDS_LEN, a);
        lerpInto(tmp.face, 0, seq.face, i * FACE_N, seq.face, j * FACE_N, FACE_N, a);
        this.last = tmp;
        drawFrame(this.ctx, this.canvas.width, this.canvas.height, tmp, this.skin);
        // which sign are we in? switch halfway through the blend into it
        let k = 0;
        while (k + 1 < seq.marks.length && f >= seq.marks[k + 1].start - this.blend / 2) k++;
        if (k !== cur) { cur = k; if (cb.onSign) cb.onSign(seq.marks[k].item, k, this.pack.signs[seq.marks[k].item.label]); }
        const m = seq.marks[k];
        if (cb.onTick) cb.onTick(k, clamp((f - m.start) / (m.end - m.start), 0, 1));
        if (f >= seq.T - 1) { if (cb.onTick) cb.onTick(k, 1); if (cb.onEnd) cb.onEnd(); return; }
        root.requestAnimationFrame(step);
      };
      root.requestAnimationFrame(step);
    }
  }

  const api = { Player, decode, buildSequence, drawFrame, TONES, FRAME_LEN, POSE_LEN, HANDS_LEN, FACE_N };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.FSLAvatar = api;
})(typeof window !== "undefined" ? window : globalThis);
