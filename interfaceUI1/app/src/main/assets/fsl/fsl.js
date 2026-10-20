/*
 * fsl.js - offline text -> sign sequencing, no dependencies.
 *
 *   FSL.translate(text, pack, {spell: "names"})  -> { items, playlist, skipped, coverage, gloss, isQuestion }
 *   new FSL.Player(stageElement).play(playlist, {rate, fadeMs, onSign, onTick, onEnd, onError})
 *
 * The matcher is a line-by-line port of fsl_t2s/matcher.py + text.py. All rules (contractions,
 * irregular verbs, dropped words, number words) come from the pack, so Python and JS cannot drift.
 * tests/parity.js checks both give identical results.
 */
(function (root) {
  "use strict";

  // ------------------------------------------------------------------ text
  function tokenize(text, rules) {
    text = text.replace(/[\u2018\u2019]/g, "'");
    const tokens = [];
    const re = /[A-Za-z0-9']+/g;
    let prevEnd = 0, m;
    while ((m = re.exec(text)) !== null) {
      const word = m[0].replace(/^'+|'+$/g, "");
      if (!word) continue;
      const gap = text.slice(prevEnd, m.index);
      const sentenceStart = tokens.length === 0 || /[.!?]/.test(gap);
      prevEnd = m.index + m[0].length;
      const low = word.toLowerCase();
      const cap = /[A-Z]/.test(word[0]) && !sentenceStart && low !== "i";
      const parts = Object.prototype.hasOwnProperty.call(rules.contractions, low)
        ? rules.contractions[low] : null;
      if (parts) {
        parts.split(" ").forEach(p => tokens.push({ text: p, cap: false }));
        continue;
      }
      let t = low;
      if (t.endsWith("'s")) t = t.slice(0, -2);
      t = t.replace(/'/g, "");
      if (t) tokens.push({ text: t, cap });
    }
    return { tokens, isQuestion: text.trim().endsWith("?") };
  }

  function stemCandidates(w) {
    const c = [];
    if (w.length > 4 && w.endsWith("ies")) c.push(w.slice(0, -3) + "y");
    if (w.length > 4 && w.endsWith("ied")) c.push(w.slice(0, -3) + "y");
    if (w.length > 3 && w.endsWith("es")) c.push(w.slice(0, -2));
    if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) c.push(w.slice(0, -1));
    if (w.length > 4 && w.endsWith("ing")) {
      const b = w.slice(0, -3);
      c.push(b, b + "e");
      if (b.length > 2 && b[b.length - 1] === b[b.length - 2]) c.push(b.slice(0, -1));
    }
    if (w.length > 4 && w.endsWith("ed")) {
      c.push(w.slice(0, -2), w.slice(0, -1));
      if (w.length > 5 && w[w.length - 3] === w[w.length - 4]) c.push(w.slice(0, -3));
    }
    if (w.length > 4 && w.endsWith("er")) c.push(w.slice(0, -2), w.slice(0, -1));
    if (w.length > 5 && w.endsWith("ly")) c.push(w.slice(0, -2));
    const seen = new Set(), out = [];
    for (const x of c) if (x && !seen.has(x)) { seen.add(x); out.push(x); }
    return out;
  }

  // ------------------------------------------------------------------ matcher
  const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);
  // a text form matches only if it maps to a gloss whose clip is actually in the pack
  const has = (pack, form) => own(pack.index, form) && own(pack.signs, pack.index[form]);

  function spellWord(word, pack) {
    const seq = [], missing = new Set();
    for (const ch of word) {
      const isDigit = /[0-9]/.test(ch);
      const table = isDigit ? pack.digits : pack.letters;
      if (own(table, ch)) seq.push([isDigit ? "digit" : "letter", ch]);
      else missing.add(ch);
    }
    if (missing.size) return { seq: [], why: "no fingerspelling clip for " + [...missing].sort().join(",") };
    return { seq, why: "" };
  }

  function single(tok, pack, spell, drop) {
    const w = tok.text, rules = pack.rules;
    if (has(pack, w)) {
      const g = pack.index[w];
      return { kind: "sign", text: w, gloss: g, match: pack.canon[g] !== w ? "alias" : "exact" };
    }
    if (own(rules.numbers, w) && has(pack, rules.numbers[w])) {
      return { kind: "sign", text: w, gloss: pack.index[rules.numbers[w]], match: "number" };
    }
    if (drop.has(w)) return { kind: "dropped", text: w };
    const base = own(rules.irregular, w) ? rules.irregular[w] : null;
    if (base && has(pack, base)) {
      return { kind: "sign", text: w, gloss: pack.index[base], match: "irregular" };
    }
    for (const cand of stemCandidates(w)) {
      if (has(pack, cand)) return { kind: "sign", text: w, gloss: pack.index[cand], match: "stem" };
    }
    const wants = (spell === "all" && /^[a-z]+$/.test(w)) ||
                  ((spell === "names" || spell === "all") && (tok.cap || /^[0-9]+$/.test(w)));
    if (wants) {
      const { seq, why } = spellWord(w, pack);
      if (seq.length) return { kind: "spell", text: w, letters: seq };
      return { kind: "skipped", text: w, reason: why };
    }
    return { kind: "skipped", text: w, reason: "not in lexicon" };
  }

  function translate(text, pack, opts) {
    const spell = (opts && opts.spell) || "names";
    const drop = new Set(pack.rules.drop);
    const { tokens, isQuestion } = tokenize(text, pack.rules);
    const items = [];
    let i = 0;
    while (i < tokens.length) {
      let done = false;
      for (let n = Math.min(pack.max_ngram, tokens.length - i); n > 1; n--) {
        const phrase = tokens.slice(i, i + n).map(t => t.text).join(" ");
        if (has(pack, phrase)) {
          items.push({ kind: "sign", text: phrase, gloss: pack.index[phrase], match: "phrase" });
          i += n; done = true; break;
        }
      }
      if (done) continue;
      items.push(single(tokens[i], pack, spell, drop));
      i += 1;
    }
    return buildResult(text, items, isQuestion, pack);
  }

  function buildResult(text, items, isQuestion, pack) {
    const playlist = [];
    for (const it of items) {
      if (it.kind === "sign") {
        const s = pack.signs[it.gloss];
        playlist.push({ label: it.gloss, file: s.file, duration: s.duration, source: it.text });
      } else if (it.kind === "spell") {
        for (const [kind, ch] of it.letters) {
          const s = (kind === "digit" ? pack.digits : pack.letters)[ch];
          playlist.push({ label: ch.toUpperCase(), file: s.file, duration: s.duration, source: it.text });
        }
      }
    }
    const counted = items.filter(i => i.kind !== "dropped");
    const covered = counted.filter(i => i.kind === "sign" || i.kind === "spell").length;
    return {
      text, items, playlist, isQuestion,
      skipped: items.filter(i => i.kind === "skipped").map(i => i.text),
      coverage: counted.length ? covered / counted.length : 1,
      gloss: summary(items),
    };
  }

  function summary(items) {
    const parts = [];
    for (const i of items) {
      if (i.kind === "sign") parts.push(i.gloss.toUpperCase() + (i.match === "exact" ? "" : ` (${i.match})`));
      else if (i.kind === "spell") parts.push("[" + i.letters.map(l => l[1].toUpperCase()).join("-") + "]");
      else if (i.kind === "skipped") parts.push(`<<${i.text}: ${i.reason}>>`);
    }
    return parts.join(" ");
  }

  /** Single sign by gloss (for the "all signs" list). */
  function single_sign(gloss, pack) {
    const s = pack.signs[gloss];
    return s ? [{ label: gloss, file: s.file, duration: s.duration, source: gloss }] : [];
  }

  // ------------------------------------------------------------------ player
  /**
   * Plays a playlist back to back with two stacked <video> elements: the next clip is
   * preloaded in the hidden one and faded in over the last fadeMs of the current clip.
   * Every clip starts/ends at the signer's rest pose, so the crossfade is nearly invisible.
   */
  class Player {
    constructor(stage, baseUrl) {
      this.videos = [...stage.querySelectorAll("video")];
      this.base = baseUrl || "";
      this.token = 0;
      this.rate = 1;
      this.videos.forEach(v => { v.muted = true; v.playsInline = true; v.preload = "auto"; });
    }

    setRate(r) {
      this.rate = r;
      this.videos.forEach(v => { v.playbackRate = r; });
    }

    stop() {
      this.token++;
      this.videos.forEach(v => { try { v.pause(); } catch (e) { /* ignore */ } });
    }

    async play(playlist, cb) {
      cb = cb || {};
      this.stop();
      const token = this.token;
      const alive = () => token === this.token;
      const fade = Math.max(0, cb.fadeMs == null ? 130 : cb.fadeMs);
      if (!playlist.length) return;

      const [A, B] = this.videos;
      for (const v of this.videos) { v.style.transition = "none"; v.style.opacity = "0"; v.style.zIndex = "1"; }
      let slot = 0;
      const load = (v, item) => new Promise(resolve => {
        const done = ok => { v.removeEventListener("canplay", onOk); v.removeEventListener("error", onErr); resolve(ok); };
        const onOk = () => done(true), onErr = () => done(false);
        v.addEventListener("canplay", onOk, { once: true });
        v.addEventListener("error", onErr, { once: true });
        v.src = this.base + item.file;
        v.load();
      });

      let pending = load(A, playlist[0]);
      for (let i = 0; i < playlist.length; i++) {
        const v = this.videos[slot], nv = this.videos[1 - slot];
        const ok = await pending;
        if (!alive()) return;
        if (!ok) {                                   // missing/broken file: report and skip it
          if (cb.onError) cb.onError(playlist[i], i);
          if (i + 1 < playlist.length) { nv.pause(); pending = load(nv, playlist[i + 1]); slot = 1 - slot; continue; }
          break;
        }
        const last = i === playlist.length - 1;
        // Preload the next clip into the other element, but only once that element is free
        // (not while it is still fading out as the outgoing clip).
        const preloadNext = () => { pending = last ? null : load(nv, playlist[i + 1]); };

        v.playbackRate = this.rate;
        if (i === 0 || fade === 0) {                 // first clip, or no crossfade: hard cut
          v.currentTime = 0;
          v.style.transition = "none"; v.style.zIndex = "3"; v.style.opacity = "1";
          for (const o of this.videos) if (o !== v) { o.pause(); o.style.opacity = "0"; o.style.zIndex = "1"; }
          preloadNext();
          try { await v.play(); } catch (e) { /* autoplay blocked: user gesture required */ }
        } else {                                     // crossfade in over the outgoing clip
          const prev = nv;
          v.currentTime = 0;
          v.style.zIndex = "3";
          v.style.transition = `opacity ${fade}ms linear`;
          v.style.opacity = "1";
          try { await v.play(); } catch (e) { /* ignore */ }
          prev.style.zIndex = "2";
          setTimeout(() => {
            if (!alive()) return;
            prev.pause(); prev.style.transition = "none"; prev.style.opacity = "0"; prev.style.zIndex = "1";
            preloadNext();
          }, fade + 30);
        }
        if (cb.onSign) cb.onSign(playlist[i], i);

        // wait until it is time to hand over to the next clip (or the clip ends)
        await new Promise(resolve => {
          const lead = last ? 0 : (fade / 1000) * this.rate;
          const check = () => {
            if (!alive()) return resolve();
            const dur = v.duration || playlist[i].duration;
            if (cb.onTick) cb.onTick(i, Math.min(1, v.currentTime / dur));
            if (v.ended || v.currentTime >= dur - lead - 0.02) return resolve();
            requestAnimationFrame(check);
          };
          v.addEventListener("ended", () => resolve(), { once: true });
          check();
        });
        if (!alive()) return;
        if (cb.onTick) cb.onTick(i, 1);
        slot = 1 - slot;
      }
      if (alive() && cb.onEnd) cb.onEnd();
    }
  }

  const api = { tokenize, stemCandidates, translate, summary, single_sign, Player };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FSL = api;
})(typeof window !== "undefined" ? window : globalThis);
