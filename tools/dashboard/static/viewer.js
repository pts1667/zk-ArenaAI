// The replay viewer: a recorded game (tools/policy_replay.py's JSON) drawn on its map, units as Zero-K's icons in their side's
// colour, moving smoothly between the once-a-second frames.  Opened by app.js from the hash:
//   #replay=/replay/DIR/FILE&t=SECONDS[&zoom=Z&at=X,Z&speed=S&play=1&orders=0]
(function () {
  'use strict';
  const { fmt, esc, token } = window.Charts;
  const TINT = ['#4b9bff', '#ff7a33'];          // the sides on the map: bright enough for any map picture
  const ORDER_NAMES = { 0: 'keep', 1: 'move', 2: 'fight', 3: 'attack unit', 4: 'attack ground', 5: 'stop', 6: 'manual fire', 7: 'jump',
    8: 'cloak on', 9: 'cloak off', 10: 'build', 11: 'assist', 12: 'repair', 13: 'reclaim', 14: 'produce', 16: 'throw', 17: 'push',
    18: 'pull', 19: 'place beacon', 20: 'wait at beacon' };
  const CLASSES = [['move', '#46d65e', 'move, jump'], ['fight', '#b57cff', 'fight'], ['attack', '#ff4b4b', 'attack'],
    ['build', '#ffd43b', 'build'], ['assist', '#45c8ff', 'assist, repair'], ['reclaim', '#ff80d2', 'reclaim']];
  const ORDER_CLASS = { 1: 0, 7: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 2: 1, 3: 2, 4: 2, 6: 2, 10: 3, 11: 4, 12: 4, 13: 5 };
  const SPEEDS = [1, 2, 4, 8, 16, 32];
  const DEATH = 2.5;                              // seconds a death stays marked
  const clock = (s) => { s = Math.max(0, Math.floor(s)); return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`; };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* no storage */ } },
  };
  const canvas = (w, h) => Object.assign(document.createElement('canvas'), { width: w, height: h });

  // Icons: each type loaded once, tinted per side into offscreen canvases (64 px, and 28 px for small draws) -------------------
  const icons = new Map();
  let iconReady = () => {};
  function iconEntry(type) {
    let e = icons.get(type);
    if (e) return e.alias || e;
    e = { ready: false, big: [], small: [], url: [] };
    icons.set(type, e);
    const img = new Image();
    img.onload = () => { try { prepareIcon(e, img); e.ready = true; } catch (err) { e.failed = true; } iconReady(); };
    img.onerror = () => { if (type !== 'default') e.alias = iconEntry('default'); iconReady(); };
    img.src = `/icon/${encodeURIComponent(type)}.png`;
    return e;
  }
  function prepareIcon(e, img) {
    const n = 64, base = canvas(n, n), g = base.getContext('2d', { willReadFrequently: true });
    const w = img.naturalWidth, h = img.naturalHeight, k = n / Math.max(w, h);
    g.imageSmoothingQuality = 'high';
    g.drawImage(img, (n - w * k) / 2, (n - h * k) / 2, w * k, h * k);
    const d = g.getImageData(0, 0, n, n), p = d.data;
    let clear = 0;
    for (let i = 3; i < p.length; i += 4) if (p[i] < 128) clear++;
    if (clear < n * n * 0.03) {                   // drawn opaque on black (the mex icon): read its dark as transparent
      for (let i = 0; i < p.length; i += 4) p[i + 3] = Math.min(255, Math.max(p[i], p[i + 1], p[i + 2]) * 2);
      g.putImageData(d, 0, 0);
    }
    for (const side of [0, 1]) {
      const c = canvas(n, n), t = c.getContext('2d');
      t.drawImage(base, 0, 0);
      t.globalCompositeOperation = 'multiply';    // white takes the side's colour, black stays black
      t.fillStyle = TINT[side];
      t.fillRect(0, 0, n, n);
      t.globalCompositeOperation = 'destination-in';
      t.drawImage(base, 0, 0);
      e.big[side] = c;
      const s = canvas(28, 28), u = s.getContext('2d');
      u.imageSmoothingQuality = 'high';
      u.drawImage(c, 0, 0, 28, 28);
      e.small[side] = s;
    }
  }
  function iconURL(type, side) {
    const e = iconEntry(type);
    if (!e.ready) return null;
    return e.url[side] || (e.url[side] = e.small[side].toDataURL());
  }

  // Loading -------------------------------------------------------------------------------------------------------------------
  let unitsPromise = null;
  const units = () => unitsPromise || (unitsPromise = fetch('/api/units').then((r) => (r.ok ? r.json() : {})).catch(() => ({})));
  const image = (src) => new Promise((ok, fail) => { const i = new Image(); i.onload = () => ok(i); i.onerror = fail; i.src = src; });

  async function load(url) {
    const [json, catalog] = await Promise.all([fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    }), units()]);
    let map = null, img = null;
    try {
      const r = await fetch('/api/map/' + encodeURIComponent(json.map.slug));
      if (r.ok) map = await r.json();
      if (map && map.image) img = await image(map.image).catch(() => null);
    } catch (e) { /* no picture: a plain ground */ }
    return prepare(json, catalog, map, img);
  }

  function prepare(json, catalog, map, img) {
    const F = {};
    json.unit_fields.forEach((k, i) => { F[k] = i; });
    const NF = json.unit_fields.length;
    const S = {};
    json.side_fields.forEach((k, i) => { S[k] = i; });
    const frames = json.frames.map((f) => {
      const u = Int32Array.from(f.units);
      const index = new Map();
      for (let o = 0; o < u.length; o += NF) index.set(u[o + F.uid], o);
      return { t: f.t, u, index, sides: f.sides, features: f.features ? Float32Array.from(f.features) : null };
    });
    json.frames = null;                           // kept as typed arrays above
    const N = frames.length;
    const defs = [];
    let defaultDef = null;
    const info = (id, name, cost) => {
      const c = catalog[name] || {};
      const air = !!c.air, mobile = c.mobile ?? true;
      return { id, name, cost: cost || c.cost || 0, human: c.human || name, desc: c.description || '', icon: c.icon || 'default',
        size: c.icon_size || 1.3, air, mobile, layer: air ? 2 : mobile ? 1 : 0 };
    };
    for (const [id, d] of Object.entries(json.defs || {})) defs[+id] = info(+id, d.name, d.cost);
    defaultDef = info(-1, 'unknown', 0);
    const deaths = [new Float32Array(0)];
    let rows = 0, featAt = new Int32Array(N), lastFeat = 0;
    for (let f = 0; f < N; f++) {
      const A = frames[f];
      rows = Math.max(rows, A.u.length / NF);
      if (A.features) lastFeat = f;
      featAt[f] = lastFeat;
      if (!f) continue;
      const P = frames[f - 1], out = [];
      for (let o = 0; o < P.u.length; o += NF) {
        if (!A.index.has(P.u[o + F.uid])) out.push(P.u[o + F.x], P.u[o + F.z], P.u[o + F.side], P.u[o + F.def]);
      }
      deaths.push(Float32Array.from(out));
    }
    const series = (field) => [0, 1].map((s) => Float32Array.from(frames, (f) => (f.sides[s] ? f.sides[s][S[field]] : 0)));
    const times = Float64Array.from(frames, (f) => f.t);
    for (const d of defs) if (d) iconEntry(d.icon);
    return {
      json, F, NF, S, frames, N, times, defs, defaultDef, deaths, featAt, maxRows: rows, img, map,
      w: json.map.width || (map && map.width) || 8192, h: json.map.height || (map && map.height) || 8192,
      end: times[N - 1] || 0,
      income: series('metal_income'), value: series('value'), units: series('units'),
      frameAt(t) {
        let lo = 0, hi = N - 1;
        if (t >= times[hi]) return hi;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; if (times[m] <= t) lo = m; else hi = m; }
        return lo;
      },
    };
  }

  // The viewer ------------------------------------------------------------------------------------------------------------------
  class Viewer {
    constructor(root) {
      this.root = root;
      this.R = null;
      this.url = null;
      this.token = 0;
      this.t = 0;
      this.playing = false;
      this.speed = store.get('viewer:speed', 4);
      this.opts = Object.assign({ orders: true, health: true, wrecks: true, spots: true, icons: 1 }, store.get('viewer:opts', {}));
      this.zoom = 1; this.cx = 0; this.cz = 0;
      this.hover = -1; this.selected = -1;
      this.frameShown = -1;
      root.innerHTML = `
<div class="v-top"><span class="v-title">Replay</span><span class="v-result"></span><span class="spacer"></span>
  <span class="dim small v-keys">space play · ←/→ 5 s · [ ] speed · O orders · F fit · Esc close</span>
  <button type="button" class="v-close" title="Close (Esc)">Close</button></div>
<div class="v-mid">
  <div class="v-stage"><canvas></canvas><div class="v-tip" hidden></div>
    <div class="v-zoom"><button type="button" data-z="in" title="Zoom in">+</button><button type="button" data-z="out" title="Zoom out">−</button><button type="button" data-z="fit" title="Fit the map (F)">⤢</button></div>
    <div class="v-hint">wheel zoom · drag pan · click a unit for its details</div><div class="v-loading">Loading…</div></div>
  <div class="v-side">
    ${[0, 1].map((s) => `<div class="v-sidecard" data-side="${s}"><div class="v-sidehead"><i class="sw" style="background:${TINT[s]}"></i><span class="name"></span><span class="won"></span></div>
      <div class="v-sub"></div><div class="v-stats"></div><div class="v-comp"></div></div>`).join('')}
    ${[['income', 'metal income'], ['value', 'army value'], ['units', 'units']].map(([k, label]) =>
      `<div class="v-mini" data-key="${k}"><div class="t-label"><span>${label}</span><span class="cur"></span></div><canvas></canvas></div>`).join('')}
    <div class="v-toggles">
      <label><input type="checkbox" data-opt="orders">orders</label><label><input type="checkbox" data-opt="health">health</label>
      <label><input type="checkbox" data-opt="wrecks">wrecks</label><label><input type="checkbox" data-opt="spots">metal spots</label>
      <label>icons <select class="v-sel" data-opt="icons"><option value="0.7">small</option><option value="1">medium</option><option value="1.4">large</option></select></label>
    </div>
    <div class="v-legend">${CLASSES.map((c) => `<span><i style="background:${c[1]}"></i>${c[2]}</span>`).join('')}</div>
    <div class="v-unit"></div>
  </div>
</div>
<div class="v-bottom"><button type="button" class="v-play" title="Play or pause (space)">▶</button>
  <div class="v-speeds">${SPEEDS.map((s) => `<button type="button" data-speed="${s}">${s}×</button>`).join('')}</div>
  <span class="v-time">0:00 / 0:00</span><canvas class="v-timeline"></canvas></div>`;
      const q = (s) => root.querySelector(s);
      this.stage = q('.v-stage');
      this.canvas = q('.v-stage canvas');
      this.g = this.canvas.getContext('2d');
      this.tip = q('.v-tip');
      this.loading = q('.v-loading');
      this.timeline = q('.v-timeline');
      this.timeText = q('.v-time');
      this.playButton = q('.v-play');
      this.unitBox = q('.v-unit');
      this.minis = Array.from(root.querySelectorAll('.v-mini')).map((el) => ({ el, key: el.dataset.key, canvas: el.querySelector('canvas'), cur: el.querySelector('.cur') }));
      this.sideCards = Array.from(root.querySelectorAll('.v-sidecard'));
      this.bind();
      iconReady = () => { this.dirty = true; this.compKey = null; this.frameShown = -1; };
    }

    bind() {
      const root = this.root;
      root.querySelector('.v-close').addEventListener('click', () => this.dismiss());
      this.playButton.addEventListener('click', () => this.setPlaying(!this.playing));
      root.querySelector('.v-speeds').addEventListener('click', (e) => { const b = e.target.closest('button[data-speed]'); if (b) this.setSpeed(+b.dataset.speed); });
      for (const input of root.querySelectorAll('[data-opt]')) {
        const k = input.dataset.opt;
        if (input.type === 'checkbox') input.checked = !!this.opts[k]; else input.value = String(this.opts[k]);
        input.addEventListener('change', () => {
          this.opts[k] = input.type === 'checkbox' ? input.checked : +input.value;
          store.set('viewer:opts', this.opts);
          this.dirty = true;
        });
      }
      root.querySelector('.v-zoom').addEventListener('click', (e) => {
        const b = e.target.closest('button');
        if (!b) return;
        const W = this.stage.clientWidth, H = this.stage.clientHeight;
        if (b.dataset.z === 'fit') this.fit(); else this.zoomAt(W / 2, H / 2, b.dataset.z === 'in' ? 1.6 : 1 / 1.6);
      });
      const st = this.stage;
      st.addEventListener('wheel', (e) => {
        e.preventDefault();
        const r = st.getBoundingClientRect();
        this.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)));
      }, { passive: false });
      let drag = null;
      st.addEventListener('pointerdown', (e) => {
        if (e.button !== 0 || e.target.closest('.v-zoom')) return;
        drag = { x: e.clientX, y: e.clientY, cx: this.cx, cz: this.cz, moved: false };
        st.setPointerCapture(e.pointerId);
      });
      st.addEventListener('pointermove', (e) => {
        const r = st.getBoundingClientRect();
        this.mouse = { x: e.clientX - r.left, y: e.clientY - r.top };
        if (drag) {
          const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
          if (Math.abs(dx) + Math.abs(dy) > 3) { drag.moved = true; st.classList.add('dragging'); }
          if (drag.moved && this.tf) { this.cx = drag.cx - dx / this.tf.s; this.cz = drag.cz - dy / this.tf.s; this.clampView(); }
        }
        this.dirty = true;
      });
      st.addEventListener('pointerup', (e) => {
        if (drag && !drag.moved && !e.target.closest('.v-zoom')) {
          const k = this.pick(this.mouse);
          this.selected = k >= 0 ? this.cur.uid[k] : -1;
          this.frameShown = -1;
        }
        drag = null;
        st.classList.remove('dragging');
        this.dirty = true;
      });
      st.addEventListener('pointerleave', () => { this.mouse = null; this.dirty = true; });
      st.addEventListener('dblclick', (e) => { const r = st.getBoundingClientRect(); this.zoomAt(e.clientX - r.left, e.clientY - r.top, 2); });
      this.scrubber(this.timeline);
      for (const m of this.minis) this.scrubber(m.canvas);
      document.addEventListener('keydown', (e) => {
        if (!this.active || e.target.closest('input, select, textarea') || e.ctrlKey || e.metaKey || e.altKey) return;
        const k = e.key;
        if (k === ' ') this.setPlaying(!this.playing);
        else if (k === 'ArrowLeft' || k === 'ArrowRight') this.seek(this.t + (k === 'ArrowLeft' ? -1 : 1) * (e.shiftKey ? 30 : 5));
        else if (k === ']' || k === '+' || k === '=') this.setSpeed(SPEEDS[Math.min(SPEEDS.length - 1, SPEEDS.indexOf(this.speed) + 1)]);
        else if (k === '[' || k === '-') this.setSpeed(SPEEDS[Math.max(0, SPEEDS.indexOf(this.speed) - 1)]);
        else if (k === 'o' || k === 'O') this.toggle('orders');
        else if (k === 'h' || k === 'H') this.toggle('health');
        else if (k === 'f' || k === 'F') this.fit();
        else if (k === 'Escape') this.dismiss();
        else if (k === 'Home') this.seek(0);
        else if (k === 'End') this.seek(Infinity);
        else return;
        e.preventDefault();
      });
      new ResizeObserver(() => { this.dirty = true; this.statics = null; }).observe(this.stage);
      new ResizeObserver(() => { this.statics = null; this.dirty = true; }).observe(this.timeline);
      window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { this.statics = null; this.dirty = true; });
    }

    scrubber(el) {
      let down = false;
      const at = (e) => {
        const r = el.getBoundingClientRect();
        if (this.R) this.seek(clamp((e.clientX - r.left) / r.width, 0, 1) * this.R.end);
      };
      el.addEventListener('pointerdown', (e) => { down = true; el.setPointerCapture(e.pointerId); at(e); });
      el.addEventListener('pointermove', (e) => { if (down) at(e); });
      el.addEventListener('pointerup', () => { down = false; });
    }

    toggle(k) {
      this.opts[k] = !this.opts[k];
      const input = this.root.querySelector(`[data-opt="${k}"]`);
      if (input) input.checked = this.opts[k];
      store.set('viewer:opts', this.opts);
      this.dirty = true;
    }

    async open(url, params) {
      this.active = true;
      this.root.hidden = false;
      document.documentElement.style.overflow = 'hidden';
      this.startLoop();
      if (url === this.url && this.R) { this.apply(params); return; }
      this.url = url;
      this.R = null;
      const token = ++this.token;
      this.loading.hidden = false;
      this.loading.textContent = 'Loading replay…';
      this.root.querySelector('.v-title').textContent = url.split('/').slice(-2).join(' / ');
      try {
        const R = await load(url);
        if (token !== this.token) return;
        this.R = R;
        this.loading.hidden = true;
        this.setup();
        this.apply(params);
      } catch (e) {
        if (token === this.token) this.loading.textContent = 'Could not load this replay: ' + e.message;
      }
    }

    close() {
      if (!this.active) return;
      this.active = false;
      this.setPlaying(false, true);
      this.root.hidden = true;
      this.tip.hidden = true;
      document.documentElement.style.overflow = '';
      cancelAnimationFrame(this.raf);
      this.raf = 0;
    }

    dismiss() {     // closed by the viewer's own button or Esc: drop the hash without a new history entry
      history.replaceState(null, '', location.pathname + location.search);
      this.close();
    }

    apply(p) {
      if (!p) return;
      if (p.has('speed')) this.setSpeed(+p.get('speed') || this.speed);
      if (p.has('orders')) { this.opts.orders = p.get('orders') !== '0'; const i = this.root.querySelector('[data-opt="orders"]'); if (i) i.checked = this.opts.orders; }
      this.fit();
      if (p.has('zoom')) this.zoom = clamp(+p.get('zoom') || 1, 0.5, 80);
      if (p.has('at')) { const [x, z] = p.get('at').split(',').map(Number); if (Number.isFinite(x) && Number.isFinite(z)) { this.cx = x; this.cz = z; } }
      this.seek(p.has('t') ? +p.get('t') || 0 : 0, true);
      this.setPlaying(p.get('play') === '1', true);
    }

    setup() {
      const R = this.R, j = R.json;
      const winner = j.result ? j.result.winner : null;
      const info = j.info || {};
      const extra = [info.iteration != null ? `iteration ${info.iteration}` : '', info.hours != null ? `${(+info.hours).toFixed(1)} h of PPO` : ''].filter(Boolean).join(' · ');
      const mapName = (R.map && R.map.name) || j.map.slug;
      this.root.querySelector('.v-title').textContent = `${j.label || j.stage} · ${mapName}${extra ? ' · ' + extra : ''}`;
      const how = j.result ? `${j.result.how} at ${clock(j.result.seconds)}` : '';
      this.root.querySelector('.v-result').innerHTML = winner == null
        ? `<span class="badge">draw</span> <span class="dim">${esc(how)}</span>`
        : `<span class="badge"><i class="sw" style="background:${TINT[winner]};border-radius:50%"></i>${esc(j.sides[winner].name)} won</span> <span class="dim">${esc(how)}</span>`;
      unitsPromise.then((catalog) => {
        this.sideCards.forEach((card, s) => {
          const side = j.sides[s] || {};
          const fac = catalog[side.factory];
          card.querySelector('.name').textContent = `${side.name || 'side ' + s}`;
          card.querySelector('.won').textContent = winner === s ? 'won' : '';
          card.querySelector('.v-sub').textContent = `side ${s} · ${side.variant || '?'} · ${fac ? fac.human : side.factory || '?'}`;
        });
      });
      this.statics = null;
      this.compKey = null;
      this.frameShown = -1;
      this.selected = -1;
      const cap = R.maxRows * 2 + 8;
      this.cur = { n: 0, uid: new Int32Array(cap), sx: new Float32Array(cap), sy: new Float32Array(cap), px: new Float32Array(cap),
        alpha: new Float32Array(cap), src: new Uint8Array(cap), off: new Int32Array(cap), at: new Map() };
      this.fit();
    }

    // The view: a zoom over the fitted map and the world point at the canvas's centre.
    fit() { if (!this.R) return; this.zoom = 1; this.cx = this.R.w / 2; this.cz = this.R.h / 2; this.dirty = true; }
    clampView() { const R = this.R; this.cx = clamp(this.cx, 0, R.w); this.cz = clamp(this.cz, 0, R.h); }
    zoomAt(mx, my, factor) {
      if (!this.tf) return;
      const { s, ox, oy } = this.tf;
      const wx = (mx - ox) / s, wz = (my - oy) / s;
      this.zoom = clamp(this.zoom * factor, 0.6, 80);
      const s2 = this.tf.s0 * this.zoom;
      this.cx = wx - (mx - this.stage.clientWidth / 2) / s2;
      this.cz = wz - (my - this.stage.clientHeight / 2) / s2;
      this.clampView();
      this.dirty = true;
    }

    seek(t, quiet) {
      if (!this.R) return;
      this.t = clamp(t, 0, this.R.end);
      this.dirty = true;
      if (!quiet && !this.playing) this.remember();
    }
    remember() {
      clearTimeout(this.hashTimer);
      this.hashTimer = setTimeout(() => {
        if (this.active && this.url) history.replaceState(null, '', `#replay=${this.url}&t=${Math.round(this.t)}`);
      }, 300);
    }
    setPlaying(on, quiet) {
      if (on && this.R && this.t >= this.R.end) this.t = 0;
      this.playing = !!on && !!this.R;
      this.playButton.textContent = this.playing ? '❚❚' : '▶';
      if (!this.playing && !quiet) this.remember();
    }
    setSpeed(s) {
      this.speed = SPEEDS.includes(s) ? s : 4;
      store.set('viewer:speed', this.speed);
      for (const b of this.root.querySelectorAll('.v-speeds button')) b.setAttribute('aria-pressed', String(+b.dataset.speed === this.speed));
    }

    startLoop() {
      if (this.raf) return;
      this.setSpeed(this.speed);
      let last = 0;
      const step = (now) => {
        if (!this.active) { this.raf = 0; return; }
        this.raf = requestAnimationFrame(step);
        const dt = last ? Math.min(0.25, (now - last) / 1000) : 0;
        last = now;
        if (!this.R) return;
        if (this.playing) {
          this.t += dt * this.speed;
          if (this.t >= this.R.end) { this.t = this.R.end; this.setPlaying(false); }
          this.dirty = true;
        }
        if (this.dirty) { this.dirty = false; this.render(); }
      };
      this.raf = requestAnimationFrame(step);
    }

    // Drawing ------------------------------------------------------------------------------------------------------------------
    render() {
      const R = this.R, F = R.F, NF = R.NF, g = this.g, c = this.canvas;
      const W = this.stage.clientWidth, H = this.stage.clientHeight, dpr = window.devicePixelRatio || 1;
      if (!W || !H) return;
      if (c.width !== Math.round(W * dpr) || c.height !== Math.round(H * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.globalAlpha = 1;
      g.fillStyle = '#15161a';
      g.fillRect(0, 0, W, H);
      const s0 = Math.min(W / R.w, H / R.h) * 0.97, s = s0 * this.zoom, ox = W / 2 - this.cx * s, oy = H / 2 - this.cz * s;
      this.tf = { s, s0, ox, oy };
      g.imageSmoothingEnabled = true;
      g.imageSmoothingQuality = 'high';
      if (R.img) g.drawImage(R.img, ox, oy, R.w * s, R.h * s);
      else {
        g.fillStyle = '#3a3d36';
        g.fillRect(ox, oy, R.w * s, R.h * s);
        g.strokeStyle = 'rgba(255,255,255,0.07)';
        g.lineWidth = 1;
        g.beginPath();
        for (let x = 512; x < R.w; x += 512) { g.moveTo(ox + x * s, oy); g.lineTo(ox + x * s, oy + R.h * s); }
        for (let z = 512; z < R.h; z += 512) { g.moveTo(ox, oy + z * s); g.lineTo(ox + R.w * s, oy + z * s); }
        g.stroke();
      }
      g.strokeStyle = 'rgba(0,0,0,0.7)';
      g.strokeRect(ox - 0.5, oy - 0.5, R.w * s + 1, R.h * s + 1);

      const t = this.t, i = R.frameAt(t), j = Math.min(i + 1, R.N - 1), A = R.frames[i], B = R.frames[j];
      const a = j > i ? clamp((t - A.t) / (B.t - A.t), 0, 1) : 0;
      const zoom = this.zoom, base = Math.min(W, H) / 125 * this.opts.icons * Math.sqrt(zoom);
      const sizeOf = (d) => clamp(d.size * base, 5, 80);

      if (this.opts.spots && R.json.spots) {
        const r = Math.max(2.2, 26 * s);
        g.lineWidth = 1.2;
        g.strokeStyle = 'rgba(255,255,255,0.8)';
        g.fillStyle = 'rgba(10,10,10,0.35)';
        g.beginPath();
        for (const [x, z] of R.json.spots) { const X = ox + x * s, Y = oy + z * s; g.moveTo(X + r, Y); g.arc(X, Y, r, 0, 7); }
        g.fill();
        g.stroke();
      }
      if (this.opts.wrecks) {
        const f = R.frames[R.featAt[i]].features;
        if (f && f.length) {
          g.fillStyle = 'rgba(214,190,140,0.85)';
          g.strokeStyle = 'rgba(30,24,12,0.8)';
          g.lineWidth = 1;
          g.beginPath();
          const k = Math.sqrt(zoom) * Math.min(W, H) / 900;
          for (let m = 0; m < f.length; m += 3) {
            const X = ox + f[m] * s, Y = oy + f[m + 1] * s, r = clamp(Math.sqrt(f[m + 2]) * 0.35 * k, 1.6, 9);
            g.moveTo(X, Y - r); g.lineTo(X + r, Y); g.lineTo(X, Y + r); g.lineTo(X - r, Y); g.closePath();
          }
          g.fill();
          g.stroke();
        }
      }

      // Where each unit is now: frame i's units moved toward frame j's, those only in j fading in, those only in i fading out.
      const cur = this.cur;
      cur.at.clear();
      let n = 0;
      const put = (src, o, x, z, alpha) => {
        const row = src ? B.u : A.u, d = R.defs[row[o + F.def]] || R.defaultDef;
        cur.uid[n] = row[o + F.uid]; cur.sx[n] = ox + x * s; cur.sy[n] = oy + z * s; cur.px[n] = sizeOf(d);
        cur.alpha[n] = alpha; cur.src[n] = src; cur.off[n] = o;
        cur.at.set(cur.uid[n], n);
        n++;
      };
      for (let o = 0; o < A.u.length; o += NF) {
        const ob = j === i ? o : B.index.get(A.u[o + F.uid]);
        if (ob === undefined) put(0, o, A.u[o + F.x], A.u[o + F.z], 1 - a);
        else put(0, o, A.u[o + F.x] + (B.u[ob + F.x] - A.u[o + F.x]) * a, A.u[o + F.z] + (B.u[ob + F.z] - A.u[o + F.z]) * a, 1);
      }
      if (j > i && a > 0) for (let o = 0; o < B.u.length; o += NF) if (!A.index.has(B.u[o + F.uid])) put(1, o, B.u[o + F.x], B.u[o + F.z], a);
      cur.n = n;
      const rowOf = (k) => (cur.src[k] ? B.u : A.u);
      const hoverK = this.mouse && !this.root.querySelector('.v-stage.dragging') ? this.pick(this.mouse) : -1;
      this.hover = hoverK >= 0 ? cur.uid[hoverK] : -1;

      // Orders, under the units: one path per kind of order.
      const paths = CLASSES.map(() => new Path2D()), ends = CLASSES.map(() => new Path2D());
      let any = false;
      for (let k = 0; k < n; k++) {
        const row = rowOf(k), o = cur.off[k], cls = ORDER_CLASS[row[o + F.order]];
        if (cls === undefined) continue;
        const uid = cur.uid[k];
        if (!this.opts.orders && uid !== this.hover && uid !== this.selected) continue;
        let X, Y;
        const tx = row[o + F.tx], tz = row[o + F.tz], target = row[o + F.target];
        if (tx > 0 || tz > 0) { X = ox + tx * s; Y = oy + tz * s; }      // -1 is no point; so is 0, 0 (a near-zero point, rounded)
        else if (target >= 0 && cur.at.has(target)) { const m = cur.at.get(target); X = cur.sx[m]; Y = cur.sy[m]; }
        else continue;
        paths[cls].moveTo(cur.sx[k], cur.sy[k]);
        paths[cls].lineTo(X, Y);
        ends[cls].moveTo(X + 2, Y);
        ends[cls].arc(X, Y, 2, 0, 7);
        any = true;
      }
      if (any) {
        g.lineWidth = 1;
        g.globalAlpha = 0.45;
        CLASSES.forEach((c, ci) => { g.strokeStyle = c[1]; g.fillStyle = c[1]; g.stroke(paths[ci]); g.fill(ends[ci]); });
        g.globalAlpha = 1;
      }

      // Units: structures, then ground, then air.
      for (let layer = 0; layer < 3; layer++) {
        for (let k = 0; k < n; k++) {
          const row = rowOf(k), o = cur.off[k], d = R.defs[row[o + F.def]] || R.defaultDef;
          if (d.layer !== layer) continue;
          const X = cur.sx[k], Y = cur.sy[k], px = cur.px[k];
          if (X < -px || Y < -px || X > W + px || Y > H + px) continue;
          const side = row[o + F.side] ? 1 : 0, hp = row[o + F.hp], build = row[o + F.build], flags = row[o + F.flags];
          let alpha = cur.alpha[k];
          if (build < 100) alpha *= 0.38;
          if (flags & 1) alpha *= 0.55;
          g.globalAlpha = alpha;
          const e = iconEntry(d.icon);
          if (e.ready) g.drawImage(px <= 24 ? e.small[side] : e.big[side], X - px / 2, Y - px / 2, px, px);
          else { g.fillStyle = TINT[side]; g.beginPath(); g.arc(X, Y, px * 0.3, 0, 7); g.fill(); }
          if (flags & 2) { g.strokeStyle = '#9fe8ff'; g.lineWidth = 1.5; g.beginPath(); g.arc(X, Y, px * 0.55, 0, 7); g.stroke(); }
          g.globalAlpha = cur.alpha[k];
          if (px >= 7 && (build < 100 || (this.opts.health && hp < 100))) {
            const bw = Math.max(6, px * 0.8), bh = Math.max(2, Math.min(4, px * 0.12)), bx = X - bw / 2, by = Y + px * 0.48;
            g.fillStyle = 'rgba(0,0,0,0.7)';
            g.fillRect(bx - 0.5, by - 0.5, bw + 1, bh + 1);
            const frac = build < 100 ? build / 100 : hp / 100;
            g.fillStyle = build < 100 ? '#e6e6e6' : hp > 60 ? '#4cd964' : hp > 30 ? '#ffcc00' : '#ff3b30';
            g.fillRect(bx, by, bw * frac, bh);
          }
        }
      }
      g.globalAlpha = 1;

      // Deaths: a cross where a unit was last seen, fading over DEATH seconds.
      for (let f = Math.max(1, i - Math.ceil(DEATH) - 1); f <= i; f++) {
        const age = t - R.frames[f].t;
        if (age < 0 || age >= DEATH) continue;
        const list = R.deaths[f], q = age / DEATH;
        for (let m = 0; m < list.length; m += 4) {
          const d = R.defs[list[m + 3]] || R.defaultDef, X = ox + list[m] * s, Y = oy + list[m + 1] * s, r = sizeOf(d) * 0.32 * (1 + q * 0.6);
          g.globalAlpha = 1 - q;
          g.beginPath();
          g.moveTo(X - r, Y - r); g.lineTo(X + r, Y + r); g.moveTo(X + r, Y - r); g.lineTo(X - r, Y + r);
          g.strokeStyle = 'rgba(0,0,0,0.75)'; g.lineWidth = 4; g.stroke();
          g.strokeStyle = TINT[list[m + 2] ? 1 : 0]; g.lineWidth = 2; g.stroke();
          g.beginPath(); g.arc(X, Y, r * 1.5 + q * r, 0, 7); g.lineWidth = 1; g.stroke();
        }
      }
      g.globalAlpha = 1;

      for (const [uid, color, width] of [[this.selected, '#ffffff', 2], [this.hover, 'rgba(255,255,255,0.75)', 1.5]]) {
        if (uid < 0 || !cur.at.has(uid)) continue;
        const k = cur.at.get(uid);
        g.strokeStyle = color; g.lineWidth = width;
        g.beginPath(); g.arc(cur.sx[k], cur.sy[k], cur.px[k] * 0.62 + 3, 0, 7); g.stroke();
      }

      this.tooltip(hoverK, i);
      if (i !== this.frameShown) { this.frameShown = i; this.panel(i); }
      this.drawCursors();
      this.timeText.textContent = `${clock(t)} / ${clock(R.end)}`;
    }

    pick(m) {
      if (!m || !this.cur) return -1;
      const cur = this.cur;
      let best = -1, bestD = Infinity;
      for (let k = 0; k < cur.n; k++) {
        const dx = cur.sx[k] - m.x, dy = cur.sy[k] - m.y, r = cur.px[k] * 0.5 + 3, d = dx * dx + dy * dy;
        if (d <= r * r && d <= bestD) { best = k; bestD = d; }
      }
      return best;
    }

    describe(k) {
      const R = this.R, F = R.F, cur = this.cur, row = cur.src[k] ? R.frames[Math.min(R.frameAt(this.t) + 1, R.N - 1)].u : R.frames[R.frameAt(this.t)].u, o = cur.off[k];
      const d = R.defs[row[o + F.def]] || R.defaultDef, side = row[o + F.side], kind = row[o + F.order];
      const flags = row[o + F.flags], build = row[o + F.build];
      let order = kind >= 0 ? ORDER_NAMES[kind] || `order ${kind}` : 'no order';
      const tx = row[o + F.tx], tz = row[o + F.tz], target = row[o + F.target];
      if (kind >= 0 && (tx > 0 || tz > 0)) order += ` → ${tx}, ${tz}`;
      else if (kind >= 0 && target >= 0) {
        const m = cur.at.get(target);
        if (m !== undefined) {
          const trow = cur.src[m] ? R.frames[Math.min(R.frameAt(this.t) + 1, R.N - 1)].u : R.frames[R.frameAt(this.t)].u;
          const td = R.defs[trow[cur.off[m] + F.def]] || R.defaultDef;
          order += ` → ${td.human} #${target}`;
        } else order += ` → #${target}`;
      }
      const sideName = (R.json.sides[side] || {}).name || `side ${side}`;
      return `<b>${esc(d.human)}</b> <span class="d">${esc(d.desc)}</span><br>` +
        `<span style="color:${TINT[side ? 1 : 0]}">●</span> ${esc(sideName)} <span class="d">#${cur.uid[k]} · ${fmt(d.cost)} metal</span><br>` +
        `hp ${row[o + F.hp]}%${build < 100 ? ` · built ${build}%` : ''}${flags & 1 ? ' · cloaked' : ''}${flags & 2 ? ' · paralyzed' : ''}<br>` +
        `<span class="d">order:</span> ${esc(order)}`;
    }

    tooltip(k, i) {
      if (k < 0 || !this.mouse) { this.tip.hidden = true; this.tipKey = null; return; }
      const key = `${this.cur.uid[k]}:${i}`;
      if (key !== this.tipKey) { this.tipKey = key; this.tip.innerHTML = this.describe(k); }
      this.tip.hidden = false;
      const W = this.stage.clientWidth, H = this.stage.clientHeight, tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
      const x = this.mouse.x + 14 + tw > W ? this.mouse.x - 14 - tw : this.mouse.x + 14;
      const y = this.mouse.y + 14 + th > H ? this.mouse.y - 14 - th : this.mouse.y + 14;
      this.tip.style.transform = `translate(${Math.round(x)}px, ${Math.round(y)}px)`;
      this.tip.style.left = '0';
      this.tip.style.top = '0';
    }

    // The side panel at frame i: stats, what each side has, the selected unit.
    panel(i) {
      const R = this.R, F = R.F, NF = R.NF, A = R.frames[i];
      this.sideCards.forEach((card, s) => {
        const v = A.sides[s] || [];
        const stat = (k, label) => `<div class="v-stat"><div class="k">${label}</div><div class="v">${fmt(v[R.S[k]])}</div></div>`;
        card.querySelector('.v-stats').innerHTML = stat('metal', 'metal') + stat('metal_income', 'm income') + stat('energy_income', 'e income') +
          stat('value', 'value') + stat('units', 'units');
      });
      const key = i + ':' + icons.size;
      if (key !== this.compKey) {
        this.compKey = key;
        [0, 1].forEach((s) => {
          const count = new Map();
          let structures = 0;
          for (let o = 0; o < A.u.length; o += NF) {
            if (A.u[o + F.side] !== s || A.u[o + F.build] < 100) continue;
            const d = R.defs[A.u[o + F.def]] || R.defaultDef;
            if (d.layer === 0) { structures++; continue; }
            count.set(d, (count.get(d) || 0) + 1);
          }
          const list = [...count].sort((x, y) => y[1] * y[0].cost - x[1] * x[0].cost);
          const shown = list.slice(0, 10).map(([d, c]) => {
            const url = iconURL(d.icon, s);
            return `<span title="${c} × ${esc(d.human)}${d.desc ? ' (' + esc(d.desc) + ')' : ''}">${url ? `<img src="${url}" alt="">` : esc(d.human)}${c}</span>`;
          }).join('');
          const more = list.length > 10 ? `<span class="more">+${list.length - 10} kinds</span>` : '';
          this.sideCards[s].querySelector('.v-comp').innerHTML = shown + more + `<span class="more">${structures} structures</span>`;
        });
      }
      if (this.selected >= 0 && this.cur && this.cur.at.has(this.selected)) this.unitBox.innerHTML = this.describe(this.cur.at.get(this.selected));
      else this.unitBox.innerHTML = '';
    }

    // The timeline and the small charts: their curves drawn once to an offscreen canvas, the cursor on top each time.
    drawStatic(el, draw) {
      const dpr = window.devicePixelRatio || 1, w = el.clientWidth, h = el.clientHeight;
      const off = canvas(Math.max(1, Math.round(w * dpr)), Math.max(1, Math.round(h * dpr)));
      const g = off.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      draw(g, w, h);
      return { off, w, h };
    }
    curves(g, w, h, pair, pad) {
      const R = this.R, top = Math.max(1e-9, ...pair.map((a) => a.reduce((m, v) => Math.max(m, v), 0))) * 1.08;
      const x = (k) => (R.times[k] / (R.end || 1)) * w, y = (v) => h - pad - (v / top) * (h - 2 * pad);
      pair.forEach((arr, s) => {
        g.strokeStyle = TINT[s];
        g.lineWidth = 1.5;
        g.lineJoin = 'round';
        g.beginPath();
        for (let k = 0; k < arr.length; k++) (k ? g.lineTo(x(k), y(arr[k])) : g.moveTo(x(k), y(arr[k])));
        g.stroke();
      });
      return top;
    }
    makeStatics() {
      const R = this.R, hair = token('--hair'), ink = token('--text-light');
      const font = getComputedStyle(document.body).fontFamily;
      this.statics = {
        timeline: this.drawStatic(this.timeline, (g, w, h) => {
          g.fillStyle = hair;
          g.fillRect(0, h - 14, w, 1);
          g.font = `10px ${font}`;
          g.fillStyle = ink;
          g.textBaseline = 'bottom';
          const every = [60, 120, 300, 600].find((m) => (R.end / m) * 46 < w) || 600;
          for (let m = 0; m <= R.end; m += every) {
            const x = (m / (R.end || 1)) * w;
            g.fillRect(x, h - 14, 1, 4);
            g.textAlign = m === 0 ? 'left' : 'center';
            if (x < w - 14) g.fillText(clock(m), x, h);
          }
          g.save();
          g.beginPath(); g.rect(0, 0, w, h - 15); g.clip();
          this.curves(g, w, h - 14, R.value, 3);
          g.restore();
        }),
      };
      for (const m of this.minis) {
        this.statics[m.key] = this.drawStatic(m.canvas, (g, w, h) => {
          g.fillStyle = hair;
          g.fillRect(0, h - 1, w, 1);
          const top = this.curves(g, w, h, R[m.key], 2);
          g.font = `10px ${font}`;
          g.fillStyle = ink;
          g.textBaseline = 'top';
          g.fillText(fmt(top / 1.08), 2, 1);
        });
      }
    }
    drawCursors() {
      if (!this.statics) this.makeStatics();
      const R = this.R, i = R.frameAt(this.t), dpr = window.devicePixelRatio || 1;
      const ink = token('--text');
      const draw = (el, st, values) => {
        if (el.width !== st.off.width || el.height !== st.off.height) { el.width = st.off.width; el.height = st.off.height; }
        const g = el.getContext('2d');
        g.setTransform(1, 0, 0, 1, 0, 0);
        g.clearRect(0, 0, el.width, el.height);
        g.drawImage(st.off, 0, 0);
        g.setTransform(dpr, 0, 0, dpr, 0, 0);
        const x = Math.round((this.t / (R.end || 1)) * st.w) + 0.5;
        g.strokeStyle = ink;
        g.lineWidth = 1;
        g.beginPath(); g.moveTo(x, 0); g.lineTo(x, st.h); g.stroke();
        g.fillStyle = ink;
        g.beginPath(); g.moveTo(x - 4, 0); g.lineTo(x + 4, 0); g.lineTo(x, 5); g.fill();
        return values;
      };
      draw(this.timeline, this.statics.timeline);
      for (const m of this.minis) {
        draw(m.canvas, this.statics[m.key]);
        const v = R[m.key];
        m.cur.innerHTML = [0, 1].map((s) => `<span style="color:${TINT[s]}">●</span> ${fmt(v[s][i])}`).join('&nbsp; ');
      }
    }
  }

  let viewer = null;
  const get = () => viewer || (viewer = new Viewer(document.getElementById('viewer')));
  window.Viewer = {
    open(url, params) { return get().open(url, params); },
    close() { if (viewer) viewer.close(); },
    get instance() { return viewer; },
  };
})();
