// Small canvas line charts for the dashboard: a few series against one x, a hover crosshair with every series' value, legend
// chips that hide a series.  Colours come from the page's CSS variables, so light and dark follow the stylesheet.
(function () {
  'use strict';

  const root = document.documentElement;
  const token = (name) => getComputedStyle(root).getPropertyValue(name).trim();
  const SERIES = ['--s1', '--s2', '--s3', '--s4', '--s5', '--s6', '--s7', '--s8'];

  function fmt(v) {
    if (v == null || !Number.isFinite(v)) return '–';
    const a = Math.abs(v);
    if (a >= 1e9) return (v / 1e9).toFixed(2) + 'B';
    if (a >= 1e6) return (v / 1e6).toFixed(2) + 'M';
    if (a >= 1e5) return (v / 1e3).toFixed(0) + 'K';
    if (a >= 1e4) return (v / 1e3).toFixed(1) + 'K';
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a === 0) return '0';
    if (a >= 0.001) return v.toPrecision(3);
    return v.toExponential(1);
  }

  // A tick's label: as many decimals as its step needs, compact past 10K.
  function tickFmt(v, step) {
    if (Math.abs(v) >= 1e4) return fmt(v);
    const mag = Math.floor(Math.log10(step) + 1e-9);
    const dec = Math.max(0, -mag + (Math.abs(step / Math.pow(10, mag) - 2.5) < 1e-6 ? 1 : 0));
    return v.toFixed(Math.min(dec, 8));
  }

  // Round tick values: steps of 1, 2 or 5 times a power of ten.
  function ticks(lo, hi, want) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / Math.max(1, want);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }

  class LineChart {
    // opts: {title, x, series: [key...], y: [lo, hi] (a floor the data can widen), onRemove, xLabel}
    constructor(host, opts) {
      this.host = host;
      this.opts = opts;
      this.rows = [];
      this.hidden = new Set();
      host.classList.add('chart');
      host.innerHTML = '<div class="chart-head"><span class="chart-title"></span><span class="chart-legend"></span>' +
        '<button type="button" class="chart-x" title="Remove this chart">×</button></div>' +
        '<div class="chart-plot"><canvas></canvas><div class="chart-tip" hidden></div></div>';
      this.canvas = host.querySelector('canvas');
      this.tip = host.querySelector('.chart-tip');
      this.plot = host.querySelector('.chart-plot');
      host.querySelector('.chart-title').textContent = opts.title || opts.series.join(', ');
      const x = host.querySelector('.chart-x');
      if (opts.onRemove) x.addEventListener('click', () => opts.onRemove(this)); else x.remove();
      this.legend();
      this.ro = new ResizeObserver(() => this.draw());
      this.ro.observe(this.plot);
      this.canvas.addEventListener('mousemove', (e) => this.hover(e));
      this.canvas.addEventListener('mouseleave', () => { this.hoverX = null; this.tip.hidden = true; this.draw(); });
      LineChart.all.add(this);
    }

    legend() {
      const box = this.host.querySelector('.chart-legend');
      box.innerHTML = '';
      if (this.opts.series.length < 2) return;
      this.opts.series.forEach((key, i) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'lg' + (this.hidden.has(key) ? ' off' : '');
        b.innerHTML = `<i style="background:var(${SERIES[i % 8]})"></i>`;
        b.append(key);
        b.title = 'Show or hide ' + key;
        b.addEventListener('click', () => {
          this.hidden.has(key) ? this.hidden.delete(key) : this.hidden.add(key);
          b.classList.toggle('off');
          this.draw();
        });
        box.append(b);
      });
    }

    destroy() { this.ro.disconnect(); LineChart.all.delete(this); this.host.remove(); }

    setData(rows) {
      const xk = this.opts.x;
      this.rows = rows;
      this.lines = this.opts.series.map((key) => {
        const xs = [], ys = [];
        for (const r of rows) {
          const x = r[xk], y = r[key];
          if (typeof x === 'number' && typeof y === 'number' && Number.isFinite(y)) { xs.push(x); ys.push(y); }
        }
        return { key, xs, ys };
      });
      this.draw();
    }

    frame() {
      const w = this.plot.clientWidth, h = this.plot.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
        this.canvas.width = Math.round(w * dpr);
        this.canvas.height = Math.round(h * dpr);
      }
      const g = this.canvas.getContext('2d');
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, w, h);
      return { g, w, h };
    }

    draw() {
      if (!this.plot.isConnected) return;
      const { g, w, h } = this.frame();
      if (w < 20 || h < 20) return;
      const font = getComputedStyle(document.body).fontFamily;
      const ink = token('--text-light'), hair = token('--hair'), surface = token('--bg');
      g.font = `11px ${font}`;
      const lines = (this.lines || []).filter((l) => !this.hidden.has(l.key) && l.xs.length);
      if (!lines.length) {
        g.fillStyle = ink;
        g.textAlign = 'center';
        g.fillText(this.lines && this.lines.length ? 'no values for these keys yet' : 'loading…', w / 2, h / 2);
        return;
      }
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const l of lines) {
        x0 = Math.min(x0, l.xs[0]); x1 = Math.max(x1, l.xs[l.xs.length - 1]);
        for (const y of l.ys) { if (y < y0) y0 = y; if (y > y1) y1 = y; }
      }
      if (this.opts.y) { y0 = Math.min(y0, this.opts.y[0]); y1 = Math.max(y1, this.opts.y[1]); }
      else { const pad = (y1 - y0) * 0.06 || Math.abs(y1) * 0.1 || 1; y0 -= pad; y1 += pad; }
      if (x1 === x0) { const d = Math.abs(x0) * 0.1 || 1; x0 -= d; x1 += d; }
      const yt = ticks(y0, y1, Math.max(2, Math.floor(h / 40)));
      if (!this.opts.y) { y0 = Math.min(y0, yt[0]); y1 = Math.max(y1, yt[yt.length - 1]); }
      const ystep = yt.length > 1 ? yt[1] - yt[0] : 1;
      const labels = yt.map((v) => tickFmt(v, ystep));
      const left = 8 + Math.max(...labels.map((s) => g.measureText(s).width)), right = 10, top = 6, bottom = 18;
      const pw = w - left - right, ph = h - top - bottom;
      const sx = (x) => left + (x - x0) / (x1 - x0) * pw;
      const sy = (y) => top + (1 - (y - y0) / (y1 - y0)) * ph;
      this.geom = { sx, x0, x1, left, pw, top, ph, sy };

      g.lineWidth = 1;
      g.strokeStyle = hair;
      g.fillStyle = ink;
      g.textAlign = 'right';
      g.textBaseline = 'middle';
      yt.forEach((v, i) => {
        const y = Math.round(sy(v)) + 0.5;
        if (y < top - 1 || y > top + ph + 1) return;
        g.beginPath(); g.moveTo(left, y); g.lineTo(left + pw, y); g.stroke();
        g.fillText(labels[i], left - 5, y);
      });
      g.textAlign = 'center';
      g.textBaseline = 'top';
      const xt = ticks(x0, x1, Math.max(2, Math.floor(pw / 80)));
      const xstep = xt.length > 1 ? xt[1] - xt[0] : 1;
      for (const v of xt) {
        const x = sx(v);
        if (x < left - 1 || x > left + pw + 1) continue;
        const label = tickFmt(v, xstep), half = g.measureText(label).width / 2;
        g.fillText(label, Math.min(Math.max(x, half), w - half - 1), top + ph + 4);
      }

      g.save();
      g.beginPath(); g.rect(left, top - 2, pw, ph + 4); g.clip();
      g.lineJoin = 'round';
      g.lineCap = 'round';
      const n = Math.max(1, this.rows.length);
      for (const l of lines) {
        const i = this.opts.series.indexOf(l.key);
        const color = token(SERIES[i % 8]);
        g.strokeStyle = color;
        g.lineWidth = 2;
        g.beginPath();
        l.xs.forEach((x, k) => (k ? g.lineTo(sx(x), sy(l.ys[k])) : g.moveTo(sx(x), sy(l.ys[k]))));
        g.stroke();
        if (l.xs.length <= 40 || l.xs.length < n / 3) {   // a few points, or a series only some rows have: mark each
          g.fillStyle = color;
          g.strokeStyle = surface;
          g.lineWidth = 1.5;
          l.xs.forEach((x, k) => { g.beginPath(); g.arc(sx(x), sy(l.ys[k]), 2.6, 0, 7); g.fill(); g.stroke(); });
        }
      }
      g.restore();
      if (this.hoverX != null) this.drawHover(g, lines);
    }

    nearest(l, x) {
      let lo = 0, hi = l.xs.length - 1;
      while (hi - lo > 1) { const m = (lo + hi) >> 1; if (l.xs[m] <= x) lo = m; else hi = m; }
      return Math.abs(l.xs[hi] - x) < Math.abs(l.xs[lo] - x) ? hi : lo;
    }

    drawHover(g, lines) {
      const { sx, sy, top, ph } = this.geom;
      const surface = token('--bg');
      // snap to the nearest x any visible series has
      let best = null;
      for (const l of lines) { const k = this.nearest(l, this.hoverX); if (best == null || Math.abs(l.xs[k] - this.hoverX) < Math.abs(best - this.hoverX)) best = l.xs[k]; }
      const x = Math.round(sx(best)) + 0.5;
      g.strokeStyle = token('--text-light');
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(x, top); g.lineTo(x, top + ph); g.stroke();
      const rows = [];
      for (const l of lines) {
        const k = this.nearest(l, best);
        const i = this.opts.series.indexOf(l.key);
        const color = token(SERIES[i % 8]);
        g.fillStyle = color; g.strokeStyle = surface; g.lineWidth = 2;
        g.beginPath(); g.arc(sx(l.xs[k]), sy(l.ys[k]), 4, 0, 7); g.fill(); g.stroke();
        const note = l.xs[k] !== best ? ` <span class="tx">@${fmt(l.xs[k])}</span>` : '';
        rows.push(`<div class="row"><i style="background:${color}"></i>${esc(l.key)}${note}<span class="v">${fmt(l.ys[k])}</span></div>`);
      }
      this.tip.innerHTML = `<div class="tx">${esc(this.opts.x)} ${fmt(best)}</div>` + rows.join('');
      this.tip.hidden = false;
      const pw = this.plot.clientWidth, tw = this.tip.offsetWidth;
      this.tip.style.left = (x + 12 + tw > pw ? x - 12 - tw : x + 12) + 'px';
      this.tip.style.top = '4px';
    }

    hover(e) {
      if (!this.geom) return;
      const r = this.canvas.getBoundingClientRect();
      const { left, pw, x0, x1 } = this.geom;
      const px = Math.min(Math.max(e.clientX - r.left, left), left + pw);
      this.hoverX = x0 + (px - left) / pw * (x1 - x0);
      this.draw();
    }
  }
  LineChart.all = new Set();

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  }

  // Redraw everything when the colour scheme flips.
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => LineChart.all.forEach((c) => c.draw()));

  window.Charts = { LineChart, fmt, tickFmt, ticks, esc, SERIES, token };
})();
