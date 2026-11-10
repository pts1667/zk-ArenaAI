// The dashboard page: the log (appended by HTMX from the `log` events) filtered, capped and kept scrolled; each result card's
// charts (static/charts.js) fed from /api/metrics and refreshed on `results` events; the running stage's clock; and the
// replay viewer (static/viewer.js) opened by the URL's hash: #replay=/replay/DIR/FILE&t=SECONDS.
(function () {
  'use strict';
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const { LineChart, fmt, esc } = window.Charts;
  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { v == null ? localStorage.removeItem(k) : localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* private window */ } },
  };
  const getJSON = (url) => fetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${url}`))));

  function duration(s) {
    s = Math.max(0, Math.round(s));
    if (s < 60) return `${s}s`;
    if (s < 3600) return `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, '0')}s`;
    return `${Math.floor(s / 3600)}h ${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}m`;
  }

  // Events ----------------------------------------------------------------------------------------------------------------------
  const conn = $('#conn');
  const setConn = (state, text) => { conn.className = 'conn ' + state; conn.lastElementChild.textContent = text; };
  const sources = new WeakSet();
  let opened = 0;
  document.body.addEventListener('htmx:sseOpen', (e) => {
    const source = e.detail.source;
    setConn('live', 'live');
    if (sources.has(source)) return;             // the browser reconnected it: the server carries on after Last-Event-ID
    if (opened++) clearLog();                     // a new EventSource: the server sends its backlog again
    sources.add(source);
    source.addEventListener('results', (m) => onResults(m.data));
  });
  document.body.addEventListener('htmx:sseError', () => setConn('lost', 'reconnecting'));

  // The log --------------------------------------------------------------------------------------------------------------------
  const MAX_LINES = 5000;
  const log = $('#log'), chips = $('#log-chips'), find = $('#log-find'), newer = $('#log-new');
  const filterStyle = document.head.appendChild(document.createElement('style'));
  const seen = new Set();
  let stick = true, unread = 0, queued = false, onlyStage = '', problems = false, query = '', autoTop = 0;

  function chip(label, stage, cls) {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'chip' + (cls ? ' ' + cls : '');
    b.textContent = label;
    if (stage != null) b.dataset.stage = stage;
    b.setAttribute('aria-pressed', 'false');
    return b;
  }
  const allChip = chip('all', '');
  const problemChip = chip('⚠ problems', null, 'problems');
  problemChip.title = 'Only lines that look like errors or warnings';
  chips.append(allChip, problemChip);
  chips.addEventListener('click', (e) => {
    const b = e.target.closest('button.chip');
    if (!b) return;
    if (b === problemChip) problems = !problems; else onlyStage = b.dataset.stage;
    applyFilter();
  });
  function applyFilter() {
    let css = '';
    if (onlyStage) css += `#log .line:not([data-stage="${CSS.escape(onlyStage)}"]){display:none}`;
    if (problems) css += '#log .line:not(.err):not(.warn){display:none}';
    filterStyle.textContent = css;
    $$('button.chip[data-stage]', chips).forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.stage === onlyStage)));
    problemChip.setAttribute('aria-pressed', String(problems));
    toBottom();
  }
  function showStage(stage) {
    if (stage && !seen.has(stage)) addStage(stage);
    onlyStage = stage;
    applyFilter();
    $('#log-section').scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  function addStage(stage) {
    seen.add(stage);
    chips.insertBefore(chip(stage, stage), problemChip);
    applyFilter();
  }
  const shown = (el) => (!onlyStage || el.dataset.stage === onlyStage) && (!problems || el.matches('.err, .warn')) && !el.classList.contains('nomatch');
  function toBottom() {
    stick = true;
    unread = 0;
    newer.hidden = true;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => { queued = false; if (stick) { log.scrollTop = log.scrollHeight; autoTop = log.scrollTop; } });
  }
  function clearLog() { log.replaceChildren(); toBottom(); }

  log.addEventListener('scroll', () => {     // only the reader scrolling up unsticks it: lines arriving don't
    if (log.scrollTop + log.clientHeight >= log.scrollHeight - 24) { stick = true; unread = 0; newer.hidden = true; }
    else if (log.scrollTop < autoTop - 4) stick = false;
    autoTop = Math.min(autoTop, log.scrollTop);
  });
  log.addEventListener('htmx:sseMessage', () => {
    const el = log.lastElementChild;
    if (!el) return;
    const stage = el.dataset.stage;
    if (stage && !seen.has(stage)) addStage(stage);
    if (query && !el.textContent.toLowerCase().includes(query)) el.classList.add('nomatch');
    for (let extra = log.childElementCount - MAX_LINES; extra > 0; extra--) log.firstElementChild.remove();
    if (stick) toBottom();
    else if (shown(el)) {
      unread++;
      newer.textContent = `↓ ${unread} new line${unread > 1 ? 's' : ''}`;
      newer.hidden = false;
    }
  });
  newer.addEventListener('click', toBottom);
  let findTimer = 0;
  find.addEventListener('input', () => {
    clearTimeout(findTimer);
    findTimer = setTimeout(() => {
      query = find.value.trim().toLowerCase();
      for (const el of log.children) el.classList.toggle('nomatch', !!query && !el.textContent.toLowerCase().includes(query));
      toBottom();
    }, 150);
  });
  document.addEventListener('click', (e) => {
    const b = e.target.closest('button.logf');
    if (b) showStage(b.dataset.stage);
  });

  // The running stage's clock, and the tab's title -------------------------------------------------------------------------------
  setInterval(() => {
    const now = Date.now();
    for (const el of $$('.elapsed[data-seconds]')) {
      if (!el._t0) el._t0 = now;
      el.textContent = duration(+el.dataset.seconds + (now - el._t0) / 1000);
    }
    const pill = $('#run-status .pill');
    if (pill) document.title = `${pill.textContent.replace(/\s+/g, ' ').trim()} · Training run`;
  }, 1000);

  // Results: a card refreshes itself on its stage's `results` event; charts and breakdowns are preserved across that --------------
  const timers = new Map();
  function onResults(stage) {
    if (timers.has(stage)) return;                 // at most one refresh a stage every 2 s
    timers.set(stage, setTimeout(() => {
      timers.delete(stage);
      const card = document.getElementById('result-' + stage);
      if (card) {
        htmx.ajax('GET', '/fragment/results/' + encodeURIComponent(stage), { target: card, swap: 'outerHTML' });
        htmx.ajax('GET', '/fragment/results', { target: '#headline', select: '#headline', swap: 'outerHTML' });
      } else {
        htmx.ajax('GET', '/fragment/results', { target: '#results', swap: 'innerHTML' });
      }
      const group = groups.get(stage);
      if (group) group.refresh();
    }, 2000));
  }

  const DEFAULTS = {
    encoder: [
      { title: 'loss', series: ['loss', 'recon'] },
      { title: 'information, nats', series: ['where_nats', 'what_nats', 'kl_nats'] },
      { title: 'probe R²', series: ['r2_own_value', 'r2_enemy_value', 'r2_enemy_known', 'r2_enemy_x', 'r2_commander_hp'] },
      { title: 'gradient norm', series: ['grad_norm'] },
    ],
    clone: [
      { title: 'loss', series: ['loss', 'ordered_bce'] },
      { title: 'held out: right', series: ['held_kind_right', 'held_anchor_right', 'held_done_right', 'held_ordered_right'], y: [0, 1] },
      { title: 'held out, z unknown: right', series: ['held_zu_kind_right', 'held_zu_anchor_right', 'held_zu_done_right'], y: [0, 1] },
      { title: 'held out: right by kind', series: ['held_right_*'], y: [0, 1] },
    ],
    ppo: [
      { title: 'test vs army script', series: ['eval_fight', 'eval_fight_draws'], y: [0, 1] },
      { title: 'KL', series: ['kl', 'kl_clone'] },
      { title: 'entropy', series: ['entropy_*'] },
      { title: 'losses', series: ['loss_pg', 'loss_value', 'loss_clone'] },
      { title: 'explained variance', series: ['explained_variance'] },
      { title: 'clip fraction', series: ['clip_fraction'] },
      { title: 'decisions / s', series: ['overall_rate', 'collect_rate', 'train_rate'] },
      { title: 'returns', series: ['return_min', 'return_max'] },
      { title: 'self-play share, beta', series: ['self_play_share', 'beta'] },
    ],
  };

  function expand(patterns, keys) {
    const out = [];
    for (const p of patterns) {
      if (p.includes('*')) {
        const re = new RegExp('^' + p.split('*').map((s) => s.replace(/[.+?^${}()|[\]\\]/g, '\\$&')).join('.*') + '$');
        for (const k of keys) if (re.test(k) && !out.includes(k)) out.push(k);
      } else if (keys.includes(p) && !out.includes(p)) out.push(p);
    }
    return out.slice(0, 8);
  }

  const groups = new Map();
  class StageCharts {
    constructor(host) {
      this.host = host;
      this.stage = host.dataset.stage;
      this.kind = host.dataset.kind;
      this.x = host.dataset.x;
      this.keys = null;
      this.rows = [];
      this.charts = [];
      host.dataset.mounted = '1';
      const list = 'keys-' + this.stage;
      host.innerHTML = `<div class="chart-tools"><input list="${esc(list)}" placeholder="add a chart: key, key2 or prefix_*" aria-label="Keys to chart">` +
        '<button type="button" class="add">Add chart</button><button type="button" class="reset" title="Back to the default charts">Reset</button>' +
        `<span class="dim count"></span><datalist id="${esc(list)}"></datalist></div><div class="chart-grid"></div>`;
      this.input = $('input', host);
      $('.add', host).addEventListener('click', () => this.add());
      this.input.addEventListener('keydown', (e) => { if (e.key === 'Enter') this.add(); });
      $('.reset', host).addEventListener('click', () => { store.set('charts:' + this.kind, null); this.build(); this.refresh(); });
      groups.set(this.stage, this);
      this.refresh();
    }

    specs() { return store.get('charts:' + this.kind, null) || DEFAULTS[this.kind] || []; }

    add() {
      const patterns = this.input.value.split(',').map((s) => s.trim()).filter(Boolean);
      const series = expand(patterns, this.keys || []);
      if (!series.length) { this.input.setCustomValidity('No numeric key matches'); this.input.reportValidity(); return; }
      this.input.setCustomValidity('');
      store.set('charts:' + this.kind, [...this.specs(), { title: patterns.join(', '), series: patterns }]);
      this.input.value = '';
      this.build();
      this.refresh();
    }

    build() {
      this.charts.forEach((c) => c.destroy());
      this.charts = [];
      const grid = $('.chart-grid', this.host);
      this.specs().forEach((spec, i) => {
        const series = expand(spec.series, this.keys || []);
        if (!series.length) return;
        const div = grid.appendChild(document.createElement('div'));
        const chart = new LineChart(div, {
          title: spec.title, x: this.x, series, y: spec.y,
          onRemove: () => { const list = this.specs().slice(); list.splice(i, 1); store.set('charts:' + this.kind, list); this.build(); },
        });
        if (this.rows.length) chart.setData(this.rows);
        this.charts.push(chart);
      });
    }

    async refresh() {
      if (this.busy) { this.again = true; return; }
      this.busy = true;
      try {
        if (!this.keys) {
          this.keys = (await getJSON(`/api/metrics/${encodeURIComponent(this.stage)}?keys=${this.x}&max=2`)).keys;
          this.listKeys();
          this.build();
        }
        const want = new Set([this.x]);
        this.charts.forEach((c) => c.opts.series.forEach((k) => want.add(k)));
        const r = await getJSON(`/api/metrics/${encodeURIComponent(this.stage)}?keys=${[...want].map(encodeURIComponent).join(',')}&max=1000`);
        this.rows = r.rows;
        if (r.keys.join() !== this.keys.join()) {          // new keys (the first test's, say): charts may take more series
          this.keys = r.keys;
          this.listKeys();
          this.build();
          this.again = true;
        } else this.charts.forEach((c) => c.setData(r.rows));
        $('.count', this.host).textContent = `${r.rows.length} rows · ${this.keys.length} numeric keys`;
      } catch (e) {
        $('.count', this.host).textContent = 'could not load: ' + e.message;
      } finally {
        this.busy = false;
        if (this.again) { this.again = false; this.refresh(); }
      }
    }

    listKeys() {
      $('datalist', this.host).replaceChildren(...this.keys.map((k) => Object.assign(document.createElement('option'), { value: k })));
    }
  }

  // A test's games by the policy's factory, its variant and the map.
  async function breakdown(host) {
    host.dataset.mounted = '1';
    let e;
    try { e = await getJSON('/api/eval/' + encodeURIComponent(host.dataset.stage)); } catch (err) { return; }
    const records = (e && e.records) || [];
    if (!records.length) return;
    const table = (title, key, most) => {
      const by = new Map();
      for (const r of records) {
        const k = r[key] ?? '?';
        const c = by.get(k) || { n: 0, won: 0, drew: 0 };
        c.n++;
        if (r.result === 'won') c.won++;
        if (r.result === 'drew') c.drew++;
        by.set(k, c);
      }
      const rows = [...by].sort((a, b) => b[1].n - a[1].n).slice(0, most);
      const body = rows.map(([k, c]) => `<tr><td>${esc(String(k).replace(/^factory/, ''))}</td><td class="num">${c.n}</td><td class="num">${c.won}</td>` +
        `<td class="num">${c.drew}</td><td class="num heat" style="--w:${(c.won / c.n * 100).toFixed(0)}%">${(c.won / c.n * 100).toFixed(0)}%</td></tr>`).join('');
      return `<div class="tablebox"><table class="grid"><tr><th>${esc(title)}</th><th class="num">games</th><th class="num">won</th><th class="num">drew</th>` +
        `<th class="num">win rate</th></tr>${body}</table></div>`;
    };
    host.innerHTML = `<div class="t-label">the ${records.length} games by</div><div class="three">${table("policy's factory", 'own_factory', 12)}` +
      `${table('variant', 'variant', 12)}${table('map', 'map', 10)}</div>`;
  }

  htmx.onLoad((el) => {
    const scope = el.parentNode || el;
    for (const host of $$('.charts[data-stage]:not([data-mounted])', scope)) new StageCharts(host);
    for (const host of $$('.breakdown[data-stage]:not([data-mounted])', scope)) breakdown(host);
  });

  // The replay viewer, by the hash ------------------------------------------------------------------------------------------------
  function route() {
    const p = new URLSearchParams(location.hash.slice(1));
    const url = p.get('replay');
    if (url) window.Viewer.open(url, p);
    else window.Viewer.close();
  }
  window.addEventListener('hashchange', route);
  document.addEventListener('DOMContentLoaded', route);

  window.Dashboard = { fmt, duration, store };
})();
