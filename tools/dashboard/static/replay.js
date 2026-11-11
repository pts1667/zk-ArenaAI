// A replay's data for the viewer (viewer.js): tools/policy_replay.py's JSON, format 2 or 3, read into typed arrays, its pages
// loaded as they're needed and dropped when far behind.  No drawing here, and no browser: node runs it too (tests/policy/
// test_dashboard.py), so what a test checks is what the page reads.
(function (root, make) {
  const api = make();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.Replay = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';
  const PAGES_KEPT = 4;               // loaded at once at most: the one shown, the next, and the two last seen
  const CHICKENS = 8;                 // the chickens' colour slot, after the eight teams'
  // An alliance's mark, by its letter: shapes that stay apart at a few pixels (a hexagon there reads as the circle).
  const SHAPES = ['circle', 'square', 'triangle', 'diamond', 'down', 'star', 'cross', 'bar'];

  // Format 2 (two sides, each its own alliance) in format 3's terms.
  function normalize(json) {
    if ((json.format || 1) >= 3) return json;
    const winner = json.result ? json.result.winner : null;
    json.teams = (json.sides || []).map((s, i) => Object.assign({}, s, {
      ally: i, plays: s.name === 'policy' ? 'policy' : 'script', outcome: winner == null ? 0 : winner === i ? 1 : -1 }));
    json.kind = '1v1';
    json.unit_fields = json.unit_fields.map((k) => (k === 'side' ? 'team' : k));
    json.team_fields = json.side_fields;
    for (const f of json.frames) { f.teams = f.sides; delete f.sides; }
    delete json.sides;
    delete json.side_fields;
    return json;
  }

  // The alliances in order of their allyteam, each with a letter, its mark and its teams; and each team's colour slot, which is
  // its number (so a team keeps its colour whatever else plays), the chickens' their own.
  function lineup(teams) {
    const by = new Map();
    teams.forEach((t, i) => {
      const a = t.ally == null ? i : t.ally;
      if (!by.has(a)) by.set(a, []);
      by.get(a).push(i);
    });
    const alliances = [...by.keys()].sort((x, y) => x - y).map((ally, k) => ({
      ally, letter: String.fromCharCode(65 + k), shape: SHAPES[k % SHAPES.length], index: k, teams: by.get(ally) }));
    const allianceOf = teams.map((t, i) => alliances.find((a) => a.teams.includes(i)));
    const slot = teams.map((t, i) => (t.plays === 'chickens' ? CHICKENS : Math.min(i, CHICKENS - 1)));
    return { alliances, allianceOf, slot };
  }

  // "1v1", "2v2", "4-way free-for-all", "3 vs chickens": what kind of game, from its teams.
  function kindName(json, alliances) {
    const sizes = alliances.map((a) => a.teams.length);
    const chickens = json.teams.filter((t) => t.plays === 'chickens').length;
    if (json.kind === 'chicken' || chickens) return `${json.teams.length - chickens} vs chickens`;
    if (alliances.length > 2) return sizes.every((s) => s === 1) ? `${alliances.length}-way free-for-all` : sizes.join('v');
    return sizes.join('v') || json.kind || '1v1';
  }

  function prepare(json, catalog, fetchPage) {
    json = normalize(json);
    catalog = catalog || {};
    const F = {};
    json.unit_fields.forEach((k, i) => { F[k] = i; });
    const NF = json.unit_fields.length;
    const S = {};
    json.team_fields.forEach((k, i) => { S[k] = i; });
    const n = json.teams.length;
    const head = json.frames;
    json.frames = null;                           // kept as typed arrays below
    const N = head.length;
    const times = Float64Array.from(head, (f) => f.t);
    const stats = head.map((f) => f.teams || []);
    const series = (field) => Array.from({ length: n }, (_, s) => Float32Array.from(stats, (t) => (t[s] ? t[s][S[field]] : 0)));
    const defs = [];
    const info = (id, name, cost) => {
      const c = catalog[name] || {};
      const air = !!c.air, mobile = c.mobile == null ? true : c.mobile;
      return { id, name, cost: cost || c.cost || 0, human: c.human || name, desc: c.description || '', icon: c.icon || 'default',
        size: c.icon_size || 1.3, air, mobile, layer: air ? 2 : mobile ? 1 : 0 };
    };
    for (const [id, d] of Object.entries(json.defs || {})) defs[+id] = info(+id, d.name, d.cost);
    const { alliances, allianceOf, slot } = lineup(json.teams);
    const pages = (json.pages || [{ first: 0, count: N, file: null }]).map((p, k) => ({ k, first: p.first, count: p.count, file: p.file, frames: null, loading: null }));
    let maxRows = 0;

    // A page's frames as typed arrays, each with its units by uid; `featAt`: for each, the frame of the page whose features hold.
    function fill(page, raw) {
      let last = 0;
      page.featAt = new Int32Array(page.count);
      page.deaths = new Array(page.count).fill(null);
      page.frames = raw.map((f, k) => {
        const u = Int32Array.from(f.units);
        const index = new Map();
        for (let o = 0; o < u.length; o += NF) index.set(u[o + F.uid], o);
        if (f.features) last = k;
        page.featAt[k] = last;
        maxRows = Math.max(maxRows, u.length / NF);
        return { t: times[page.first + k], teams: stats[page.first + k], u, index, features: f.features ? Float32Array.from(f.features) : null };
      });
    }
    if (!json.pages) fill(pages[0], head);

    const pageOf = (i) => {
      let lo = 0, hi = pages.length - 1;
      while (lo < hi) { const m = (lo + hi + 1) >> 1; if (pages[m].first <= i) lo = m; else hi = m - 1; }
      return pages[lo];
    };
    let keep = [0];                 // the pages the last `ensure` wanted
    function load(page) {
      if (page.frames) return Promise.resolve(page);
      if (!page.loading) {
        page.loading = Promise.resolve(fetchPage(page.file)).then((p) => { fill(page, p.frames); page.loading = null; evict(keep); return page; },
          (e) => { page.loading = null; throw e; });
      }
      return page.loading;
    }
    function evict(keep) {
      const loaded = pages.filter((p) => p.frames && p.file);
      while (loaded.length > PAGES_KEPT) {
        let far = -1, worst = -1;
        loaded.forEach((p, m) => { const d = Math.min(...keep.map((k) => Math.abs(p.k - k))); if (d > worst) { worst = d; far = m; } });
        if (worst <= 0) break;
        const [p] = loaded.splice(far, 1);
        p.frames = null; p.featAt = null; p.deaths = null;
      }
    }

    const R = {
      json, F, NF, S, n, N, times, stats, defs, defaultDef: info(-1, 'unknown', 0), alliances, allianceOf, slot, pages,
      kindName: kindName(json, alliances),
      marks: alliances.length > 2 || alliances.some((a) => a.teams.length > 1),     // whether the map needs alliance marks: a 1v1 doesn't
      end: times[N - 1] || 0,
      income: series('metal_income'), value: series('value'), units: series('units'),
      get maxRows() { return maxRows; },
      frameAt(t) {
        let lo = 0, hi = N - 1;
        if (t >= times[hi]) return hi;
        while (hi - lo > 1) { const m = (lo + hi) >> 1; if (times[m] <= t) lo = m; else hi = m; }
        return lo;
      },
      // Frame i's units, or null while its page isn't loaded (`ensure` loads it).
      frame(i) {
        const p = pageOf(i);
        return p.frames ? p.frames[i - p.first] : null;
      },
      features(i) {
        const p = pageOf(i);
        return p.frames ? p.frames[p.featAt[i - p.first]].features : null;
      },
      // Units seen in frame i - 1 and gone in frame i: x, z, team, def, four numbers a unit.
      deaths(i) {
        const p = pageOf(i), k = i - p.first;
        if (!i || !p.frames) return null;
        if (p.deaths[k]) return p.deaths[k];
        const A = p.frames[k], P = R.frame(i - 1);
        if (!P) return null;
        const out = [];
        for (let o = 0; o < P.u.length; o += NF) if (!A.index.has(P.u[o + F.uid])) out.push(P.u[o + F.x], P.u[o + F.z], P.u[o + F.team], P.u[o + F.def]);
        return (p.deaths[k] = Float32Array.from(out));
      },
      // Loads the pages frames i and i + 1 are in, and the next one ahead; resolves when the first two are there.
      ensure(i) {
        const a = pageOf(i), b = pageOf(Math.min(i + 1, N - 1)), ahead = pages[Math.min(b.k + 1, pages.length - 1)];
        keep = [a.k, b.k, ahead.k];
        const want = [load(a), load(b)];
        load(ahead).catch(() => {});
        evict(keep);
        return Promise.all(want);
      },
      loaded(i) { return !!pageOf(i).frames && !!pageOf(Math.min(i + 1, N - 1)).frames; },
    };
    return R;
  }

  return { normalize, lineup, kindName, prepare, SHAPES, CHICKENS, PAGES_KEPT };
});
