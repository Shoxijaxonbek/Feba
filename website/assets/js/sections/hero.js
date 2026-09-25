// Hero key numbers: dev score, events found, full run time vs budget, false alarms.
import { h, fmtNum } from '../util.js';
import { loadIndex, loadMetrics, loadAllSamples } from '../store.js';

const DEFAULT_BUDGET = 3; // x video duration, from the task rules

function tile(label, value, sub, meter) {
  return h('div', { class: 'kpi' },
    h('dt', {}, label),
    h('dd', {}, h('span', { class: 'kpi-value' }, value), sub ? h('span', { class: 'kpi-sub' }, sub) : null, meter || null));
}

/** Runtime rows from metrics.json, falling back to the `timing` block of each sample result. */
export async function runtimeRows() {
  const [metrics, all] = await Promise.all([loadMetrics(), loadAllSamples()]);
  if (Array.isArray(metrics?.timing) && metrics.timing.length) return metrics.timing;
  return all.filter((x) => x.result.timing).map(({ entry, result }) => ({
    video: entry.id, duration: result.timing.video_sec ?? result.duration ?? entry.duration,
    part_a_sec: result.timing.part_a_sec, part_b_sec: result.timing.part_b_sec, budget_sec: result.timing.budget_sec,
  }));
}

/** Official Score A on our dev labels: the C3902 row of the "Dev sets" table in metrics.json. */
function devScore(metrics) {
  const table = (metrics?.ablations || []).find((a) => /^Dev sets/.test(a.name));
  const row = table?.rows?.find((r) => /^C3902/.test(r.variant)) || table?.rows?.[0];
  return Number.isFinite(row?.score_a) ? row.score_a : metrics?.score_a;
}

export async function initHero() {
  const box = document.getElementById('hero-kpis');
  const [idx, all, timing, metrics] = await Promise.all([loadIndex(), loadAllSamples(), runtimeRows(), loadMetrics()]);
  const results = new Map(all.map((x) => [x.entry.id, x.result]));

  const events = idx.reduce((n, e) => n + (Number.isFinite(e.n_events) ? e.n_events : results.get(e.id)?.events?.length || 0), 0);
  const footage = idx.reduce((n, e) => n + (Number(e.duration) || Number(results.get(e.id)?.duration) || 0), 0);

  // whole run (Part A + Part B) against the time budget, over all sample videos
  const rows = timing.filter((t) => Number(t.duration) > 0 && Number.isFinite(t.part_a_sec) && Number.isFinite(t.part_b_sec));
  const dur = rows.reduce((n, t) => n + Number(t.duration), 0);
  const ratio = dur ? rows.reduce((n, t) => n + t.part_a_sec + t.part_b_sec, 0) / dur : null;
  const budgetRows = rows.filter((t) => Number(t.budget_sec) > 0);
  const budget = budgetRows.length
    ? budgetRows.reduce((n, t) => n + Number(t.budget_sec), 0) / budgetRows.reduce((n, t) => n + Number(t.duration), 0)
    : DEFAULT_BUDGET;

  // alarms = separate runs of risk >= 0.5 (what the Part B metric counts)
  let alarms = 0;
  for (const { result } of all) {
    let prev = 0;
    for (const [, r] of Array.isArray(result.risk) ? result.risk : []) {
      if (r >= 0.5 && prev < 0.5) alarms += 1;
      prev = r;
    }
  }
  const score = devScore(metrics);
  const minutes = fmtNum(footage / 60, 1);

  const meter = ratio != null
    ? h('span', { class: 'meter meter--budget', role: 'img', 'aria-label': `${ratio.toFixed(1)} of a ${budget.toFixed(1)} times budget` },
        h('span', { class: 'meter-fill', style: { width: `${Math.min(100, (ratio / budget) * 100)}%` } }))
    : null;

  box.replaceChildren(...[
    Number.isFinite(score) ? tile('Event detection score', score.toFixed(2),
      'official metric (Score A) on our own labels of C3902') : null,
    tile('Traffic events found', fmtNum(events), footage ? `in ${minutes} min of 4K video, day and dusk` : null),
    ratio != null ? tile('Full run time', `${ratio.toFixed(1)}×`, `video length, Part A + B · limit ${fmtNum(budget, 1)}×`, meter) : null,
    all.some((x) => Array.isArray(x.result.risk) && x.result.risk.length)
      ? tile('False accident alarms', fmtNum(alarms), `in ${minutes} min of normal traffic`) : null,
  ].filter(Boolean));
  box.removeAttribute('aria-busy');
}
