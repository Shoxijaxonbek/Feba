// Hero key numbers: videos processed, events detected, footage length, runtime vs budget.
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
    part_a_sec: result.timing.part_a_sec, part_b_sec: result.timing.part_b_sec,
  }));
}

export async function initHero() {
  const box = document.getElementById('hero-kpis');
  const [idx, all, timing] = await Promise.all([loadIndex(), loadAllSamples(), runtimeRows()]);
  const results = new Map(all.map((x) => [x.entry.id, x.result]));

  const events = idx.reduce((n, e) => n + (Number.isFinite(e.n_events) ? e.n_events : results.get(e.id)?.events?.length || 0), 0);
  const footage = idx.reduce((n, e) => n + (Number(e.duration) || Number(results.get(e.id)?.duration) || 0), 0);
  const labels = new Set(all.flatMap((x) => (x.result.events || []).map((e) => (Array.isArray(e) ? e[2] : e.label))));

  const rows = timing.filter((t) => Number(t.duration) > 0 && Number.isFinite(t.part_a_sec));
  const dur = rows.reduce((n, t) => n + Number(t.duration), 0);
  const ratioA = dur ? rows.reduce((n, t) => n + t.part_a_sec, 0) / dur : null;
  const budgetRows = rows.filter((t) => Number(t.budget_sec) > 0);
  const budget = budgetRows.length
    ? budgetRows.reduce((n, t) => n + Number(t.budget_sec), 0) / budgetRows.reduce((n, t) => n + Number(t.duration), 0)
    : DEFAULT_BUDGET;

  const riskSamples = all.flatMap((x) => (Array.isArray(x.result.risk) ? x.result.risk : []));
  const alarmed = riskSamples.filter((p) => p[1] >= 0.5).length;

  const meter = ratioA != null
    ? h('span', { class: 'meter meter--budget', role: 'img', 'aria-label': `${ratioA.toFixed(2)} of a ${budget.toFixed(1)} times budget` },
        h('span', { class: 'meter-fill', style: { width: `${Math.min(100, (ratioA / budget) * 100)}%` } }))
    : null;

  box.replaceChildren(...[
    tile('Sample videos processed', fmtNum(idx.length), footage ? `${fmtNum(footage / 60, 1)} min of 4K footage` : null),
    tile('Events detected', fmtNum(events), labels.size ? `${labels.size} of 14 event classes` : null),
    tile('Part A runtime', ratioA != null ? `${ratioA.toFixed(2)}×` : '–', `of video length · budget ${fmtNum(budget, 1)}×`, meter),
    riskSamples.length ? tile('Time above risk alarm', `${fmtNum((100 * alarmed) / riskSamples.length, 1)} %`, 'risk ≥ 0.5; normal traffic stays below') : null,
  ].filter(Boolean));
  box.removeAttribute('aria-busy');
}
