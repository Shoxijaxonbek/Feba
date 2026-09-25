// Metrics from data/metrics.json: headline scores, per-class F1, ablations, runtime.
import { h, fill, fmtNum, fmtTime } from '../util.js';
import { classOrder, CLASSES } from '../classes.js';
import { chip } from '../result-view.js';
import { loadMetrics } from '../store.js';
import { runtimeRows } from './hero.js';

const f2 = (x) => (Number.isFinite(x) ? x.toFixed(2) : '–');

function scoreTile(label, value, sub) {
  return h('div', { class: 'kpi' }, h('dt', {}, label),
    h('dd', {}, h('span', { class: 'kpi-value' }, value), sub ? h('span', { class: 'kpi-sub' }, sub) : null));
}

function perClassTable(perClass) {
  const labels = Object.keys(perClass).sort(classOrder);
  const mean = (m) => {
    const xs = [m.f1_03, m.f1_05, m.f1_07].filter(Number.isFinite);
    return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
  };
  const missing = CLASSES.filter((c) => !(c.id in perClass)).map((c) => c.id);
  return h('div', {},
    h('div', { class: 'table-scroll' }, h('table', { class: 'data-table data-table--num' },
      h('caption', { class: 'sr-only' }, 'Per-class F1 at temporal-IoU thresholds 0.3, 0.5 and 0.7'),
      h('thead', {}, h('tr', {}, ['Class', 'F1 @ 0.3', 'F1 @ 0.5', 'F1 @ 0.7', 'Mean F1', 'TP', 'FP', 'FN']
        .map((c) => h('th', { scope: 'col' }, c)))),
      h('tbody', {}, labels.map((label) => {
        const m = perClass[label] || {};
        const avg = mean(m);
        return h('tr', {},
          h('th', { scope: 'row' }, chip(label)),
          h('td', {}, f2(m.f1_03)), h('td', {}, f2(m.f1_05)), h('td', {}, f2(m.f1_07)),
          h('td', {}, h('span', { class: 'bar-cell' },
            h('span', { class: 'bar-track', 'aria-hidden': 'true' }, h('span', { class: 'bar-fill', style: { width: `${(avg ?? 0) * 100}%` } })),
            f2(avg))),
          h('td', {}, m.tp ?? '–'), h('td', {}, m.fp ?? '–'), h('td', {}, m.fn ?? '–'));
      })))),
    missing.length ? h('p', { class: 'hint' }, `Classes without dev-set results: ${missing.join(', ')}.`) : null);
}

function ablationTable(ab) {
  const rows = Array.isArray(ab.rows) ? ab.rows : [];
  const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter((k) => k !== 'variant');
  const title = (k) => ({ score_a: 'Score A', score_b: 'Score B', sec_per_min: 'Sec / video min' }[k] || k.replace(/_/g, ' '));
  const best = Object.fromEntries(keys.filter((k) => k.startsWith('score')).map((k) => [k, Math.max(...rows.map((r) => r[k]).filter(Number.isFinite))]));
  return h('div', { class: 'card' },
    h('h4', { class: 'card-title' }, ab.name || 'Ablation'),
    h('div', { class: 'table-scroll' }, h('table', { class: 'data-table data-table--num' },
      h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, 'Variant'), keys.map((k) => h('th', { scope: 'col' }, title(k))))),
      h('tbody', {}, rows.map((r) => h('tr', {},
        h('th', { scope: 'row' }, r.variant ?? '–'),
        keys.map((k) => h('td', { class: best[k] === r[k] ? 'is-best' : null },
          typeof r[k] === 'number' ? (k.startsWith('score') ? f2(r[k]) : fmtNum(r[k], 1)) : (r[k] ?? '–')))))))));
}

function timingTable(rows) {
  return h('div', { class: 'table-scroll' }, h('table', { class: 'data-table data-table--num' },
    h('thead', {}, h('tr', {}, ['Video', 'Duration', 'Part A', 'Part B', 'Budget', 'Part A / video', 'Within budget']
      .map((c) => h('th', { scope: 'col' }, c)))),
    h('tbody', {}, rows.map((t) => {
      const total = (t.part_a_sec || 0) + (t.part_b_sec || 0);
      const budget = t.budget_sec || (t.duration ? t.duration * 3 : null);
      return h('tr', {},
        h('th', { scope: 'row' }, t.video ?? '–'),
        h('td', {}, fmtTime(t.duration, false)), h('td', {}, fmtTime(t.part_a_sec, false)), h('td', {}, fmtTime(t.part_b_sec, false)),
        h('td', {}, fmtTime(budget, false)),
        h('td', {}, t.duration && t.part_a_sec ? `${(t.part_a_sec / t.duration).toFixed(2)}×` : '–'),
        h('td', {}, budget ? (total <= budget ? h('span', { class: 'ok' }, '✓ yes') : h('span', { class: 'bad' }, '✗ no')) : '–'));
    }))));
}

export async function initMetrics() {
  const [m, timing] = await Promise.all([loadMetrics(), runtimeRows()]);
  const body = document.getElementById('metrics-body');
  if (!m) {
    fill(body, h('p', { class: 'empty' }, 'Evaluation results will appear here once data/metrics.json is published.'),
      timing.length ? h('div', { class: 'card' }, h('h3', {}, 'Runtime'), timingTable(timing)) : null);
    return;
  }
  const perClass = m.per_class && typeof m.per_class === 'object' ? m.per_class : null;
  const ablations = Array.isArray(m.ablations) ? m.ablations : [];
  fill(body,
    h('dl', { class: 'kpis kpis--3' },
      scoreTile('Part A score', f2(m.score_a), 'macro-F1, mean over tIoU 0.3 / 0.5 / 0.7'),
      scoreTile('Part B score', Number.isFinite(m.score_b) ? f2(m.score_b) : 'not scored', Number.isFinite(m.score_b) ? null : 'not evaluated yet'),
      scoreTile('Dev set', perClass ? `${Object.keys(perClass).length} classes` : '–', m.dev_set || null)),
    perClass ? h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, 'Per-class F1'),
      h('p', { class: 'hint' }, 'A predicted segment counts as a hit when its temporal IoU with a same-class label reaches the threshold.')), perClassTable(perClass)) : null,
    ablations.length ? h('h3', { class: 'sub-head' }, 'Ablations') : null,
    ablations.length ? h('div', { class: 'ablations' }, ablations.map(ablationTable)) : null,
    timing.length ? h('div', { class: 'card' }, h('div', { class: 'card-head' }, h('h3', {}, 'Runtime'),
      h('p', { class: 'hint' }, 'Wall-clock time on our machine; the budget is 3× the video duration.')), timingTable(timing)) : null);
}
