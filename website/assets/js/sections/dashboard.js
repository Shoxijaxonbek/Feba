// Operator dashboard: aggregates over every sample result.
import { h, fill, fmtNum, fmtDuration } from '../util.js';
import { classInfo, classColor, classVar, classOrder } from '../classes.js';
import { plotLazy, layout } from '../plot.js';
import { createTimeline } from '../timeline.js';
import { chip } from '../result-view.js';
import { loadAllSamples, normaliseResult } from '../store.js';

const ALARM = 0.5;

function alarmEpisodes(risk) {
  let n = 0, on = false;
  for (const [, v] of risk) {
    if (v >= ALARM && !on) n++;
    on = v >= ALARM;
  }
  return n;
}

const tile = (label, value, sub) => h('div', { class: 'kpi' }, h('dt', {}, label),
  h('dd', {}, h('span', { class: 'kpi-value' }, value), sub ? h('span', { class: 'kpi-sub' }, sub) : null));

const chartCard = (title, hint, el) => h('div', { class: 'card' },
  h('div', { class: 'card-head' }, h('h3', {}, title), hint ? h('p', { class: 'hint' }, hint) : null), el);

export async function initDashboard() {
  const body = document.getElementById('dashboard-body');
  const all = (await loadAllSamples()).map(({ entry, result }) => ({ id: entry.id, r: normaliseResult(result) }));
  if (!all.length) {
    fill(body, h('p', { class: 'empty' }, 'The dashboard fills in once sample results are published.'));
    return;
  }
  const events = all.flatMap(({ id, r }) => r.events.map((e) => ({ ...e, video: id })));
  const minutes = all.reduce((n, { r }) => n + r.duration, 0) / 60;
  const labels = [...new Set(events.map((e) => e.label))].sort(classOrder);
  const perClass = new Map(labels.map((l) => [l, events.filter((e) => e.label === l).length]));
  const top = labels.reduce((best, l) => (perClass.get(l) > (perClass.get(best) ?? -1) ? l : best), labels[0]);
  const alarms = all.reduce((n, { r }) => n + alarmEpisodes(r.risk), 0);

  // events per class (horizontal bars, class colours, value at the tip)
  const classEl = h('div', { class: 'chart', style: { height: `${Math.max(160, labels.length * 36 + 48)}px` }, role: 'img', 'aria-label': 'Number of events per class' });
  plotLazy(classEl, (t) => {
    const ordered = [...labels].reverse(); // first class on top
    return {
      data: [{ type: 'bar', orientation: 'h', y: ordered.map((l) => classInfo(l).name), x: ordered.map((l) => perClass.get(l)),
        marker: { color: ordered.map(classColor) }, width: 0.6, text: ordered.map((l) => String(perClass.get(l))),
        textposition: 'outside', cliponaxis: false, textfont: { color: t.text, size: 12 },
        hovertemplate: '%{y}: %{x} events<extra></extra>' }],
      layout: layout(t, { margin: { l: 120, r: 36, t: 8, b: 32 }, xaxis: { title: { text: 'events' }, rangemode: 'tozero' },
        yaxis: { showline: false, ticks: '' } }),
    };
  });

  // events per minute, stacked by class
  const maxMin = Math.max(1, Math.ceil(Math.max(...all.map(({ r }) => r.duration)) / 60));
  const minuteEl = h('div', { class: 'chart', role: 'img', 'aria-label': 'Events starting in each minute of video, stacked by class' });
  plotLazy(minuteEl, (t) => ({
    data: labels.map((l) => {
      const y = Array.from({ length: maxMin }, () => 0);
      for (const e of events) if (e.label === l) y[Math.min(maxMin - 1, Math.floor(e.start / 60))]++;
      return { type: 'bar', name: classInfo(l).name, x: y.map((_, i) => `${i}–${i + 1}`), y,
        marker: { color: classColor(l), line: { color: t.surface, width: 1 } }, hovertemplate: `${classInfo(l).name}: %{y}<extra>minute %{x}</extra>` };
    }),
    layout: layout(t, { barmode: 'stack', bargap: 0.35, showlegend: true, xaxis: { title: { text: 'minute of video (all samples summed)' }, type: 'category' },
      yaxis: { title: { text: 'events starting' } } }),
  }));

  // per crosswalk
  const cw = events.filter((e) => e.info?.crosswalk);
  const crossings = [...new Set(cw.map((e) => String(e.info.crosswalk)))].sort();
  const cwLabels = labels.filter((l) => cw.some((e) => e.label === l));
  const cwEl = h('div', { class: 'chart', style: { height: `${Math.max(160, crossings.length * 48 + 80)}px` }, role: 'img', 'aria-label': 'Events per crosswalk, stacked by class' });
  if (cw.length) {
    plotLazy(cwEl, (t) => ({
      data: cwLabels.map((l) => ({
        type: 'bar', orientation: 'h', name: classInfo(l).name, y: crossings.map((c) => `Crosswalk ${c}`),
        x: crossings.map((c) => cw.filter((e) => e.label === l && String(e.info.crosswalk) === c).length),
        marker: { color: classColor(l), line: { color: t.surface, width: 1 } }, width: 0.55,
        hovertemplate: `%{y} · ${classInfo(l).name}: %{x}<extra></extra>`,
      })),
      layout: layout(t, { barmode: 'stack', showlegend: true, margin: { l: 100, r: 16, t: 8, b: 32 },
        xaxis: { title: { text: 'events' } }, yaxis: { autorange: 'reversed', showline: false, ticks: '' } }),
    }));
  }

  // combined timeline: one row per video; clicking opens the video in Results
  const maxDur = Math.max(...all.map(({ r }) => r.duration));
  const combined = createTimeline({
    duration: maxDur, ariaLabel: 'Events in all sample videos',
    rows: all.map(({ id, r }) => ({
      label: id, color: 'var(--muted)', count: r.events.length,
      items: r.events.map((e) => ({ start: e.start, end: e.end, title: `${classInfo(e.label).name} · ${id}`, color: classVar(e.label),
        detail: e.info?.crosswalk ? `crosswalk ${e.info.crosswalk}` : '', data: { id } })),
    })),
    onSeek: (time, item) => item && dispatchEvent(new CustomEvent('rs:open-result', { detail: { id: item.data.id, t: time } })),
  });

  // table twin of the class chart
  const table = h('div', { class: 'table-scroll' }, h('table', { class: 'data-table data-table--num' },
    h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, 'Class'), all.map(({ id }) => h('th', { scope: 'col' }, id)), h('th', { scope: 'col' }, 'Total'))),
    h('tbody', {}, labels.map((l) => h('tr', {}, h('th', { scope: 'row' }, chip(l)),
      all.map(({ id }) => h('td', {}, events.filter((e) => e.label === l && e.video === id).length)),
      h('td', {}, h('strong', {}, perClass.get(l))))))));

  fill(body,
    h('dl', { class: 'kpis kpis--6' },
      tile('Videos', fmtNum(all.length), fmtDuration(minutes * 60)),
      tile('Events', fmtNum(events.length), `${labels.length} classes`),
      tile('Events per minute', fmtNum(events.length / Math.max(minutes, 1e-9), 1)),
      tile('Most frequent', top ? classInfo(top).name : '–', top ? `${perClass.get(top)} events` : null),
      tile('Crosswalk events', fmtNum(cw.length), crossings.length ? `at ${crossings.length} crosswalks` : null),
      tile('Risk alarms', fmtNum(alarms), 'episodes with risk ≥ 0.5')),
    h('div', { class: 'grid-2' },
      chartCard('Events per class', null, classEl),
      cw.length ? chartCard('Events per crosswalk', 'Pedestrian events by the nearest crosswalk (A main approach, B far carriageway, C slip road).', cwEl) : null),
    chartCard('Events per minute', 'When events start, summed over all sample videos.', minuteEl),
    chartCard('Combined timeline', 'Every event in every sample video. Select a bar to open that video in Results at that moment.', combined.el),
    chartCard('Events per class and video', null, table));
}
