// Exploratory data analysis from data/eda/eda.json.
import { h, fill, fmtNum, fmtTime, fmtDuration, figureImg, segmented } from '../util.js';
import { objectClasses, signalInfo, theme } from '../classes.js';
import { plotLazy, layout, alpha } from '../plot.js';
import { createTimeline } from '../timeline.js';
import { loadEda } from '../store.js';

// Categorical slots for per-video series (same validated order as the object classes).
const SERIES = { light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'], dark: ['#3987e5', '#d95926', '#199e70', '#c98500'] };
const seriesColor = (i) => SERIES[theme()][i % 4];

const IMAGES = [
  ['vehicle_heatmap', 'Vehicle heatmap', 'Where vehicle ground points were seen, summed over both videos.'],
  ['person_heatmap', 'Pedestrian heatmap', 'Where pedestrians walk. Note the band between the zebras.'],
  ['flow_field', 'Lane-direction field', 'Median heading per cell, learned from vehicle trajectories. Only coherent cells are used for wrong_way.'],
  ['trajectories', 'Trajectories', 'A sample of vehicle and pedestrian tracks, coloured by class.'],
  ['scene_layout', 'Scene layout', 'Hand-annotated carriageway, islands, crosswalks A/B/C, stop line, queue zone, junction box and bus stop.'],
  ['signal_heads', 'Signal heads', 'The lamp boxes read for the main vehicle signal and the pedestrian signal.'],
];

const card = (title, hint, ...body) => h('div', { class: 'card' },
  h('div', { class: 'card-head' }, h('h3', {}, title), hint ? h('p', { class: 'hint' }, hint) : null), ...body);

function metaTable(videos) {
  return h('div', { class: 'table-scroll' }, h('table', { class: 'data-table' },
    h('thead', {}, h('tr', {}, ['Video', 'Resolution', 'Frame rate', 'Duration', 'Codec', 'Bitrate', 'Size']
      .map((c) => h('th', { scope: 'col' }, c)))),
    h('tbody', {}, videos.map((v) => h('tr', {},
      h('th', { scope: 'row' }, v.id),
      h('td', { class: 'mono' }, v.width && v.height ? `${v.width} × ${v.height}` : '–'),
      h('td', { class: 'mono' }, v.fps ? `${fmtNum(v.fps, 2)} fps` : '–'),
      h('td', { class: 'mono' }, fmtTime(v.duration)),
      h('td', {}, v.codec || '–'),
      h('td', { class: 'mono' }, v.bitrate_mbps ? `${fmtNum(v.bitrate_mbps)} Mbit/s` : '–'),
      h('td', { class: 'mono' }, v.size_gb ? `${fmtNum(v.size_gb, 2)} GB` : '–'))))));
}

function brightnessChart(videos) {
  const withData = videos.filter((v) => v.brightness?.t?.length);
  if (!withData.length) return null;
  const all = withData.flatMap((v) => v.brightness.mean);
  const el = h('div', { class: 'chart', role: 'img', 'aria-label': 'Mean frame brightness over time per video' });
  plotLazy(el, (t) => ({
    data: withData.map((v, i) => ({
      x: v.brightness.t, y: v.brightness.mean, name: v.id, type: 'scatter', mode: 'lines',
      line: { color: seriesColor(i), width: 2 }, hovertemplate: `${v.id} · %{x:.0f} s · %{y:.1f}<extra></extra>`,
    })),
    layout: layout(t, {
      showlegend: withData.length > 1, hovermode: 'x unified',
      xaxis: { title: { text: 'time (s)' } }, yaxis: { title: { text: 'mean luma (0–255)' } },
    }),
  }));
  return card('Lighting over time',
    `Mean frame brightness stays between ${fmtNum(Math.min(...all), 0)} and ${fmtNum(Math.max(...all), 0)}: daytime footage with no exposure jumps, so fixed colour thresholds for the signal lamps are safe.`,
    el);
}

function speedChart(speeds) {
  const counts = speeds?.vehicle_px_s;
  const bins = speeds?.bins;
  if (!Array.isArray(counts) || !counts.length) return null;
  // bins may be edges (n + 1 values) or centres (n values)
  const centres = Array.isArray(bins) && bins.length === counts.length + 1
    ? counts.map((_, i) => (bins[i] + bins[i + 1]) / 2)
    : Array.isArray(bins) && bins.length === counts.length ? bins : counts.map((_, i) => i);
  const width = centres.length > 1 ? (centres[1] - centres[0]) * 0.86 : 1;
  const el = h('div', { class: 'chart', role: 'img', 'aria-label': 'Histogram of vehicle speeds in pixels per second' });
  plotLazy(el, (t) => ({
    data: [{ x: centres, y: counts, type: 'bar', width, marker: { color: seriesColor(0) },
      hovertemplate: '%{x:.0f} px/s · %{y:,} observations<extra></extra>' }],
    layout: layout(t, { bargap: 0.05, xaxis: { title: { text: 'vehicle speed (px/s, 1920-px scene)' } }, yaxis: { title: { text: 'observations' } } }),
  }));
  return card('Vehicle speeds', 'Speed of every tracked vehicle observation. The spike near zero is queues, the bus stop and parked cars.', el);
}

function countsBuild(series) {
  const objs = objectClasses(Object.keys(series)).filter((o) => Array.isArray(series[o.id]));
  return (t) => ({
    data: objs.map((o) => ({
      x: series.t, y: series[o.id], name: o.name, type: 'scatter', mode: 'lines', stackgroup: 'one',
      line: { color: o.color, width: 1 }, fillcolor: alpha(o.color, 0.55),
      hovertemplate: `${o.name}: %{y}<extra></extra>`,
    })),
    layout: layout(t, { showlegend: true, hovermode: 'x unified', xaxis: { title: { text: 'time (s)' } }, yaxis: { title: { text: 'objects in view' } } }),
  });
}

function redShapes(sig, duration, t) {
  if (!Array.isArray(sig)) return [];
  return sig.map(([t0, state], i) => [t0, i + 1 < sig.length ? sig[i + 1][0] : duration, signalInfo(state).id])
    .filter(([, , id]) => id === 'red')
    .map(([x0, x1]) => ({ type: 'rect', xref: 'x', yref: 'paper', x0, x1, y0: 0, y1: 1, layer: 'below',
      fillcolor: alpha(t.critical, 0.09), line: { width: 0 } }));
}

function queueBuild(q, sig, duration) {
  return (t) => ({
    data: [
      { x: q.t, y: q.vehicles, name: 'vehicles in queue zone', type: 'scatter', mode: 'lines', line: { color: seriesColor(0), width: 2 },
        hovertemplate: 'in zone: %{y}<extra></extra>' },
      { x: q.t, y: q.stationary, name: 'stationary', type: 'scatter', mode: 'lines', line: { color: seriesColor(1), width: 2 },
        hovertemplate: 'stationary: %{y}<extra></extra>' },
    ],
    layout: layout(t, { showlegend: true, hovermode: 'x unified', shapes: redShapes(sig, duration, t),
      xaxis: { title: { text: 'time (s) · shaded: main signal red' } }, yaxis: { title: { text: 'vehicles' }, rangemode: 'tozero' } }),
  });
}

function cycleCard(sc) {
  if (!sc) return null;
  const phases = [
    ['green', 'Green', sc.green_s], ['green_flash', 'Flashing green', sc.flash_s], ['amber', 'Amber', sc.amber_s],
    ['red', 'Red', sc.red_s], ['red_amber', 'Red + amber', sc.red_amber_s],
  ].filter(([, , sec]) => Number(sec) > 0);
  const total = Number(sc.cycle_s) || phases.reduce((n, p) => n + p[2], 0);
  if (!total || !phases.length) return null;
  const bar = h('div', { class: 'cycle-bar', role: 'img', 'aria-label': `Signal cycle of ${total} s: ${phases.map(([, n, sec]) => `${n} ${sec} s`).join(', ')}` },
    phases.map(([id, , sec]) => h('span', { class: `cycle-seg tl-sig--${id}`, style: { flex: `${sec} 0 0`, background: signalInfo(id).color } })));
  const legend = h('ul', { class: 'cycle-legend' }, phases.map(([id, name, sec]) => h('li', {},
    h('i', { class: 'swatch', style: { background: signalInfo(id).color }, 'aria-hidden': 'true' }), `${name} `, h('span', { class: 'mono' }, `${sec} s`))));
  const green = (Number(sc.green_s) || 0) + (Number(sc.flash_s) || 0);
  const stats = h('dl', { class: 'mini-stats' },
    h('div', {}, h('dt', {}, 'Cycle'), h('dd', {}, `${fmtNum(total, 0)} s`)),
    h('div', {}, h('dt', {}, 'Green share'), h('dd', {}, `${fmtNum((100 * green) / total, 0)} %`)),
    h('div', {}, h('dt', {}, 'Red share'), h('dd', {}, `${fmtNum((100 * (Number(sc.red_s) || 0)) / total, 0)} %`)),
    h('div', {}, h('dt', {}, 'Cycles per hour'), h('dd', {}, fmtNum(3600 / total, 0))));
  return card('Signal cycle', 'Read from the lamp colours of the main vehicle signal. The cycle is fixed-time, so it repeats exactly.', bar, legend, stats);
}

export async function initEda() {
  const eda = await loadEda();
  const body = document.getElementById('eda-body');
  if (!eda) {
    fill(body, h('p', { class: 'empty' }, 'EDA results will appear here (data/eda/eda.json).'));
    return;
  }
  const videos = Array.isArray(eda.videos) ? eda.videos : [];
  const durations = Object.fromEntries(videos.map((v) => [v.id, v.duration]));
  const ids = [...new Set([
    ...videos.map((v) => v.id),
    ...Object.keys(eda.counts_over_time || {}), ...Object.keys(eda.queue || {}), ...Object.keys(eda.signal_cycle?.timeline || {}),
  ])];

  // per-video panel: one selector scopes the counts chart, the queue chart and the signal strip
  const countsEl = h('div', { class: 'chart chart--tall', role: 'img', 'aria-label': 'Objects in view over time, stacked by class' });
  const queueEl = h('div', { class: 'chart', role: 'img', 'aria-label': 'Queue length over time with red signal phases shaded' });
  const sigBox = h('div', { class: 'sig-strip' });
  const countsCard = card('Objects in view', 'Detections per second by COCO class (1 Hz).', countsEl);
  const queueCard = card('Queue vs signal', 'Vehicles in the approach queue zone. The queue builds on red and clears after green, so congestion only counts queues during green.', queueEl);
  const sigCard = card('Signal timeline', 'Main signal state over the whole video.', sigBox);

  function showVideo(id) {
    const duration = durations[id] || eda.counts_over_time?.[id]?.t?.at(-1) || eda.queue?.[id]?.t?.at(-1) || 1;
    const series = eda.counts_over_time?.[id];
    const q = eda.queue?.[id];
    const sig = eda.signal_cycle?.timeline?.[id];
    countsCard.hidden = !series?.t?.length;
    if (series?.t?.length) plotLazy(countsEl, countsBuild(series));
    queueCard.hidden = !q?.t?.length;
    if (q?.t?.length) plotLazy(queueEl, queueBuild(q, sig, duration));
    sigCard.hidden = !sig?.length;
    if (sig?.length) fill(sigBox, createTimeline({ duration, rows: [], signal: sig, ariaLabel: `Signal timeline of ${id}` }).el);
  }

  const picker = ids.length > 1
    ? h('div', { class: 'toolbar' }, h('span', { class: 'toolbar-label' }, 'Video'),
        segmented(ids.map((id) => ({ id, label: id })), ids[0], showVideo, 'Video for the per-video charts'))
    : null;

  const images = eda.images || {};
  const findings = Array.isArray(eda.findings) ? eda.findings.filter((f) => f && (f.title || f.text)) : [];

  fill(body,
    videos.length ? card('The footage', `${videos.length} videos, ${fmtDuration(videos.reduce((n, v) => n + (Number(v.duration) || 0), 0))} in total. Fixed camera, same view in every video.`, metaTable(videos)) : null,
    h('div', { class: 'grid-2' }, brightnessChart(videos), speedChart(eda.speeds)),
    ids.length ? h('div', { class: 'stack' }, h('h3', { class: 'sub-head' }, 'Per video'), picker, countsCard, h('div', { class: 'grid-2' }, queueCard, sigCard)) : null,
    cycleCard(eda.signal_cycle),
    h('h3', { class: 'sub-head' }, 'What the camera sees'),
    h('div', { class: 'img-grid' }, IMAGES.filter(([key]) => key in images).map(([key, title, caption]) =>
      figureImg(images[key], title, h('span', {}, h('strong', {}, `${title}. `), caption)))),
    findings.length ? h('h3', { class: 'sub-head' }, 'Findings that shaped the solution') : null,
    findings.length ? h('ol', { class: 'findings' }, findings.map((f) => h('li', { class: 'card finding' },
      f.image ? figureImg(f.image, f.title || 'finding') : null,
      h('h4', {}, f.title || ''), f.text ? h('p', {}, f.text) : null))) : null,
  );
  if (ids.length) showVideo(ids[0]);
}
