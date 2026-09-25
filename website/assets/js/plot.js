// Plotly is ~3.5 MB, so it is loaded from jsDelivr only when the first chart scrolls into view.
// Charts are registered with a build(tokens) function and rebuilt when the theme changes.
import { h, whenVisible } from './util.js';

const PLOTLY_URL = 'https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js';
let loading = null;
const charts = new Map(); // element -> build function

export function loadPlotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  loading ??= new Promise((resolve, reject) => {
    const tag = document.createElement('script');
    tag.src = PLOTLY_URL;
    tag.async = true;
    tag.onload = () => resolve(window.Plotly);
    tag.onerror = () => { loading = null; reject(new Error('Plotly failed to load')); };
    document.head.append(tag);
  });
  return loading;
}

/** Current theme colours, read from the CSS tokens. */
export function tokens() {
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  return {
    text: v('--text'), text2: v('--text-2'), muted: v('--muted'), grid: v('--grid'), axis: v('--axis'),
    surface: v('--surface'), border: v('--border-strong'), accent: v('--accent'),
    critical: v('--critical'), font: v('--font-sans'),
  };
}

/** '#rrggbb' + alpha -> 'rgba(...)' (for area washes). */
export function alpha(hex, a) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
}

const CONFIG = { responsive: true, displaylogo: false, displayModeBar: false, scrollZoom: false };

/** Shared layout; `extra` is merged one level deep (xaxis/yaxis/legend/margin objects are merged too). */
export function layout(t, extra = {}) {
  const axis = {
    gridcolor: t.grid, linecolor: t.axis, tickcolor: t.axis, zeroline: false, showline: true,
    tickfont: { color: t.muted, size: 11 }, title: { font: { color: t.text2, size: 12 } }, automargin: true,
  };
  const base = {
    font: { family: t.font, color: t.text2, size: 12 },
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    margin: { l: 48, r: 12, t: 12, b: 40 },
    xaxis: axis, yaxis: axis,
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', font: { color: t.text2, size: 12 }, bgcolor: 'rgba(0,0,0,0)' },
    hoverlabel: { bgcolor: t.surface, bordercolor: t.border, font: { color: t.text, family: t.font, size: 12 } },
    showlegend: false,
  };
  const out = { ...base, ...extra };
  for (const k of ['xaxis', 'yaxis', 'legend', 'margin']) {
    if (extra[k]) out[k] = { ...base[k], ...extra[k] };
  }
  return out;
}

/** Render now. build(tokens) -> {data, layout}; onReady(el, Plotly) runs after the first render. */
export async function plot(el, build, onReady) {
  let Plotly;
  try {
    Plotly = await loadPlotly();
  } catch {
    el.replaceChildren(h('p', { class: 'chart-error' }, 'The chart library could not be loaded (offline?). The tables on this page show the same data.'));
    return;
  }
  const { data, layout: lay } = build(tokens());
  await Plotly.react(el, data, lay, CONFIG);
  const first = !charts.has(el);
  charts.set(el, build);
  if (first && onReady) onReady(el, Plotly);
}

/** Render when the element nears the viewport. */
export const plotLazy = (el, build, onReady) => whenVisible(el, () => plot(el, build, onReady));

export function rethemeAll() {
  if (!window.Plotly) return;
  for (const [el, build] of charts) {
    if (!el.isConnected) { charts.delete(el); continue; }
    const { data, layout: lay } = build(tokens());
    window.Plotly.react(el, data, lay, CONFIG);
  }
}

export function purge(el) {
  if (window.Plotly && charts.has(el)) window.Plotly.purge(el);
  charts.delete(el);
}
