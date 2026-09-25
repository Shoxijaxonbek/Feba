// Cached loaders for the files described in DATA_CONTRACT.md. Every loader resolves to null
// (or an empty value) when a file is missing, so sections can render their fallbacks.
import { getJSON, getText } from './util.js';

const cache = new Map();
const once = (key, fn) => {
  if (!cache.has(key)) cache.set(key, fn());
  return cache.get(key);
};

export const loadSite = () => once('site', () => getJSON('data/site.json'));
export const loadEda = () => once('eda', () => getJSON('data/eda/eda.json'));
export const loadMetrics = () => once('metrics', () => getJSON('data/metrics.json'));
export const loadReport = () => once('report', () => getText('data/report.md'));

export const loadIndex = () => once('index', async () => {
  const idx = await getJSON('data/samples/index.json');
  return Array.isArray(idx) ? idx.filter((v) => v && v.id) : [];
});

export const loadSample = (id) => once(`sample:${id}`, () => getJSON(`data/samples/${encodeURIComponent(id)}.json`));

/** Every sample listed in the index that has a result file, as {entry, result}. */
export const loadAllSamples = () => once('all', async () => {
  const idx = await loadIndex();
  const results = await Promise.all(idx.map((e) => loadSample(e.id)));
  return idx.map((entry, i) => ({ entry, result: results[i] })).filter((x) => x.result);
});

/** Failure cases from index entries (`failures`) and data/failures.json (array or {failures}). */
export const loadFailures = () => once('failures', async () => {
  const [idx, file] = await Promise.all([loadIndex(), getJSON('data/failures.json')]);
  const fromFile = Array.isArray(file) ? file : Array.isArray(file?.failures) ? file.failures : [];
  const fromIndex = idx.flatMap((e) => (Array.isArray(e.failures) ? e.failures.map((f) => ({ video: e.id, ...f })) : []));
  return [...fromIndex, ...fromFile].filter((f) => f && (f.title || f.text));
});

/** Normalise a result so components can rely on arrays being present and sorted. */
export function normaliseResult(r) {
  const events = (Array.isArray(r?.events) ? r.events : [])
    .map((e) => (Array.isArray(e) ? { start: e[0], end: e[1], label: e[2] } : { ...e }))
    .filter((e) => isFinite(e.start) && isFinite(e.end) && e.label)
    .map((e, i) => ({ ...e, idx: i }));
  const risk = (Array.isArray(r?.risk) ? r.risk : []).filter((p) => Array.isArray(p) && isFinite(p[0]) && isFinite(p[1]));
  const signal = (Array.isArray(r?.signal) ? r.signal : []).filter((p) => Array.isArray(p) && isFinite(p[0]))
    .sort((a, b) => a[0] - b[0]);
  const lastT = Math.max(0, ...events.map((e) => e.end), risk.at(-1)?.[0] ?? 0, r?.counts?.t?.at(-1) ?? 0);
  return { ...r, events, risk, signal, duration: Number(r?.duration) > 0 ? Number(r.duration) : lastT };
}
