// Small DOM, formatting and loading helpers shared by all sections.

/** Build an element: h('a', {href, class, onclick, dataset, style, 'aria-label'}, ...children). */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'style' && typeof v === 'object') {
      for (const [p, val] of Object.entries(v)) el.style.setProperty(p, val);
    } else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  append(el, children);
  return el;
}

/** Replace el's children, skipping null/false like h() does. */
export function fill(el, ...children) {
  el.replaceChildren();
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

/** Same as h() for SVG elements. */
export function s(tag, attrs = {}, ...children) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  append(el, children);
  return el;
}

/** Fetch JSON; resolves to null on network errors, non-2xx responses and invalid JSON. */
export async function getJSON(url) {
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

export async function getText(url) {
  try {
    const res = await fetch(url, { cache: 'no-cache' });
    return res.ok ? await res.text() : null;
  } catch {
    return null;
  }
}

/** Run fn once when el comes within `margin` of the viewport. */
export function whenVisible(el, fn, margin = '300px') {
  if (!('IntersectionObserver' in window)) return void fn();
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      fn();
    }
  }, { rootMargin: margin });
  io.observe(el);
}

/** 83.4 -> "1:23.4" (tenths optional). */
export function fmtTime(sec, tenths = true) {
  if (sec == null || !isFinite(sec)) return '–';
  const neg = sec < 0;
  sec = Math.abs(sec);
  const m = Math.floor(sec / 60);
  const rest = sec - m * 60;
  const ss = tenths ? rest.toFixed(1).padStart(4, '0') : String(Math.floor(rest)).padStart(2, '0');
  return `${neg ? '-' : ''}${m}:${ss}`;
}

export function fmtDuration(sec) {
  if (sec == null || !isFinite(sec)) return '–';
  if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)} s`;
  return `${Math.floor(sec / 60)} min ${Math.round(sec % 60)} s`;
}

export const fmtNum = (x, digits = 0) =>
  x == null || !isFinite(x) ? '–' : Number(x).toLocaleString('en-US', { maximumFractionDigits: digits, minimumFractionDigits: digits });

export const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

/** Segmented control (a row of toggle buttons); calls onChange(id) when the selection changes. */
export function segmented(items, selected, onChange, label) {
  const buttons = items.map((it) => h('button', {
    type: 'button', class: 'seg-btn', 'aria-pressed': String(it.id === selected), dataset: { id: it.id },
    onclick: (e) => {
      if (e.currentTarget.getAttribute('aria-pressed') === 'true') return;
      selectSegment(group, it.id);
      onChange(it.id);
    },
  }, it.label));
  const group = h('div', { class: 'seg', role: 'group', 'aria-label': label }, buttons);
  return group;
}

/** Mark a segment as selected without firing onChange. */
export function selectSegment(group, id) {
  group.querySelectorAll('.seg-btn').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === id)));
}

/** A link value is usable if it is a non-empty string other than "#". */
export const isRealLink = (url) => typeof url === 'string' && url.trim() !== '' && url.trim() !== '#';

/** Placeholder shown where a video or image is not available yet. */
export function placeholder(caption, path, kind = 'image') {
  return h('div', { class: `ph ph--${kind}`, role: 'img', 'aria-label': caption },
    h('span', { class: 'ph-icon', 'aria-hidden': 'true', html: kind === 'video' ? ICON.film : ICON.image }),
    h('span', { class: 'ph-caption' }, caption),
    path ? h('code', { class: 'ph-path' }, path) : null);
}

/** <img> that swaps itself for a placeholder when the file is missing. */
export function figureImg(src, alt, caption) {
  const fig = h('figure', { class: 'fig' });
  const media = src
    ? h('img', { src, alt, loading: 'lazy', decoding: 'async',
        onerror: () => media.replaceWith(placeholder(`${alt} (not generated yet)`, src)) })
    : placeholder(`${alt} (not generated yet)`);
  fig.append(media);
  if (caption) fig.append(h('figcaption', {}, caption));
  return fig;
}

/** Evaluate a step function given as sorted [[t, value], ...] change points. */
export function stepAt(points, t) {
  let v = points?.[0]?.[1];
  for (const [x, val] of points || []) {
    if (x > t) break;
    v = val;
  }
  return v;
}

/** Linear lookup in a sorted [[t, y], ...] series (nearest sample at or before t). */
export function sampleAt(series, t) {
  if (!series?.length) return null;
  let lo = 0, hi = series.length - 1;
  if (t <= series[0][0]) return series[0][1];
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (series[mid][0] <= t) lo = mid; else hi = mid - 1;
  }
  return series[lo][1];
}

export const ICON = {
  github: '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.52-1.33-1.28-1.69-1.28-1.69-1.05-.72.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.69 5.38-5.26 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z"/></svg>',
  linkedin: '<svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.14 1.45-2.14 2.94v5.67H9.35V9h3.41v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28ZM5.34 7.43a2.06 2.06 0 1 1 0-4.12 2.06 2.06 0 0 1 0 4.12ZM7.12 20.45H3.56V9h3.56v11.45ZM22.22 0H1.77C.79 0 0 .77 0 1.73v20.54C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.73V1.73C24 .77 23.2 0 22.22 0Z"/></svg>',
  globe: '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9.5"/><path d="M2.5 12h19M12 2.5c2.6 2.8 3.9 6 3.9 9.5s-1.3 6.7-3.9 9.5c-2.6-2.8-3.9-6-3.9-9.5S9.4 5.3 12 2.5Z"/></svg>',
  play: '<svg viewBox="0 0 16 16" width="12" height="12" fill="currentColor"><path d="M4 2.5v11l9-5.5-9-5.5Z"/></svg>',
  image: '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="2"/><path d="m21 16-5-5-9 9"/></svg>',
  film: '<svg viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/></svg>',
  alert: '<svg viewBox="0 0 16 16" width="14" height="14" fill="currentColor"><path d="M8 1 15 14H1L8 1Zm-.75 5v4h1.5V6h-1.5Zm0 5.5V13h1.5v-1.5h-1.5Z"/></svg>',
  link: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M10 14a4.5 4.5 0 0 0 6.4 0l3-3a4.5 4.5 0 0 0-6.4-6.4l-1 1"/><path d="M14 10a4.5 4.5 0 0 0-6.4 0l-3 3a4.5 4.5 0 0 0 6.4 6.4l1-1"/></svg>',
  download: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v12m0 0-5-5m5 5 5-5M4 20h16"/></svg>',
  box: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m12 2 9 5v10l-9 5-9-5V7l9-5Z"/><path d="m3 7 9 5 9-5M12 12v10"/></svg>',
  doc: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"/><path d="M14 2v6h6M8 13h8M8 17h6"/></svg>',
  upload: '<svg viewBox="0 0 24 24" width="32" height="32" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M12 16V4m0 0-5 5m5-5 5 5M4 20h16"/></svg>',
};
