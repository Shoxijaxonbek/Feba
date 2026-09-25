// The result viewer shared by "Results on sample videos" and the live demo: annotated video,
// "now" panel, clickable event timeline with signal strip, risk curve with playhead, sortable
// event table, per-class example gallery and failure cases. Everything seeks the same clock.
import { h, fill, fmtTime, placeholder, whenVisible, stepAt, sampleAt, clamp, ICON } from './util.js';
import { classInfo, classVar, classColor, classOrder, signalInfo, objectClasses } from './classes.js';
import { createTimeline, tickStep } from './timeline.js';
import { plot, layout, alpha, purge } from './plot.js';
import { normaliseResult } from './store.js';

const ALARM = 0.5;
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function chip(label, extra) {
  return h('span', { class: 'chip', style: { '--cls': classVar(label) } },
    h('i', { class: 'swatch', 'aria-hidden': 'true' }), classInfo(label).name, extra ? h('span', { class: 'chip-extra' }, extra) : null);
}

/** {"crosswalk": "B", "approach": "main"} -> "crosswalk B · approach: main" */
export function describeInfo(info) {
  if (!info || typeof info !== 'object') return '';
  return Object.entries(info)
    .filter(([, v]) => v != null && v !== '')
    .map(([k, v]) => (k === 'crosswalk' ? `crosswalk ${v}` : `${k.replace(/_/g, ' ')}: ${Array.isArray(v) ? v.join(', ') : v}`))
    .join(' · ');
}

/**
 * Mount a result (shape of data/samples/<id>.json) into `container`.
 * opts: {videoUrl, poster, name, failures, gallery}. Returns {destroy, seek}.
 */
export function mountResult(container, raw, opts = {}) {
  const r = normaliseResult(raw);
  const D = r.duration || 1;
  const name = opts.name || r.id || 'video';
  const videoUrl = opts.videoUrl ?? r.annotated_video;
  const st = { t: 0, raf: 0, pendingSeek: null, nowKey: '' };
  const cleanups = [];

  // ---- video ----
  const videoBox = h('div', { class: 'rv-video' });
  let video = null;
  if (videoUrl) {
    video = h('video', { controls: true, playsinline: true, preload: 'none', poster: opts.poster || null,
      'aria-label': `Annotated video of ${name} with tracks, zones and detected events` });
    videoBox.append(video);
    const showMissing = () => {
      if (!video) return;
      video.replaceWith(placeholder('Annotated video not available yet. The timeline, chart and table still work: clicks move the playhead.', videoUrl, 'video'));
      video = null;
    };
    video.addEventListener('error', showMissing);
    video.addEventListener('loadedmetadata', () => {
      if (st.pendingSeek != null) { video.currentTime = st.pendingSeek; st.pendingSeek = null; }
    });
    video.addEventListener('timeupdate', () => video && update(video.currentTime));
    video.addEventListener('seeked', () => video && update(video.currentTime));
    video.addEventListener('play', loop);
    whenVisible(videoBox, ensureSrc, '200px');
  } else {
    videoBox.append(placeholder('No annotated video in this result', null, 'video'));
  }

  function ensureSrc() {
    if (video && !video.getAttribute('src')) {
      video.preload = 'metadata';
      video.src = videoUrl;
    }
  }

  function loop() {
    cancelAnimationFrame(st.raf);
    const tick = () => {
      if (!video || video.paused || video.ended) return;
      update(video.currentTime);
      st.raf = requestAnimationFrame(tick);
    };
    st.raf = requestAnimationFrame(tick);
  }

  function seek(t, { play = true, reveal = false } = {}) {
    t = clamp(t, 0, D);
    if (video) {
      ensureSrc();
      if (video.readyState >= 1) video.currentTime = t; else st.pendingSeek = t;
      if (play) video.play().catch(() => {});
    }
    update(t);
    if (reveal) {
      const rect = videoBox.getBoundingClientRect();
      if (rect.top < 60 || rect.bottom > innerHeight) {
        videoBox.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' });
      }
    }
  }

  // ---- now panel ----
  const now = h('div', { class: 'rv-now' });
  const counts = r.counts && Array.isArray(r.counts.t) ? r.counts : null;
  const countKeys = counts ? objectClasses(Object.keys(counts)).filter((o) => Array.isArray(counts[o.id])) : [];

  function renderNow(t) {
    const sig = r.signal.length ? signalInfo(stepAt(r.signal, t)) : null;
    const risk = sampleAt(r.risk, t);
    const active = r.events.filter((e) => t >= e.start && t <= e.end);
    let countText = '';
    if (counts) {
      const i = clamp(Math.round(t), 0, counts.t.length - 1);
      countText = countKeys.map((o) => [counts[o.id][i], o.name.toLowerCase()]).filter(([n]) => n > 0).map(([n, label]) => `${n} ${label}`).join(' · ');
    }
    const key = [t.toFixed(1), sig?.id, risk?.toFixed(2), active.map((e) => e.idx).join(','), countText].join('|');
    if (key === st.nowKey) return;
    st.nowKey = key;
    fill(now,
      h('div', { class: 'now-item' }, h('span', { class: 'now-label' }, 'Time'),
        h('span', { class: 'now-value mono' }, fmtTime(t), h('span', { class: 'now-sub' }, ` / ${fmtTime(D)}`))),
      sig ? h('div', { class: 'now-item' }, h('span', { class: 'now-label' }, 'Main signal'),
        h('span', { class: 'sig-pill' }, h('i', { style: { background: sig.color }, 'aria-hidden': 'true' }), sig.name)) : null,
      risk != null ? h('div', { class: 'now-item' }, h('span', { class: 'now-label' }, 'Accident risk (Part B)'),
        h('span', { class: 'now-value mono' }, risk.toFixed(2),
          risk >= ALARM ? h('span', { class: 'badge badge--critical', html: `${ICON.alert} alarm` }) : null),
        h('span', { class: 'meter', 'aria-hidden': 'true' }, h('span', { class: `meter-fill${risk >= ALARM ? ' is-alarm' : ''}`, style: { width: `${risk * 100}%` } }), h('i', { class: 'meter-mark' }))) : null,
      h('div', { class: 'now-item now-item--wide' }, h('span', { class: 'now-label' }, 'Active events'),
        active.length ? h('span', { class: 'chips' }, active.map((e) => chip(e.label, e.info?.crosswalk ? `· ${e.info.crosswalk}` : null))) : h('span', { class: 'now-sub' }, 'none')),
      countText ? h('div', { class: 'now-item now-item--wide' }, h('span', { class: 'now-label' }, 'In view'), h('span', { class: 'now-sub' }, countText)) : null,
    );
  }

  // ---- timeline ----
  const byClass = new Map();
  for (const e of r.events) (byClass.get(e.label) || byClass.set(e.label, []).get(e.label)).push(e);
  const rows = [...byClass.keys()].sort(classOrder).map((label) => ({
    label: classInfo(label).name, color: classVar(label), count: byClass.get(label).length,
    items: byClass.get(label).map((e) => ({ start: e.start, end: e.end, title: classInfo(label).name,
      detail: [describeInfo(e.info), e.tracks?.length ? `track ${e.tracks.join(', ')}` : ''].filter(Boolean).join(' · '), data: e })),
  }));
  const timeline = createTimeline({ duration: D, rows, signal: r.signal, ariaLabel: `Events in ${name}`, onSeek: (t) => seek(t) });

  // ---- risk chart ----
  const riskEl = h('div', { class: 'chart chart--risk', role: 'img', 'aria-label': `Accident-risk score over time for ${name}. Click to seek.` });
  const riskHead = h('div', { class: 'chart-playhead', hidden: true, 'aria-hidden': 'true' });
  const riskWrap = h('div', { class: 'chart-wrap' }, riskEl, riskHead);
  let riskReady = false;
  const labelWidth = () => parseFloat(getComputedStyle(timeline.el).getPropertyValue('--tl-label')) || 120;
  let lastLabelW = 0;

  const riskBuild = (t) => {
    lastLabelW = labelWidth();
    const step = tickStep(D);
    const tickvals = [];
    for (let x = 0; x <= D + 1e-6; x += step) tickvals.push(x);
    const shapes = [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: ALARM, y1: ALARM, line: { color: t.critical, width: 1.5, dash: 'dash' } }];
    for (const e of r.events) {
      if (e.label === 'accident' || e.label === 'near_miss') {
        shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: e.start, x1: e.end, y0: 0, y1: 1,
          fillcolor: alpha(classColor(e.label), 0.16), line: { width: 0 }, layer: 'below' });
      }
    }
    return {
      data: [{ x: r.risk.map((p) => p[0]), y: r.risk.map((p) => p[1]), type: 'scatter', mode: 'lines', name: 'risk',
        line: { color: t.accent, width: 2 }, fill: 'tozeroy', fillcolor: alpha(t.accent, 0.1),
        hovertemplate: '%{x:.1f} s · risk %{y:.2f}<extra></extra>' }],
      layout: layout(t, {
        margin: { l: lastLabelW, r: 0, t: 8, b: 28 }, hovermode: 'x', shapes,
        xaxis: { range: [0, D], tickvals, ticktext: tickvals.map((x) => fmtTime(x, false)), automargin: false, showspikes: false },
        yaxis: { range: [0, 1.02], tickvals: [0, 0.25, 0.5, 0.75, 1], automargin: false, fixedrange: true },
        annotations: [{ xref: 'paper', x: 1, y: ALARM, yanchor: 'bottom', xanchor: 'right', showarrow: false,
          text: 'alarm 0.5', font: { color: t.text2, size: 11 } }],
      }),
    };
  };

  function placeRiskHead(t) {
    const fl = riskEl._fullLayout;
    if (!riskReady || !fl?.xaxis?.l2p) return;
    riskHead.hidden = false;
    riskHead.style.left = `${fl._size.l + fl.xaxis.l2p(clamp(t, 0, D))}px`;
    riskHead.style.top = `${fl._size.t}px`;
    riskHead.style.height = `${fl._size.h}px`;
  }

  if (r.risk.length) {
    whenVisible(riskWrap, () => plot(riskEl, riskBuild, (el) => {
      riskReady = true;
      el.on('plotly_click', (ev) => ev.points?.[0] && seek(ev.points[0].x));
      el.on('plotly_afterplot', () => placeRiskHead(st.t));
      placeRiskHead(st.t);
    }));
    const onResize = () => { if (riskReady && labelWidth() !== lastLabelW) plot(riskEl, riskBuild); };
    addEventListener('resize', onResize);
    cleanups.push(() => removeEventListener('resize', onResize));
  }

  // ---- table ----
  const table = eventTable(r.events, (e) => seek(e.start, { reveal: true }));

  // ---- gallery & failures ----
  const gallery = opts.gallery === false ? null : eventGallery(r.events, name, (e) => seek(e.start, { reveal: true }));
  const failures = (opts.failures || []).filter((f) => !f.video || f.video === r.id);

  function update(t) {
    st.t = t;
    timeline.setTime(t);
    placeRiskHead(t);
    renderNow(t);
    table.setTime(t);
  }

  const classesList = rows.length
    ? rows.map((row) => `${row.count} ${row.label.toLowerCase()}`).join(', ')
    : 'no events';

  container.replaceChildren(
    h('div', { class: 'rv' },
      h('div', { class: 'rv-top' }, videoBox, now),
      h('div', { class: 'card rv-card' },
        h('div', { class: 'card-head' }, h('h3', {}, 'Event timeline'),
          h('p', { class: 'hint' }, `${r.events.length} events: ${classesList}. Click a bar to play from its start.`)),
        rows.length ? timeline.el : h('p', { class: 'empty' }, 'No events were detected in this video.')),
      h('div', { class: 'card rv-card' },
        h('div', { class: 'card-head' }, h('h3', {}, 'Accident risk (Part B)'),
          h('p', { class: 'hint' }, 'Probability that an accident starts within 5 s, computed causally from past frames only. Dashed line: alarm level 0.5. Click the chart to seek.')),
        r.risk.length ? riskWrap : h('p', { class: 'empty' }, 'This result has no risk curve.')),
      h('div', { class: 'card rv-card' },
        h('div', { class: 'card-head' }, h('h3', {}, 'All events'),
          h('p', { class: 'hint' }, 'Sort by any column. Select a row to jump to the event.')),
        r.events.length ? table.el : h('p', { class: 'empty' }, 'No events.')),
      gallery,
      failures.length ? failureBlock(failures, (f) => seek(f.start, { reveal: true })) : null,
    ),
  );
  update(0);

  return {
    seek,
    destroy() {
      cancelAnimationFrame(st.raf);
      cleanups.forEach((fn) => fn());
      purge(riskEl);
      video?.pause();
    },
  };
}

function eventTable(events, onPick) {
  const cols = [
    { key: 'label', title: 'Class', cmp: (a, b) => classOrder(a.label, b.label) || a.start - b.start },
    { key: 'start', title: 'Start', cmp: (a, b) => a.start - b.start },
    { key: 'end', title: 'End', cmp: (a, b) => a.end - b.end },
    { key: 'dur', title: 'Duration', cmp: (a, b) => (a.end - a.start) - (b.end - b.start) },
    { key: 'where', title: 'Details', cmp: (a, b) => describeInfo(a.info).localeCompare(describeInfo(b.info)) },
    { key: 'tracks', title: 'Tracks', cmp: (a, b) => (a.tracks?.[0] ?? 0) - (b.tracks?.[0] ?? 0) },
  ];
  let sort = { key: 'start', dir: 1 };
  const tbody = h('tbody');
  const heads = cols.map((c) => h('th', { scope: 'col', 'aria-sort': 'none' },
    h('button', { type: 'button', class: 'sort-btn', onclick: () => {
      sort = { key: c.key, dir: sort.key === c.key ? -sort.dir : 1 };
      render();
    } }, c.title, h('span', { class: 'sort-ind', 'aria-hidden': 'true' }))));
  let rowEls = [];

  function render() {
    const col = cols.find((c) => c.key === sort.key);
    const list = [...events].sort((a, b) => col.cmp(a, b) * sort.dir);
    heads.forEach((th, i) => th.setAttribute('aria-sort', cols[i].key === sort.key ? (sort.dir > 0 ? 'ascending' : 'descending') : 'none'));
    rowEls = list.map((e) => {
      const tr = h('tr', { onclick: (ev) => { if (!ev.target.closest('button')) onPick(e); } },
        h('td', {}, chip(e.label)),
        h('td', {}, h('button', { type: 'button', class: 'play-btn mono', 'aria-label': `Play ${classInfo(e.label).name} from ${fmtTime(e.start)}`, onclick: () => onPick(e) },
          h('span', { html: ICON.play, 'aria-hidden': 'true' }), fmtTime(e.start))),
        h('td', { class: 'mono' }, fmtTime(e.end)),
        h('td', { class: 'mono' }, `${(e.end - e.start).toFixed(1)} s`),
        h('td', {}, describeInfo(e.info) || '–'),
        h('td', { class: 'mono' }, e.tracks?.length ? e.tracks.join(', ') : '–'));
      return { tr, e };
    });
    tbody.replaceChildren(...rowEls.map((x) => x.tr));
  }
  render();

  return {
    el: h('div', { class: 'table-scroll table-scroll--tall' },
      h('table', { class: 'data-table data-table--events' }, h('thead', {}, h('tr', {}, heads)), tbody)),
    setTime(t) {
      for (const { tr, e } of rowEls) tr.classList.toggle('is-active', t >= e.start && t <= e.end);
    },
  };
}

function eventGallery(events, name, onPick) {
  const byClass = new Map();
  for (const e of events) (byClass.get(e.label) || byClass.set(e.label, []).get(e.label)).push(e);
  if (!byClass.size) return null;
  const groups = [...byClass.keys()].sort(classOrder).map((label) => {
    const list = byClass.get(label);
    // prefer events with thumbnails, then longer ones (usually clearer examples)
    const picks = [...list].sort((a, b) => (!!b.thumb - !!a.thumb) || (b.end - b.start) - (a.end - a.start)).slice(0, 4)
      .sort((a, b) => a.start - b.start);
    return h('div', { class: 'gal-group' },
      h('h4', {}, chip(label, `${list.length}`)),
      h('div', { class: 'gal-grid' }, picks.map((e) => {
        const alt = `${classInfo(label).name} in ${name} at ${fmtTime(e.start)}`;
        const tile = h('span', { class: 'gal-ph', style: { '--cls': classVar(label) }, 'aria-hidden': 'true' },
          h('span', {}, classInfo(label).name), h('span', { class: 'mono' }, fmtTime(e.start)));
        const img = e.thumb ? h('img', { src: e.thumb, alt, loading: 'lazy', decoding: 'async', onerror: () => img.replaceWith(tile) }) : tile;
        return h('button', { type: 'button', class: 'gal-item', 'aria-label': `${alt}. Play from here.`, onclick: () => onPick(e) },
          img,
          h('span', { class: 'gal-cap' }, h('span', { class: 'mono' }, `${fmtTime(e.start)}–${fmtTime(e.end)}`), describeInfo(e.info) ? ` · ${describeInfo(e.info)}` : ''));
      })));
  });
  return h('div', { class: 'card rv-card' },
    h('div', { class: 'card-head' }, h('h3', {}, 'Examples of each class'),
      h('p', { class: 'hint' }, 'Up to four examples per class. Select one to play it.')),
    groups);
}

function failureBlock(failures, onPick) {
  return h('div', { class: 'card rv-card rv-failures' },
    h('div', { class: 'card-head' }, h('h3', {}, 'Honest failure cases'),
      h('p', { class: 'hint' }, 'Where the pipeline is wrong on this video, and why.')),
    h('div', { class: 'fail-grid' }, failures.map((f) => h('article', { class: 'fail' },
      f.image ? h('img', { src: f.image, alt: f.title || 'failure case', loading: 'lazy', onerror: (e) => e.target.remove() }) : null,
      h('div', { class: 'fail-meta' },
        f.kind ? h('span', { class: 'badge' }, f.kind) : null,
        f.label ? chip(f.label) : null),
      f.title ? h('h4', {}, f.title) : null,
      f.text ? h('p', {}, f.text) : null,
      Number.isFinite(f.start) ? h('button', { type: 'button', class: 'btn btn-small', onclick: () => onPick(f) },
        h('span', { html: ICON.play, 'aria-hidden': 'true' }), ` Watch from ${fmtTime(f.start)}`) : null))));
}
