// Interactive event timeline: one labelled row per group, a shared time axis, an optional
// signal-state strip and a playhead. Bars are buttons (keyboard focusable); clicking a bar
// or empty track space calls onSeek(time, item).
import { h, fmtTime, clamp } from './util.js';
import { signalInfo } from './classes.js';

const LANE_H = 12;
const LANE_GAP = 3;
const MAX_LANES = 4;

/** Greedy interval partitioning so overlapping events in one row sit on separate lanes. */
function assignLanes(items) {
  const ends = [];
  return [...items].sort((a, b) => a.start - b.start).map((it) => {
    let lane = ends.findIndex((end) => end <= it.start);
    if (lane === -1) lane = ends.length < MAX_LANES ? ends.length : ends.indexOf(Math.min(...ends));
    ends[lane] = Math.max(ends[lane] ?? 0, it.end);
    return { ...it, lane };
  });
}

export function tickStep(duration) {
  const steps = [5, 10, 15, 30, 60, 120, 300, 600, 1800];
  return steps.find((st) => duration / st <= 8) || 3600;
}

/**
 * rows: [{label, color, count?, items: [{start, end, title, detail, color?, data}]}]
 * signal: optional [[t, state], ...] change points; duration: seconds covered by the axis.
 */
export function createTimeline({ duration, rows, signal, onSeek, ariaLabel }) {
  const pct = (t) => `${(clamp(t, 0, duration) / duration) * 100}%`;
  const bars = [];

  const tip = h('div', { class: 'tl-tip', role: 'tooltip', hidden: true });
  const playhead = h('div', { class: 'tl-playhead', hidden: true, 'aria-hidden': 'true' });
  const root = h('div', { class: `tl${onSeek ? '' : ' tl--static'}`, role: 'group', 'aria-label': ariaLabel || 'Event timeline' });

  const step = tickStep(duration);
  const ticks = [];
  for (let t = 0, i = 0; t <= duration + 1e-6; t += step, i++) {
    ticks.push(h('span', { class: `tl-tick${i % 2 ? ' tl-tick--odd' : ''}`, style: { left: pct(t) } }, fmtTime(t, false)));
  }
  root.append(h('div', { class: 'tl-row tl-row--axis', 'aria-hidden': 'true' },
    h('div', { class: 'tl-label' }), h('div', { class: 'tl-track tl-track--axis' }, ticks)));

  const seekFromTrack = (ev) => {
    if (ev.target.closest('.tl-bar')) return;
    const r = ev.currentTarget.getBoundingClientRect();
    onSeek?.(((ev.clientX - r.left) / r.width) * duration, null);
  };

  if (signal?.length) {
    const segs = signal.map(([t0, state], i) => {
      const t1 = i + 1 < signal.length ? signal[i + 1][0] : duration;
      const info = signalInfo(state);
      return h('span', {
        class: `tl-sig tl-sig--${info.id}`, title: `${info.name} ${fmtTime(t0)}–${fmtTime(t1)}`,
        style: { left: pct(t0), width: pct(t1 - t0), background: info.color },
      });
    });
    root.append(h('div', { class: 'tl-row tl-row--signal' },
      h('div', { class: 'tl-label' }, 'Main signal'),
      h('div', { class: 'tl-track tl-track--signal', role: 'img', 'aria-label': `Main signal state, ${signal.length} phases`, onclick: seekFromTrack }, segs)));
  }

  for (const row of rows) {
    const items = assignLanes(row.items);
    const lanes = Math.max(1, ...items.map((i) => i.lane + 1));
    const track = h('div', { class: 'tl-track', style: { height: `${lanes * LANE_H + (lanes - 1) * LANE_GAP + 8}px` }, onclick: seekFromTrack });
    for (const it of items) {
      const bar = h('button', {
        type: 'button', class: 'tl-bar',
        style: { left: pct(it.start), width: pct(it.end - it.start), top: `${4 + it.lane * (LANE_H + LANE_GAP)}px`,
          '--bar': it.color || row.color },
        'aria-label': `${it.title}, ${fmtTime(it.start)} to ${fmtTime(it.end)}${it.detail ? `, ${it.detail}` : ''}. Press to play from here.`,
        onclick: () => onSeek?.(it.start, it),
        onmouseenter: (e) => showTip(e.currentTarget, it),
        onfocus: (e) => showTip(e.currentTarget, it),
        onmouseleave: hideTip, onblur: hideTip,
      });
      bars.push({ el: bar, it });
      track.append(bar);
    }
    root.append(h('div', { class: 'tl-row' },
      h('div', { class: 'tl-label', title: row.label },
        h('i', { class: 'swatch', style: { background: row.color }, 'aria-hidden': 'true' }),
        h('span', { class: 'tl-name' }, row.label),
        row.count != null ? h('span', { class: 'tl-count' }, row.count) : null),
      track));
  }

  const layer = h('div', { class: 'tl-layer', 'aria-hidden': 'true' }, playhead);
  root.append(layer, tip);

  function showTip(bar, it) {
    tip.replaceChildren(
      h('strong', {}, it.title),
      h('span', {}, `${fmtTime(it.start)} – ${fmtTime(it.end)} · ${(it.end - it.start).toFixed(1)} s`),
      it.detail ? h('span', { class: 'tl-tip-detail' }, it.detail) : null);
    tip.hidden = false;
    const rb = bar.getBoundingClientRect();
    const rr = root.getBoundingClientRect();
    const tw = tip.offsetWidth;
    const center = rb.left + Math.min(rb.width, 40) / 2 - rr.left;
    tip.style.left = `${clamp(center - tw / 2, 0, rr.width - tw)}px`;
    tip.style.top = `${rb.top - rr.top - tip.offsetHeight - 6}px`;
  }
  function hideTip() { tip.hidden = true; }

  return {
    el: root,
    setTime(t) {
      playhead.style.left = pct(t);
      playhead.hidden = t == null;
      for (const { el, it } of bars) el.classList.toggle('is-active', t >= it.start && t <= it.end);
    },
  };
}
