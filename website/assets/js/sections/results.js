// Results on the sample videos: a video selector plus the shared result viewer.
import { h, fill, fmtNum, fmtTime, segmented, selectSegment } from '../util.js';
import { loadIndex, loadSample, loadFailures } from '../store.js';
import { mountResult } from '../result-view.js';

function metaLine(entry, result) {
  const t = result?.timing;
  const parts = [
    entry.file || entry.id,
    entry.width && entry.height ? `${entry.width} × ${entry.height}` : null,
    entry.fps ? `${fmtNum(entry.fps, 2)} fps` : null,
    `${fmtTime(entry.duration ?? result?.duration, false)} long`,
    `${fmtNum(result?.events?.length ?? entry.n_events)} events`,
    t?.part_a_sec && t?.video_sec ? `Part A took ${fmtTime(t.part_a_sec, false)} (${(t.part_a_sec / t.video_sec).toFixed(2)}× video length)` : null,
  ];
  return h('p', { class: 'meta-line' }, parts.filter(Boolean).join(' · '));
}

export async function initResults() {
  const [idx, failures] = await Promise.all([loadIndex(), loadFailures()]);
  const pickerBox = document.getElementById('results-picker');
  const view = document.getElementById('results-view');
  if (!idx.length) {
    fill(view, h('p', { class: 'empty' }, 'Sample results will appear here (data/samples/index.json).'));
    return;
  }
  let current = null;
  let currentId = null;
  let token = 0;

  async function show(id, seekTo) {
    const mine = ++token;
    const entry = idx.find((e) => e.id === id);
    current?.destroy();
    current = null;
    currentId = id;
    fill(view, h('p', { class: 'loading' }, `Loading ${id}…`));
    const result = await loadSample(id);
    if (mine !== token) return; // a newer selection won
    if (!result) {
      fill(view, h('p', { class: 'empty' }, `No result file for ${id} yet (data/samples/${id}.json).`));
      return;
    }
    const box = h('div');
    fill(view, metaLine(entry, result), box);
    current = mountResult(box, result, {
      name: id,
      videoUrl: result.annotated_video || entry.annotated_video,
      poster: entry.poster,
      failures,
    });
    if (seekTo != null) current.seek(seekTo, { reveal: true });
  }

  const picker = segmented(idx.map((e) => ({ id: e.id, label: e.id })), idx[0].id, (id) => show(id), 'Sample video');
  fill(pickerBox, h('span', { class: 'toolbar-label' }, 'Sample video'), picker);

  // the dashboard's combined timeline asks to open a video at a given time
  addEventListener('rs:open-result', async (e) => {
    const { id, t } = e.detail;
    if (!idx.some((x) => x.id === id)) return;
    selectSegment(picker, id);
    if (id === currentId && current) current.seek(t, { reveal: true });
    else await show(id, t);
  });

  await show(idx[0].id);
}
