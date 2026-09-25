// Live demo: upload an .mp4 to the same-origin API, poll the job, render the result with the
// shared result viewer. Degrades to a "backend offline" notice on static hosting.
import { h, fill, fmtNum, fmtTime, ICON } from '../util.js';
import { loadIndex, loadSample } from '../store.js';
import { mountResult } from '../result-view.js';

const API = '/api';
const MAX_MB = 500;
const MAX_SEC = 120;
const POLL_MS = 2000;

const STAGES = {
  queued: 'Waiting in the queue', uploading: 'Uploading', decoding: 'Decoding frames', detecting: 'Detecting road users',
  tracking: 'Tracking road users', signal: 'Reading the traffic light', rules: 'Applying the event rules',
  events: 'Applying the event rules', risk: 'Scoring accident risk', rendering: 'Rendering the annotated video',
  render: 'Rendering the annotated video', done: 'Done',
};
const stageText = (s) => {
  const text = STAGES[s] || String(s || 'working').replace(/_/g, ' ');
  return text[0].toUpperCase() + text.slice(1);
};

class OfflineError extends Error {}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** GET /api/health -> {ok, queue, ...} from the demo server; null on static hosting (404) or network errors. */
async function probeBackend() {
  try {
    const res = await fetch(`${API}/health`, { cache: 'no-store' });
    const body = res.ok ? await res.json() : null;
    return body?.ok ? body : null;
  } catch {
    return null;
  }
}

function upload(file, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API}/jobs`);
    xhr.responseType = 'json';
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300 && body?.job_id) return resolve(body.job_id);
      if (xhr.status === 404 || xhr.status === 405 || xhr.status === 501) return reject(new OfflineError());
      if (xhr.status === 413) return reject(new Error(`The file is too large (limit ${MAX_MB} MB).`));
      const detail = typeof body?.detail === 'string' ? body.detail : body?.error;
      reject(new Error(detail || `The server rejected the upload (HTTP ${xhr.status}).`));
    };
    xhr.onerror = () => reject(new OfflineError());
    const form = new FormData();
    form.append('video', file);
    xhr.send(form);
  });
}

/** Reads the duration locally; null when the browser cannot decode the container. */
function localDuration(file) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(file);
    const v = document.createElement('video');
    const done = (d) => { URL.revokeObjectURL(url); resolve(d); };
    const timer = setTimeout(() => done(null), 5000);
    v.preload = 'metadata';
    v.onloadedmetadata = () => { clearTimeout(timer); done(Number.isFinite(v.duration) ? v.duration : null); };
    v.onerror = () => { clearTimeout(timer); done(null); };
    v.src = url;
  });
}

export async function initDemo() {
  const root = document.getElementById('demo-body');
  const status = h('p', { class: 'backend-status', role: 'status' }, h('span', { class: 'dot dot--pending', 'aria-hidden': 'true' }), 'Checking whether the demo server is reachable…');
  const input = h('input', { type: 'file', id: 'demo-file', accept: 'video/mp4,.mp4', class: 'sr-only' });
  const zone = h('label', { for: 'demo-file', class: 'dropzone' },
    h('span', { class: 'dz-icon', html: ICON.upload, 'aria-hidden': 'true' }),
    h('span', { class: 'dz-title' }, 'Drop an .mp4 here or choose a file'),
    h('span', { class: 'dz-sub' }, `Up to ${MAX_SEC / 60} minutes and ${MAX_MB} MB`));
  const picked = h('p', { class: 'picked', hidden: true });
  const runBtn = h('button', { type: 'button', class: 'btn btn-primary', disabled: true }, 'Analyse video');
  const sampleBtn = h('button', { type: 'button', class: 'btn' }, 'Try with a sample clip');
  const bar = h('span', { class: 'progress-fill' });
  const progressBar = h('div', { class: 'progress', role: 'progressbar', 'aria-valuemin': 0, 'aria-valuemax': 100, 'aria-valuenow': 0, 'aria-label': 'Job progress' }, bar);
  const stage = h('p', { class: 'stage' });
  const progress = h('div', { class: 'job', hidden: true, 'aria-live': 'polite' }, stage, progressBar);
  const message = h('div', { class: 'notice', hidden: true, role: 'alert' });
  const resultBox = h('div', { class: 'demo-result' });
  let file = null;
  let view = null;
  let busy = false;
  let online = null; // /api/health response, once probed

  fill(root,
    h('div', { class: 'demo-grid' },
      h('div', { class: 'card demo-upload' }, status, input, zone, picked,
        h('div', { class: 'btn-row' }, runBtn, sampleBtn), progress, message),
      h('aside', { class: 'card demo-notes' },
        h('h3', {}, 'Before you upload'),
        h('ul', { class: 'tick-list' },
          h('li', {}, `.mp4 video, at most ${MAX_SEC / 60} minutes and ${MAX_MB} MB.`),
          h('li', {}, 'Footage from the same junction camera works best: the scene layout and lane directions are specific to that view.'),
          h('li', {}, 'The demo server runs on CPU, so a 2-minute clip takes a few minutes. Keep this tab open to watch the progress.')))),
    resultBox);

  const setBusy = (b) => {
    busy = b;
    runBtn.disabled = b || !file;
    sampleBtn.disabled = b;
    input.disabled = b;
    zone.classList.toggle('is-disabled', b);
  };
  const say = (kind, ...content) => {
    message.className = `notice notice--${kind}`;
    fill(message, ...content);
    message.hidden = false;
  };
  const setProgress = (p, text) => {
    const pct = Math.round(Math.max(0, Math.min(1, p || 0)) * 100);
    progress.hidden = false;
    bar.style.width = `${pct}%`;
    progressBar.setAttribute('aria-valuenow', pct);
    stage.textContent = `${text} · ${pct} %`;
  };
  const offlineNotice = () => say('info',
    h('strong', {}, 'The demo server is offline. '),
    'This copy of the site is static (for example GitHub Pages), so it cannot process uploads. ',
    'Start the backend from the repository and open the site from it, or use “Try with a sample clip” to see a precomputed result.');

  function clearResult() {
    view?.destroy();
    view = null;
    fill(resultBox);
  }

  async function choose(f) {
    message.hidden = true;
    picked.hidden = true;
    file = null;
    runBtn.disabled = true;
    if (!f) return;
    const isMp4 = /\.mp4$/i.test(f.name) || f.type === 'video/mp4';
    if (!isMp4) return say('error', 'Please choose an .mp4 file.');
    if (f.size > MAX_MB * 1024 * 1024) return say('error', `This file is ${fmtNum(f.size / 1048576)} MB; the limit is ${MAX_MB} MB.`);
    const dur = await localDuration(f);
    if (dur != null && dur > MAX_SEC + 0.5) return say('error', `This clip is ${fmtTime(dur, false)} long; the limit is ${MAX_SEC / 60} minutes. Trim it and try again.`);
    file = f;
    picked.hidden = false;
    picked.textContent = `${f.name} · ${fmtNum(f.size / 1048576, 1)} MB${dur != null ? ` · ${fmtTime(dur, false)}` : ''}`;
    runBtn.disabled = busy;
  }

  async function run() {
    if (!file || busy) return;
    setBusy(true);
    message.hidden = true;
    clearResult();
    const started = Date.now();
    try {
      setProgress(0, 'Uploading');
      const id = await upload(file, (p) => setProgress(p, 'Uploading'));
      let failures = 0;
      for (;;) {
        await sleep(POLL_MS);
        let job;
        try {
          const res = await fetch(`${API}/jobs/${encodeURIComponent(id)}`, { cache: 'no-store' });
          if (res.status === 404) throw new Error('The server no longer knows this job (it may have restarted).');
          if (!res.ok) throw new TypeError(`HTTP ${res.status}`);
          job = await res.json();
          failures = 0;
        } catch (err) {
          if (!(err instanceof TypeError) || ++failures > 3) throw err;
          continue; // transient network hiccup: keep polling
        }
        if (job.status === 'error') throw new Error(job.error || 'Processing failed on the server.');
        const elapsed = fmtTime((Date.now() - started) / 1000, false);
        setProgress(job.status === 'done' ? 1 : job.progress, `${stageText(job.status === 'queued' ? 'queued' : job.stage)} · ${elapsed} elapsed`);
        if (job.status === 'done') break;
      }
      const res = await fetch(`${API}/jobs/${encodeURIComponent(id)}/result`, { cache: 'no-store' });
      if (!res.ok) throw new Error(`Could not fetch the result (HTTP ${res.status}).`);
      const result = await res.json();
      progress.hidden = true;
      say('success', `Done in ${fmtTime((Date.now() - started) / 1000, false)}. ${result.events?.length ?? 0} events found.`);
      view = mountResult(resultBox, result, {
        name: file.name, videoUrl: result.annotated_video || `${API}/jobs/${encodeURIComponent(id)}/video`, poster: result.poster,
        gallery: (result.events || []).some((e) => e.thumb),
      });
    } catch (err) {
      progress.hidden = true;
      if (err instanceof OfflineError && !online) offlineNotice();
      else if (err instanceof OfflineError) say('error', h('strong', {}, 'Could not reach the demo server. '), 'Check your connection and try again.');
      else say('error', h('strong', {}, 'Something went wrong. '), err.message || String(err));
    } finally {
      setBusy(false);
    }
  }

  async function trySample() {
    const idx = await loadIndex();
    const entry = idx[0];
    const result = entry && (await loadSample(entry.id));
    clearResult();
    if (!result) return say('error', 'No sample result is available yet.');
    say('info', `Showing the precomputed result for sample ${entry.id}. Upload your own clip above to run the pipeline.`);
    view = mountResult(resultBox, result, {
      name: entry.id, videoUrl: result.annotated_video || entry.annotated_video, poster: entry.poster, gallery: false,
    });
  }

  input.addEventListener('change', () => choose(input.files[0]));
  runBtn.addEventListener('click', run);
  sampleBtn.addEventListener('click', trySample);
  for (const ev of ['dragenter', 'dragover']) {
    zone.addEventListener(ev, (e) => { e.preventDefault(); if (!busy) zone.classList.add('is-over'); });
  }
  for (const ev of ['dragleave', 'drop']) zone.addEventListener(ev, () => zone.classList.remove('is-over'));
  zone.addEventListener('drop', (e) => {
    e.preventDefault();
    if (!busy) choose(e.dataTransfer.files[0]);
  });

  online = await probeBackend();
  const queued = Number.isFinite(online?.queue) && online.queue > 0 ? ` ${online.queue} job(s) ahead of you.` : '';
  fill(status, h('span', { class: `dot ${online ? 'dot--ok' : 'dot--off'}`, 'aria-hidden': 'true' }),
    online ? `Demo server online.${queued}` : 'Demo server offline on this copy of the site.');
  if (!online) offlineNotice();
}
