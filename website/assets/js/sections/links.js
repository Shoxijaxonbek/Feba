// Project links from data/site.json. Missing or "#" links render as "coming soon".
import { h, fill, isRealLink, ICON } from '../util.js';
import { loadSite } from '../store.js';

const ITEMS = [
  ['repo', 'Source code', 'Pipeline, rules, evaluation and this website.', ICON.github],
  ['space', 'Live site and demo', 'This website with the upload demo, hosted on Hugging Face Spaces.', ICON.globe],
  ['weights', 'Model weights', 'Detector weights and the scene configuration files.', ICON.box],
  ['predictions', 'Predictions on the samples', 'predictions_samples.json in the official submission format.', ICON.doc],
  ['report_pdf', 'Report (PDF)', 'The written report for the jury.', ICON.download],
];

/** Relative links are checked with HEAD so a missing file shows as "not published yet". */
async function exists(url) {
  if (/^https?:/i.test(url)) return true;
  try {
    return (await fetch(url, { method: 'HEAD', cache: 'no-store' })).ok;
  } catch {
    return false;
  }
}

export async function initLinks() {
  const site = await loadSite();
  const links = site?.links || {};
  const cards = await Promise.all(ITEMS.map(async ([key, title, text, icon]) => {
    const url = links[key];
    const ok = isRealLink(url) && (await exists(url));
    const inner = [h('span', { class: 'link-icon', html: icon, 'aria-hidden': 'true' }),
      h('span', { class: 'link-text' }, h('strong', {}, title), h('span', {}, text)),
      h('span', { class: 'link-state' }, ok ? 'Open' : 'Coming soon')];
    return ok
      ? h('a', { class: 'card link-card', href: url, target: '_blank', rel: 'noopener' }, inner)
      : h('div', { class: 'card link-card is-pending', 'aria-disabled': 'true' }, inner);
  }));
  fill(document.getElementById('links-body'), h('div', { class: 'link-grid' }, cards));
}
