// Report: data/report.md rendered with marked (jsDelivr), plus the PDF link from site.json.
import { h, fill, isRealLink, ICON } from '../util.js';
import { loadReport, loadSite } from '../store.js';

const MARKED_URL = 'https://cdn.jsdelivr.net/npm/marked@12.0.2/lib/marked.esm.js';

export async function initReport() {
  const [md, site] = await Promise.all([loadReport(), loadSite()]);
  const body = document.getElementById('report-body');
  const pdf = site?.links?.report_pdf;
  const pdfLink = isRealLink(pdf)
    ? h('p', {}, h('a', { class: 'btn', href: pdf, target: '_blank', rel: 'noopener' }, h('span', { html: ICON.download, 'aria-hidden': 'true' }), ' Download the report (PDF)'))
    : null;

  if (!md) {
    fill(body, h('div', { class: 'prose' },
      h('p', {}, 'The one-page report (what we built, what worked, what did not, what next) will appear here once data/report.md is published.')), pdfLink);
    return;
  }
  const text = md.replace(/<!--[\s\S]*?-->/g, '').trim();
  let html;
  try {
    const { marked } = await import(MARKED_URL);
    html = marked.parse(text, { gfm: true });
  } catch {
    html = null; // offline: fall back to plain text below
  }
  const prose = h('div', { class: 'prose' });
  if (html) prose.innerHTML = html;
  else prose.append(h('pre', { class: 'prose-raw' }, text));
  prose.querySelectorAll('table').forEach((t) => {
    const wrap = h('div', { class: 'table-scroll' });
    t.replaceWith(wrap);
    wrap.append(t);
    t.classList.add('data-table');
  });
  prose.querySelectorAll('a[href^="http"]').forEach((a) => { a.target = '_blank'; a.rel = 'noopener'; });
  fill(body, prose, pdfLink);
}
