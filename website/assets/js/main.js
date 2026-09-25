// Entry point: theme, navigation, scroll-spy, and section start-up.
import { installClassCss } from './classes.js';
import { rethemeAll } from './plot.js';
import { initHero } from './sections/hero.js';
import { initTeam } from './sections/team.js';
import { initApproach } from './sections/approach.js';
import { initEda } from './sections/eda.js';
import { initResults } from './sections/results.js';
import { initMetrics } from './sections/metrics.js';
import { initDemo } from './sections/demo.js';
import { initDashboard } from './sections/dashboard.js';
import { initReport } from './sections/report.js';
import { initLinks } from './sections/links.js';

const root = document.documentElement;
const THEME_KEY = 'rs-theme';

function storedTheme() {
  try { return localStorage.getItem(THEME_KEY); } catch { return null; }
}

function applyTheme(mode, persist) {
  root.dataset.theme = mode;
  const btn = document.getElementById('theme-toggle');
  btn?.setAttribute('aria-pressed', String(mode === 'dark'));
  btn?.setAttribute('aria-label', mode === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  if (persist) {
    try { localStorage.setItem(THEME_KEY, mode); } catch { /* private mode: theme just isn't remembered */ }
  }
  rethemeAll();
}

function setupTheme() {
  const mq = matchMedia('(prefers-color-scheme: dark)');
  applyTheme(storedTheme() || (mq.matches ? 'dark' : 'light'), false);
  mq.addEventListener('change', (e) => { if (!storedTheme()) applyTheme(e.matches ? 'dark' : 'light', false); });
  document.getElementById('theme-toggle').addEventListener('click', () => {
    applyTheme(root.dataset.theme === 'dark' ? 'light' : 'dark', true);
  });
}

function setupNav() {
  const nav = document.getElementById('site-nav');
  const menuBtn = document.getElementById('menu-toggle');
  const setOpen = (open) => {
    nav.classList.toggle('is-open', open);
    menuBtn.setAttribute('aria-expanded', String(open));
  };
  menuBtn.addEventListener('click', () => setOpen(!nav.classList.contains('is-open')));
  nav.addEventListener('click', (e) => { if (e.target.closest('a')) setOpen(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') setOpen(false); });

  // scroll-spy: highlight the section occupying the upper part of the viewport
  const links = new Map([...nav.querySelectorAll('a[href^="#"]')].map((a) => [a.getAttribute('href').slice(1), a]));
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      links.forEach((a) => a.removeAttribute('aria-current'));
      links.get(e.target.id)?.setAttribute('aria-current', 'true');
    }
  }, { rootMargin: '-45% 0px -50% 0px' });
  links.forEach((_, id) => { const sec = document.getElementById(id); if (sec) io.observe(sec); });
}

async function start() {
  installClassCss();
  setupTheme();
  setupNav();
  const sections = [initHero, initTeam, initApproach, initEda, initResults, initMetrics, initDemo, initDashboard, initReport, initLinks];
  // Each section fails on its own: one broken data file must not blank the whole page.
  await Promise.all(sections.map((init) => Promise.resolve().then(init).catch((err) => console.error(`[${init.name}]`, err))));
  // content above the target grew while loading, so land on a deep link again
  if (location.hash.length > 1) document.getElementById(location.hash.slice(1))?.scrollIntoView();
}

start();
