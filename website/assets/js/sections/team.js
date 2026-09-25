// Team cards from data/site.json. Null fields are skipped; "#" links render as "link pending".
import { h, fill, isRealLink, ICON } from '../util.js';
import { loadSite } from '../store.js';

const LINKS = [
  ['github', 'GitHub', ICON.github],
  ['linkedin', 'LinkedIn', ICON.linkedin],
  ['portfolio', 'Portfolio', ICON.globe],
  ['youtube', 'YouTube', ICON.youtube],
  ['telegram', 'Telegram', ICON.telegram],
];

const initials = (name) => (name || '?').split(/\s+/).map((w) => w[0]).slice(0, 2).join('').toUpperCase();

function avatar(m) {
  const fallback = h('span', { class: 'avatar avatar--initials', 'aria-hidden': 'true' }, initials(m.name));
  if (!m.photo) return fallback;
  const img = h('img', { class: 'avatar', src: m.photo, alt: `Photo of ${m.name}`, loading: 'lazy', width: 72, height: 72,
    onerror: () => img.replaceWith(fallback) });
  return img;
}

function linkRow(m) {
  const items = LINKS.filter(([key]) => m[key]).map(([key, label, icon]) => (isRealLink(m[key])
    ? h('a', { class: 'icon-link', href: m[key], target: '_blank', rel: 'noopener', 'aria-label': `${m.name} on ${label}` },
        h('span', { html: icon, 'aria-hidden': 'true' }), label)
    : h('span', { class: 'icon-link is-pending', title: `${label} link pending` },
        h('span', { html: icon, 'aria-hidden': 'true' }), label)));
  return items.length ? h('div', { class: 'links-row' }, items) : null;
}

function memberCard(m) {
  const projects = Array.isArray(m.projects) ? m.projects.filter((p) => p && p.title) : [];
  const contributions = Array.isArray(m.contributions) ? m.contributions.filter(Boolean) : [];
  return h('article', { class: 'card member' },
    h('header', { class: 'member-head' }, avatar(m),
      h('div', {}, h('h3', {}, m.name || 'Team member'), m.role ? h('p', { class: 'member-role' }, m.role) : null)),
    contributions.length ? h('div', {}, h('h4', { class: 'mini-head' }, 'Contributed'),
      h('ul', { class: 'tick-list' }, contributions.map((c) => h('li', {}, c)))) : null,
    projects.length ? h('div', {}, h('h4', { class: 'mini-head' }, 'Previous projects'),
      h('ul', { class: 'project-list' }, projects.map((p) => h('li', {},
        isRealLink(p.url) ? h('a', { href: p.url, target: '_blank', rel: 'noopener' }, p.title) : h('strong', {}, p.title),
        p.blurb ? h('span', {}, ` · ${p.blurb}`) : null)))) : null,
    linkRow(m));
}

export async function initTeam() {
  const site = await loadSite();
  const team = site?.team;
  const box = document.getElementById('team-grid');
  if (!team) {
    fill(box, h('p', { class: 'empty' }, 'Team details will appear here (data/site.json).'));
    return;
  }
  if (team.tagline) document.getElementById('team-tagline').textContent = team.tagline;
  const members = Array.isArray(team.members) ? team.members.filter(Boolean) : [];
  fill(box, members.length ? members.map(memberCard) : h('p', { class: 'empty' }, 'No team members listed yet.'));
  if (team.name) document.querySelectorAll('[data-team-name]').forEach((el) => { el.textContent = team.name; });
}
