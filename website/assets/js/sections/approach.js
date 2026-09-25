// Per-class rule summary, generated from the class registry so names and colours match everywhere.
import { h, fill } from '../util.js';
import { CLASSES } from '../classes.js';
import { chip } from '../result-view.js';

export function initApproach() {
  const emitted = CLASSES.filter((c) => c.emitted);
  const skipped = CLASSES.filter((c) => !c.emitted);
  fill(document.getElementById('rules-list'),
    h('dl', { class: 'rules' }, emitted.map((c) => h('div', { class: 'rule' },
      h('dt', {}, chip(c.id)), h('dd', {}, c.rule)))),
    h('div', { class: 'rule rule--skipped' },
      h('p', { class: 'chips' }, skipped.map((c) => chip(c.id))),
      h('p', {}, h('strong', {}, 'Not predicted. '),
        'The score is a macro-F1 over classes, and a class we predict wrongly adds a zero to the average, ',
        'so we only emit classes we can detect reliably. Road obstacles and fire/smoke are also not COCO classes, so the detector cannot see them.')));
}
