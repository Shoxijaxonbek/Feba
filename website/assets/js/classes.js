// Single source of truth for colour and identity: event classes, signal states, object classes.
// Event-class order is the official label order; colours were chosen so that neighbours in this
// order stay distinguishable (incl. protan/deutan simulation) in both light and dark themes.

const GREY = { light: '#898781', dark: '#898781' };

export const CLASSES = [
  { id: 'accident', name: 'Accident', light: '#e34948', dark: '#e66767', emitted: true,
    rule: 'Ground footprints of two road users touch while closing in, both stop within 2.5 s and stay stopped ≥ 4 s, outside the signal queue.' },
  { id: 'near_miss', name: 'Near miss', light: '#4a3aa7', dark: '#9085e9', emitted: true,
    rule: 'The causal conflict score (required deceleration, held over three processed frames) stays above the alarm level for ≥ 0.4 s and no contact follows.' },
  { id: 'red_light', name: 'Red light', light: '#eb6834', dark: '#d95926', emitted: true,
    rule: 'Vehicle crosses the stop line after the main signal has been red ≥ 1 s and enters the junction box still on red; ends when it leaves the box.' },
  { id: 'wrong_way', name: 'Wrong way', light: '#1baf7a', dark: '#199e70', emitted: true,
    rule: 'Vehicle moves against the learned lane direction for ≥ 1.5 s and ≥ 80 px, on the carriageways only: inside the junction box several legal flows cross, so the learned direction means nothing there.' },
  { id: 'illegal_u_turn', name: 'Illegal U-turn', light: '#2a78d6', dark: '#3987e5', emitted: true,
    rule: 'One continuous track (no ID switch, no frame-edge clipping) whose start and end headings differ by ≥ 150°; the segment is the contiguous turning stretch around the point where the heading passes 90°.' },
  { id: 'stopped_vehicle', name: 'Stopped vehicle', light: '#eda100', dark: '#c98500', emitted: true,
    rule: 'Vehicle stationary ≥ 10 s on a carriageway while the traffic around it keeps moving (queues and jams stand still together). Junction box, queue and stop zones and buses at the bus stop are excluded; fragmented tracks at one spot are joined.' },
  { id: 'jaywalking', name: 'Jaywalking', light: '#e87ba4', dark: '#d55181', emitted: true,
    rule: 'Pedestrian on the carriageway ≥ 25 px inside the curb, ≥ 40 px from every zebra, not on an island or at the bus stop, for ≥ 2 s; the segment covers the whole time the person is on the road outside a crossing.' },
  { id: 'failure_to_yield', name: 'Failure to yield', light: '#008300', dark: '#008300', emitted: true,
    rule: 'A moving vehicle’s ground edge is inside a zebra while a pedestrian is on the roadway part of the same zebra within 1.5 vehicle widths. The event is the vehicle’s passage through the crossing.' },
  { id: 'illegal_turn', name: 'Illegal turn', ...GREY, emitted: false },
  { id: 'solid_line_crossing', name: 'Solid line crossing', ...GREY, emitted: false },
  { id: 'stop_line', name: 'Stop line', light: '#8e4a9e', dark: '#a45cb6', emitted: true,
    rule: 'Vehicle stationary ≥ 2 s on red with its front more than 0.6 of its own length past the stop line (a bonnet over the line is not flagged); ends at the next green or when it drives off.' },
  { id: 'congestion', name: 'Congestion', light: '#a0622b', dark: '#a86a32', emitted: true,
    rule: '≥ 6 vehicles stationary or crawling in the approach while its signal is green, for ≥ 10 s.' },
  { id: 'road_obstacle', name: 'Road obstacle', ...GREY, emitted: false },
  { id: 'fire_smoke', name: 'Fire / smoke', ...GREY, emitted: false },
];

const BY_ID = new Map(CLASSES.map((c, i) => [c.id, { ...c, order: i }]));

export function classInfo(id) {
  return BY_ID.get(id) || { id, name: String(id).replace(/_/g, ' '), ...GREY, emitted: false, order: 99 };
}

export const classOrder = (a, b) => classInfo(a).order - classInfo(b).order;

export function theme() {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light';
}

export const classColor = (id) => classInfo(id)[theme()];

// CSS custom properties --c-<id> for both themes, so markup can colour chips without JS lookups.
export function installClassCss() {
  const decl = (mode) => CLASSES.map((c) => `--c-${c.id}:${c[mode]};`).join('');
  const style = document.createElement('style');
  style.textContent = `:root{${decl('light')}}:root[data-theme="dark"]{${decl('dark')}}`;
  document.head.append(style);
}

export const classVar = (id) => (BY_ID.has(id) ? `var(--c-${id})` : GREY.light);

// Main vehicle signal. Aliases cover state names a future pipeline version might emit.
const SIGNAL = {
  red: { name: 'Red', color: '#d03b3b' },
  amber: { name: 'Amber', color: '#fab219' },
  red_amber: { name: 'Red + amber', color: '#ec835a' },
  green: { name: 'Green', color: '#0ca30c' },
  green_flash: { name: 'Flashing green', color: '#7ccf6f' },
  unknown: { name: 'Unknown', color: '#898781' },
};
const SIGNAL_ALIAS = { yellow: 'amber', 'red+amber': 'red_amber', redamber: 'red_amber',
  flashing_green: 'green_flash', green_flashing: 'green_flash', flash: 'green_flash', off: 'unknown' };

export function signalInfo(state) {
  const key = String(state ?? 'unknown').toLowerCase();
  const id = SIGNAL[key] ? key : SIGNAL_ALIAS[key] || 'unknown';
  return { id, ...SIGNAL[id] };
}

// COCO object classes used in counts charts (default categorical order, validated per theme).
const OBJECTS = [
  { id: 'car', name: 'Cars', light: '#2a78d6', dark: '#3987e5' },
  { id: 'person', name: 'Pedestrians', light: '#eb6834', dark: '#d95926' },
  { id: 'bus', name: 'Buses', light: '#1baf7a', dark: '#199e70' },
  { id: 'truck', name: 'Trucks', light: '#eda100', dark: '#c98500' },
  { id: 'motorcycle', name: 'Motorcycles', light: '#e87ba4', dark: '#d55181' },
  { id: 'bicycle', name: 'Bicycles', light: '#008300', dark: '#008300' },
];

export function objectClasses(keys) {
  const known = OBJECTS.filter((o) => keys.includes(o.id));
  const extra = keys.filter((k) => k !== 't' && !OBJECTS.some((o) => o.id === k))
    .map((id) => ({ id, name: id, ...GREY }));
  return [...known, ...extra].map((o) => ({ ...o, color: o[theme()] }));
}
