// Helpers every page loads from /common.js (served read-only by src/datafiles.py), so one fix reaches every tab.
const $ = (id) => document.getElementById(id);
// text made safe for innerHTML and for attribute values in either kind of quotes
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const get = (u) => fetch(u, { cache: "no-store" }).then(async (r) => r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${u}`)));
const pct = (x) => x == null ? "n/a" : `${(100 * x).toFixed(1)}%`;
// run an async render; if it fails, show the error in the element with id `box` instead of failing silently
const guard = (fn, box) => fn().catch((e) => { $(box).innerHTML = `<p class="err">${esc(e.message)}</p>`; });
