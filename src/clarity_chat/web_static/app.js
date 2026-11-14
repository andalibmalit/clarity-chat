// Clarity's page: the core (v0.3.0 item B6). The secret, the requests, the event stream,
// the page load and the live tab. The screens (Group C) build on `clarity` below.
//
// Rules this file keeps (tests/test_web.py checks the first):
// - Server HTML goes into the page only through swap(); text only through textContent.
// - The page never measures text: it joins a reply by the numbers the server sends (`at`,
//   `end`, `reply_end`), since JavaScript counts UTF-16 units and the server code points.
"use strict";

const KEY_NAME = "clarity-key";
const TAB = crypto.randomUUID(); // in memory only: a duplicated tab is a new tab

const clarity = {
  key: null, // the start's secret
  live: false,
  state: null, // the latest /state
  writing: { k: null, text: "", end: 0 }, // the reply being written, joined by numbers
  listeners: [], // screens' event handlers (Group C)
  buffer: null, // events held while the page loads; null once it has
};
window.clarity = clarity;

function readKey() {
  // A key in the link's fragment wins over a stored one, then leaves the address bar.
  const match = /^#key=([A-Za-z0-9_-]+)$/.exec(location.hash);
  if (match) {
    try {
      sessionStorage.setItem(KEY_NAME, match[1]);
    } catch (e) {
      // storage blocked: the key lives in memory, and a reload needs the link again
    }
    history.replaceState(null, "", location.pathname);
    return match[1];
  }
  try {
    return sessionStorage.getItem(KEY_NAME);
  } catch (e) {
    return null;
  }
}

function show(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

function noKey() {
  // Board state A3: no secret, or one from an earlier start.
  try {
    sessionStorage.removeItem(KEY_NAME);
  } catch (e) {}
  clarity.key = null;
  document.body.dataset.page = "no-key";
  show("status", "Open Clarity from the link in the terminal.");
}

function stopped() {
  document.body.dataset.page = "stopped";
  show("status", "Clarity has stopped.");
}

async function request(path, init = {}) {
  const response = await fetch(path, {
    ...init,
    headers: { ...(init.headers || {}), Authorization: `Bearer ${clarity.key}` },
  });
  if (response.status === 401) {
    noKey();
    throw new Error("no key");
  }
  return response;
}

async function command(name, args = "") {
  const response = await request("/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command: name, args, id: Date.now(), tab: TAB }),
  });
  if (response.status === 409) {
    notLive();
    throw new Error("not the live tab");
  }
  return response.json();
}

async function view(name, params = {}) {
  const query = new URLSearchParams(params).toString();
  const response = await request(`/view/${name}${query ? `?${query}` : ""}`);
  return response.text();
}

// The page's one Trusted Types policy: the CSP lets nothing else make HTML from a string.
const POLICY = window.trustedTypes
  ? window.trustedTypes.createPolicy("clarity", { createHTML: (html) => html })
  : null;

function swap(element, html) {
  // The one place server HTML enters the page: parsed, then its nodes moved in.
  const parsed = new DOMParser().parseFromString(POLICY ? POLICY.createHTML(html) : html, "text/html");
  element.replaceChildren(...parsed.body.childNodes);
}

async function download() {
  const response = await request("/export");
  if (!response.ok) return response.text();
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "export.jsonl";
  link.click();
  URL.revokeObjectURL(link.href);
  return null;
}

function notLive() {
  // Board state A4: another tab is live; this one can take over.
  clarity.live = false;
  document.body.dataset.page = "not-live";
  show("status", "Clarity is open in another tab.");
  const button = document.getElementById("use-here");
  if (button) button.hidden = false;
}

async function useHere() {
  await request("/tab", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tab: TAB }),
  });
  load();
}

function apply(event) {
  // Join the reply being written by numbers; hand everything to the screens.
  if (clarity.buffer !== null) {
    clarity.buffer.push(event);
    return;
  }
  if (event.event === "reply_start") {
    clarity.writing = { k: event.k, text: "", end: 0 };
  } else if (event.event === "text") {
    if (event.at >= clarity.writing.end) {
      clarity.writing.text += event.text;
      clarity.writing.end = event.end;
    }
    show("writing", clarity.writing.text);
  } else if (event.event === "record" && event.record.type === "call") {
    document.body.dataset.lastCall = event.record.call_id;
  } else if (event.event === "not_live") {
    if (event.live !== TAB) notLive();
  } else if (event.event === "stopped") {
    stopped();
  }
  for (const listener of clarity.listeners) listener(event);
}

async function stream(onHello) {
  // The event stream: JSON lines over fetch (EventSource can't send the secret).
  const response = await request(`/events?tab=${encodeURIComponent(TAB)}`);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline);
      buffer = buffer.slice(newline + 1);
      if (!line) continue;
      const event = JSON.parse(line);
      if (event.event === "hello") onHello();
      else if (event.event !== "heartbeat") apply(event);
    }
  }
}

async function load() {
  // The stream first, so nothing sent while the page reads is lost; then /state; then
  // the events buffered meanwhile, joined by their numbers.
  clarity.buffer = [];
  await new Promise((resolve) => connect(resolve));
  const answer = await command("state");
  const state = answer.result;
  clarity.state = state;
  clarity.live = true;
  const k = state.running ? state.running.k : null;
  clarity.writing = { k, text: state.reply_so_far, end: state.reply_end };
  show("writing", clarity.writing.text);
  const buffered = clarity.buffer;
  clarity.buffer = null;
  for (const event of buffered) apply(event);
  document.body.dataset.page = "ready";
  show("status", state.conversation_id ? `Conversation ${state.conversation_id}` : "Ready.");
}

let retry = 0;
function connect(onHello) {
  stream(() => {
    retry = 0;
    onHello();
  })
    .then(() => reconnect())
    .catch((error) => {
      if (clarity.key && error.message !== "no key") reconnect();
    });
}

function reconnect() {
  // A dropped stream: try again, waiting longer each time, then load the page again.
  if (!clarity.key || document.body.dataset.page === "stopped") return;
  retry += 1;
  if (retry > 6) {
    stopped();
    return;
  }
  setTimeout(() => load().catch(() => {}), Math.min(250 * 2 ** retry, 8000));
}

clarity.command = command;
clarity.view = view;
clarity.swap = swap;
clarity.download = download;
clarity.useHere = useHere;

document.addEventListener("DOMContentLoaded", () => {
  clarity.key = readKey();
  if (!clarity.key) {
    noKey();
    return;
  }
  const button = document.getElementById("use-here");
  if (button) button.addEventListener("click", () => useHere());
  load().catch(() => {});
});
