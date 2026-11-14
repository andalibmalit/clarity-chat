// Clarity's page. The core (v0.3.0 item B6): the secret, the requests, the event stream,
// the page load and the live tab. The shell (C1): the top bar, the Clarity menu, the
// picker and the row under the message box, the first screen and a model starting, About,
// Quit, and the plain pages. Later screens build on `clarity` below; `clarity.on` holds
// their handlers for controls the shell already draws (Chats, Appearance, Settings, Send).
//
// Rules this file keeps (tests/test_web.py checks the first):
// - Server HTML goes into the page only through swap(); text only through textContent.
// - The page never measures text: it joins a reply by the numbers the server sends (`at`,
//   `end`, `reply_end`), since JavaScript counts UTF-16 units and the server code points.
"use strict";

const KEY_NAME = "clarity-key";
const TAB = crypto.randomUUID(); // in memory only: a duplicated tab is a new tab
const CRASH = "Clarity stopped after an unexpected error: ";

const clarity = {
  key: null, // the start's secret
  live: false,
  state: null, // the latest /state
  writing: { k: null, text: "", end: 0 }, // the reply being written, joined by numbers
  listeners: [], // screens' event handlers
  buffer: null, // events held while the page loads; null once it has
  on: {}, // later screens' handlers: chats, appearance, settings, send
};
window.clarity = clarity;

function $(id) {
  return document.getElementById(id);
}

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

function noKey() {
  // Board state A3: no secret, or one from an earlier start.
  try {
    sessionStorage.removeItem(KEY_NAME);
  } catch (e) {}
  clarity.key = null;
  document.body.dataset.page = "no-key";
}

function stopped(reason) {
  // Board state A5: Quit, a lost connection, or an unexpected error, shown whole.
  document.body.dataset.page = "stopped";
  const crash = Boolean(reason);
  $("stopped-plain").hidden = crash;
  $("stopped-why").hidden = crash;
  $("stopped-crash").hidden = !crash;
  $("stopped-error").hidden = !crash;
  if (crash) $("stopped-error").textContent = reason.startsWith(CRASH) ? reason.slice(CRASH.length) : reason;
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
  const answer = await response.json();
  if (answer.stopped) stopped(answer.error);
  return answer;
}

async function view(name, params = {}) {
  const query = new URLSearchParams(params).toString();
  const response = await request(`/view/${name}${query ? `?${query}` : ""}`);
  if (!response.ok) throw new Error(await response.text());
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
    $("writing").textContent = clarity.writing.text;
  } else if (event.event === "record" && event.record.type === "call") {
    document.body.dataset.lastCall = event.record.call_id;
  } else if (event.event === "not_live") {
    if (event.live !== TAB) notLive();
  } else if (event.event === "stopped") {
    stopped(event.reason);
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
  // the events buffered meanwhile, joined by their numbers; then the screens.
  clarity.buffer = [];
  await new Promise((resolve) => connect(resolve));
  const answer = await command("state");
  const state = answer.result;
  clarity.state = state;
  clarity.live = true;
  const k = state.running ? state.running.k : null;
  clarity.writing = { k, text: state.reply_so_far, end: state.reply_end };
  $("writing").textContent = clarity.writing.text;
  const buffered = clarity.buffer;
  clarity.buffer = null;
  for (const event of buffered) apply(event);
  await refresh();
  document.body.dataset.page = "ready";
  if (!clarity.state.conversation_id && clarity.state.model !== "starting") {
    showMenu($("picker-button")); // the first screen: choose a profile to start (B1)
  }
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
  // A dropped stream: try again, waiting longer each time, then load the page again. A
  // Clarity that doesn't answer has stopped (A5).
  if (!clarity.key || document.body.dataset.page === "stopped") return;
  retry += 1;
  if (retry > 3) {
    stopped();
    return;
  }
  setTimeout(() => load().catch(() => {}), 250 * 2 ** retry);
}

// --- The shell (C1)

let drawing = Promise.resolve();
function refresh() {
  // Redraw from reads, one redraw at a time, so the last one shows the latest state.
  drawing = drawing.then(draw).catch(() => {});
  return drawing;
}

async function draw() {
  const answer = await command("state");
  if (!answer.result) return;
  clarity.state = answer.result;
  const [thread, row] = await Promise.all([view("thread"), view("picker")]);
  if (openMenu && openMenu.menu.id === "picker-menu") hideMenu(false);
  swap($("thread"), thread);
  swap($("composer-row"), row);
  const state = clarity.state;
  const starting = state.model === "starting";
  const chat = Boolean(state.conversation_id) && !starting;
  $("title").hidden = !chat;
  $("title-text").textContent = state.title || "New chat";
  $("title").classList.toggle("fresh", !state.title);
  $("message").disabled = !chat && !starting; // you can type while a model starts
  $("send").disabled = !chat || state.model !== "ready";
  const waits = $("picker-button").getAttribute("aria-disabled") === "true";
  $("new-chat").setAttribute("aria-disabled", String(waits));
  const commit = state.clarity.commit || "";
  const ref = /^[0-9a-f]{40}$/.test(commit) ? commit : `v${state.clarity.version}`;
  $("help").href = `${$("help").dataset.base}${ref}/docs`; // the docs at this version
}

function notice(title, text) {
  // A refusal's sentence, whole, above the message box (B3, B4).
  if (title === null) {
    $("notice").replaceChildren();
    return;
  }
  const box = document.createElement("div");
  box.className = "problem";
  const line = document.createElement("p");
  line.setAttribute("role", "alert");
  const strong = document.createElement("strong");
  strong.textContent = `! ${title}`;
  line.append(strong, ` ${text}`);
  box.append(line);
  $("notice").replaceChildren(box);
}

async function start(name, args) {
  // A new chat: with a profile from the picker, or the same profile (New chat).
  const label = name === "profile" ? args : clarity.state.profile;
  notice(null);
  const answer = await command(name, args);
  if (answer.error && !answer.stopped) notice(`${label} didn’t start.`, answer.error);
  await refresh();
  if (!answer.error) $("message").focus();
}

async function about() {
  swap($("about-body"), await view("about"));
  $("about").showModal();
}

async function quit() {
  // A reply being written stops first, and its part is kept, as Stop does.
  await command("cancel");
  const answer = await command("quit");
  if (answer.result) stopped();
}

function act(element) {
  if (element.getAttribute("aria-disabled") === "true") return;
  if (element.dataset.profile !== undefined) {
    start("profile", element.dataset.profile);
    return;
  }
  const action = element.dataset.action;
  if (action === "new") {
    if (clarity.state.conversation_id) start("new", "");
    else showMenu($("picker-button"));
  } else if (action === "about") {
    about();
  } else if (action === "quit") {
    quit();
  } else if (clarity.on[action]) {
    clarity.on[action](element);
  }
}

// Menus (the Clarity menu, the picker): a button with aria-haspopup and aria-controls
// opens its menu; arrows, Home and End move, Escape closes and returns to the button, Tab
// or a click elsewhere closes it.
let openMenu = null;

function itemsOf(menu) {
  return [...menu.querySelectorAll('[role^="menuitem"]')];
}

function showMenu(button, which = "first") {
  hideMenu(false);
  const menu = button && $(button.getAttribute("aria-controls"));
  if (!menu) return;
  menu.hidden = false;
  button.setAttribute("aria-expanded", "true");
  openMenu = { button, menu };
  const items = itemsOf(menu);
  const ticked = items.find((item) => item.getAttribute("aria-checked") === "true");
  const target = which === "last" ? items[items.length - 1] : ticked || items[0];
  if (target) target.focus();
}

function hideMenu(returnFocus) {
  if (!openMenu) return;
  const { button, menu } = openMenu;
  openMenu = null;
  menu.hidden = true;
  button.setAttribute("aria-expanded", "false");
  if (returnFocus) button.focus();
}

document.addEventListener("click", (event) => {
  const button = event.target.closest('[aria-haspopup="menu"]');
  if (button) {
    if (button.getAttribute("aria-disabled") === "true") return;
    if (openMenu && openMenu.button === button) hideMenu(false);
    else showMenu(button);
    return;
  }
  const item = event.target.closest('[role^="menuitem"]');
  if (item && openMenu && openMenu.menu.contains(item)) {
    hideMenu(true);
    act(item);
    return;
  }
  if (openMenu && !openMenu.menu.contains(event.target)) hideMenu(false);
  const control = event.target.closest("[data-action]");
  if (control) act(control);
});

document.addEventListener("keydown", (event) => {
  const button = event.target.closest && event.target.closest('[aria-haspopup="menu"]');
  if (button && (event.key === "ArrowDown" || event.key === "ArrowUp")) {
    if (button.getAttribute("aria-disabled") === "true") return;
    event.preventDefault();
    showMenu(button, event.key === "ArrowUp" ? "last" : "first");
    return;
  }
  if (!openMenu || !openMenu.menu.contains(event.target)) return;
  const items = itemsOf(openMenu.menu);
  const at = items.indexOf(event.target);
  const to = { ArrowDown: at + 1, ArrowUp: at - 1, Home: 0, End: items.length - 1 }[event.key];
  if (to !== undefined) {
    event.preventDefault();
    items[(to + items.length) % items.length].focus();
  } else if (event.key === "Escape") {
    event.preventDefault();
    hideMenu(true);
  } else if (event.key === "Tab") {
    hideMenu(false);
  }
});

clarity.listeners.push((event) => {
  // A model starting or ready, a reply starting, your message written: redraw from reads.
  const yours = event.event === "record" && event.record.type === "turn" && event.record.role === "user";
  if (event.event === "status" || event.event === "reply_start" || yours) refresh();
});

clarity.command = command;
clarity.view = view;
clarity.swap = swap;
clarity.download = download;
clarity.useHere = useHere;
clarity.refresh = refresh;
clarity.notice = notice;

document.addEventListener("DOMContentLoaded", () => {
  clarity.key = readKey();
  if (!clarity.key) {
    noKey();
    return;
  }
  $("use-here").addEventListener("click", () => useHere());
  $("about-close").addEventListener("click", () => $("about").close());
  $("about").addEventListener("close", () => $("clarity-button").focus());
  $("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    if (clarity.on.send) clarity.on.send();
  });
  load().catch(() => {});
});
