// Clarity's page. The core (v0.3.0 item B6): the secret, the requests, the event stream,
// the page load and the live tab. The shell (C1): the top bar, the Clarity menu, the
// picker and the row under the message box, the first screen and a model starting, About,
// Quit, and the plain pages. The thread (C2): sending, writing and Stop, Edit, the
// counter, Copy, Regenerate, marks and notes, and failed and refused calls. Later screens
// build on `clarity` below; `clarity.on` holds their handlers for controls already drawn
// (Chats, Appearance, Settings, the replies pop-out, Details).
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
  on: {}, // later screens' handlers: chats, appearance, settings, replies, details
  sending: false, // this page waits for a command that writes or starts
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

let drawn = 0;
function refresh() {
  // Redraw from reads. The newest redraw wins: an older one still waiting (a view whose read
  // queued just before a reply started waits for the whole reply) is dropped when it lands.
  drawn += 1;
  return draw(drawn).catch(() => {});
}

async function draw(number) {
  const answer = await command("state");
  if (!answer.result) return;
  const [thread, row] = await Promise.all([view("thread"), view("picker")]);
  if (number !== drawn) return;
  clarity.state = answer.result;
  if (openMenu && openMenu.menu.id === "picker-menu") hideMenu(false);
  openPop = null; // the thread's popovers are drawn again, closed
  swap($("thread"), thread);
  swap($("composer-row"), row);
  const state = clarity.state;
  const starting = state.model === "starting";
  const chat = Boolean(state.conversation_id) && !starting;
  const again = starting && state.running && state.running.command === "resume"; // same chat
  $("title").hidden = !chat && !again;
  $("title-text").textContent = state.title || "New chat";
  $("title").classList.toggle("fresh", !state.title);
  $("message").disabled = !chat && !starting; // you can type while a model starts
  $("send").disabled = !chat || state.model !== "ready";
  const waits = $("picker-button").getAttribute("aria-disabled") === "true";
  $("new-chat").setAttribute("aria-disabled", String(waits));
  showWriting(state.running);
  if (state.unanswered && clarity.filled !== state.unanswered.turn_id) {
    clarity.filled = state.unanswered.turn_id; // your failed message, back in the box once
    if (!$("message").value) $("message").value = state.unanswered.text;
  }
  clearTimeout(clarity.poll);
  if (state.running && !clarity.sending) clarity.poll = setTimeout(refresh, 1000); // after a reload
  const commit = state.clarity.commit || "";
  const ref = /^[0-9a-f]{40}$/.test(commit) ? commit : `v${state.clarity.version}`;
  $("help").href = `${$("help").dataset.base}${ref}/docs`; // the docs at this version
}

function notice(strong, text, action = null, label = "") {
  // A refusal's sentence, whole, above the message box (B3, B4, E5, E7), with the one thing
  // to do next, if any. A status, not an alert, when nothing failed (strong null).
  if (text === null) {
    $("notice").replaceChildren();
    return;
  }
  const line = document.createElement("p");
  line.setAttribute("role", strong ? "alert" : "status");
  if (strong) {
    const bold = document.createElement("strong");
    bold.textContent = strong;
    line.append(bold, " ");
  }
  line.append(text);
  const box = document.createElement("div");
  box.className = strong ? "problem" : "status";
  box.append(line);
  if (action) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "btn";
    button.dataset.action = action;
    button.textContent = label;
    box.append(button);
  }
  $("notice").replaceChildren(box);
}

async function start(name, args) {
  // A new chat: with a profile from the picker, or the same profile (New chat).
  const label = name === "profile" ? args : clarity.state.profile;
  notice(null, null);
  clarity.sending = true;
  const answer = await command(name, args).finally(() => (clarity.sending = false));
  if (answer.error && !answer.stopped) notice(`! ${label} didn’t start.`, answer.error);
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
  } else if (THREAD[action]) {
    THREAD[action](element, element.closest("[data-call]")?.dataset.call);
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
  if (openPop && !openPop.contains(event.target) && !event.target.closest(".bm")) hidePop(false);
  const version = event.target.closest("[data-switch]");
  if (version) {
    command("switch", version.dataset.switch).then(refresh);
    return;
  }
  const unmark = event.target.closest("[data-unmark]");
  if (unmark) {
    annotate("unmark", unmark.closest("[data-call]").dataset.call, unmark.dataset.unmark);
    return;
  }
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
  if (event.key === "Escape" && openPop && openPop.contains(event.target)) {
    hidePop(true);
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

// --- The thread (C2)

const WRITES = new Set(["say", "regen", "edit"]);

function showWriting(running) {
  // "Writing…" or "Writing reply 2 of 3" in a status region (text only), the reply so far
  // beside it, and Stop where Send was. The reply's text itself is not announced.
  const on = Boolean(running) && WRITES.has(running.command);
  $("writing-box").hidden = !on;
  $("send").hidden = on;
  $("stop").hidden = !on;
  if (!on) return;
  const several = running.n > 1;
  $("writing-status").textContent = several ? `Writing reply ${running.k} of ${running.n}` : "Writing…";
  $("stop-label").textContent = several && running.k < running.n ? "Stop the rest" : "Stop";
  const steps = [];
  for (let i = 1; several && i <= running.n; i += 1) {
    const step = document.createElement("span");
    if (i > running.k) step.className = "off";
    steps.push(step);
  }
  $("writing-dots").replaceChildren(...steps);
}

function failedShown() {
  const line = $("thread").querySelector("[data-failed]");
  return line ? line.dataset.failed : null;
}

async function write(name, args, keep = null) {
  // A command that writes replies. A failure the thread doesn't draw (it draws a failed
  // message from its call) shows here, with its sentence whole.
  notice(null, null);
  const before = failedShown();
  clarity.sending = true;
  clarity.writing = { k: null, text: "", end: 0 };
  $("writing").textContent = "";
  showWriting({ command: name, k: 1, n: 1 });
  const answer = await command(name, args).finally(() => (clarity.sending = false));
  await refresh();
  if (answer.stopped) return;
  const failure = answer.error ? [answer.type, answer.error] : answer.result && answer.result.stopped ? ["", answer.result.stopped] : null;
  if (failure && failedShown() === before) refused(name, ...failure);
  if (answer.error && keep !== null && !$("message").value) $("message").value = keep;
}

function refused(name, type, sentence) {
  if (type === "CancelledBeforeReply") notice(null, sentence);
  else if (type === "ContextError" || type === "ConfigError") notice("!", sentence);
  else if (clarity.state.model === "stopped") notice("! Failed:", sentence, "restart", "Start the model again");
  else if (name === "regen") notice("! Failed:", sentence, "regen", "Try again");
  else notice("! Failed:", sentence);
}

function send() {
  const text = $("message").value;
  if (!text || $("send").disabled) return;
  $("message").value = "";
  write("say", text, text);
}

let openPop = null;

function showPop(button) {
  hidePop(false);
  openPop = $(button.getAttribute("aria-controls"));
  openPop.hidden = false;
  button.setAttribute("aria-expanded", "true");
  openPop.querySelector("button").focus();
}

function hidePop(returnFocus) {
  if (!openPop) return;
  const button = openPop.closest("li").querySelector(".bm");
  openPop.hidden = true;
  button.setAttribute("aria-expanded", "false");
  openPop = null;
  if (returnFocus) button.focus();
}

async function annotate(kind, call, text = "") {
  // /mark, /unmark or /note on this reply by its call ID; the popover opens again after.
  const answer = await command(kind, `${call} ${text}`.trim());
  await refresh();
  if (answer.error) notice("!", answer.error);
  const button = document.querySelector(`li[data-call="${call}"] .bm`);
  if (button) showPop(button);
}

function editing(item, on) {
  item.querySelector(".msg").hidden = on;
  item.querySelector(".msgbar").hidden = on;
  item.querySelector("[data-edit]").hidden = !on;
  if (on) item.querySelector("[data-edit] textarea").focus();
  else item.querySelector('[data-action="edit"]').focus();
}

function noteForm(pop, on) {
  pop.querySelector("[data-view]").hidden = on;
  pop.querySelector("[data-note]").hidden = !on;
  if (on) pop.querySelector("[data-note] textarea").focus();
}

const THREAD = {
  copy(element) {
    const status = element.closest(".tb").querySelector(".copied");
    const text = element.closest("li").querySelector(".msg").textContent;
    navigator.clipboard.writeText(text).then(
      () => (status.textContent = "Copied"),
      () => (status.textContent = "Couldn’t copy"),
    );
    setTimeout(() => (status.textContent = ""), 1600);
  },
  regen: () => write("regen", ""),
  restart: () => start("resume", clarity.state.conversation_id),
  edit: (element) => editing(element.closest("li"), true),
  "edit-cancel": (element) => editing(element.closest("li"), false),
  marks: (element) => (openPop && openPop.id === element.getAttribute("aria-controls") ? hidePop(true) : showPop(element)),
  "pop-close": () => hidePop(true),
  mark: (element, call) => annotate("mark", call),
  "note-edit": (element) => noteForm(element.closest(".pop"), true),
  "note-cancel": (element) => noteForm(element.closest(".pop"), false),
};

document.addEventListener("submit", (event) => {
  const form = event.target;
  event.preventDefault();
  if (form.id === "composer") {
    send();
  } else if (form.matches("[data-edit]")) {
    const text = form.querySelector("textarea").value;
    if (text) write("edit", text, text);
  } else if (form.matches("[data-note]")) {
    const text = form.querySelector("textarea").value.trim();
    if (text) annotate("note", form.closest("[data-call]").dataset.call, text);
  }
});

clarity.listeners.push((event) => {
  if (event.event === "reply_start") showWriting({ command: "regen", k: event.k, n: event.n });
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
  $("stop").addEventListener("click", () => command("cancel"));
  load().catch(() => {});
});
