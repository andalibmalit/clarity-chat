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
  failedCall: null, // the last failed call's ID, from its record on the stream
  alerted: new Set(), // sentences already announced, so a redraw doesn't repeat them
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

function swap(element, html, prepare = null) {
  // The one place server HTML enters the page: parsed, then its nodes moved in. `prepare`
  // sees the parsed nodes first (to set attributes, never to add HTML).
  const parsed = new DOMParser().parseFromString(POLICY ? POLICY.createHTML(html) : html, "text/html");
  if (prepare) prepare(parsed.body);
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
  // A tab showing A4 waits for "Use it here": polling would take over when the live tab
  // drops its stream for a moment (a reload). Checked again when the wait is over, since a
  // tab learns it isn't live from a command's answer, after its stream has closed.
  const waiting = () => !clarity.key || ["stopped", "not-live"].includes(document.body.dataset.page);
  if (waiting()) return;
  retry += 1;
  if (retry > 3) {
    stopped();
    return;
  }
  setTimeout(() => waiting() || load().catch(() => {}), 250 * 2 ** retry);
}

// --- The shell (C1)

let drawn = 0;
function refresh() {
  // Redraw from reads. The newest redraw wins: an older one still waiting (a view whose read
  // queued just before a reply started waits for the whole reply) is dropped when it lands.
  drawn += 1;
  return draw(drawn).catch((error) => {
    if (document.body.dataset.page === "ready") notice("!", error.message);
  });
}

async function draw(number) {
  const answer = await command("state");
  if (!answer.result) return;
  const [thread, row] = await Promise.all([view("thread"), view("picker")]);
  if (number !== drawn) return;
  clarity.state = answer.result;
  if (openMenu && openMenu.menu.id === "picker-menu") hideMenu(false);
  openPop = null; // the thread's popovers are drawn again, closed
  openList = null;
  swap($("thread"), thread, alertOnce);
  shade(); // the thread is new: shade its reply again if Probabilities is open
  swap($("composer-row"), row);
  const state = clarity.state;
  const starting = state.model === "starting";
  const chat = Boolean(state.conversation_id) && !starting;
  const again = starting && state.running && state.running.command === "resume"; // same chat
  $("title").hidden = !chat && !again;
  $("title-text").textContent = state.title || "New chat";
  $("title").classList.toggle("fresh", !state.title);
  $("message").disabled = !chat && !starting; // you can type while a model starts
  $("send").disabled = !chat && !starting;
  // While a model starts, Send waits and says why (a stopped server isn't a wait: Send then
  // fails with the backend's sentence, and the failed line offers to start it again).
  $("send-waits").textContent = starting ? "Waits until the model has started." : "";
  if (starting) $("send").setAttribute("aria-disabled", "true");
  else $("send").removeAttribute("aria-disabled");
  const waits = $("picker-button").getAttribute("aria-disabled") === "true";
  $("new-chat").setAttribute("aria-disabled", String(waits));
  showWriting(state.running);
  if (panelOf) showPanel(panelOf.name, panelOf.params);
  countBranches(chat);
  if (state.unanswered && clarity.filled !== state.unanswered.turn_id) {
    clarity.filled = state.unanswered.turn_id; // your failed message, back in the box once
    if (!$("message").value) $("message").value = state.unanswered.text;
  }
  clearTimeout(clarity.poll);
  if (changing(state.running) && !clarity.sending) clarity.poll = setTimeout(waitForEnd, 1000);
  const commit = state.clarity.commit || "";
  const ref = /^[0-9a-f]{40}$/.test(commit) ? commit : `v${state.clarity.version}`;
  $("help").href = `${$("help").dataset.base}${ref}/docs`; // the docs at this version
  clarity.docs = `${$("help").dataset.base}${ref}/docs/`;
}

// Commands that only read: one waiting its turn doesn't change the chat (web.READS).
const READS = new Set(["history", "branches", "compare", "prompt", "probs", "settings"]);
const changing = (running) => Boolean(running) && !READS.has(running.command);

async function waitForEnd() {
  // After a reload mid-reply, nothing tells this page the command ended: read /state alone
  // each second, and redraw everything once it has.
  const answer = await command("state").catch(() => null);
  if (!answer || !answer.result) return;
  if (changing(answer.result.running)) {
    showWriting(answer.result.running);
    clarity.poll = setTimeout(waitForEnd, 1000);
  } else {
    refresh();
  }
}

function alertOnce(body) {
  // A start warning or a failed line is announced when it first appears, not again on each
  // redraw (a mark, a note, a switch).
  for (const line of body.querySelectorAll("[data-alert]")) {
    if (clarity.alerted.has(line.textContent)) continue;
    clarity.alerted.add(line.textContent);
    line.setAttribute("role", "alert");
  }
}

function notice(strong, text, action = null, label = "", call = null) {
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
  if (call) {
    box.dataset.call = call;
    const details = document.createElement("button");
    details.type = "button";
    details.className = "linkbtn";
    details.dataset.action = "details";
    details.textContent = "Details";
    box.append(details);
  }
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
  if (openList && !openList.box.contains(event.target) && event.target !== openList.button) {
    closeReplies(false);
  }
  const version = event.target.closest("[data-switch]");
  if (version) {
    if (!$("compare").hidden) closeCompare(); // picking a reply in Compare goes back to it
    const turn = version.dataset.reveal;
    command("switch", version.dataset.switch).then(async (answer) => {
      await refresh();
      if (answer.error) notice("!", answer.error);
      else if (turn) reveal(turn);
    });
    return;
  }
  const token = event.target.closest(TOKENS);
  if (token) {
    pickToken(token);
    return;
  }
  const pane = event.target.closest("#panel .seg [data-pane]");
  if (pane) {
    // The prompt as Text, Whitespace or Token IDs: one pane shown at a time.
    for (const button of pane.parentElement.children) {
      button.setAttribute("aria-pressed", String(button === pane));
    }
    for (const pre of pane.closest("details").querySelectorAll("pre[data-pane]")) {
      pre.hidden = pre.dataset.pane !== pane.dataset.pane;
    }
    return;
  }
  const set = event.target.closest("[data-set]");
  if (set) {
    if (set.getAttribute("aria-disabled") !== "true") setting(set.dataset.set);
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
  if (event.key === "Escape" && openList && openList.box.contains(event.target)) {
    closeReplies(true);
    return;
  }
  const token = event.target.closest && event.target.closest(TOKENS);
  if (token && (event.key === "ArrowRight" || event.key === "ArrowLeft")) {
    event.preventDefault();
    const tokens = [...token.parentElement.querySelectorAll(".tok")];
    const at = tokens.indexOf(token) + (event.key === "ArrowRight" ? 1 : -1);
    if (tokens[at]) pickToken(tokens[at]);
    return;
  }
  if (event.key === "Escape" && $("panel").contains(event.target)) {
    closePanel();
    return;
  }
  if (event.key === "Escape" && $("compare").contains(event.target)) {
    closeCompare();
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
  if (!on) {
    $("writing-status").textContent = ""; // the region stays in the page, empty
    $("writing-dots").replaceChildren();
    return;
  }
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
  // message from its call) shows here, with its sentence whole. Focus goes to Stop while
  // it writes, then to the new reply's toolbar (V3Writing).
  notice(null, null);
  const before = failedShown();
  clarity.sending = true;
  clarity.failedCall = null;
  clarity.writing = { k: null, text: "", end: 0 };
  $("writing").textContent = "";
  showWriting({ command: name, k: 1, n: 1 });
  $("stop").focus();
  const answer = await command(name, args).finally(() => (clarity.sending = false));
  await refresh();
  if (answer.stopped) return;
  const failure = answer.error ? [answer.type, answer.error] : answer.result && answer.result.stopped ? ["", answer.result.stopped] : null;
  if (failure && failedShown() === before) refused(name, ...failure);
  if (answer.error && name === "edit") reopenEdit(keep);
  else if (answer.error && keep !== null && !$("message").value) $("message").value = keep;
  const toolbar = document.querySelector(".reply.last .tb button");
  if (!answer.error && toolbar) toolbar.focus();
  else if (document.activeElement === document.body) $("message").focus();
}

function reopenEdit(text) {
  // A failed Edit: the edit box again, with your text, so Send retries the edit.
  const item = $("thread").querySelector("[data-edit]")?.closest("li");
  if (!item) return;
  editing(item, true);
  item.querySelector("[data-edit] textarea").value = text;
}

function refused(name, type, sentence) {
  const call = clarity.failedCall; // the failed call's record came on the stream
  if (type === "CancelledBeforeReply") notice(null, sentence);
  else if (type === "ContextError" || type === "ConfigError") notice("!", sentence);
  else if (clarity.state.model === "stopped") notice("! Failed:", sentence, "restart", "Start the model again", call);
  else if (name === "regen") notice("! Failed:", sentence, "regen", "Try again", call);
  else notice("! Failed:", sentence, null, "", call);
}

function send() {
  const text = $("message").value;
  if (!text || $("send").disabled || $("send").getAttribute("aria-disabled") === "true") return;
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
  const button = document.querySelector(`li[data-call="${CSS.escape(call)}"] .bm`);
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

// The replies pop-out and Compare (C3), drawn by the `replies` and `compare` views.
let openList = null; // the open pop-out and the button that opened it
let compareFor = null; // the reply whose message Compare shows

async function openReplies(button) {
  // Opens below the reply's toolbar, with focus on its first control (V3Replies, focus).
  const again = openList && openList.button === button;
  closeReplies(false);
  if (again) return;
  const call = button.closest("[data-call]").dataset.call;
  const box = document.createElement("div");
  box.className = "listpop";
  box.setAttribute("role", "dialog");
  box.setAttribute("aria-label", "Replies to this message");
  box.dataset.call = call;
  try {
    swap(box, await view("replies", { call }));
  } catch (error) {
    notice("!", error.message);
    return;
  }
  button.closest(".tb").after(box);
  button.setAttribute("aria-expanded", "true");
  openList = { button, box };
  box.scrollIntoView({ block: "nearest" }); // the thread scrolls to show all of it
  (box.querySelector(".go") || box.querySelector("#more-n") || box.querySelector("button")).focus();
}

function closeReplies(returnFocus) {
  if (!openList) return;
  const { button, box } = openList;
  openList = null;
  box.remove();
  button.setAttribute("aria-expanded", "false");
  if (returnFocus) button.focus();
}

async function openCompare(call) {
  // Compare takes the thread's place; Back to the chat (or Escape) returns to the thread,
  // to the replies button that led here.
  closeReplies(false);
  try {
    swap($("compare"), await view("compare", { call }));
  } catch (error) {
    notice("!", error.message);
    return;
  }
  compareFor = call;
  $("threadarea").hidden = $("cpwrap").hidden = true;
  $("compare").hidden = false;
  $("compare-back").focus();
}

function closeCompare() {
  $("compare").hidden = true;
  $("threadarea").hidden = $("cpwrap").hidden = false;
  const button = document.querySelector(`li[data-call="${CSS.escape(compareFor || "")}"] [data-action="replies"]`);
  (button || $("message")).focus();
}

clarity.on.replies = (element) => openReplies(element);

// The side panel (C4a Settings, C4b Details): a view drawn beside the thread. It's drawn
// again after every redraw, so its controls wait, or stop waiting, with the page.
let panelOf = null; // { name, params, opener }

async function showPanel(name, params, opener) {
  let html;
  try {
    html = await view(name, params);
  } catch (error) {
    notice("!", error.message);
    return;
  }
  // Read what's focused and open when the new panel lands, not when it was asked for.
  const focused = document.activeElement && $("panel").contains(document.activeElement)
    ? document.activeElement.id
    : null;
  const sections = new Map([...$("panel").querySelectorAll("details[data-key]")].map((d) => [d.dataset.key, d.open]));
  swap($("panel"), html);
  for (const section of $("panel").querySelectorAll("details[data-key]")) {
    if (sections.has(section.dataset.key)) section.open = sections.get(section.dataset.key); // as left
  }
  const opening = !panelOf || panelOf.name !== name || opener;
  const same = panelOf && panelOf.name === name && panelOf.params.call === params.call;
  panelOf = {
    name,
    params,
    opener: opener || (panelOf && panelOf.opener),
    refusal: same && !opener ? panelOf.refusal : null, // a replay's refusal, kept on redraws
  };
  const refused = $("panel").querySelector(".refused");
  if (panelOf.refusal && refused) {
    refused.textContent = panelOf.refusal;
    refused.hidden = false;
  }
  $("panel").hidden = false;
  for (const link of $("panel").querySelectorAll("a[data-docs]")) {
    link.href = `${clarity.docs || ""}${link.dataset.docs}`; // the guide at this version
  }
  shade();
  if (name === "branches") markedOnly(Boolean(clarity.markedOnly));
  const again = focused && $(focused);
  if (again) again.focus();
  else if (opening) $("panel-title").focus();
}

function closePanel() {
  // Focus goes back to the control that opened the panel, or the one drawn in its place.
  const opener = panelOf && panelOf.opener;
  panelOf = null;
  $("panel").hidden = true;
  shade(); // clears the shading
  let back = opener;
  if (opener && !opener.isConnected) {
    const call = opener.closest("[data-call]")?.dataset.call;
    const scope = call ? document.querySelector(`[data-call="${CSS.escape(call)}"]`) : document;
    back = scope && scope.querySelector(`[data-action="${opener.dataset.action}"]`);
  }
  (back || $("message")).focus();
}

function refuseRow(row, box, sentence) {
  // A refusal shows its sentence under its row, and the box keeps what you typed.
  const refused = row.querySelector(".refused");
  refused.textContent = sentence;
  refused.hidden = false;
  const note = row.querySelector(".note");
  note.textContent = `Not applied: the next reply still uses ${box.dataset.inUse}.`;
  note.hidden = false;
  box.setAttribute("aria-invalid", "true");
  const slider = row.querySelector('input[type="range"]');
  if (slider) slider.value = slider.getAttribute("value");
}

async function setting(args, row = null, box = null) {
  // /set NAME VALUE, or /set NAME (back to the profile's value).
  const answer = await command("set", args);
  if (answer.error && row) {
    refuseRow(row, box, answer.error);
    return;
  }
  if (answer.error) notice("!", answer.error);
  await refresh();
}

clarity.on.settings = (element) => showPanel("settings", {}, element);

// Branches (C4d): the top bar's count of branch ends, shown from the second one, and the
// panel of branch ends and marked replies ("Marked only" filters it in the page).
async function countBranches(chat) {
  // /branches waits its turn (behind a reply, say), so one count at a time, and one more
  // after it if a redraw asked meanwhile; the button keeps its count until then.
  if (clarity.counting) {
    clarity.recount = true;
    return;
  }
  clarity.counting = true;
  try {
    const answer = chat ? await command("branches", "all") : null;
    const ends = answer && answer.result ? answer.result.ends.length : 0;
    $("branches-count").textContent = String(ends);
    $("branches-button").hidden = ends < 2;
  } finally {
    clarity.counting = false;
  }
  if (clarity.recount) {
    clarity.recount = false;
    countBranches(Boolean(clarity.state.conversation_id));
  }
}

function markedOnly(on) {
  clarity.markedOnly = on;
  const button = $("marked-only");
  if (!button) return;
  button.setAttribute("aria-pressed", String(on));
  for (const part of $("panel").querySelectorAll("[data-list]")) {
    part.hidden = (part.dataset.list === "marked") !== on;
  }
}

function reveal(turn) {
  // A marked reply picked in Branches: its branch is drawn; scroll to it and outline it.
  const item = $("thread").querySelector(`li[data-turn="${CSS.escape(turn)}"]`);
  if (!item) return;
  item.scrollIntoView({ block: "center" });
  item.classList.add("revealed");
}

// Probabilities (C4c): while its section is open, the reply is drawn shaded, token by
// token, from the panel's copy; picking a token shows its chance and alternatives. Tokens
// that can't be placed in the reply are listed in the panel instead.
const TOKENS = "#thread .msg.shaded .tok, #panel .toklist .tok";

function shade() {
  // Unhides only the reply it hid (Edit hides one too).
  for (const old of $("thread").querySelectorAll(".msg.shaded")) old.remove();
  for (const msg of $("thread").querySelectorAll(".msg[data-shaded]")) {
    msg.hidden = false;
    delete msg.dataset.shaded;
  }
  const section = $("panel").hidden ? null : $("panel").querySelector('details[data-key="probs"]');
  const shaded = section && section.open && section.querySelector(".msg.shaded");
  if (!shaded) return;
  const item = $("thread").querySelector(`li[data-call="${CSS.escape(section.dataset.call)}"]`);
  if (!item) return;
  const copy = shaded.cloneNode(true);
  copy.hidden = false;
  copy.setAttribute("role", "group");
  copy.setAttribute("aria-label", "The reply, token by token; arrow keys move between them");
  const msg = item.querySelector(".msg");
  msg.hidden = true;
  msg.dataset.shaded = "";
  msg.after(copy);
}

function pickToken(token) {
  // One token selected, its details shown in the panel; one Tab stop for the tokens.
  const tokens = [...token.parentElement.querySelectorAll(".tok")];
  for (const t of tokens) {
    t.setAttribute("aria-pressed", String(t === token));
    t.tabIndex = t === token ? 0 : -1;
  }
  for (const info of $("panel").querySelectorAll(".tokinfo")) {
    info.hidden = info.dataset.i !== token.dataset.i;
  }
  token.focus();
}

document.addEventListener("toggle", (event) => {
  if (event.target.matches && event.target.matches('#panel details[data-key="probs"]')) shade();
}, true);

// Details (C4b): one call's record, from the toolbar, the facts line or a failed line.
clarity.on.details = (element) => {
  showPanel("details", { call: element.closest("[data-call]").dataset.call }, element);
};

async function replay(call, button) {
  // Replay re-sends the call; Stop sends /cancel. The section says "Replaying…" until the
  // command ends, then the panel is drawn again with the replay's result.
  const section = button.closest("details");
  const status = document.createElement("p");
  status.className = "st";
  status.setAttribute("role", "status");
  status.textContent = "Replaying…";
  const stop = document.createElement("button");
  stop.type = "button";
  stop.className = "btn";
  stop.dataset.action = "replay-stop";
  stop.textContent = "Stop";
  button.replaceWith(status, stop);
  stop.focus();
  clarity.sending = true;
  if (panelOf) panelOf.refusal = null;
  const answer = await command("replay", call).finally(() => (clarity.sending = false));
  if (panelOf && answer.error) panelOf.refusal = answer.error;
  await showPanel("details", { call });
  await refresh();
  const again = $("panel").querySelector('[data-action="replay"]');
  if (again) again.focus();
  else if (section) section.focus();
}

document.addEventListener("input", (event) => {
  const slider = event.target.closest('#panel input[type="range"]');
  if (slider) $(slider.dataset.mirror).value = slider.value; // the box follows the slider
});

document.addEventListener("change", (event) => {
  // A row sends /set when a slider is released, or on Enter in a box or leaving it.
  const input = event.target.closest("#panel input");
  if (!input) return;
  const box = input.type === "range" ? $(input.dataset.mirror) : input;
  if (input.getAttribute("aria-disabled") === "true") {
    showPanel(panelOf.name, panelOf.params); // it waits: back to the value in use
    return;
  }
  // An empty box (or one a number box can't read) would send /set NAME, which puts the
  // setting back: refused here instead, and nothing is sent.
  if (!box.value.trim()) refuseRow(box.closest(".srow"), box, "Type a value; an empty box changes nothing.");
  else setting(`${box.dataset.name} ${box.value}`, box.closest(".srow"), box);
});

const THREAD = {
  "panel-close": () => closePanel(),
  branches: (element) => showPanel("branches", {}, element),
  "marked-only": () => markedOnly(!clarity.markedOnly),
  replay: (element, call) => replay(call, element),
  "replay-stop": () => command("cancel"),
  "reset-all": async (element) => {
    for (const name of element.dataset.names.split(" ")) {
      const answer = await command("set", name);
      if (answer.error) notice("!", answer.error);
    }
    await refresh();
  },
  "replies-close": () => closeReplies(true),
  again() {
    closeReplies(false);
    write("regen", "same");
  },
  compare: (element, call) => openCompare(call),
  "compare-back": () => closeCompare(),
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
  restart: () => start("resume", clarity.state.log_id),
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
  } else if (form.matches("[data-more]")) {
    const n = form.querySelector("input").value;
    closeReplies(false);
    write("regen", n);
  } else if (form.matches("[data-note]")) {
    const text = form.querySelector("textarea").value.trim();
    if (text) annotate("note", form.closest("[data-call]").dataset.call, text);
  }
});

clarity.listeners.push((event) => {
  if (event.event === "record" && event.record.type === "call" && event.record.error) {
    clarity.failedCall = event.record.call_id;
  }
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
