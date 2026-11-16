// Clarity's page. The core (v0.3.0 item B6): the secret, the requests, the event stream,
// the page load and the live tab. The shell (C1): the top bar, the Clarity menu, the
// picker and the row under the message box, the first screen and a model starting, About,
// Quit, and the plain pages. The thread (C2): sending, writing and Stop, Edit, the
// counter, Copy, Regenerate, marks and notes, and failed and refused calls. Later screens
// build on `clarity` below; THREAD holds every control's handler, by its data-action.
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
  foundOpen: new Set(), // the "From documents" lists left open, kept on redraws (C3)
  start: null, // the first /state's `start`: this Clarity's start
  writing: { k: null, text: "", end: 0 }, // the reply being written, joined by numbers
  listeners: [], // screens' event handlers
  buffer: null, // events held while the page loads; null once it has
  sending: false, // this page waits for a command that writes or starts
  failedCall: null, // the last failed call's ID, from its record on the stream
  focusBox: false, // a start from the picker or New chat is running: focus to the box
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
    throw handled("no key");
  }
  return response;
}

function handled(message) {
  // A refusal the page has already acted on (the no-key or not-live page): callers'
  // unanswered() leaves it alone, so it isn't also shown as a dropped connection.
  const error = new Error(message);
  error.handled = true;
  return error;
}

async function command(name, args = "") {
  const response = await request("/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ command: name, args, id: Date.now(), tab: TAB }),
  });
  if (response.status === 409) {
    notLive();
    throw handled("not the live tab");
  }
  // A refusal before the command table (a body too big, a 500) is plain text, not JSON.
  const text = await response.text();
  let answer;
  try {
    answer = JSON.parse(text);
  } catch {
    answer = { error: text || `Clarity answered ${response.status}.`, type: "Refused" };
  }
  if (answer.stopped) stopped(answer.error);
  return answer;
}

function sameStart(state) {
  // Another start's value means another program answers on this port (Clarity stopped and
  // something took the port): the page stops, and draws or loads nothing it sends.
  if (!state || typeof state.start !== "string") return false;
  if (clarity.start === null) clarity.start = state.start;
  return state.start === clarity.start;
}

function unanswered(error) {
  // A command whose request failed (a dropped connection): an answer the callers can show,
  // marked `dropped`. One command() already acted on (handled) is only `handled`: the
  // caller stops, showing nothing.
  if (error.handled) return { handled: true };
  return { error: `Clarity didn't answer (${error.message}).`, type: "Refused", dropped: true };
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
  // Saved under the file's own name, from the response's Content-Disposition.
  const response = await request("/export");
  if (!response.ok) return response.text();
  const blob = await response.blob();
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  const named = /filename="?([^";]+)"?/.exec(response.headers.get("Content-Disposition") || "");
  link.download = named ? named[1] : "export.jsonl";
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
  const controller = new AbortController();
  clarity.dropStream = () => controller.abort(); // as a dropped connection does (tests)
  const response = await request(`/events?tab=${encodeURIComponent(TAB)}`, { signal: controller.signal });
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
  notice(null, null); // a sentence from before the reload, or before the connection dropped
  await new Promise((resolve) => connect(resolve));
  const answer = await command("state");
  if (answer.stopped) return; // command() showed why, the error whole: not another start
  const state = answer.result;
  if (!sameStart(state)) {
    stopped();
    return;
  }
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
  // A dropped stream: try again, waiting longer each time, then load the page again. After
  // 3 tries the page says Clarity isn't answering and keeps asking (lost()).
  // A tab showing A4 waits for "Use it here": polling would take over when the live tab
  // drops its stream for a moment (a reload). Checked again when the wait is over, since a
  // tab learns it isn't live from a command's answer, after its stream has closed.
  const waiting = () => !clarity.key || ["stopped", "not-live"].includes(document.body.dataset.page);
  if (waiting()) return;
  retry += 1;
  if (retry > 3) {
    lost();
    return;
  }
  setTimeout(() => waiting() || load().catch(() => {}), 250 * 2 ** retry);
}

function lost() {
  // Clarity isn't answering: say so and keep asking quietly, every 2 seconds. Once it
  // answers, load the page again. "Clarity has stopped" is only for Quit, or a server
  // that says it stopped.
  document.body.dataset.page = "lost";
  setTimeout(async () => {
    if (document.body.dataset.page !== "lost") return;
    const answer = await command("state").catch(() => null);
    if (answer && answer.result) {
      if (sameStart(answer.result)) location.reload();
      else stopped();
    }
    else if (answer && answer.stopped) stopped(answer.error);
    else if (document.body.dataset.page === "lost") lost();
  }, 2000);
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
  if (!sameStart(answer.result)) {
    stopped();
    return;
  }
  const [thread, row, chips] = await Promise.all([view("thread"), view("picker"), view("pending")]);
  if (number !== drawn) return;
  clarity.state = answer.result;
  if (openMenu && openMenu.menu.id === "picker-menu") hideMenu(false);
  openPop = null; // the thread's popovers are drawn again, closed
  openList = null;
  swap($("thread"), thread, alertOnce);
  rove($("thread"));
  if (clarity.differOpen) showDiffer(true);
  for (const key of clarity.foundOpen) showFound(key, true);
  shade(); // the thread is new: shade its reply again if Probabilities is open
  swap($("composer-row"), row);
  swap($("chips"), chips);
  $("chips").hidden = !$("chips").querySelector(".chip");
  const state = clarity.state;
  const starting = state.model === "starting";
  const chat = Boolean(state.conversation_id) && !starting;
  const again = starting && state.running && state.running.command === "resume"; // same chat
  // Nothing to rename or export before the first message (design walkthrough, finding 9).
  const said = state.turns.some((turn) => turn.role === "user");
  $("title").hidden = (!chat && !again) || !said || clarity.renaming;
  if (changing(state.running)) $("title").setAttribute("aria-disabled", "true");
  else $("title").removeAttribute("aria-disabled");
  // Its reason, shown when pressed while it waits (E1); hidden again on each redraw.
  $("title-waits").textContent = changing(state.running) ? waitsFor(state.running) : "";
  $("title-waits").classList.add("vh");
  $("title-text").textContent = state.title || "New chat";
  $("title").classList.toggle("fresh", !state.title);
  $("message").disabled = !chat && !starting; // you can type while a model starts
  if (clarity.focusBox && starting) {
    clarity.focusBox = false; // once: a later redraw mustn't take the focus back
    $("message").focus(); // a start from the picker (QA 4)
  }
  $("send").disabled = !chat && !starting;
  $("clip").disabled = !chat && !starting; // starting, it waits and says why (showWriting)
  // While a model starts, Send waits and says why (a stopped server isn't a wait: Send then
  // fails with the backend's sentence, and the failed line offers to start it again).
  $("send-waits").textContent = starting ? "Waits until the model has started." : "";
  if (!starting) $("send-waits").classList.add("vh"); // shown only while it waits
  if (starting) $("send").setAttribute("aria-disabled", "true");
  else $("send").removeAttribute("aria-disabled");
  const waits = $("picker-button").getAttribute("aria-disabled") === "true";
  $("new-chat").setAttribute("aria-disabled", String(waits));
  showWriting(state.running);
  readingLine(state);
  if (panelOf) showPanel(panelOf.name, panelOf.params);
  if (!$("drawer").hidden) await showDrawer();
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

// Commands that only read: one waiting its turn doesn't change the chat (commands.READS).
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
  return box;
}

async function start(name, args) {
  // A new chat: with a profile from the picker, or the same profile (New chat).
  const label = name === "profile" ? args : clarity.state.profile;
  closeDrawer(false);
  notice(null, null);
  clarity.sending = true;
  clarity.focusBox = true; // the box takes text while the model starts: focus goes there
  const answer = await command(name, args)
    .catch(unanswered)
    .finally(() => {
      clarity.sending = false;
      clarity.focusBox = false;
    });
  if (answer.handled) return answer;
  await refresh(); // first: a redraw that fails says so, and the sentence below stays
  if (!answer.error) {
    notKept(answer.result);
    $("message").focus();
  } else if (!answer.stopped) {
    notice(`! ${label} didn’t start.`, answer.error);
    $("picker-button").focus(); // to choose again
  }
  return answer;
}

async function about() {
  swap($("about-body"), await view("about"));
  $("about").showModal();
}

async function quit() {
  // A reply being written stops first, and its part is kept, as Stop does.
  await command("cancel");
  const answer = await command("quit");
  if (!answer.result) return;
  stopped();
  const dropped = answer.result.dropped || [];
  $("stopped-kept").textContent = dropped.length ? notKeptSentence(dropped) : "";
  $("stopped-kept").hidden = !dropped.length;
}

function notKeptSentence(dropped) {
  // "Not kept: NAMES, …", SENTENCES' words as the terminal prints them (the close's must A).
  const names = dropped.map((a) => a.name).join(", ");
  return $("notice").dataset.notKept.replace("{names}", () => names);
}

function notKept(result) {
  // New chat, a profile, a chat or Start the model again dropped what was waiting: say so.
  if (result && result.dropped && result.dropped.length) notice(null, notKeptSentence(result.dropped));
}

function act(element) {
  if (element.getAttribute("aria-disabled") === "true") {
    if (element.id === "clip") $("clip-waits").textContent = element.dataset.why || "";
    // A control that waits with a screen-reader reason shows it when pressed (E1, extended).
    const why = $(element.getAttribute("aria-describedby") || "");
    if (why) why.classList.remove("vh");
    return;
  }
  if (element.dataset.profile !== undefined) {
    start("profile", element.dataset.profile);
    return;
  }
  const action = element.dataset.action;
  if (action === "new") {
    if (clarity.state.conversation_id) start("new", "");
    else newWithoutAChat();
  } else if (action === "about") {
    about();
  } else if (action === "quit") {
    quit();
  } else if (THREAD[action]) {
    THREAD[action](element, element.closest("[data-call]")?.dataset.call);
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

async function openPicker(which = "first") {
  // The picker's row is read again on each open, so a profile saved since the last redraw
  // is listed. If the read fails, the list drawn last opens.
  const row = await view("picker").catch(() => null);
  if (row !== null) swap($("composer-row"), row);
  if ($("picker-button").getAttribute("aria-disabled") !== "true") showMenu($("picker-button"), which);
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
    if (button.getAttribute("aria-disabled") === "true") {
      // A menu that waits says why on screen, not only to a screen reader (E1).
      const why = $(button.getAttribute("aria-describedby") || "");
      if (why) why.classList.remove("vh");
      return;
    }
    if (openMenu && openMenu.button === button) hideMenu(false);
    else if (button.id === "picker-button") openPicker();
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
    const caption = pane.closest("details").querySelector("[data-caption]");
    if (caption) caption.hidden = pane.dataset.pane === "ids"; // nothing is marked in the IDs
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
  const row = event.target.closest("[data-resume]");
  if (row) {
    openChat(row);
    return;
  }
  const copy = event.target.closest("[data-copy]");
  if (copy) {
    const status = copy.nextElementSibling;
    navigator.clipboard.writeText(copy.dataset.copy).then(
      () => (status.textContent = "Copied"),
      () => (status.textContent = "Couldn’t copy"),
    );
    setTimeout(() => (status.textContent = ""), 1600);
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
    const which = event.key === "ArrowUp" ? "last" : "first";
    if (button.id === "picker-button") openPicker(which);
    else showMenu(button, which);
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
  if (event.key === "Escape" && $("drawer").contains(event.target)) {
    closeDrawer(true);
    return;
  }
  if (event.key === "Escape" && $("rename").contains(event.target)) {
    event.preventDefault();
    renaming(false, true);
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
  // The paperclip waits while anything that changes the chat runs, and says why when
  // pressed (clipWhy).
  const why = clipWhy(running);
  $("clip").dataset.why = why;
  if (why) $("clip").setAttribute("aria-disabled", "true");
  else $("clip").removeAttribute("aria-disabled");
  $("clip-waits").textContent = why; // shown while it waits, not only when pressed (QA 2)
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

async function write(name, args, keep = null, from = null) {
  // A command that writes replies. A failure the thread doesn't draw (it draws a failed
  // message from its call) shows here, with its sentence whole. Sent from the message box,
  // focus stays there; otherwise it goes to Stop while it writes, then to the new reply's
  // toolbar, on the button that started it (`from`; V3Writing, V3Access).
  notice(null, null);
  const before = failedShown();
  clarity.sending = true;
  clarity.failedCall = null;
  clarity.writing = { k: null, text: "", end: 0 };
  $("writing").textContent = "";
  $("finished").textContent = "";
  showWriting({ command: name, k: 1, n: 1 });
  $(name === "say" ? "message" : "stop").focus(); // Send is hidden now: never drop the focus
  const answer = await command(name, args)
    .catch(unanswered)
    .finally(() => (clarity.sending = false));
  if (answer.handled) return;
  if (answer.dropped) showWriting(null); // not answered: the page can't wait on it
  if (!answer.error) finished();
  await refresh();
  if (answer.stopped) return;
  const failure = answer.error ? [answer.type, answer.error] : answer.result && answer.result.stopped ? ["", answer.result.stopped] : null;
  if (failure && failedShown() === before) refused(name, ...failure);
  if (answer.error && name === "edit") reopenEdit(keep);
  else if (answer.error && keep !== null && !$("message").value) $("message").value = keep;
  // Focus still on Stop (now hidden, so on the page) goes to the new reply's toolbar; focus
  // you kept in the message box stays there.
  const toolbar = document.querySelector(".reply.last .tb");
  const button = toolbar && (toolbar.querySelector(`[data-action="${from}"]`) || toolbar.querySelector("button"));
  const left = document.activeElement === document.body || document.activeElement === $("stop");
  if (!answer.error && button && left) button.focus();
  else if (document.activeElement === document.body) $("message").focus();
}

function finished() {
  // "Reply finished", for a screen reader only (sighted users see the reply and its
  // toolbar); gone after a few seconds.
  $("finished").textContent = "Reply finished";
  clearTimeout(clarity.finishedTimer);
  clarity.finishedTimer = setTimeout(() => ($("finished").textContent = ""), 4000);
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
  else if (type === "ContextError" || type === "ConfigError" || type === "Refused") notice("!", sentence);
  else if (clarity.state.model === "stopped") notice("! Failed:", sentence, "restart", "Start the model again", call);
  else if (name === "regen") notice("! Failed:", sentence, "regen", "Try again", call);
  else notice("! Failed:", sentence, null, "", call);
}

function send() {
  const text = $("message").value;
  // While a reply is written, Send is hidden (Stop is in its place) and Enter waits too.
  if ($("send").getAttribute("aria-disabled") === "true") {
    // It waits (a model starting, from the picker too, which is still sending its start):
    // the reason shows beside Send, not only to a screen reader, and the text stays for
    // Enter once it's ready (board B2; QA 1). The redraw after the start hides it again.
    $("send-waits").classList.remove("vh");
    return;
  }
  if (clarity.sending || $("send").hidden) {
    // A reply being written, or a command this page sent (a Replay): said beside Send, and
    // the text stays (fix review, batch 6). The redraw after it hides the line again.
    const what = $("send").hidden ? "reply" : "command";
    $("send-waits").textContent = `Waits until the ${what} is done.`;
    $("send-waits").classList.remove("vh");
    return;
  }
  if ($("send").disabled) return;
  if (!text) {
    // No text: the command table says why ("Type a message to send.", or S6's with the
    // chips), beside Send, as its waits are (E1, F12); one sentence, the server's (could 17).
    command("say", "")
      .catch(unanswered)
      .then((answer) => {
        if (!answer.error) return;
        $("send-waits").textContent = answer.error;
        $("send-waits").classList.remove("vh");
      });
    return;
  }
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
  const button = openPop.closest("[data-call]").querySelector(".bm"); // a reply, or a card
  openPop.hidden = true;
  button.setAttribute("aria-expanded", "false");
  openPop = null;
  if (returnFocus) button.focus();
}

async function annotate(kind, call, text = "") {
  // /mark, /unmark or /note on this reply by its call ID; the popover opens again after,
  // in the thread or on its Compare card (E3), which is drawn again too.
  const answer = await command(kind, `${call} ${text}`.trim());
  await refresh();
  const comparing = !$("compare").hidden;
  if (comparing) {
    const html = await view("compare", { call: compareFor }).catch((error) => {
      // Said in Compare, where you are (the notice line is under the hidden thread): could 14.
      const lost = unanswered(error);
      if (!lost.handled) compareProblem(lost.error);
      return null;
    });
    if (html !== null) swap($("compare"), html);
  }
  if (answer.error) notice("!", answer.error);
  const root = comparing ? $("compare") : $("thread");
  const button = root.querySelector(`[data-call="${CSS.escape(call)}"] .bm`);
  if (button) showPop(button);
}

function editing(item, on) {
  item.querySelector(".msg").hidden = on;
  item.querySelector(".msgbar").hidden = on;
  for (const line of item.querySelectorAll(":scope > .att")) line.hidden = on; // "Keeps" says it (F9)
  item.querySelector("[data-edit]").hidden = !on;
  const keeps = item.querySelector("[data-edit] .keeps");
  if (keeps && !on) {
    // Cancel brings every chip back (the board's ⑥).
    for (const chip of keeps.querySelectorAll(".chip")) chip.hidden = false;
    keeps.querySelector(".lead").textContent = keeps.dataset.lead;
    keeps.querySelector(".att:not(.lead)").hidden = false;
  }
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
  rove(box);
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

// The side panel (C4a Settings, C4b Details): a view drawn beside the thread. It's drawn
// again after every redraw, so its controls wait, or stop waiting, with the page.
let panelOf = null; // { name, params, opener }

let panelAsked = 0;
let panelOpening = 0; // the call asked by a control (an opener) still on its way, if any
async function showPanel(name, params, opener) {
  // The newest call wins; a redraw (no opener) for a panel closed or changed meanwhile
  // draws nothing, so a late view can't reopen it or take the focus. A redraw never
  // outranks a panel you asked for that's still on its way (fix review, batch 6, item 7):
  // it isn't asked at all, and the panel you asked for redraws itself when it lands.
  if (!opener && panelOpening) return;
  const asked = ++panelAsked;
  if (opener) panelOpening = asked;
  let html;
  try {
    html = await view(name, params);
  } catch (error) {
    if (asked === panelAsked) notice("!", error.message);
    return;
  } finally {
    if (panelOpening === asked) panelOpening = 0;
  }
  if (asked !== panelAsked || (!opener && (!panelOf || panelOf.name !== name))) return;
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
  $("branches-button").setAttribute("aria-expanded", String(name === "branches"));
  docsLinks($("panel"));
  shade();
  if (name === "branches") markedOnly(Boolean(clarity.markedOnly));
  const again = focused && $(focused);
  if (again) again.focus();
  else if (opening) $("panel-title").focus();
}

function docsLinks(root) {
  for (const link of root.querySelectorAll("a[data-docs]")) {
    link.href = `${clarity.docs || ""}${link.dataset.docs}`; // the guide at this version
  }
}

function closePanel() {
  // Focus goes back to the control that opened the panel, or the one drawn in its place.
  // A view still on its way is now older than this close, so it can't reopen the panel.
  panelAsked += 1;
  panelOpening = 0;
  const opener = panelOf && panelOf.opener;
  panelOf = null;
  $("panel").hidden = true;
  $("branches-button").setAttribute("aria-expanded", "false");
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
  const sentence = answer.error && answer.error.replace(/^\/set: /, ""); // no command line here
  if (answer.error && row) {
    refuseRow(row, box, sentence);
    return;
  }
  if (answer.error) notice("!", sentence);
  await refresh();
}

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
    $("branches-button").setAttribute("aria-label", `Branches, ${ends}`); // its name when narrow
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
  const answer = await command("replay", call)
    .catch((error) => {
      const lost = unanswered(error);
      if (!lost.handled) notice("!", lost.error);
      return lost;
    })
    .finally(() => (clarity.sending = false));
  if (answer.handled) return;
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

// --- Attach (v0.3.1 C1): the paperclip, the sheet and the chips (board V31Attach)
//
// The file goes to /upload as it is; the sheet is the `attach` view, drawn from `/attach ID
// [range] check`, so every number and sentence is the server's. A changed range asks for
// the sheet again (the extraction is cached: no second reading). Add sends the checked
// choice's command, and only once its count is the boxes' range (Checkpoint 3, must 1).

// phase: uploading, reading, sheet or stopped; attempt numbers each file, asked each read.
const attaching = { upload: null, name: "", phase: null, attempt: 0, asked: 0, abort: null, timer: null, adding: false };

function readingLine(state) {
  // A file read that this page didn't start (a reload or another tab's, mid-read) still
  // shows, with its Cancel (the whole-branch review's could 13; the plan's S1).
  const running = state.running;
  const on = Boolean(state.reading) && Boolean(running) && running.command === "attach" && !attaching.phase;
  $("reading-line").hidden = !on;
  if (!on) {
    $("reading-line").replaceChildren();
    return;
  }
  const cancel = sheetButton("Cancel", "reading-cancel", "linkbtn");
  cancel.dataset.upload = (running.args || "").split(" ")[0];
  $("reading-line").replaceChildren(`Reading ${state.reading.name}…`, " ", cancel);
}

function newWithoutAChat() {
  // New chat with no chat open (the first screen, or a model starting): the picker, where
  // you choose the profile; it waits during a start and says why (E1, extended).
  closeDrawer(false);
  if ($("picker-button").getAttribute("aria-disabled") === "true") {
    const why = $("picker-waits");
    if (why) why.classList.remove("vh");
  } else {
    openPicker();
  }
}

function compareProblem(sentence) {
  const line = document.createElement("p");
  line.setAttribute("role", "alert");
  const bold = document.createElement("strong");
  bold.textContent = "!";
  line.append(bold, " ", sentence);
  const box = document.createElement("div");
  box.className = "problem";
  box.append(line);
  $("compare").prepend(box);
}

function clipWhy(running) {
  // The paperclip waits while a model starts or anything that changes the chat runs (a
  // reply, a Replay), and says why when pressed (Checkpoint 3, must 2 and could 13).
  if (clarity.state && clarity.state.model === "starting") return "Waits until the model has started.";
  if (changing(running) || clarity.sending) return waitsFor(running);
  return "";
}

function waitsFor(running) {
  return running && WRITES.has(running.command) ? "Waits until the reply is done." : "Waits until the command is done.";
}

function pickFile() {
  // In a chat-mode chat the picker opens in the press itself, so the browser keeps the
  // press's permission to open it (the whole-branch review's should 6). In text mode
  // nothing opens: the session's refusal says why, as a refusal (Checkpoint 3's should 5,
  // the walkthrough's F12).
  const why = clipWhy(clarity.state.running); // a Replay this page sent, not yet drawn
  if (why) {
    $("clip-waits").textContent = why;
    return;
  }
  $("clip-waits").textContent = "";
  if (clarity.state.mode === "chat") {
    // A fresh input for each press, with the same id (the QA driver's upload step): Firefox
    // sometimes never opened a second picker on one input (CI, run 43766644268).
    const fresh = $("attach-file").cloneNode();
    $("attach-file").replaceWith(fresh);
    fresh.click();
    return;
  }
  command("attach")
    .catch(unanswered)
    .then((answer) => answer.error && notice("!", answer.error));
}

function sheetHead(title, close) {
  // The sheet's head, as the template draws it: a document icon, the title and, once
  // nothing is being read, ×.
  const head = document.createElement("div");
  head.className = "shhd";
  const h2 = document.createElement("h2");
  h2.id = "attach-title";
  h2.tabIndex = -1;
  h2.append(icon("doc"), title);
  head.append(h2);
  if (close) {
    const x = sheetButton("", "attach-close", "iconbtn");
    x.setAttribute("aria-label", "Close");
    x.append(icon("close"));
    head.append(x);
  }
  return head;
}

function icon(name) {
  const svg = $("clip").querySelector("svg").cloneNode(true); // the page's own icon markup
  svg.querySelector("use").setAttribute("href", `#i-${name}`);
  return svg;
}

function sheetButton(label, action, cls) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = cls;
  button.dataset.action = action;
  button.textContent = label;
  return button;
}

function sheetReading(name) {
  // "Reading NAME…" with Cancel, while the file goes up and its text is read (S1).
  const status = document.createElement("p");
  status.className = "reading";
  status.setAttribute("role", "status");
  const spin = document.createElement("span");
  spin.className = "spin";
  spin.setAttribute("aria-hidden", "true");
  status.append(spin, `Reading ${name}…`);
  const actions = document.createElement("div");
  actions.className = "actions";
  const cancel = sheetButton("Cancel", "attach-cancel", "btn ghost");
  actions.append(cancel);
  $("attach-body").replaceChildren(sheetHead(name, false), status, actions);
  if (!$("attach-sheet").open) $("attach-sheet").showModal();
  cancel.focus();
  // A slow file: after a few seconds, how long it may take (F15; the sentence is SENTENCES').
  const attempt = attaching.attempt;
  setTimeout(() => {
    if (attempt !== attaching.attempt || !["uploading", "reading"].includes(attaching.phase)) return;
    const slow = document.createElement("p");
    slow.className = "d2";
    slow.textContent = $("attach-sheet").dataset.slow;
    status.after(slow);
  }, 3000);
}

function sheetEnded(sentence, refused = true) {
  // Not attached: a refusal (an alert), or a read you stopped (a status), with Close. The
  // title is the read refusals' one, which the template draws (could 16).
  const line = document.createElement("p");
  line.setAttribute("role", refused ? "alert" : "status");
  if (refused) {
    const bold = document.createElement("strong");
    bold.textContent = "!";
    line.append(bold, " ");
  }
  line.append(sentence);
  const box = document.createElement("div");
  box.className = refused ? "problem" : "status";
  box.append(line);
  const actions = document.createElement("div");
  actions.className = "actions";
  const spacer = document.createElement("span");
  spacer.className = "sp";
  actions.append(spacer, sheetButton("Choose another file", "attach-pick", "btn ghost"), sheetButton("Close", "attach-close", "btn"));
  $("attach-body").replaceChildren(sheetHead("Not attached", true), box, actions);
  if (!$("attach-sheet").open) $("attach-sheet").showModal();
  actions.querySelector("[data-action=attach-close]").focus();
}

async function attachFile(file) {
  attaching.attempt += 1;
  const attempt = attaching.attempt;
  Object.assign(attaching, { upload: null, name: file.name, phase: "uploading", abort: new AbortController() });
  sheetReading(file.name);
  try {
    const query = new URLSearchParams({ tab: TAB, name: file.name });
    // The type is set here: a File body would send its own (application/pdf), refused 415.
    const response = await request(`/upload?${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
      signal: attaching.abort.signal,
    });
    if (attempt !== attaching.attempt || attaching.phase !== "uploading") return; // stopped
    if (!response.ok) {
      const text = await response.text();
      if (text === "Clarity is open in another tab.") notLive();
      else sheetEnded(text);
      return;
    }
    attaching.upload = (await response.json()).upload;
    attaching.phase = "reading";
    await loadSheet({}, true);
  } catch (error) {
    if (attempt !== attaching.attempt || attaching.phase !== "uploading") return; // Cancel aborted it
    const lost = unanswered(error);
    if (!lost.handled) sheetEnded(lost.error);
  }
}

async function cancelRead(draw = true) {
  // Cancel stops only this file's read (Checkpoint 3's must 2). An upload still sending is
  // aborted: nothing reads it. Otherwise /cancel names the upload: its read stops if it
  // runs, and is refused if it hasn't started (the whole-branch review's should 4). The
  // sheet then says what the read's own answer says, once it has stopped.
  const { phase, upload, name } = attaching;
  if (phase !== "uploading" && phase !== "reading") return;
  if (phase === "uploading") {
    attaching.phase = "stopped";
    attaching.abort.abort();
    if (draw) sheetEnded($("attach-sheet").dataset.stopped.replace("{name}", () => name), false); // SENTENCES
    return;
  }
  attaching.phase = "cancelling";
  const cancel = $("attach-body").querySelector("[data-action=attach-cancel]");
  if (cancel) cancel.setAttribute("aria-disabled", "true"); // pressed once: the answer comes
  const answer = await command("cancel", upload).catch(unanswered);
  if (answer.handled || attaching.phase !== "cancelling") return;
  if (answer.error && draw) {
    attaching.phase = "stopped";
    sheetEnded(answer.error); // the cancel didn't reach Clarity: said, not swallowed
  }
}

function boxesHold(range) {
  // Whether the range boxes still say `range` (a read with no range: always).
  if (range.first === undefined) return true;
  const from = $("attach-from");
  const to = $("attach-to");
  return Boolean(from && to) && from.value === range.first && to.value === range.last;
}

async function loadSheet(range, first = false) {
  attaching.asked += 1;
  const asked = attaching.asked;
  const attempt = attaching.attempt;
  const current = () => asked === attaching.asked && attempt === attaching.attempt && attaching.phase !== "stopped";
  // A cancelled read's answer is its refusal (SENTENCES' "stopped"): drawn as it comes.
  let html;
  try {
    html = await view("attach", { upload: attaching.upload, ...range });
  } catch (error) {
    if (!current()) return;
    const lost = unanswered(error); // handled: the page already shows the no-key page
    if (!lost.handled) sheetEnded(lost.error);
    return;
  } finally {
    // A redraw while the read ran drew "attach" running. It has ended: draw what runs now,
    // before the sheet, so the picker opens when pressed (Firefox CI, run 43755817667).
    if (changing(clarity.state.running)) await refresh();
  }
  // Only the newest answer, for the range the boxes still hold, is drawn: an older one, or
  // one for keys typed over since, would show a count for another range (must 1).
  if (!current() || !$("attach-sheet").open || !boxesHold(range)) return;
  if (attaching.phase === "cancelling") {
    // Cancel was pressed while this answer was on its way: the read stops here, as asked,
    // whatever the answer (the final fix review's could 9; the surface finder's should 3).
    attaching.phase = "stopped";
    sheetEnded($("attach-sheet").dataset.stopped.replace("{name}", () => attaching.name), false); // SENTENCES
    return;
  }
  attaching.phase = "sheet";
  // The preview stays as you left it; a new sheet opens as the server drew it (F7).
  const opened = $("attach-preview") ? !$("attach-preview").hidden : null;
  const focus = document.activeElement;
  const typing = focus && ["attach-from", "attach-to"].includes(focus.id) ? focus : null;
  swap($("attach-body"), html);
  if (typing) {
    // The box being typed in stays, caret and all: it holds what was counted.
    const fresh = $(typing.id);
    typing.max = fresh.max;
    fresh.replaceWith(typing);
  }
  docsLinks($("attach-body"));
  rove($("attach-body"));
  const preview = $("attach-preview"); // a refusal's sheet has none (the final fix review, 1)
  if (preview) showPreview(opened === null ? !preview.hidden : opened);
  updateAdd();
  const back = focus && focus.id && $(focus.id);
  if (first || !back) $("attach-title").focus();
  else back.focus();
}

function checkedChoice() {
  return $("attach-body").querySelector("[role=radio][aria-checked=true]");
}

function updateAdd() {
  // Add sends only a choice that fits and was counted for what the sheet shows; otherwise
  // it waits, and the line beside it, or the choice's own note, says why.
  const add = $("attach-body").querySelector("[data-action=attach-add]");
  if (!add) return;
  const choice = checkedChoice();
  const ok = Boolean(choice) && choice.dataset.ok === "true";
  if (ok) add.removeAttribute("aria-disabled");
  else add.setAttribute("aria-disabled", "true");
  const labels = { new: "Start new chat", search: "Search it" };
  add.textContent = (choice && labels[choice.dataset.choice]) || "Add to message";
  const why = $("attach-why");
  why.dataset.choose = why.dataset.choose || why.textContent; // the server's sentence
  let text = "";
  if (!choice) text = why.dataset.choose;
  else if (choice.dataset.counted === "false") text = rangeTyped() ? "Counting…" : "Waits for a number in both boxes.";
  why.textContent = text;
  why.hidden = !text;
  add.setAttribute("aria-describedby", text ? "attach-why" : choice.dataset.note);
}

function showPreview(on) {
  const preview = $("attach-preview");
  if (!preview) return;
  preview.hidden = !on;
  const button = $("attach-body").querySelector("[data-action=attach-preview]");
  button.setAttribute("aria-expanded", String(on));
  button.textContent = on ? "Hide extracted text" : "Show extracted text";
}

function framingForm(on) {
  $("framing-form").hidden = !on;
  if ($("framing-shown")) $("framing-shown").hidden = on; // the box holds it while editing (⑦)
  const edit = $("attach-body").querySelector("[data-action=framing-edit]");
  edit.hidden = on;
  edit.setAttribute("aria-expanded", String(on));
  if (on) $("framing-line").focus();
  else edit.focus();
}

async function framingCommand(args) {
  // /framing for this chat, then the sheet again, counted with the line it now sends.
  const answer = await command("framing", args).catch(unanswered);
  if (answer.handled) return;
  if (answer.error) {
    notice("!", answer.error);
    return;
  }
  const choice = checkedChoice();
  const part = choice && choice.dataset.choice === "part" && $("attach-from") && rangeTyped();
  const unit = part ? $("attach-body").querySelector(".range").dataset.unit : null;
  await loadSheet(part ? { unit, first: $("attach-from").value, last: $("attach-to").value } : {});
}

function rangeTyped() {
  return /^[0-9]{1,9}$/.test($("attach-from").value) && /^[0-9]{1,9}$/.test($("attach-to").value);
}

function rangeChanged() {
  clearTimeout(attaching.timer);
  if (!rangeTyped()) return; // Add waits, saying why (updateAdd)
  const unit = $("attach-body").querySelector(".range").dataset.unit;
  loadSheet({ unit, first: $("attach-from").value, last: $("attach-to").value });
}

function counting() {
  // A range typed chooses the part, and its count is now another range's: Add waits.
  const part = $("attach-body").querySelector("[data-choice=part]");
  for (const radio of $("attach-body").querySelectorAll("[role=radio]")) {
    radio.setAttribute("aria-checked", String(radio === part));
  }
  part.dataset.ok = "false";
  part.dataset.counted = "false";
  updateAdd();
}

document.addEventListener("input", (event) => {
  if (!["attach-from", "attach-to"].includes(event.target.id)) return;
  counting();
  clearTimeout(attaching.timer);
  attaching.timer = setTimeout(rangeChanged, 400); // typing: one read once it settles
});

document.addEventListener("change", (event) => {
  if (["attach-from", "attach-to"].includes(event.target.id)) rangeChanged();
});

async function searchIt(choice) {
  // "Search it" (v0.3.2 C2): the file into your documents (unless the same file is there),
  // then /search with the chat's list and it; the notice says so, with its control-token
  // count from /search's answer.
  const existing = choice.dataset.existing;
  const name = existing || $("attach-search-name").value.trim();
  if (!existing) {
    const words = ["add", choice.dataset.args, name, ...(choice.dataset.replace === "true" ? ["--replace"] : [])];
    const added = await command("documents", words.join(" ")).catch(unanswered);
    if (added.error || added.handled) return added;
  }
  const answer = await command("search", [...new Set([...searchedNames(), name])].join(" ")).catch(unanswered);
  if (answer.error || answer.handled) return answer;
  const box = $("attach-search-note");
  const lines = [box.dataset.searched.replace("{name}", () => name)];
  const count = (answer.result.searching.find((d) => d.name === name) || {}).control_token_parts;
  if (count) lines.push(count === 1 ? box.dataset.controlOne : box.dataset.control.replace("{n}", () => String(count)));
  clarity.searchNotice = lines;
  return answer;
}

async function addAttachment() {
  const choice = checkedChoice();
  if (!choice || choice.dataset.ok !== "true" || attaching.adding) return;
  // One press: a second can't send a second /new or attach (the whole-branch review, 5).
  attaching.adding = true;
  $("attach-body").querySelector("[data-action=attach-add]").setAttribute("aria-disabled", "true");
  try {
    await adding(choice);
  } finally {
    attaching.adding = false;
  }
}

async function adding(choice) {
  let answer;
  let started = null; // Start new chat's /new: what it dropped is said once it's done
  if (choice.dataset.choice === "new") {
    answer = started = await command("new").catch(unanswered);
    if (!answer.error && !answer.handled) answer = await command("attach", choice.dataset.args).catch(unanswered);
  } else if (choice.dataset.choice === "search") {
    answer = await searchIt(choice);
  } else {
    answer = await command("attach", choice.dataset.args).catch(unanswered);
  }
  if (answer.handled) return;
  if (started && !started.error) notKept(started.result); // the old chat's chips: not kept
  if (answer.error) {
    const line = $("attach-body").querySelector(".sheet-error");
    line.textContent = `! ${answer.error}`;
    line.hidden = false;
    updateAdd();
    return;
  }
  $("attach-sheet").close();
  await refresh();
  if (clarity.searchNotice) {
    const [first, ...rest] = clarity.searchNotice;
    clarity.searchNotice = null;
    notice(null, [first, ...rest.map((line) => `! ${line}`)].join(" "));
  }
  $("message").focus();
}

// --- Chats and export (C5)

async function showDrawer(opener = null) {
  // The Chats drawer, from /view/chats. Redrawn with the page, it keeps a refusal under the
  // row it was for, and the row being opened.
  let html;
  try {
    html = await view("chats");
  } catch (error) {
    notice("!", error.message);
    return;
  }
  if (!opener && $("drawer").hidden) return; // a redraw for a drawer closed meanwhile
  const active = document.activeElement;
  const held = active && $("drawer").contains(active) ? active.dataset.resume || active.className : null;
  swap($("drawer"), html);
  if (held) {
    // A redraw keeps the focus on the same row or button.
    const again = [...$("drawer").querySelectorAll("button")].find(
      (b) => (b.dataset.resume || b.className) === held,
    );
    if (again) again.focus();
  }
  const refusal = clarity.drawerRefusal;
  for (const row of $("drawer").querySelectorAll("[data-resume]")) {
    row.classList.toggle("busy", row.dataset.resume === clarity.opening);
    if (refusal && row.dataset.resume === refusal.id) {
      const problem = row.nextElementSibling;
      problem.querySelector(".why").textContent = refusal.text;
      problem.hidden = false;
    }
  }
  $("drawer").hidden = false;
  $("scrim").hidden = false;
  $("chats-handle").setAttribute("aria-expanded", "true");
  if (opener) {
    clarity.drawerOpener = opener;
    const current = $("drawer").querySelector('.crow[aria-current="true"]');
    (current || $("drawer").querySelector(".crow") || $("drawer").querySelector(".iconbtn")).focus();
  }
}

function closeDrawer(returnFocus) {
  if ($("drawer").hidden) return;
  $("drawer").hidden = true;
  $("scrim").hidden = true;
  $("chats-handle").setAttribute("aria-expanded", "false");
  clarity.drawerRefusal = null;
  const opener = clarity.drawerOpener;
  if (returnFocus) (opener && opener.isConnected ? opener : $("message")).focus();
}

async function openChat(row) {
  // Picking a row sends /resume. A refusal before anything starts stays under the row and
  // the chat you were in stays as it was (L6); once open, the drawer closes.
  if (row.getAttribute("aria-disabled") === "true" || clarity.opening) return;
  if (row.getAttribute("aria-current") === "true") {
    closeDrawer(true); // the chat that's open: nothing to reopen
    return;
  }
  clarity.opening = row.dataset.resume;
  clarity.drawerRefusal = null;
  for (const problem of $("drawer").querySelectorAll(".problem")) problem.hidden = true;
  row.classList.add("busy");
  notice(null, null);
  clarity.sending = true;
  const answer = await command("resume", row.dataset.resume)
    .catch(unanswered)
    .finally(() => {
      clarity.sending = false;
      clarity.opening = null;
    });
  if (answer.handled) return;
  if (answer.error && answer.type === "ConfigError") {
    clarity.drawerRefusal = { id: row.dataset.resume, text: answer.error };
    await refresh();
    const shown = $("drawer").querySelector(`[data-resume="${CSS.escape(row.dataset.resume)}"]`);
    if (shown) shown.focus();
    return;
  }
  closeDrawer(false);
  await refresh(); // first: a redraw that fails says so, and the sentence below stays
  if (answer.error && !answer.stopped) notice("! Didn’t open.", answer.error);
  else if (!answer.error) {
    notKept(answer.result);
    $("message").focus();
  }
}

function showDiffer(on) {
  // "Show what changed" after a resume: closed at first (L3), kept as left on redraws.
  clarity.differOpen = on;
  const button = $("thread").querySelector('[data-action="differ"]');
  if (!button) return;
  button.setAttribute("aria-expanded", String(on));
  $("differ").hidden = !on;
}

async function restartModel() {
  // "Start the model again": /resume for this chat. A refusal says so, with the server's
  // output after its first line in a box that scrolls (C23).
  notice(null, null);
  clarity.sending = true;
  const answer = await command("resume", clarity.state.log_id)
    .catch(unanswered)
    .finally(() => (clarity.sending = false));
  if (answer.handled) return;
  await refresh(); // first: a redraw that fails says so, and the sentence below stays
  if (!answer.error) notKept(answer.result);
  if (answer.error && !answer.stopped) {
    const [first, ...rest] = answer.error.split("\n");
    const box = notice("! The model didn’t start again.", first);
    if (rest.length) {
      const output = document.createElement("pre");
      output.className = "outbox";
      output.tabIndex = 0;
      output.setAttribute("aria-label", "llama-server’s output");
      output.textContent = rest.join("\n");
      box.append(output);
    }
  }
}

function renaming(on, returnFocus = false) {
  // Rename… turns the title into a box, its text selected; Save sends /title and waits
  // while the box is empty; Cancel, Escape or leaving the box keeps the old title.
  clarity.renaming = on;
  $("rename").hidden = !on;
  $("title").hidden = on || !clarity.state.conversation_id;
  if (on) {
    $("rename-text").value = clarity.state.title || "";
    renameWaits();
    $("rename-text").focus();
    $("rename-text").select();
  } else if (returnFocus) {
    $("title").focus();
  }
}

function renameWaits() {
  if ($("rename-text").value.trim()) $("rename-save").removeAttribute("aria-disabled");
  else $("rename-save").setAttribute("aria-disabled", "true");
}

async function rename() {
  const text = $("rename-text").value.trim();
  if (!text) return;
  const answer = await command("title", text).catch(unanswered);
  if (answer.handled) return;
  renaming(false, true);
  await refresh(); // first: a redraw that fails says so, and the sentence below stays
  if (answer.error) notice("!", answer.error);
}

async function openDocuments(params = {}, focus = "documents-title") {
  // The Documents… page (v0.3.2 C1), drawn by the documents view: the library, Add's
  // panel (an upload and its name), a notice, and the try box (a message to try).
  closeDrawer(false);
  let html;
  try {
    html = await view("documents", params);
  } catch (error) {
    notice("!", error.message);
    return;
  }
  swap($("documents-body"), html);
  if (!$("documents-sheet").open) $("documents-sheet").showModal();
  ($(focus) || $("documents-title")).focus();
}

function documentsRefused(sentence) {
  // A command's refusal, shown in the page above the library; the rest stays as drawn.
  const refused = $("documents-body").querySelector(".refused");
  refused.textContent = `! ${sentence}`;
  refused.hidden = false;
  refused.scrollIntoView({ block: "nearest" });
}

async function documentsCommand(name, args, after) {
  // A command from the page; its refusal shows there, its result redraws the page.
  const answer = await command(name, args).catch(unanswered);
  if (answer.handled) return;
  if (answer.error) {
    documentsRefused(answer.error);
    return;
  }
  await openDocuments(after(answer.result));
  if (name === "search") refresh();
}

async function documentsUpload(file) {
  // Add…: the file goes to /upload as the paperclip's does, then the page draws Add's panel.
  const query = new URLSearchParams({ tab: TAB, name: file.name }).toString();
  let response;
  try {
    response = await request(`/upload?${query}`, {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: file,
    });
  } catch (error) {
    const lost = unanswered(error);
    if (!lost.handled) documentsRefused(lost.error);
    return;
  }
  if (!response.ok) {
    const text = await response.text();
    if (text === "Clarity is open in another tab.") notLive();
    else documentsRefused(text);
    return;
  }
  await openDocuments({ upload: (await response.json()).upload }, "documents-name");
}

function documentsAdd(form) {
  const name = $("documents-name").value.trim();
  const upload = form.dataset.upload;
  documentsCommand("documents", `add ${upload} ${name}`, () => ({})); // the notice is its result
}

function documentsSearch(names, after) {
  documentsCommand("search", names.length ? names.join(" ") : "none", after);
}

async function openChooser() {
  // The chooser reads the library when it opens, as the picker reads its profiles.
  const row = await view("picker", { library: "1" }).catch(() => null);
  if (row !== null) swap($("composer-row"), row);
  if ($("search-button")) showMenu($("search-button"));
}

function showFound(key, on) {
  const line = $("thread").querySelector(`[data-found="${CSS.escape(key)}"]`);
  if (on) clarity.foundOpen.add(key);
  else clarity.foundOpen.delete(key);
  if (!line) return;
  line.querySelector("[data-action=found]").setAttribute("aria-expanded", String(on));
  line.querySelector(".parts").hidden = !on;
}

async function fixCommand(name, args) {
  // A command from a button in the thread: a refusal is a notice; the thread is drawn again.
  const answer = await command(name, args).catch(unanswered);
  if (answer.handled) return;
  if (answer.error) notice("!", answer.error);
  await refresh();
}

async function chooserCommand(name, args) {
  // A command from the chooser: the row under the box is drawn again from /state and the
  // chooser opens again on it; a refusal is a notice.
  const answer = await command(name, args).catch(unanswered);
  if (answer.handled) return;
  if (answer.error) notice("!", answer.error);
  await refresh();
  await openChooser();
}

function searchedNames() {
  return (clarity.state.searching || []).map((d) => d.name);
}

async function exportChat() {
  // Export writes the file at once; the sheet is the result, or the refusal with Try again.
  closeDrawer(false);
  try {
    await command("export");
  } catch (error) {
    const lost = unanswered(error);
    if (!lost.handled) notice("!", lost.error); // the sheet would show the last export's
    return;
  }
  let html;
  try {
    html = await view("export");
  } catch (error) {
    notice("!", error.message);
    return;
  }
  swap($("export-body"), html);
  docsLinks($("export-body"));
  if (!$("export-sheet").open) $("export-sheet").showModal();
  $("export-title").focus();
}

document.addEventListener("input", (event) => {
  if (event.target.id === "rename-text") renameWaits();
  if (event.target.id === "more-n") $("more-make").textContent = `Make ${event.target.value}`; // "Make 3", as Again's
});

document.addEventListener("mousedown", (event) => {
  // Save and Cancel keep the focus in the rename box, so leaving it means leaving the form
  // (Safari doesn't focus a clicked button).
  if (event.target.closest("#rename button")) event.preventDefault();
});

document.addEventListener("focusout", (event) => {
  // Leaving the rename box (not for its own buttons) keeps the old title.
  const form = $("rename");
  if (form.hidden || !form.contains(event.target)) return;
  setTimeout(() => {
    if (clarity.renaming && !form.contains(document.activeElement)) renaming(false);
  }, 0);
});

// --- Appearance and access (C6)

// Appearance lives in the browser's own storage, so Clarity writes no file for it.
const THEMES = {
  notebook: { cls: "", desc: "Notebook: serif text on warm paper. The default." },
  neutral: { cls: "t-neutral", desc: "Neutral: plain type on white or near-black." },
  sky: { cls: "t-sky", desc: "Sky: sea and sky around the page.", scenic: true },
  space: { cls: "t-space", desc: "Space: stars and galaxies around the page.", scenic: true },
  hc: { cls: "t-hc", desc: "High contrast: black and white, with heavier borders and focus rings." },
};
const REDUCE = window.matchMedia("(prefers-reduced-motion: reduce)");

function stored(name, fallback, allowed) {
  try {
    const value = localStorage.getItem(`clarity.${name}`);
    return allowed.includes(value) ? value : fallback;
  } catch {
    return fallback; // storage blocked: the defaults
  }
}

function store(name, value) {
  try {
    localStorage.setItem(`clarity.${name}`, value);
  } catch {
    // storage blocked: the choice lasts until the page closes
  }
}

const appearance = {
  theme: stored("theme", "notebook", Object.keys(THEMES)),
  look: stored("look", "system", ["light", "dark", "system"]),
  motion: stored("motion", "default", ["default", "on", "off"]),
};

function applyAppearance() {
  // A theme is a class of tokens on the page; Light or dark sets the scheme, and Match
  // system leaves it to the browser. Motion: the switch's choice, else the system's.
  const root = document.documentElement;
  const theme = THEMES[appearance.theme];
  root.className = ["clarity", theme.cls, theme.scenic ? "scenic" : "", appearance.motion === "default" ? "" : `motion-${appearance.motion}`].filter(Boolean).join(" ");
  if (appearance.look === "system") delete root.dataset.scheme;
  else root.dataset.scheme = appearance.look;
  const on = appearance.motion === "default" ? !REDUCE.matches : appearance.motion === "on";
  $("motion-row").hidden = !theme.scenic; // nothing moves in the other themes
  $("motion").setAttribute("aria-checked", String(on));
  $("motion-word").textContent = on ? "On" : "Off";
  $("motion-why").hidden = !(appearance.motion === "default" && REDUCE.matches);
  $("theme-desc").textContent = theme.desc;
  for (const input of $("appearance").querySelectorAll("input")) {
    input.checked = input.value === appearance[input.name];
  }
}

document.addEventListener("change", (event) => {
  const input = event.target.closest("#appearance input");
  if (!input) return;
  appearance[input.name] = input.value; // changes apply at once; there is no Save
  store(input.name, input.value);
  applyAppearance();
});

REDUCE.addEventListener("change", () => applyAppearance());
applyAppearance(); // before the page shows, so it never flashes the default

function toggleMotion() {
  appearance.motion = $("motion").getAttribute("aria-checked") === "true" ? "off" : "on";
  store("motion", appearance.motion);
  applyAppearance();
}

// Toolbars and the replies' rows are one Tab stop each; the arrow keys move inside, and Tab
// comes back to the control last used there.
const RADIOS = '.opts[role="radiogroup"]'; // the attach sheet's choices: arrows choose too
const ROVING = `.tb[role="toolbar"], .listpop .rows, ${RADIOS}`;

function rove(root) {
  for (const group of root.querySelectorAll(ROVING)) {
    const items = itemsIn(group);
    const kept = items.find((item) => item.dataset.action === group.dataset.last || item.dataset.switch === group.dataset.last);
    const checked = group.matches(RADIOS) && items.find((item) => item.getAttribute("aria-checked") === "true");
    items.forEach((item) => (item.tabIndex = -1));
    if (items.length) (checked || kept || items[0]).tabIndex = 0;
  }
}

function itemsIn(group) {
  return [...group.querySelectorAll("button:not([disabled])")];
}

document.addEventListener("focusin", (event) => {
  const group = event.target.closest && event.target.closest(ROVING);
  if (!group || event.target.tagName !== "BUTTON") return;
  for (const item of itemsIn(group)) item.tabIndex = item === event.target ? 0 : -1;
  group.dataset.last = event.target.dataset.action || event.target.dataset.switch || "";
});

$("message").addEventListener("keydown", (event) => {
  // Enter sends and Shift+Enter starts a new line, the browser chat convention; Enter
  // while an input method is composing belongs to it.
  if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
  event.preventDefault();
  send();
});

document.addEventListener("keydown", (event) => {
  const group = event.target.closest && event.target.closest(ROVING);
  if (!group) return;
  const radios = group.matches(RADIOS);
  const key = (radios && { ArrowLeft: "ArrowUp", ArrowRight: "ArrowDown" }[event.key]) || event.key;
  const keys = group.matches(".tb") ? ["ArrowLeft", "ArrowRight"] : ["ArrowUp", "ArrowDown"];
  if (![...keys, "Home", "End"].includes(key)) return;
  const items = itemsIn(group);
  const at = items.indexOf(event.target);
  if (at < 0) return;
  event.preventDefault();
  const to = { [keys[0]]: Math.max(at - 1, 0), [keys[1]]: Math.min(at + 1, items.length - 1), Home: 0, End: items.length - 1 }[key];
  items[to].focus();
  if (radios) act(items[to]); // the radio pattern: the arrows choose as they move
});

const THREAD = {
  // Details (C4b): one call's record, from the toolbar, the facts line or a failed line.
  details: (element, call) => showPanel("details", { call }, element),
  // Your message's attachment line: Details for its reply, at Attachments (F16); "Show in
  // Details" under "From documents", at Documents (C3).
  attached: async (element) => {
    await showPanel("details", { call: element.dataset.reply }, element);
    const key = element.dataset.section || "attachments";
    const section = $("panel").querySelector(`details[data-key=${key}]`);
    if (!section) return;
    section.open = true;
    section.querySelector("summary").focus();
    section.scrollIntoView({ block: "nearest" });
  },
  settings: (element) => showPanel("settings", {}, element),
  replies: (element) => openReplies(element),
  appearance() {
    applyAppearance();
    $("appearance").showModal();
    $("appearance").querySelector('input[name="theme"]:checked').focus();
  },
  chats(element) {
    if (!$("drawer").hidden && element.id === "chats-handle") closeDrawer(true);
    else showDrawer(element);
  },
  "panel-close": () => closePanel(),
  branches: (element) => showPanel("branches", {}, element),
  "marked-only": () => markedOnly(!clarity.markedOnly),
  replay: (element, call) => replay(call, element),
  "replay-stop": () => command("cancel"),
  "reset-all": async (element) => {
    for (const name of element.dataset.names.split(" ")) {
      const answer = await command("set", name);
      if (answer.error) notice("!", answer.error.replace(/^\/set: /, ""));
    }
    await refresh();
  },
  "replies-close": () => closeReplies(true),
  again() {
    closeReplies(false);
    write("regen", "same", null, "replies");
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
  regen: () => write("regen", "", null, "regen"),
  restart: () => restartModel(),
  "appearance-close": () => $("appearance").close(),
  motion: () => toggleMotion(),
  "chats-close": () => closeDrawer(true),
  differ: () => showDiffer(!clarity.differOpen),
  rename: () => renaming(true),
  "rename-cancel": () => renaming(false, true),
  export: () => exportChat(),
  "export-close": () => $("export-sheet").close(),
  documents: () => {
    hideMenu(false);
    openDocuments();
  },
  "search-menu": (element) => {
    if (openMenu && openMenu.button === element) hideMenu(true);
    else openChooser();
  },
  "search-toggle": (element) => {
    // The chooser (v0.3.2 C2): a row ticked or unticked sends /search with the new list.
    const name = element.dataset.name;
    const names = searchedNames();
    const next = element.getAttribute("aria-checked") === "true" ? names.filter((n) => n !== name) : [...names, name];
    chooserCommand("search", next.length ? next.join(" ") : "none");
  },
  "search-rebuild": (element) => chooserCommand("documents", `rebuild ${element.dataset.name}`),
  found: (element) => {
    const key = element.closest("[data-found]").dataset.found;
    showFound(key, element.getAttribute("aria-expanded") !== "true");
  },
  // A refused Send's buttons (W6): each fixes its own line; the box goes with the last.
  "refused-rebuild": (element) => fixCommand("documents", `rebuild ${element.dataset.name}`),
  "refused-stop": (element) => {
    const rest = searchedNames().filter((n) => n !== element.dataset.name);
    fixCommand("search", rest.length ? rest.join(" ") : "none");
  },
  "new-with-documents": async (element) => {
    // A full context: a new chat on this profile, then /search with this chat's documents.
    const answer = await start("new", "");
    if (answer && !answer.error && !answer.handled) await fixCommand("search", element.dataset.names);
  },
  "documents-close": () => $("documents-sheet").close(),
  "documents-add": () => {
    // A fresh input each press, as the paperclip's (Firefox keeps a used one's file).
    const input = $("documents-file").cloneNode();
    $("documents-file").replaceWith(input);
    input.click();
  },
  "documents-add-cancel": () => openDocuments(),
  "documents-rebuild": (element) => {
    const name = element.dataset.name;
    // The notice is its result; a message being tried stays (the review's could 4).
    const tried = $("documents-try").value;
    documentsCommand("documents", `rebuild ${name}`, () => (tried.trim() ? { try: tried } : {}));
  },
  "documents-search-here": (element) => {
    const name = element.dataset.name;
    documentsSearch([...searchedNames(), name], () => ({ searched: name }));
  },
  "documents-stop": (element) => {
    const name = element.dataset.name;
    const tried = $("documents-try").value;
    documentsSearch(searchedNames().filter((n) => n !== name), () => ({ try: tried }));
  },
  "attach-pick": () => pickFile(),
  "attach-choose": (element) => {
    // A choice made while a range is counted stays: that count's answer is dropped (could 15).
    clearTimeout(attaching.timer);
    attaching.asked += 1;
    for (const radio of $("attach-body").querySelectorAll("[role=radio]")) {
      radio.setAttribute("aria-checked", String(radio === element));
    }
    updateAdd();
    if (element.dataset.counted === "false") rangeChanged(); // the part: count its boxes (should 4)
  },
  "attach-preview": () => showPreview($("attach-preview").hidden),
  "attach-replace": (element) => {
    // Q12: Replace, once its sentence says what's overwritten; Search it then replaces.
    $("attach-replace-note").hidden = false;
    element.setAttribute("aria-expanded", "true");
    $("attach-search").dataset.replace = "true";
    $("attach-search-name").value = element.closest(".name").querySelector("input").defaultValue;
  },
  "attach-add": () => addAttachment(),
  "attach-cancel": () => cancelRead(),
  "reading-cancel": async (element) => {
    element.setAttribute("aria-disabled", "true");
    const answer = await command("cancel", element.dataset.upload).catch(unanswered);
    if (answer.error) notice("!", answer.error);
    await refresh();
  }, // only this read stops, and the sheet says so (ruling 8)
  "attach-close": () => $("attach-sheet").close(),
  detach: async (element) => {
    // By the number it was drawn with: every × waits until this one's redraw lands, so a
    // second press is on chips drawn afresh (the whole-branch review's should 5).
    for (const x of $("chips").querySelectorAll("[data-action=detach]")) x.setAttribute("aria-disabled", "true");
    const answer = await command("detach", element.dataset.n).catch(unanswered);
    if (answer.handled) return;
    if (answer.error) notice("!", answer.error);
    await refresh();
    $("message").focus();
  },
  download: async () => {
    // A refusal shows in the sheet.
    const why = await download();
    const refused = $("export-body").querySelector(".refused");
    refused.textContent = why || "";
    refused.hidden = !why;
  },
  edit: (element) => editing(element.closest("li"), true),
  "edit-cancel": (element) => editing(element.closest("li"), false),
  "framing-edit": () => framingForm(true),
  "framing-cancel": () => framingForm(false),
  "framing-back": () => framingCommand(""),
  "edit-drop": (element) => {
    // Q2's ×: the chip leaves this edit only; Send names it with --drop (the board's ⑥).
    const keeps = element.closest(".keeps");
    element.closest(".chip").hidden = true;
    const left = keeps.querySelectorAll(".chip:not([hidden])").length;
    keeps.querySelector(".lead").textContent = left ? keeps.dataset.lead : keeps.dataset.none;
    keeps.querySelector(".att:not(.lead)").hidden = !left;
    (keeps.querySelector(".chip:not([hidden]) .x") || keeps.closest("form").querySelector("textarea")).focus();
  },
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
  } else if (form.id === "rename") {
    rename();
  } else if (form.id === "framing-form") {
    // The framing line's Edit (C4, the board's ⑦): an empty line saves as none.
    const line = $("framing-line").value.trim();
    framingCommand(line || "none");
  } else if (form.matches("[data-edit]")) {
    const text = form.querySelector("textarea").value;
    const dropped = [...form.querySelectorAll(".keeps .chip[hidden] .x")].flatMap((x) => ["--drop", x.dataset.n]);
    if (text) write("edit", [...dropped, text].join(" "), text);
  } else if (form.matches("[data-more]")) {
    const n = form.querySelector("input").value;
    closeReplies(false);
    write("regen", n, null, "replies");
  } else if (form.matches("[data-note]")) {
    const text = form.querySelector("textarea").value.trim();
    if (text) annotate("note", form.closest("[data-call]").dataset.call, text);
  } else if (form.id === "documents-add-form") {
    documentsAdd(form);
  } else if (form.id === "documents-try-form") {
    const tried = $("documents-try").value.trim(); // an empty try runs nothing
    if (!$("documents-try").disabled && tried) openDocuments({ try: tried }, "documents-try");
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
clarity.refresh = refresh;

document.addEventListener("DOMContentLoaded", () => {
  clarity.key = readKey();
  if (!clarity.key) {
    noKey();
    return;
  }
  $("use-here").addEventListener("click", () => useHere());
  $("about-close").addEventListener("click", () => $("about").close());
  $("about").addEventListener("close", () => $("clarity-button").focus());
  $("appearance").addEventListener("close", () => $("clarity-button").focus());
  $("stop").addEventListener("click", () => command("cancel"));
  document.addEventListener("change", (event) => {
    if (event.target.id === "documents-file") {
      const file = event.target.files[0];
      if (file) documentsUpload(file);
      return;
    }
    // #attach-file is replaced on each press (pickFile), so its change is heard here.
    if (event.target.id !== "attach-file") return;
    const file = event.target.files[0];
    event.target.value = ""; // the same file can be chosen again
    if (file) attachFile(file);
  });
  $("attach-sheet").addEventListener("cancel", (event) => {
    // Escape while reading stops the read, and the sheet says so (must 2); then it closes.
    if (attaching.phase !== "uploading" && attaching.phase !== "reading") return;
    event.preventDefault();
    cancelRead();
  });
  $("attach-sheet").addEventListener("close", () => {
    cancelRead(false); // closed while reading, however: the read stops too
    Object.assign(attaching, { phase: null, upload: null });
    $("clip").focus();
  });
  load().catch(() => {});
});
