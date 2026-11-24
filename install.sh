#!/bin/sh
# Clarity's install command, for a Mac with Apple silicon and 16 GB of memory or more:
#
#   curl --proto '=https' --tlsv1.2 -fsSL https://raw.githubusercontent.com/andalibmalit/clarity-chat/v0.4.2/install.sh | sh
#
# It checks this Mac first and writes nothing until those checks pass. Then it installs uv
# if it's missing (its installer checked by sha256 first), puts Clarity in ~/Clarity at
# this release's tag, and runs `clarity setup`, which asks once before it downloads
# anything, then the web interface. Run again, it does only what's missing.
#
# The whole script is one function, called on its last line, so a download cut short runs
# nothing. Under `curl … | sh` the script itself is sh's input, so setup's question and the
# web interface read the terminal (/dev/tty) instead. uv's installer, git, uv and Clarity get
# only HOME and PATH.
set -eu

VERSION=v0.4.2
REPO_URL=https://github.com/andalibmalit/clarity-chat
UV_VERSION=0.11.19
# The sha256 of https://astral.sh/uv/0.11.19/install.sh (the owner's run, 2026-11-22).
UV_INSTALLER_SHA256=ef8cf0575d37cf3c72e05f153dd72a845a87a7bb9be86184d5fe931b8c426250
# What a whole install keeps (llama.cpp and its build, the converter's environment, Gemma
# 4 E4B and bge-m3, Clarity, uv and Python), about 10 GB, plus 5 GB to spare: 15 GB (10^9
# bytes, as setup counts) in df -Pk's 1024-byte blocks.
NEED_KB=14648438
GETTING_STARTED=https://github.com/andalibmalit/clarity-chat/blob/main/docs/getting-started.md

say() {
    printf '%s\n' "$*"
}

refuse() {
    printf 'clarity install: %s\n' "$*" >&2
    exit 1
}

is_clarity() {
    [ -f "$1/defaults.toml" ] && [ -f "$1/server.toml" ] &&
        grep -q '^name = "clarity-chat"' "$1/pyproject.toml" 2>/dev/null
}

main() {
    # State 23: setup asks one question, so there must be a terminal to ask on. `true`, not
    # `:`: a redirection that fails on a special builtin like `:` ends a POSIX sh (dash).
    if ! true 2>/dev/null </dev/tty; then
        refuse "The install needs a terminal to ask you one question. Paste the command into Terminal."
    fi
    # State 1: the only platform with live tests.
    system=$(uname -s)
    machine=$(uname -m)
    if [ "$system" != Darwin ] || [ "$machine" != arm64 ]; then
        refuse "Clarity's install command is for a Mac with Apple silicon; this computer is $system on $machine. To install by hand, follow \"Install by hand\" in Getting started: $GETTING_STARTED"
    fi
    # State 2: the first model beside bge-m3 (the 16 GB floor).
    memory=$(sysctl -n hw.memsize)
    if [ "$memory" -lt 17179869184 ]; then
        refuse "Clarity needs 16 GB of memory; this Mac has $((memory / 1073741824)) GB."
    fi
    # State 7: room for the whole install, before anything is written.
    free_kb=$(df -Pk "$HOME" | awk 'NR == 2 { print $4 }')
    case "$free_kb" in  # `[` errors on a word, which counts as false: refuse it first
        '' | *[!0-9]*) refuse "The install couldn't tell how much disk your home folder has free (df said \"$free_kb\"), so it stopped." ;;
    esac
    if [ "$free_kb" -lt "$NEED_KB" ]; then
        refuse "The install needs $((NEED_KB * 1024 / 1000000000)) GB free in your home folder (with 5 GB to spare); it has $((free_kb * 1024 / 1000000000)) GB free."
    fi
    # State 3: Apple's command line tools bring git and the compiler llama.cpp's build needs.
    if ! xcode-select -p >/dev/null 2>&1; then
        xcode-select --install >/dev/null 2>&1 || true
        refuse "Apple's command line tools aren't installed. Their installer has opened: click Install, then paste the command again."
    fi
    # States 4 and 4b: uv, from its own installer at a pinned version, checked first. A run
    # again in the same window finds the uv it installed before the shell profile is read.
    uv=$(command -v uv || true)
    if [ -z "$uv" ] && [ -x "$HOME/.local/bin/uv" ]; then
        uv="$HOME/.local/bin/uv"
    fi
    if [ -z "$uv" ]; then
        # uv's installer checks its archive's sha256 with sha256sum, and skips the check if
        # it finds none; macOS keeps it in /sbin, which a PATH may leave out.
        installer_path="$PATH:/usr/sbin:/sbin"
        if ! env PATH="$installer_path" sh -c 'command -v sha256sum' >/dev/null 2>&1; then
            refuse "uv's installer can't check its download on this Mac: it needs sha256sum, which this Mac doesn't have. So the install stopped before fetching it. Install uv with Homebrew (brew install uv), which checks its download, then paste the command again."
        fi
        say "Installing uv $UV_VERSION, which runs Clarity (its installer checked by sha256) …"
        # mktemp makes it readable only by you (0600); it's removed on every way out.
        installer=$(mktemp "${TMPDIR:-/tmp}/clarity-uv-installer.XXXXXX")
        trap 'rm -f "$installer"' EXIT
        trap 'exit 130' HUP INT TERM
        curl --proto '=https' --tlsv1.2 -fsSL "https://astral.sh/uv/$UV_VERSION/install.sh" -o "$installer"
        if [ "$(shasum -a 256 "$installer" | cut -d ' ' -f 1)" != "$UV_INSTALLER_SHA256" ]; then
            refuse "uv's installer doesn't match its checked sha256, so the install stopped and ran nothing."
        fi
        env -i HOME="$HOME" PATH="$installer_path" sh "$installer"
        rm -f "$installer"
        trap - EXIT HUP INT TERM
        uv="$HOME/.local/bin/uv"
    fi
    # States 5 and 6: ~/Clarity, at this release's tag; a folder that isn't Clarity is left,
    # and one at another version is said, not updated. git gets none of the user's config,
    # and never prompts (a moved repository would ask for a password).
    clarity="$HOME/Clarity"
    if [ -e "$clarity" ]; then
        if ! is_clarity "$clarity"; then
            # shellcheck disable=SC2088  # the sentence names the folder as the user knows it
            refuse "~/Clarity exists and isn't Clarity, so the install leaves it alone. Move or rename it, then paste the command again."
        fi
        # Every tag at its commit, a line each: describe would name only one of several.
        tags=$(env -i HOME="$HOME" PATH="$PATH" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
            GIT_TERMINAL_PROMPT=0 git -C "$clarity" tag --points-at HEAD 2>/dev/null || true)
        if ! printf '%s\n' "$tags" | grep -qxF "$VERSION"; then
            # Tags have no spaces, so echo joins the lines with one.
            # shellcheck disable=SC2086
            at=$(echo $tags | sed 's/ / and /g')
            # shellcheck disable=SC2088  # as above
            say "~/Clarity holds Clarity at ${at:-a commit with no release tag}, not $VERSION, and the install doesn't update it."
        fi
        say "Clarity is already in ~/Clarity; checking its setup."
    else
        say "Putting Clarity $VERSION in ~/Clarity …"
        env -i HOME="$HOME" PATH="$PATH" GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
            GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$VERSION" "$REPO_URL" "$clarity"
    fi
    cd "$clarity"
    # --inexact: a hand install's extras (--extra hf, the browser group) stay installed.
    env -i HOME="$HOME" PATH="$PATH" "$uv" sync --locked --inexact --group setup
    env -i HOME="$HOME" PATH="$PATH" "$uv" run --group setup clarity setup </dev/tty
    env -i HOME="$HOME" PATH="$PATH" "$uv" run clarity ui </dev/tty
}

main "$@"
