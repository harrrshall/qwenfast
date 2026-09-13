#!/bin/sh
# A single-holder lock for the public-server supervisor, in POSIX sh.
#
# Why this exists: if two copies of `remote_public_server.sh` run at once, both
# supervise "the server", both bind :8000, and each one's health poll sees the
# *other's* process failing to bind, so each kills its own child and restarts
# it in a loop neither can see. The watchdog's whole job is to make a dead
# server come back; two watchdogs make a live server die.
#
# `mkdir` is the primitive, not a plain pidfile: creating a directory is atomic
# on every POSIX filesystem, so exactly one of N racing supervisors can win, and
# `test -f && echo $$ >` — the obvious version — has a window between the test
# and the write in which both win.
#
# A crashed holder must not lock the host out forever, so the winner writes its
# pid inside and a later arrival treats the lock as stale when that pid is gone.
#
# Usage — as a library:
#   . /home/supervisor_lock.sh
#   lock_acquire /home/qwenfast-results/public/supervisor.lock || exit 3
#   trap 'lock_release /home/qwenfast-results/public/supervisor.lock' INT TERM EXIT
#
# Usage — as a command (this is what the tests drive):
#   sh supervisor_lock.sh acquire <lockdir> [pid]   # 0 = acquired, 3 = held
#   sh supervisor_lock.sh release <lockdir>
#   sh supervisor_lock.sh status  <lockdir>         # 0 = held (prints pid), 1 = free
#
# `LOCK_FORCE=1` steals a lock held by a live process. It exists for the one
# case the stale check cannot cover — a supervisor wedged rather than dead — and
# is deliberately not the default, because "just force it" is how two of these
# end up running again.

lock_pidfile() { printf '%s/pid' "$1"; }

# NOTE: POSIX sh has no local variables, so every helper below uses a distinct
# prefix. The obvious `_pid` everywhere is a real bug: `lock_acquire` would call
# `lock_alive`, which overwrites `_pid` with the *previous* holder's, and then
# write that back into the pidfile — leaving a lock permanently attributed to a
# process that no longer exists, i.e. permanently stale.

# 0 if the lock is held by a process that still exists.
lock_alive() {
  _la_lp=$(lock_pidfile "$1")
  [ -f "$_la_lp" ] || return 1
  _la_pid=$(tr -dc '0-9' < "$_la_lp" 2>/dev/null)
  [ -n "$_la_pid" ] || return 1
  kill -0 "$_la_pid" 2>/dev/null || return 1
  return 0
}

lock_holder() {
  _lh_lp=$(lock_pidfile "$1")
  [ -f "$_lh_lp" ] && tr -dc '0-9' < "$_lh_lp" 2>/dev/null
}

# How long to wait for a just-created lock directory to grow its pidfile before
# concluding the creator died between `mkdir` and the write. `mkdir` and the
# `printf` that follows it are two syscalls, not one, and every racing starter
# looks at the lock in that window.
LOCK_PID_WAIT_TRIES=${LOCK_PID_WAIT_TRIES:-40}   # x 0.05s = 2s

# Wait for $1/pid to appear. 0 if it did, 1 if it never showed up.
lock_await_pidfile() {
  _lap_n=0
  while [ "$_lap_n" -lt "$LOCK_PID_WAIT_TRIES" ]; do
    [ -s "$(lock_pidfile "$1")" ] && return 0
    _lap_n=$((_lap_n + 1))
    sleep 0.05 2>/dev/null || sleep 1
  done
  [ -s "$(lock_pidfile "$1")" ]
}

# Take a lock we have decided is stale. Returns 0 only if we can prove we own
# the result: two supervisors can reach this at the same moment, and `mkdir`
# alone does not settle it because both may have removed and recreated the
# directory. Reading our own pid back out does settle it.
lock_steal() {
  _lst_dir=$1
  _lst_pid=$2
  rm -rf "$_lst_dir" 2>/dev/null
  mkdir "$_lst_dir" 2>/dev/null || return 3
  printf '%s\n' "$_lst_pid" > "$(lock_pidfile "$_lst_dir")" 2>/dev/null || return 3
  # Settle the race: if someone else stole it back in between, their pid is here.
  [ "$(lock_holder "$_lst_dir")" = "$_lst_pid" ] || return 3
  return 0
}

# lock_acquire <lockdir> [pid] -> 0 acquired, 3 held by a live process
lock_acquire() {
  _lka_dir=$1
  _lka_pid=${2:-$$}
  _lka_parent=$(dirname "$_lka_dir")
  [ -d "$_lka_parent" ] || mkdir -p "$_lka_parent" 2>/dev/null

  if mkdir "$_lka_dir" 2>/dev/null; then
    printf '%s\n' "$_lka_pid" > "$(lock_pidfile "$_lka_dir")"
    return 0
  fi

  # The directory exists. It is NOT necessarily stale: the winner of a race
  # creates the directory first and writes its pid a moment later, and treating
  # that gap as "stale" is how twelve racing supervisors all decide they won.
  if ! lock_await_pidfile "$_lka_dir"; then
    printf 'supervisor_lock: lock dir has no pidfile after %ss; treating as stale\n' \
      "$((LOCK_PID_WAIT_TRIES / 20))" >&2
    lock_steal "$_lka_dir" "$_lka_pid"
    return $?
  fi

  if lock_alive "$_lka_dir"; then
    if [ "${LOCK_FORCE:-0}" = "1" ]; then
      printf 'supervisor_lock: FORCING lock held by live pid %s\n' \
        "$(lock_holder "$_lka_dir")" >&2
      printf '%s\n' "$_lka_pid" > "$(lock_pidfile "$_lka_dir")"
      return 0
    fi
    return 3
  fi

  printf 'supervisor_lock: clearing stale lock (pid %s is gone)\n' \
    "$(lock_holder "$_lka_dir")" >&2
  lock_steal "$_lka_dir" "$_lka_pid"
  return $?
}

# Only ever releases a lock this process owns, so a supervisor that lost a race
# and exited cannot delete the winner's lock on its way out.
lock_release() {
  _lkr_dir=$1
  _lkr_pid=${2:-$$}
  _lkr_held=$(lock_holder "$_lkr_dir")
  if [ -z "$_lkr_held" ] || [ "$_lkr_held" = "$_lkr_pid" ] || [ "${LOCK_FORCE:-0}" = "1" ]; then
    rm -rf "$_lkr_dir" 2>/dev/null
    return 0
  fi
  return 1
}

# Command-line form, active only when this file is *run* rather than sourced.
case "${1:-}" in
  acquire)
    lock_acquire "$2" "${3:-$$}"
    _rc=$?
    [ "$_rc" -eq 0 ] && printf 'acquired %s by %s\n' "$2" "${3:-$$}" || printf 'held by %s\n' "$(lock_holder "$2")"
    exit $_rc
    ;;
  release)
    lock_release "$2" "${3:-$(lock_holder "$2")}"
    exit $?
    ;;
  status)
    if lock_alive "$2"; then printf '%s\n' "$(lock_holder "$2")"; exit 0; fi
    exit 1
    ;;
esac
