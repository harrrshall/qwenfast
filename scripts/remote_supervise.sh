#!/bin/sh
# Generic supervisor for one long running model server on the gpu box.
#
#   NAME=big PORT=8000 sh /home/remote_supervise.sh <command> [args...]
#
# the command is started as a child. the loop polls http://127.0.0.1:$PORT$HEALTH_PATH every
# HEALTH_INTERVAL seconds; a server that dies, or stays unhealthy for HEALTH_FAIL_LIMIT polls after
# its STARTUP_GRACE, is drained (SIGTERM, DRAIN_TIMEOUT) and restarted with exponential backoff.
# one supervisor per NAME (scripts/supervisor_lock.sh), a size rotated log, and an atomically
# replaced heartbeat json the local agent daemon reads to tell "loading" from "dead".
#
# unlike remote_public_server.sh this never frees the whole gpu between restarts: two servers share
# the device here, so a restart only ever kills its own process group.

set -u

NAME=${NAME:?NAME is required}
PORT=${PORT:?PORT is required}
LOG_DIR=${LOG_DIR:?LOG_DIR is required}
HEALTH_PATH=${HEALTH_PATH:-/health}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-10}
HEALTH_FAIL_LIMIT=${HEALTH_FAIL_LIMIT:-6}
STARTUP_GRACE=${STARTUP_GRACE:-1500}
DRAIN_TIMEOUT=${DRAIN_TIMEOUT:-30}
BACKOFF_MIN=${BACKOFF_MIN:-5}
BACKOFF_MAX=${BACKOFF_MAX:-300}
LOG_MAX_BYTES=${LOG_MAX_BYTES:-52428800}
LOG_KEEP=${LOG_KEEP:-5}
WARMUP_CMD=${WARMUP_CMD:-}
LOCK_DIR=${LOCK_DIR:-$LOG_DIR/supervisor.lock}
LOCK_LIB=${LOCK_LIB:-$(cd "$(dirname "$0")" && pwd)/supervisor_lock.sh}

HEARTBEAT="$LOG_DIR/heartbeat.json"
SERVER_LOG="$LOG_DIR/server.log"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"
mkdir -p "$LOG_DIR"

[ "$#" -gt 0 ] || { echo "usage: NAME=.. PORT=.. $0 <command> [args...]" >&2; exit 2; }
[ -f "$LOCK_LIB" ] || { echo "FATAL: $LOCK_LIB missing" >&2; exit 2; }
# shellcheck disable=SC1090
. "$LOCK_LIB"

log() { printf '%s %s: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$NAME" "$*" >> "$WATCHDOG_LOG"; }

if ! lock_acquire "$LOCK_DIR" "$$"; then
  log "already supervised by pid $(lock_holder "$LOCK_DIR"); exiting"
  exit 0
fi
log "supervisor pid $$ holds $LOCK_DIR; command: $*"

RESTARTS=0; UNHEALTHY=0; BACKOFF=$BACKOFF_MIN; SERVER_PID=0; STARTED_AT=0

heartbeat() {
  cat > "$HEARTBEAT.tmp" <<EOF
{"name":"$NAME","ts":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","epoch":$(date +%s),"state":"$1","pid":${SERVER_PID:-0},"restarts":$RESTARTS,"consecutive_unhealthy":$UNHEALTHY,"backoff_s":$BACKOFF,"port":$PORT,"started_at":${STARTED_AT:-0},"detail":"${2:-}"}
EOF
  mv "$HEARTBEAT.tmp" "$HEARTBEAT"
}

rotate_log() {
  [ -f "$SERVER_LOG" ] || return 0
  size=$(wc -c < "$SERVER_LOG" 2>/dev/null | tr -d ' ')
  [ -n "$size" ] && [ "$size" -ge "$LOG_MAX_BYTES" ] || return 0
  i=$LOG_KEEP
  while [ "$i" -gt 1 ]; do
    prev=$((i - 1)); [ -f "$SERVER_LOG.$prev" ] && mv "$SERVER_LOG.$prev" "$SERVER_LOG.$i"; i=$prev
  done
  mv "$SERVER_LOG" "$SERVER_LOG.1"
}

stop_server() {
  [ "${SERVER_PID:-0}" -gt 0 ] || return 0
  kill -0 "$SERVER_PID" 2>/dev/null || return 0
  heartbeat draining "SIGTERM"
  # the child runs in its own process group (setsid), so vllm's engine core workers go too
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null
  w=0
  while kill -0 "$SERVER_PID" 2>/dev/null && [ "$w" -lt "$DRAIN_TIMEOUT" ]; do sleep 1; w=$((w + 1)); done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    log "drain timeout; SIGKILL group $SERVER_PID"
    kill -9 -- "-$SERVER_PID" 2>/dev/null || kill -9 "$SERVER_PID" 2>/dev/null
  fi
  # stragglers from the group (vllm spawns an engine core process)
  sleep 2
  kill -9 -- "-$SERVER_PID" 2>/dev/null
}

cleanup() { log "supervisor stopping"; stop_server; heartbeat stopped "supervisor exited"; lock_release "$LOCK_DIR" "$$"; exit 0; }
trap cleanup INT TERM HUP

while true; do
  rotate_log
  echo "===== $(date -u) $NAME starting (restart #$RESTARTS)" >> "$SERVER_LOG"
  setsid "$@" >> "$SERVER_LOG" 2>&1 < /dev/null &
  SERVER_PID=$!
  STARTED_AT=$(date +%s); UNHEALTHY=0; WARMED=0
  log "server pid $SERVER_PID started"
  heartbeat starting ""

  while kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep "$HEALTH_INTERVAL" & wait $!
    age=$(( $(date +%s) - STARTED_AT ))
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT$HEALTH_PATH" 2>/dev/null)
    if [ "$code" = "200" ]; then
      [ "$UNHEALTHY" -gt 0 ] && log "healthy again after $UNHEALTHY bad polls"
      if [ "$WARMED" = "0" ]; then
        WARMED=1; log "healthy after ${age}s"
        [ -n "$WARMUP_CMD" ] && ( sh -c "$WARMUP_CMD" >> "$WATCHDOG_LOG" 2>&1; log "warm-up done" ) &
      fi
      UNHEALTHY=0; BACKOFF=$BACKOFF_MIN
      heartbeat healthy ""
      rotate_log
      continue
    fi
    if [ "$age" -lt "$STARTUP_GRACE" ] && [ "$WARMED" = "0" ]; then
      heartbeat loading "http $code at ${age}s"
      continue
    fi
    UNHEALTHY=$((UNHEALTHY + 1))
    log "unhealthy poll $UNHEALTHY/$HEALTH_FAIL_LIMIT (http $code)"
    heartbeat unhealthy "http $code"
    if [ "$UNHEALTHY" -ge "$HEALTH_FAIL_LIMIT" ]; then
      log "recycling pid $SERVER_PID"; stop_server; break
    fi
  done

  wait "$SERVER_PID" 2>/dev/null; RC=$?
  stop_server
  RESTARTS=$((RESTARTS + 1))
  log "server exited rc=$RC after $(( $(date +%s) - STARTED_AT ))s; restart #$RESTARTS in ${BACKOFF}s"
  heartbeat restarting "rc=$RC"
  sleep "$BACKOFF" & wait $!
  BACKOFF=$((BACKOFF * 2)); [ "$BACKOFF" -gt "$BACKOFF_MAX" ] && BACKOFF=$BACKOFF_MAX
done
