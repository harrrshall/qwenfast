#!/bin/sh
# Public qwenfast endpoint with a supervising watchdog. Run it in the foreground of a
# long-lived session on the GPU host (that process is the supervisor; keep it alive).
#
#   sh /home/remote_public_server.sh
#
# What this adds over remote_demo_server.sh: the server is a *child*, not `exec`'d, so when it
# dies (CUDA OOM, a device-side assert, the engine thread going away) the loop restarts it with
# exponential backoff instead of leaving the endpoint dead until someone notices. /health is
# polled every 10 s; a server that answers 503 for HEALTH_FAIL_LIMIT consecutive polls is killed
# and restarted, because the worst failure is an engine that keeps answering HTTP after its
# device thread has died.
#
# Secrets are read from files (never from the command line, which shows up in `ps` and in logs):
#   /home/.secrets/api_keys.json   {"keys":[{"key":…,"name":…,"rpm":…,"tpm":…,"max_tokens":…}]}
#   /home/.secrets/admin_key       one line — reads /admin/usage
#   /home/.secrets/demo_key        one line — used only by the Vercel chat proxy
#
# Configuration is the best measured serving point: preset fastest, spec k=3 (≤16 seqs),
# mixed forward with the 8,192-token chunk (a 2048-token chunk measured slower; graphs on,
# the graphed mixed step is not the default), 128 seqs, ctx 4096.
#
# Reliability features of this loop:
#   1. a single-holder lock (scripts/supervisor_lock.sh) so a second supervisor
#      refuses to start instead of fighting the first for port 8000 in an
#      endless restart loop;
#   2. a graceful drain: SIGTERM, then up to DRAIN_TIMEOUT seconds for in-flight
#      streams to finish, and only then SIGKILL. The server itself stops
#      accepting and answers /health 503 the moment it gets the TERM;
#   3. `--min-completion-tokens` in place of a hard `--max-prompt-tokens`, so a
#      long conversation gets a shorter answer instead of a 400.

set -u

LOG_DIR=${LOG_DIR:-/home/qwenfast-results/public}
RESULTS_DIR=${RESULTS_DIR:-/home/qwenfast-results}
SECRETS_DIR=${SECRETS_DIR:-/home/.secrets}
PORT=${PORT:-8000}
HEALTH_INTERVAL=${HEALTH_INTERVAL:-10}
HEALTH_FAIL_LIMIT=${HEALTH_FAIL_LIMIT:-6}      # 6 x 10 s of unhealthy before we recycle
BACKOFF_MIN=${BACKOFF_MIN:-5}
BACKOFF_MAX=${BACKOFF_MAX:-300}
LOG_MAX_BYTES=${LOG_MAX_BYTES:-52428800}       # 50 MB per server log before rotation
LOG_KEEP=${LOG_KEEP:-5}
STARTUP_GRACE=${STARTUP_GRACE:-900}            # weight load + graph capture can take minutes
DRAIN_TIMEOUT=${DRAIN_TIMEOUT:-30}             # seconds in-flight streams get on a restart
LOCK_DIR=${LOCK_DIR:-$LOG_DIR/supervisor.lock}

HEARTBEAT="$LOG_DIR/heartbeat.json"
SERVER_LOG="$LOG_DIR/server.log"
WATCHDOG_LOG="$LOG_DIR/watchdog.log"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"

# -- single-supervisor lock ------------------------------------------------
# Sourced, so `lock_acquire`/`lock_release` are functions here. Held for the
# whole life of this script and released by `cleanup`.
LOCK_LIB=${LOCK_LIB:-/home/supervisor_lock.sh}
if [ -f "$LOCK_LIB" ]; then
  # shellcheck disable=SC1090
  . "$LOCK_LIB"
else
  echo "FATAL: $LOCK_LIB missing — upload scripts/supervisor_lock.sh first" >&2
  exit 2
fi

log_early() {
  printf '%s watchdog: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_DIR/watchdog.log"
}

if ! lock_acquire "$LOCK_DIR" "$$"; then
  log_early "REFUSING TO START: another supervisor (pid $(lock_holder "$LOCK_DIR")) already holds $LOCK_DIR."
  log_early "If that pid is gone this lock would have been cleared automatically; to override, LOCK_FORCE=1."
  exit 3
fi
log_early "acquired supervisor lock $LOCK_DIR (pid $$)"

log() {
  printf '%s watchdog: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$WATCHDOG_LOG"
}

rotate_log() {
  # Size-based rotation, kept in shell so there is no logrotate dependency on the host.
  [ -f "$SERVER_LOG" ] || return 0
  size=$(wc -c < "$SERVER_LOG" 2>/dev/null | tr -d ' ')
  [ -n "$size" ] || return 0
  [ "$size" -lt "$LOG_MAX_BYTES" ] && return 0
  i=$LOG_KEEP
  while [ "$i" -gt 1 ]; do
    prev=$((i - 1))
    [ -f "$SERVER_LOG.$prev" ] && mv "$SERVER_LOG.$prev" "$SERVER_LOG.$i"
    i=$prev
  done
  mv "$SERVER_LOG" "$SERVER_LOG.1"
  log "rotated $SERVER_LOG (was $size bytes)"
}

heartbeat() {
  # state, restart count, last health, pid — one JSON object, atomically replaced so a reader
  # never sees a half-written file.
  cat > "$HEARTBEAT.tmp" <<EOF
{"ts":"$(date -u +%Y-%m-%dT%H:%M:%SZ)","epoch":$(date +%s),"state":"$1","pid":${SERVER_PID:-0},
 "restarts":$RESTARTS,"consecutive_unhealthy":$UNHEALTHY,"backoff_s":$BACKOFF,"port":$PORT,
 "started_at":${STARTED_AT:-0},"detail":"${2:-}"}
EOF
  mv "$HEARTBEAT.tmp" "$HEARTBEAT"
}

. /home/venv_vllm/bin/activate
export PYTHONPATH=/home/engine HF_HOME=/home/hf HF_HUB_OFFLINE=1 VLLM_USE_DEEP_GEMM=0
export CUDA_HOME=/home/venv_vllm/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
FP8=$(ls -d /home/hf/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/*/ | head -1)
cd /home || exit 1

KEYS_FILE="$SECRETS_DIR/api_keys.json"
ADMIN_FILE="$SECRETS_DIR/admin_key"
DEMO_FILE="$SECRETS_DIR/demo_key"

AUTH_ARGS=""
if [ -f "$KEYS_FILE" ]; then
  AUTH_ARGS="$AUTH_ARGS --api-keys-file $KEYS_FILE"
  log "keys file present: $KEYS_FILE ($(python3 -c 'import json,sys;print(len(json.load(open(sys.argv[1]))["keys"]))' "$KEYS_FILE" 2>/dev/null || echo '?') keys)"
else
  log "WARNING: $KEYS_FILE missing — the endpoint will be UNAUTHENTICATED"
fi
[ -f "$ADMIN_FILE" ] && AUTH_ARGS="$AUTH_ARGS --admin-key-file $ADMIN_FILE" || log "WARNING: no admin key; /admin/usage will be unreachable"
[ -f "$DEMO_FILE" ] && AUTH_ARGS="$AUTH_ARGS --demo-key-file $DEMO_FILE" || log "note: no demo key; the Vercel chat proxy will need a public key"

# -- serving configuration -------------------------------------------------
# Overridable from the environment so the coordinator can change capacity
# without editing the script (and so a bad value is one restart, not a commit).
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
#: Replaces --max-prompt-tokens. The server now CLAMPS max_tokens into the room
#: the prompt leaves and only refuses when less than this is left, so "reserve
#: 256 tokens for a reply" is the whole of the old 3072-token prompt cap.
MIN_COMPLETION_TOKENS=${MIN_COMPLETION_TOKENS:-256}
MAX_STREAMS_PER_KEY=${MAX_STREAMS_PER_KEY:-8}
MAX_STREAMS_PER_IP=${MAX_STREAMS_PER_IP:-8}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-300}

# Content logging. Private to this host; off unless LOG_CONTENT=1.
CONTENT_ARGS=""
if [ "${LOG_CONTENT:-0}" = "1" ]; then
  CONTENT_ARGS="--log-content --content-log-db $RESULTS_DIR/queries.sqlite"
  [ -n "${CONTENT_RETENTION_DAYS:-}" ] && \
    CONTENT_ARGS="$CONTENT_ARGS --content-retention-days $CONTENT_RETENTION_DAYS"
fi

RESTARTS=0
UNHEALTHY=0
BACKOFF=$BACKOFF_MIN
SERVER_PID=0
STARTED_AT=0

# Stop the server child politely, then firmly. The server turns SIGTERM into
# "refuse new requests, finish the streams already running, exit" — so the wait
# below is the drain, and SIGKILL is only ever reached by a process that ignored
# it. Never touches a pid that is not our own child.
stop_server() {
  [ "${SERVER_PID:-0}" -gt 0 ] || return 0
  kill -0 "$SERVER_PID" 2>/dev/null || return 0
  log "draining server pid $SERVER_PID (up to ${DRAIN_TIMEOUT}s for in-flight streams)"
  heartbeat draining "SIGTERM sent"
  kill -TERM "$SERVER_PID" 2>/dev/null
  _waited=0
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    [ "$_waited" -ge "$DRAIN_TIMEOUT" ] && break
    sleep 1
    _waited=$((_waited + 1))
  done
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    log "drain timeout after ${_waited}s; SIGKILL pid $SERVER_PID"
    kill -9 "$SERVER_PID" 2>/dev/null
  else
    log "server drained cleanly in ${_waited}s"
  fi
}

cleanup() {
  log "shutting down (signal); stopping server pid ${SERVER_PID:-0}"
  stop_server
  heartbeat stopped "supervisor exited"
  lock_release "$LOCK_DIR" "$$"
  exit 0
}
trap cleanup INT TERM

log "starting supervisor; model=$FP8 port=$PORT log=$SERVER_LOG"

while true; do
  rotate_log
  sh /home/remote_gpu_free.sh >> "$WATCHDOG_LOG" 2>&1

  echo "===== $(date -u) public server starting (restart #$RESTARTS)" >> "$SERVER_LOG"
  # shellcheck disable=SC2086  # AUTH_ARGS is intentionally word-split
  python -m qwenfast.runtime.serve \
    --model "$FP8" --served-model-name qwen3.8-27b --preset fastest \
    --spec-k 3 --spec-max-batch 16 --mixed-forward --mixed-graphs --overlap --overlap-min-fill 0.75 --prefill-chunk-tokens 8192 --detok-workers 0 \
    --max-num-seqs "$MAX_NUM_SEQS" --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 --port "$PORT" --http-keep-alive-timeout 300 \
    --default-max-tokens 1024 --max-output-tokens 4096 \
    --min-completion-tokens "$MIN_COMPLETION_TOKENS" \
    --max-inflight-requests 256 --max-request-bytes 1000000 --max-messages 128 \
    --max-streams-per-key "$MAX_STREAMS_PER_KEY" --max-streams-per-ip "$MAX_STREAMS_PER_IP" \
    --request-timeout "$REQUEST_TIMEOUT" --drain-timeout "$DRAIN_TIMEOUT" \
    --usage-db "$RESULTS_DIR/usage.sqlite" --gpu-rate-inr-per-hour 188.73 \
    $CONTENT_ARGS \
    $AUTH_ARGS ${PUBLIC_EXTRA:-} >> "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  STARTED_AT=$(date +%s)
  UNHEALTHY=0
  WARMED=0; log "server pid $SERVER_PID started"
  heartbeat starting ""

  # -- supervise ------------------------------------------------------------
  while kill -0 "$SERVER_PID" 2>/dev/null; do
    sleep "$HEALTH_INTERVAL"
    now=$(date +%s)
    age=$((now - STARTED_AT))
    code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "http://127.0.0.1:$PORT/health" 2>/dev/null)
    if [ "$code" = "200" ]; then
      if [ "$UNHEALTHY" -gt 0 ]; then log "healthy again after $UNHEALTHY bad polls"; fi
      if [ "${WARMED:-0}" = "0" ]; then
        WARMED=1; log "warm-up: 3 greedy requests (JIT/graph warm) in background"
        ( DK=$(cat /home/.secrets/demo_key 2>/dev/null); for L in 40 200 600; do
            curl -s -o /dev/null --max-time 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
              -H "Authorization: Bearer $DK" -H "Content-Type: application/json" \
              -d "{\"model\":\"qwen3.8-27b\",\"messages\":[{\"role\":\"user\",\"content\":\"Write about $L words on the sea.\"}],\"temperature\":0,\"max_tokens\":$L}"
          done; echo "$(date -u +%FT%TZ) watchdog: warm-up done" >> "$WATCHDOG_LOG" ) &
      fi
      UNHEALTHY=0
      BACKOFF=$BACKOFF_MIN   # a run that reaches healthy resets the backoff ladder
      heartbeat healthy ""
      rotate_log
      continue
    fi
    if [ "$age" -lt "$STARTUP_GRACE" ]; then
      heartbeat loading "http $code at ${age}s (grace ${STARTUP_GRACE}s)"
      continue
    fi
    UNHEALTHY=$((UNHEALTHY + 1))
    log "unhealthy poll $UNHEALTHY/$HEALTH_FAIL_LIMIT (http $code)"
    heartbeat unhealthy "http $code"
    if [ "$UNHEALTHY" -ge "$HEALTH_FAIL_LIMIT" ]; then
      log "health limit reached; recycling pid $SERVER_PID"
      stop_server
      break
    fi
  done

  wait "$SERVER_PID" 2>/dev/null
  RC=$?
  RESTARTS=$((RESTARTS + 1))
  log "server exited rc=$RC after $(( $(date +%s) - STARTED_AT ))s; restart #$RESTARTS in ${BACKOFF}s"
  heartbeat restarting "rc=$RC"
  sleep "$BACKOFF"
  BACKOFF=$((BACKOFF * 2))
  [ "$BACKOFF" -gt "$BACKOFF_MAX" ] && BACKOFF=$BACKOFF_MAX
done
