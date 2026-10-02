#!/bin/sh
# the two background services of qwenfast code, kept alive by the os:
#
#   qwenfast-agent  the daemon: openai compatible gateway + router on 127.0.0.1:7788, gpu box
#                   keeper (resume, idle pause, ssh tunnel), background task runner
#   qwenfast-code   the qfc server on 127.0.0.1:4097: sessions, tools, agents; every tui attaches
#
# macos: the daemon is a launchd agent. the server runs under a small supervisor loop started from
# your terminal, because macos privacy rules stop a launchd job from opening projects under
# ~/Desktop, ~/Documents or ~/Downloads (the open() silently waits for a consent prompt nobody
# sees); started from a terminal it inherits that terminal's access. `qfc` restarts it after a
# reboot. set QFC_SERVER_LAUNCHD=1 to run it under launchd instead, after giving
# ~/.qwenfast-code/bin/qfc-bin full disk access.
# linux: systemd --user units, or, where there is no user systemd (a container, a minimal vm), the
# same supervisor loop for both.
#
#   sh services.sh install | uninstall | start | stop | restart | status
set -u
QFC_HOME=${QFC_HOME:-$HOME/.qwenfast-code}
LOGS="$QFC_HOME/logs"
mkdir -p "$LOGS"
NODE="$QFC_HOME/toolchain/node/bin/node"
AGENT_CMD="$NODE $QFC_HOME/daemon/src/daemon.ts"
SERVER_CMD="$QFC_HOME/bin/qfc-server"
MODE=fallback
SERVER_LAUNCHD=${QFC_SERVER_LAUNCHD:-0}
if [ "$(uname -s)" = Darwin ]; then MODE=launchd
elif command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then MODE=systemd
fi

# -- launchd ----------------------------------------------------------------------------------
plist() { # label program-and-args... ; idle sleep is held off while the daemon runs
  label=$1; shift
  f="$HOME/Library/LaunchAgents/$label.plist"
  mkdir -p "$HOME/Library/LaunchAgents"
  {
    echo '<?xml version="1.0" encoding="UTF-8"?>'
    echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">'
    echo '<plist version="1.0"><dict>'
    echo "  <key>Label</key><string>$label</string>"
    echo '  <key>ProgramArguments</key><array>'
    for a in "$@"; do echo "    <string>$a</string>"; done
    echo '  </array>'
    echo "  <key>EnvironmentVariables</key><dict>"
    echo "    <key>PATH</key><string>$QFC_HOME/toolchain/node/bin:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>"
    echo "    <key>HOME</key><string>$HOME</string>"
    echo "    <key>QFC_HOME</key><string>$QFC_HOME</string>"
    echo "    <key>QFA_HOME</key><string>$QFC_HOME/agent</string>"
    echo "  </dict>"
    echo "  <key>WorkingDirectory</key><string>$HOME</string>"
    echo '  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/><key>ThrottleInterval</key><integer>10</integer>'
    echo "  <key>StandardOutPath</key><string>$LOGS/$label.out.log</string>"
    echo "  <key>StandardErrorPath</key><string>$LOGS/$label.err.log</string>"
    echo '</dict></plist>'
  } > "$f"
}
lctl_load() {
  f="$HOME/Library/LaunchAgents/$1.plist"
  launchctl bootout "gui/$(id -u)/$1" 2>/dev/null || true
  i=0; while launchctl print "gui/$(id -u)/$1" >/dev/null 2>&1 && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done
  launchctl bootstrap "gui/$(id -u)" "$f" 2>/dev/null || { sleep 2; launchctl bootstrap "gui/$(id -u)" "$f"; }
}

# -- systemd --user ---------------------------------------------------------------------------
unit() { # name description command
  d="$HOME/.config/systemd/user"; mkdir -p "$d"
  cat > "$d/$1.service" <<EOF
[Unit]
Description=$2

[Service]
Environment=QFC_HOME=$QFC_HOME QFA_HOME=$QFC_HOME/agent PATH=$QFC_HOME/toolchain/node/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$3
Restart=always
RestartSec=5
StandardOutput=append:$LOGS/$1.out.log
StandardError=append:$LOGS/$1.err.log

[Install]
WantedBy=default.target
EOF
}

# -- fallback supervisor ----------------------------------------------------------------------
fb_start() { # name command
  pidf="$QFC_HOME/run/$1.pid"; mkdir -p "$QFC_HOME/run"
  if [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null; then return 0; fi
  SETSID=""; command -v setsid >/dev/null 2>&1 && SETSID=setsid
  QFA_HOME="$QFC_HOME/agent" nohup $SETSID sh -c "trap 'exit 0' TERM; while true; do $2 >> '$LOGS/$1.out.log' 2>> '$LOGS/$1.err.log'; sleep 5; done" \
    > /dev/null 2>&1 < /dev/null &
  echo $! > "$pidf"
}
fb_stop() {
  pidf="$QFC_HOME/run/$1.pid"
  [ -f "$pidf" ] || return 0
  p=$(cat "$pidf"); pkill -TERM -P "$p" 2>/dev/null; kill -TERM "$p" 2>/dev/null; rm -f "$pidf"
}

case "${1:-status}" in
  install)
    case $MODE in
      launchd)
        plist ai.qwenfast.agent /usr/bin/caffeinate -i $AGENT_CMD
        if [ "$SERVER_LAUNCHD" = 1 ]; then plist ai.qwenfast.code $SERVER_CMD; else rm -f "$HOME/Library/LaunchAgents/ai.qwenfast.code.plist"; fi ;;
      systemd)
        unit qwenfast-agent "qwenfast code daemon (gateway, router, gpu box keeper)" "$AGENT_CMD"
        unit qwenfast-code "qwenfast code server (sessions)" "$SERVER_CMD"
        systemctl --user daemon-reload
        systemctl --user enable qwenfast-agent qwenfast-code >/dev/null 2>&1
        command -v loginctl >/dev/null && loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -q yes || \
          echo "note: run 'loginctl enable-linger $(id -un)' so the services keep running after you log out" ;;
    esac
    echo "services installed ($MODE)" ;;
  uninstall)
    sh "$0" stop
    case $MODE in
      launchd) rm -f "$HOME/Library/LaunchAgents/ai.qwenfast.agent.plist" "$HOME/Library/LaunchAgents/ai.qwenfast.code.plist" ;;
      systemd) systemctl --user disable qwenfast-agent qwenfast-code >/dev/null 2>&1
               rm -f "$HOME/.config/systemd/user/qwenfast-agent.service" "$HOME/.config/systemd/user/qwenfast-code.service"
               systemctl --user daemon-reload ;;
    esac ;;
  start)
    case $MODE in
      launchd) launchctl print "gui/$(id -u)/ai.qwenfast.agent" >/dev/null 2>&1 || lctl_load ai.qwenfast.agent
               if [ -f "$HOME/Library/LaunchAgents/ai.qwenfast.code.plist" ]; then
                 launchctl print "gui/$(id -u)/ai.qwenfast.code" >/dev/null 2>&1 || lctl_load ai.qwenfast.code
               else fb_start qwenfast-code "$SERVER_CMD"; fi ;;
      systemd) systemctl --user start qwenfast-agent qwenfast-code ;;
      fallback) fb_start qwenfast-agent "$AGENT_CMD"; fb_start qwenfast-code "$SERVER_CMD" ;;
    esac ;;
  stop)
    case $MODE in
      launchd) for l in ai.qwenfast.agent ai.qwenfast.code; do launchctl bootout "gui/$(id -u)/$l" 2>/dev/null || true; done
               fb_stop qwenfast-code ;;
      systemd) systemctl --user stop qwenfast-agent qwenfast-code ;;
      fallback) fb_stop qwenfast-agent; fb_stop qwenfast-code ;;
    esac ;;
  restart) sh "$0" stop; sleep 2; sh "$0" start ;;
  status)
    echo "mode $MODE"
    case $MODE in
      launchd) printf 'ai.qwenfast.agent: '; launchctl print "gui/$(id -u)/ai.qwenfast.agent" 2>/dev/null | awk '/^\tstate =/{s=$3} /^\tpid =/{p=$3} END{print (s?s:"not loaded") (p?" pid "p:"")}'
               f="$QFC_HOME/run/qwenfast-code.pid"
               if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then echo "qwenfast-code: running (supervised)"; else echo "qwenfast-code: stopped"; fi ;;
      systemd) systemctl --user --no-pager status qwenfast-agent qwenfast-code | grep -E "●|Active:" ;;
      fallback) for n in qwenfast-agent qwenfast-code; do
                  f="$QFC_HOME/run/$n.pid"; if [ -f "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then echo "$n: running"; else echo "$n: stopped"; fi
                done ;;
    esac ;;
  *) echo "usage: services.sh install|uninstall|start|stop|restart|status" >&2; exit 2 ;;
esac
