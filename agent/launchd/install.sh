#!/bin/sh
# installs the agent daemon as a per-user launchd service that restarts on crash, on login and on
# reboot. idle sleep is held off while it runs (caffeinate -i); a closed lid still sleeps the mac.
#   sh agent/launchd/install.sh          install or reinstall
#   sh agent/launchd/uninstall.sh        remove
set -eu
LABEL=ai.qwenfast.agent
SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$SRC_DIR/.." && pwd)
NODE=$(command -v node)
HOME_DIR=${QFA_HOME:-$HOME/.qwenfast-agent}
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
mkdir -p "$HOME/Library/LaunchAgents" "$HOME_DIR/logs"
[ -d "$SRC_DIR/node_modules" ] || (cd "$SRC_DIR" && npm ci --no-audit --no-fund)

# macos privacy rules stop a launchd job from reading ~/Desktop, ~/Documents and ~/Downloads
# (the open() just hangs on a consent prompt nobody sees). so the service runs from a copy of the
# app and of the three secrets it needs, under $HOME_DIR. rerun this script after changing src/.
AGENT_DIR="$HOME_DIR/app"
mkdir -p "$AGENT_DIR"
rsync -a --delete --exclude test --exclude launchd "$SRC_DIR/" "$AGENT_DIR/"
mkdir -p "$HOME_DIR/secrets" && chmod 700 "$HOME_DIR/secrets"
for f in agent_key jarvislabs.env jarvislabs.backup.env agent_box_id; do
  if [ -f "$REPO/.secrets/$f" ] && [ ! -f "$HOME_DIR/secrets/$f" -o "$f" != agent_box_id ]; then
    cp "$REPO/.secrets/$f" "$HOME_DIR/secrets/$f"; chmod 600 "$HOME_DIR/secrets/$f"
  fi
done
cat > "$PLIST" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/usr/bin/caffeinate</string><string>-i</string>
    <string>$NODE</string><string>$AGENT_DIR/src/daemon.ts</string>
  </array>
  <key>WorkingDirectory</key><string>$AGENT_DIR</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key><string>$(dirname "$NODE"):$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    <key>HOME</key><string>$HOME</string>
    <key>QFA_HOME</key><string>$HOME_DIR</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$HOME_DIR/logs/launchd.out.log</string>
  <key>StandardErrorPath</key><string>$HOME_DIR/logs/launchd.err.log</string>
</dict>
</plist>
PL
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
# bootout returns before the job is gone; bootstrapping too early fails with "5: input/output error"
i=0
while launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1 && [ $i -lt 30 ]; do sleep 1; i=$((i + 1)); done
launchctl bootstrap "gui/$(id -u)" "$PLIST" || { sleep 3; launchctl bootstrap "gui/$(id -u)" "$PLIST"; }
launchctl enable "gui/$(id -u)/$LABEL"
echo "installed $LABEL ($PLIST)"
echo "health: curl -s http://127.0.0.1:7788/health"
