#!/bin/sh
# stops and removes the agent daemon's launchd service. tasks on disk are kept.
LABEL=ai.qwenfast.agent
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && echo "stopped $LABEL" || echo "$LABEL was not loaded"
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
