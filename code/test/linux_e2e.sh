#!/bin/sh
# runs inside the container: install from scratch, then a real task through qfc against the gpu box.
# secrets arrive via a read-only mount at /secrets (agent_key, agent_box_id, jarvislabs.backup.env)
# and the ssh key the tunnel uses at /home/ubuntu/.ssh.
set -eu
mkdir -p .secrets && cp /secrets/* .secrets/ && chmod 600 .secrets/*
sh code/install.sh
qfc status
i=0
until qfc status | grep -q "big .* up via tunnel" && qfc status | grep -q "small .* up via tunnel"; do
  i=$((i + 1)); [ $i -gt 90 ] && { echo "models never came up"; qfc status; exit 1; }; sleep 10
done
qfc status
mkdir -p /tmp/proj && cd /tmp/proj && git init -q . && git config user.email t@t && git config user.name t
printf 'def mean(xs):\n    return sum(xs) / len(xs)\n' > stats.py
git add -A && git commit -qm init
t0=$(date +%s)
qfc run "Add median(xs) to stats.py, raise ValueError on empty input for mean and median, and write unittest tests in test_stats.py. Make sure the tests pass." </dev/null | tail -8
echo "qfc run took $(( $(date +%s) - t0 ))s"
python3 -m unittest -v test_stats 2>&1 | tail -3
qfc session list | head -4
grep -E "gateway .* -> |tunnel" "$HOME/.qwenfast-code/agent/logs/daemon.log" | tail -4
