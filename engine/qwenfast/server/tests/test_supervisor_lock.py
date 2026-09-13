"""`scripts/supervisor_lock.sh` — the single-supervisor lock, driven as a subprocess.

The hazard this guards against: two copies of `remote_public_server.sh` running at once, each
one's health poll seeing the other's process failing to bind :8000, and each killing and
restarting its own child in a loop neither supervisor can see.

Tested through `sh` rather than by reading the file, because the thing being asserted *is* the
shell semantics (`mkdir` atomicity, `kill -0` liveness, POSIX's lack of local variables).

    PYTHONPATH=engine <venv>/bin/pytest engine/qwenfast/server/tests/test_supervisor_lock.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "supervisor_lock.sh"

pytestmark = pytest.mark.skipif(
    not SCRIPT.exists() or shutil.which("sh") is None,
    reason="POSIX sh and scripts/supervisor_lock.sh are required",
)

#: A pid that is essentially certain not to exist, for the stale-lock cases.
DEAD_PID = "4194303"


def sh(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["sh", str(SCRIPT), *args],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        timeout=30,
    )


def test_a_second_supervisor_is_refused_while_the_first_is_alive(tmp_path):
    lock = tmp_path / "supervisor.lock"
    # Our own pid is, by construction, a live process.
    first = sh("acquire", str(lock), str(os.getpid()))
    assert first.returncode == 0, first.stderr

    second = sh("acquire", str(lock), "12345")
    assert second.returncode == 3, second.stdout + second.stderr
    assert "held by" in second.stdout

    # The holder is unchanged — a refused starter must not steal the pidfile.
    status = sh("status", str(lock))
    assert status.returncode == 0
    assert status.stdout.strip() == str(os.getpid())


def test_a_stale_lock_from_a_crashed_supervisor_is_reclaimed(tmp_path):
    """A supervisor killed with SIGKILL leaves the lock behind; the server must not stay down."""
    lock = tmp_path / "supervisor.lock"
    assert sh("acquire", str(lock), DEAD_PID).returncode == 0
    assert sh("status", str(lock)).returncode == 1, "a dead holder is not 'held'"

    taken = sh("acquire", str(lock), str(os.getpid()))
    assert taken.returncode == 0
    assert "clearing stale lock" in taken.stderr
    assert sh("status", str(lock)).stdout.strip() == str(os.getpid())


def test_release_only_works_for_the_owner(tmp_path):
    lock = tmp_path / "supervisor.lock"
    sh("acquire", str(lock), str(os.getpid()))

    assert sh("release", str(lock), "999999").returncode == 1
    assert lock.exists(), "a non-owner must not be able to release the lock"

    assert sh("release", str(lock), str(os.getpid())).returncode == 0
    assert not lock.exists()


def test_lock_force_overrides_a_live_holder(tmp_path):
    lock = tmp_path / "supervisor.lock"
    sh("acquire", str(lock), str(os.getpid()))
    forced = sh("acquire", str(lock), "4242", env={"LOCK_FORCE": "1"})
    assert forced.returncode == 0
    assert "FORCING" in forced.stderr
    # `status` prints only for a *live* holder, and 4242 is not one, so read the
    # pidfile rather than asking whether the lock is currently held.
    assert (lock / "pid").read_text().strip() == "4242"


def test_the_lock_survives_being_acquired_and_released_repeatedly(tmp_path):
    """Regression: the pidfile must record the *acquirer*, not a previous holder.

    POSIX sh has no local variables, so an earlier version of this script had `lock_alive`
    clobber `lock_acquire`'s `_pid` — and every acquisition after the first wrote a dead pid
    into the file, leaving a lock that read as permanently stale and therefore never locked
    anything at all.
    """
    lock = tmp_path / "supervisor.lock"
    for pid in ("111", "222", "333"):
        # Each round starts from a stale lock, which is the path through `lock_alive`.
        assert sh("acquire", str(lock), pid).returncode == 0
        holder = sh("status", str(lock))
        # These pids are not alive, so `status` says free — read the pidfile directly.
        assert (lock / "pid").read_text().strip() == pid, holder.stdout


def test_only_one_of_many_racing_supervisors_wins(tmp_path):
    """`mkdir` is the primitive precisely so that a race has exactly one winner.

    The winner must *stay alive* while the others look, because that is what a supervisor
    does — it holds the lock for its whole life. An earlier version of this test let the
    winner exit immediately, which made the lock legitimately stale and every loser
    correctly reclaim it; the failure was in the test, but it did expose a real one in the
    script next door: a racer that finds a lock directory with no pidfile yet (the winner is
    between its `mkdir` and its write) must treat that as *held*, not stale. Without the
    wait in `lock_await_pidfile`, five of twelve starters here all believed they had won.
    """
    lock = tmp_path / "supervisor.lock"
    racer = textwrap.dedent(
        f"""
        . {SCRIPT}
        if lock_acquire {lock} "$$"; then echo WON; sleep 3; else echo LOST; fi
        """
    )
    procs = [
        subprocess.Popen(["sh", "-c", racer], stdout=subprocess.PIPE, text=True)
        for _ in range(12)
    ]
    outcomes = [p.communicate(timeout=60)[0].strip().splitlines()[-1] for p in procs]
    assert outcomes.count("WON") == 1, outcomes
    assert outcomes.count("LOST") == 11


def test_the_watchdog_sources_the_lock_and_refuses_to_start_twice(tmp_path):
    """The wiring, not just the library: `remote_public_server.sh` must exit 3 on a held lock.

    The script is not run for real here (it would load a 27B model); it is run with a
    `LOCK_LIB` and `LOG_DIR` pointed at a temporary directory and a lock already held, which
    exercises exactly the block that decides whether to continue.
    """
    watchdog = SCRIPT.parent / "remote_public_server.sh"
    if not watchdog.exists():  # pragma: no cover
        pytest.skip("watchdog script not present")

    log_dir = tmp_path / "public"
    log_dir.mkdir(parents=True)
    lock = log_dir / "supervisor.lock"
    assert sh("acquire", str(lock), str(os.getpid())).returncode == 0

    proc = subprocess.run(
        ["sh", str(watchdog)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "LOG_DIR": str(log_dir),
            "RESULTS_DIR": str(tmp_path / "results"),
            "SECRETS_DIR": str(tmp_path / "secrets"),
            "LOCK_LIB": str(SCRIPT),
            "LOCK_DIR": str(lock),
        },
    )
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "REFUSING TO START" in (proc.stdout + proc.stderr)
