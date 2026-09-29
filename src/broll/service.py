"""Keeping `broll serve` running on a Mac, as a launchd agent.

Editing Joe's agents reach the library at http://127.0.0.1:8000 whenever they
cut a video, so the server has to be up without anyone double-clicking
anything. A per-user launchd agent starts it at login and restarts it if it
dies.

Two details that decide whether it works at all:

* launchd starts agents with a bare PATH (/usr/bin:/bin:/usr/sbin:/sbin), which
  has no ffmpeg on a Homebrew Mac, so the agent's PATH is written out
  explicitly with ffmpeg's folder in it.
* keys are read from ``.env`` in the working directory and in BROLL_HOME, so
  the agent runs from the checkout it was installed from - the same folder
  the double-click launcher uses - and finds the same keys.

Installing never stops a server someone else started: if the port is already
taken, the agent is written but not started, and takes over at the next login.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import BROLL_HOME_ENV

LABEL = "com.editingjoe.broll"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_INBOX = "~/Broll Inbox"
BASE_PATH = ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]

Runner = Callable[[list[str]], subprocess.CompletedProcess]


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True)


def agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path() -> Path:
    return agents_dir() / f"{LABEL}.plist"


def log_path() -> Path:
    return Path.home() / "Library" / "Logs" / "editing-joe" / "broll.log"


def _domain() -> str:
    return f"gui/{os.getuid()}"


def program_arguments(host: str, port: int) -> list[str]:
    """The `broll` next to this interpreter, so the agent runs this install."""
    script = Path(sys.executable).with_name("broll")
    head = [str(script)] if script.exists() else [sys.executable, "-m", "broll.cli"]
    return head + ["serve", "--host", host, "--port", str(port)]


def working_directory() -> Path:
    """The checkout this package runs from (for its .env), else BROLL_HOME."""
    from .config import broll_home

    repo = Path(__file__).resolve().parents[2]
    return repo if (repo / "pyproject.toml").exists() else broll_home()


def service_path() -> str:
    """PATH for the agent: ffmpeg's folder first, then the system's."""
    parts: list[str] = []
    for tool in ("ffmpeg", "ffprobe"):
        found = shutil.which(tool)
        if found and str(Path(found).parent) not in parts:
            parts.append(str(Path(found).parent))
    for extra in ("/opt/homebrew/bin", "/usr/local/bin", *BASE_PATH):
        if extra not in parts:
            parts.append(extra)
    return ":".join(parts)


def build_plist(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    inbox: str = DEFAULT_INBOX,
) -> dict:
    env = {
        "PATH": service_path(),
        "BROLL_INBOX": str(Path(inbox).expanduser()),
        # Unbuffered, so the log shows what happened when it happened.
        "PYTHONUNBUFFERED": "1",
    }
    if os.environ.get(BROLL_HOME_ENV):
        env[BROLL_HOME_ENV] = str(Path(os.environ[BROLL_HOME_ENV]).expanduser())
    log = str(log_path())
    return {
        "Label": LABEL,
        "ProgramArguments": program_arguments(host, port),
        "WorkingDirectory": str(working_directory()),
        "EnvironmentVariables": env,
        "RunAtLoad": True,
        "KeepAlive": True,
        # If it cannot start (a port clash, a broken install), try again every
        # half minute rather than launchd's default ten seconds of log spam.
        "ThrottleInterval": 30,
        "StandardOutPath": log,
        "StandardErrorPath": log,
    }


def port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def is_loaded(runner: Runner = _run) -> bool:
    return runner(["launchctl", "print", f"{_domain()}/{LABEL}"]).returncode == 0


@dataclass
class ServiceReport:
    installed: bool = False
    loaded: bool = False
    started: bool = False
    port_busy: bool = False
    answering: bool | None = None
    pid: int | None = None
    plist: str = ""
    log: str = ""
    messages: list[str] = field(default_factory=list)


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise RuntimeError(
            "`broll service` installs a launchd agent, which only exists on macOS. "
            "On a Linux server use systemd - see docs/DEPLOY.md."
        )


def install(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    inbox: str = DEFAULT_INBOX,
    start: bool = True,
    runner: Runner = _run,
) -> ServiceReport:
    _require_macos()
    report = ServiceReport(plist=str(plist_path()), log=str(log_path()))
    log_path().parent.mkdir(parents=True, exist_ok=True)
    Path(inbox).expanduser().mkdir(parents=True, exist_ok=True)
    agents_dir().mkdir(parents=True, exist_ok=True)

    if is_loaded(runner):
        # Our own agent: stop it so the new definition takes effect, and give
        # the old process a moment to let go of the port.
        runner(["launchctl", "bootout", f"{_domain()}/{LABEL}"])
        for _ in range(20):
            if not port_in_use(host, port):
                break
            time.sleep(0.25)

    with plist_path().open("wb") as handle:
        plistlib.dump(build_plist(host, port, inbox), handle)
    report.installed = True

    if not start:
        report.messages.append("Written but not started (--no-start). It starts at the next login.")
        return report
    if port_in_use(host, port):
        report.port_busy = True
        report.messages.append(
            f"Something is already serving on {host}:{port}, so it was left running and the "
            "agent was not started. The agent takes over at the next login, or run "
            "`broll service install` again once that server has stopped."
        )
        return report
    done = runner(["launchctl", "bootstrap", _domain(), str(plist_path())])
    if done.returncode != 0:
        report.messages.append(f"launchctl bootstrap failed: {(done.stderr or done.stdout).strip()}")
        return report
    report.loaded = report.started = True
    return report


def uninstall(runner: Runner = _run) -> ServiceReport:
    _require_macos()
    report = ServiceReport(plist=str(plist_path()), log=str(log_path()))
    if is_loaded(runner):
        runner(["launchctl", "bootout", f"{_domain()}/{LABEL}"])
        report.messages.append("Stopped.")
    if plist_path().exists():
        plist_path().unlink()
        report.messages.append(f"Removed {plist_path()}.")
    if not report.messages:
        report.messages.append("Not installed.")
    return report


def status(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, runner: Runner = _run) -> ServiceReport:
    _require_macos()
    report = ServiceReport(plist=str(plist_path()), log=str(log_path()))
    report.installed = plist_path().exists()
    printed = runner(["launchctl", "print", f"{_domain()}/{LABEL}"])
    report.loaded = printed.returncode == 0
    for line in (printed.stdout or "").splitlines():
        line = line.strip()
        if line.startswith("pid = ") and line[6:].strip().isdigit():
            report.pid = int(line[6:].strip())
    report.port_busy = port_in_use(host, port)
    report.answering = False
    if report.port_busy:
        answer = _answers(host, port)
        report.answering = answer is not None
        if answer == "old":
            report.messages.append(
                "The server on the port is the library but predates /api/health - "
                "restart it so it runs the current code."
            )
    return report


def _answers(host: str, port: int) -> str | None:
    """"current" if the library answers /api/health, "old" if it is the library
    without it (started before an update), None if it is something else."""
    import json
    import urllib.error
    import urllib.request

    base = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(f"{base}/api/health", timeout=5) as response:
            if json.load(response).get("service") == "broll-librarian":
                return "current"
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            try:
                with urllib.request.urlopen(f"{base}/api/clients", timeout=5) as response:
                    if "clients" in json.load(response):
                        return "old"
            except Exception:  # noqa: BLE001
                return None
    except Exception:  # noqa: BLE001 - anything but a good answer means "not the library"
        return None
    return None
