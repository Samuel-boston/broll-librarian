"""The launchd agent that keeps the library up for Editing Joe's agents.

Nothing here touches the real launchd: launchctl is replaced by a recorder, and
the agent and log folders point into a temporary directory.
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
from pathlib import Path

import pytest

from broll import service


class Launchctl:
    """Records every launchctl call; `loaded` says what `launchctl print` answers."""

    def __init__(self, loaded: bool = False, pid: int | None = None):
        self.loaded = loaded
        self.pid = pid
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str]) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        verb = cmd[1]
        if verb == "print":
            out = f"state = running\n\tpid = {self.pid}\n" if self.loaded and self.pid else ""
            return subprocess.CompletedProcess(cmd, 0 if self.loaded else 113, out, "")
        if verb == "bootstrap":
            self.loaded = True
        if verb == "bootout":
            self.loaded = False
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def verbs(self) -> list[str]:
        return [c[1] for c in self.calls]


@pytest.fixture()
def mac(tmp_path, monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "darwin")
    monkeypatch.setattr(service, "agents_dir", lambda: tmp_path / "LaunchAgents")
    monkeypatch.setattr(service, "log_path", lambda: tmp_path / "Logs" / "editing-joe" / "broll.log")
    monkeypatch.setattr(service, "port_in_use", lambda host, port: False)
    return tmp_path


def test_the_agent_serves_on_loopback_with_the_inbox_and_one_log(mac):
    plist = service.build_plist(inbox=str(mac / "Broll Inbox"))
    args = plist["ProgramArguments"]
    assert plist["Label"] == "com.editingjoe.broll"
    assert args[args.index("serve"):args.index("serve") + 5] == ["serve", "--host", "127.0.0.1", "--port", "8000"]
    assert plist["EnvironmentVariables"]["BROLL_INBOX"] == str(mac / "Broll Inbox")
    assert plist["StandardOutPath"] == plist["StandardErrorPath"] == str(service.log_path())
    assert plist["RunAtLoad"] is True and plist["KeepAlive"] is True


def test_the_agent_is_gentle_on_a_laptop_that_runs_it_all_day(mac):
    plist = service.build_plist()
    args = " ".join(plist["ProgramArguments"])
    assert "--max-jobs 1" in args, "one file indexed at a time across every client"
    assert "--lazy-model" in args, "the embedding model is loaded when first needed"
    assert "--idle-poll 5" in args
    assert plist["Nice"] == 10
    env = plist["EnvironmentVariables"]
    assert env["OMP_NUM_THREADS"] == env["VECLIB_MAXIMUM_THREADS"] == "2"


def test_the_agent_can_find_ffmpeg_even_with_launchd_s_bare_path(mac):
    path = service.build_plist()["EnvironmentVariables"]["PATH"].split(":")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        assert str(Path(ffmpeg).parent) in path
    assert "/usr/bin" in path


def test_the_agent_runs_from_the_checkout_so_it_reads_the_same_keys(mac):
    folder = Path(service.build_plist()["WorkingDirectory"])
    assert (folder / "pyproject.toml").exists()


def test_the_agent_keeps_a_non_default_library_home(mac, monkeypatch):
    monkeypatch.setenv("BROLL_HOME", str(mac / "elsewhere"))
    assert service.build_plist()["EnvironmentVariables"]["BROLL_HOME"] == str(mac / "elsewhere")


def test_install_writes_the_agent_and_starts_it(mac):
    launchctl = Launchctl()
    report = service.install(inbox=str(mac / "inbox"), runner=launchctl)
    assert report.installed and report.started
    assert launchctl.verbs() == ["print", "bootstrap"]
    written = plistlib.loads(service.plist_path().read_bytes())
    assert written["Label"] == "com.editingjoe.broll"
    assert (mac / "inbox").is_dir(), "the drop folders' parent exists before anything watches it"
    assert service.log_path().parent.is_dir()


def test_install_leaves_a_server_already_on_the_port_alone(mac, monkeypatch):
    monkeypatch.setattr(service, "port_in_use", lambda host, port: True)
    launchctl = Launchctl()
    report = service.install(inbox=str(mac / "inbox"), runner=launchctl)
    assert report.installed and report.port_busy and not report.started
    assert "bootstrap" not in launchctl.verbs(), "someone else's server must not be fought over"
    assert any("already serving" in m for m in report.messages)
    assert service.plist_path().exists(), "it takes over at the next login"


def test_reinstalling_restarts_our_own_agent_with_the_new_definition(mac):
    launchctl = Launchctl(loaded=True)
    service.install(inbox=str(mac / "inbox"), runner=launchctl)
    assert launchctl.verbs() == ["print", "bootout", "bootstrap"]


def test_uninstall_stops_and_removes_the_agent(mac):
    launchctl = Launchctl()
    service.install(inbox=str(mac / "inbox"), runner=launchctl)
    report = service.uninstall(runner=launchctl)
    assert "bootout" in launchctl.verbs()
    assert not service.plist_path().exists()
    assert any("Removed" in m for m in report.messages)


def test_status_says_whether_it_is_installed_and_running(mac):
    launchctl = Launchctl(loaded=True, pid=4242)
    service.install(inbox=str(mac / "inbox"), runner=launchctl)
    report = service.status(runner=launchctl)
    assert report.installed and report.loaded
    assert report.pid == 4242
    assert report.answering is False, "nothing answers on the port in this test"


def test_on_anything_but_a_mac_it_points_at_systemd(monkeypatch):
    monkeypatch.setattr(service.sys, "platform", "linux")
    with pytest.raises(RuntimeError, match="systemd"):
        service.install()


# -- gentle serving ------------------------------------------------------------


class SlowPipeline:
    """Counts how many files are being indexed at the same moment, across workers."""

    running = 0
    most = 0

    async def ingest(self, discovered, force=False, overwrite_corrections=False):
        import asyncio

        from broll.ingest.pipeline import IngestResult

        SlowPipeline.running += 1
        SlowPipeline.most = max(SlowPipeline.most, SlowPipeline.running)
        await asyncio.sleep(0.05)
        SlowPipeline.running -= 1
        return IngestResult(status="indexed", filename=discovered.filename)


async def test_one_gate_holds_every_client_to_one_file_at_a_time(broll_home):
    import asyncio

    from broll.config import WorkspaceConfig
    from broll.db.store import Store
    from broll.jobs.worker import Worker

    gate = asyncio.Semaphore(1)
    workers, stores = [], []
    for client_id in ("adam", "luqman"):
        config = WorkspaceConfig(id=client_id, name=client_id)
        config.provider.vision = "mock"
        config.ingest.concurrency = 4
        config.save()
        store = Store.for_config(config)
        for n in range(3):
            store.enqueue("index_source", {"filename": f"{client_id}{n}.mp4", "origin": "local"})
        stores.append(store)
        workers.append(Worker(config, store, SlowPipeline(), gate=gate))
    SlowPipeline.running = SlowPipeline.most = 0
    try:
        stats = await asyncio.gather(*(w.run(drain=True) for w in workers))
    finally:
        for store in stores:
            store.close()
    assert sum(s.done for s in stats) == 6
    assert SlowPipeline.most == 1, "two clients' workers must share the one slot"


async def test_a_worker_told_to_stop_while_waiting_its_turn_claims_nothing(workspace, store):
    import asyncio

    from broll.jobs.queue import queue_stats
    from broll.jobs.worker import Worker

    for n in range(2):
        store.enqueue("index_source", {"filename": f"clip{n}.mp4", "origin": "local"})
    gate = asyncio.Semaphore(1)
    await gate.acquire()                      # another client's file holds the slot
    worker = Worker(workspace, store, SlowPipeline(), gate=gate, idle_poll_s=0.05)
    running = asyncio.create_task(worker.run(drain=False))
    await asyncio.sleep(0.2)
    worker.stop()
    await asyncio.wait_for(running, timeout=2)
    gate.release()
    assert queue_stats(store).queued == 2, "a stopping worker must not start another file"


def test_a_lazy_server_does_not_load_the_model_at_start(workspace, monkeypatch):
    from fastapi.testclient import TestClient

    from broll.web import app as web

    class Model:
        warmed = 0

        def warm_up(self):
            Model.warmed += 1

    monkeypatch.setattr(web, "get_embedder", lambda config: Model())
    with TestClient(web.create_studio_app([workspace], run_worker=True,
                                          gentle=web.Gentle(max_jobs=1, lazy_model=True, idle_poll_s=5))):
        pass
    assert Model.warmed == 0
    with TestClient(web.create_studio_app([workspace], run_worker=True)):
        pass
    assert Model.warmed >= 1, "the default still loads it up front"
