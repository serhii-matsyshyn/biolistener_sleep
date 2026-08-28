"""The protocol API plans, guards, drives and reports a session."""
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _wait_idle(api, timeout=30.0):
    """The runner owns a thread; the next test cannot start until it exits."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with api._state.lock:
            running = api._state.running
            runner = api._state.runner
        if not running:
            return True
        if runner is not None:
            runner.request_stop("test_teardown")
        time.sleep(0.2)
    return False


@pytest.fixture
def client(tmp_path, monkeypatch):
    import app.runtime.protocol_api as api
    assert _wait_idle(api), "a previous session never stopped"
    monkeypatch.setattr(api, "SESSION_ROOT", tmp_path)
    api._state.__init__()
    app = FastAPI()
    app.include_router(api.router)
    try:
        yield TestClient(app)
    finally:
        _wait_idle(api)


SHORT = {"protocol": "validation", "compress_factor": 0.03, "audio_cues": False}


class TestPlan:
    def test_returns_the_block_plan_without_starting_anything(self, client):
        from app.protocol.spec import build_full
        expected = build_full("abba")
        p = client.get("/api/protocol/plan", params={"protocol": "full"}).json()
        assert len(p["blocks"]) == len(expected.blocks)
        assert p["total_duration_s"] == pytest.approx(expected.total_duration_s)
        assert client.get("/api/protocol/status").json()["running"] is False

    def test_the_plan_reports_which_arm_order_comes_next(self, client):
        p = client.get("/api/protocol/plan", params={"protocol": "full"}).json()
        assert p["suggested_order"] in ("abba", "baab")
        assert set(p["order_history"]) == {"abba", "baab"}

    def test_full_protocol_needs_a_checkpoint(self, client):
        p = client.get("/api/protocol/plan", params={"protocol": "full"}).json()
        assert not p["ok"]
        assert any("checkpoint" in x for x in p["problems"])

    def test_validation_protocol_needs_none(self, client):
        p = client.get("/api/protocol/plan",
                       params={"protocol": "validation", "audio_cues": False}).json()
        assert p["ok"], p["problems"]

    def test_a_compressed_plan_warns_but_does_not_block(self, client):
        p = client.get("/api/protocol/plan",
                       params={"protocol": "validation", "compress_factor": 0.03,
                               "audio_cues": False}).json()
        assert p["warnings"] and p["ok"]

    def test_a_bad_compression_factor_is_rejected(self, client):
        r = client.get("/api/protocol/plan",
                       params={"protocol": "validation", "compress_factor": 5})
        assert r.status_code == 400


class TestGuards:
    def test_start_refuses_a_protocol_that_fails_preflight(self, client):
        r = client.post("/api/protocol/start", json={"protocol": "full"})
        assert r.status_code == 400

    def test_stop_without_a_session_is_rejected(self, client):
        assert client.post("/api/protocol/stop").status_code == 409

    def test_rating_without_a_session_is_rejected(self, client):
        assert client.post("/api/protocol/kss", json={"value": 5}).status_code == 409

    def test_a_rating_must_be_on_the_scale(self, client):
        assert client.post("/api/protocol/kss", json={"value": 12}).status_code == 422


class TestRunning:
    def test_only_one_session_runs_at_a_time(self, client):
        # Both loops would own the mask and the amplifier.
        assert client.post("/api/protocol/start", json=SHORT).status_code == 200
        try:
            assert client.post("/api/protocol/start", json=SHORT).status_code == 409
        finally:
            client.post("/api/protocol/stop")

    def test_status_reports_progress_while_running(self, client):
        client.post("/api/protocol/start", json=SHORT)
        try:
            deadline = time.time() + 20
            while time.time() < deadline:
                s = client.get("/api/protocol/status").json()
                if s["tick"]:
                    break
                time.sleep(0.3)
            assert s["running"] and s["current"]["block"]
            assert s["tick"]["session_duration_s"] > 0
        finally:
            client.post("/api/protocol/stop")

    def test_the_live_socket_pushes_the_snapshot(self, client):
        client.post("/api/protocol/start", json=SHORT)
        try:
            with client.websocket_connect("/api/protocol/live") as ws:
                snap = ws.receive_json()
            assert {"running", "current", "tick", "blocks"} <= set(snap)
        finally:
            client.post("/api/protocol/stop")

    def test_stopping_ends_the_session_and_records_the_reason(self, client):
        client.post("/api/protocol/start", json=SHORT)
        time.sleep(3)
        assert client.post("/api/protocol/stop").status_code == 200
        deadline = time.time() + 20
        while time.time() < deadline:
            s = client.get("/api/protocol/status").json()
            if not s["running"]:
                break
            time.sleep(0.3)
        assert not s["running"]
        assert s["status"] == "aborted" and s["abort_reason"]
        assert s["session_dir"]

    def test_a_rating_is_stored_against_the_running_block(self, client):
        client.post("/api/protocol/start", json=SHORT)
        try:
            time.sleep(3)
            r = client.post("/api/protocol/kss", json={"value": 6})
            assert r.status_code == 200
            assert r.json()["recorded"]["value"] == 6
            assert client.get("/api/protocol/status").json()["kss"][0]["value"] == 6
        finally:
            client.post("/api/protocol/stop")


class TestReporting:
    def test_sessions_are_listed_after_a_run(self, client):
        client.post("/api/protocol/start", json=SHORT)
        time.sleep(3)
        client.post("/api/protocol/stop")
        deadline = time.time() + 20
        while time.time() < deadline:
            if not client.get("/api/protocol/status").json()["running"]:
                break
            time.sleep(0.3)
        assert client.get("/api/protocol/sessions").json()["sessions"]

    def test_gates_are_reported_for_a_recorded_session(self, client):
        client.post("/api/protocol/start", json=SHORT)
        time.sleep(3)
        client.post("/api/protocol/stop")
        deadline = time.time() + 20
        while time.time() < deadline:
            s = client.get("/api/protocol/status").json()
            if not s["running"]:
                break
            time.sleep(0.3)
        name = s["session_dir"].rstrip("/").split("/")[-1]
        g = client.get(f"/api/protocol/sessions/{name}/gates").json()
        # Stopped early, so most gates cannot be evaluated - and an
        # unevaluated gate must count as a failure, not as absence.
        assert "gates" in g and g["passed"] is False

    def test_an_unknown_session_is_a_404(self, client):
        assert client.get("/api/protocol/sessions/nope/gates").status_code == 404
