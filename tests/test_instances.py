"""Per-application storage: two instances, two databases, one process.

Every ``create_app(db_path)`` owns its own SQLite connection, lock and state.
These tests run two applications side by side against two different files and
check that graphs, graph versions, registry generations, sealed packages and
audit logs never cross between them; that cross-instance lookups stay 404;
that interleaved operations on both instances keep independent generation
counters; that shutting one instance down (its connection closes) leaves the
other instance fully readable and writable; that an instance can be restarted
against the same file with all state intact; and that an explicit path wins
over ``RISK_API_DB`` while an omitted path reads the environment at startup.
Finally, a rolled-back replacement on one instance never disturbs a
successful replacement committed by the other.
"""

import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

# Clean graph: nothing sensitive in-graph, so the registry alone decides.
GRAPH = {
    "nodes": [
        {
            "id": "users",
            "type": "source",
            "columns": [
                {"name": "uid", "sensitive": False},
                {"name": "email", "sensitive": False},
                {"name": "country", "sensitive": False},
            ],
        },
        {"id": "only_cn", "type": "filter", "input": "users",
         "references": ["country"]},
        {
            "id": "export_users",
            "type": "projection",
            "input": "only_cn",
            "columns": [
                {"name": "user_id", "copy": "uid"},
                {"name": "mail", "copy": "email"},
                {"name": "bucket", "constant": "ALL"},
            ],
        },
    ],
    "export": "export_users",
}
ROWS = {
    "users": [
        {"uid": 1, "email": "a@x.example", "country": "CN"},
        {"uid": 2, "email": "b@x.example", "country": "US"},
    ]
}
CONDITIONS = {"only_cn": {"country": "CN"}}
ENTRIES = [{"source": "users", "column": "email"}]


@pytest.fixture
def instances(tmp_path):
    """Two running (lifespan-started) apps on two different databases."""
    db_a = str(tmp_path / "a.db")
    db_b = str(tmp_path / "b.db")
    app_a = create_app(db_a)
    app_b = create_app(db_b)
    with TestClient(app_a, raise_server_exceptions=False) as ca, \
            TestClient(app_b, raise_server_exceptions=False) as cb:
        yield ca, cb, app_a, app_b, db_a, db_b


def register(client, payload=GRAPH):
    resp = client.post("/api/graphs", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def put_registry(client, entries, generation):
    return client.put(
        "/api/sensitive-columns",
        json={"entries": entries, "expected_generation": generation},
    )


def seal_ok(client, gid):
    resp = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ------------------------------------------------------------- ownership --


def test_each_instance_holds_its_own_store_connection_and_lock(instances):
    _ca, _cb, app_a, app_b, _da, _db = instances
    store_a = app_a.state.store
    store_b = app_b.state.store
    assert store_a is not store_b
    assert store_a.lock is not store_b.lock
    assert store_a._path != store_b._path
    assert store_a._conn is not store_b._conn


def test_importing_the_app_never_opens_a_database(tmp_path, monkeypatch):
    # No RISK_API_DB and no import-time disk access: a fresh import in a clean
    # cwd leaves no data directory behind.
    monkeypatch.delenv("RISK_API_DB", raising=False)
    workdir = tmp_path / "nowhere"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    import importlib

    importlib.import_module("app.main")
    assert not (workdir / "data").exists()


# ----------------------------------------------------- state isolation --


def test_graphs_versions_and_conclusions_never_cross(instances):
    ca, cb, _aa, _ab, _da, _db = instances
    gid_a = register(ca)

    # The graph registered in A simply does not exist in B.
    assert cb.get(f"/api/graphs/{gid_a}").status_code == 404
    assert cb.get(f"/api/graphs/{gid_a}").json()["error"]["code"] == (
        "GRAPH_NOT_FOUND"
    )

    # A marks the registry (generation 1): its graph is blocked.
    assert put_registry(ca, ENTRIES, 0).status_code == 200
    got_a = ca.get(f"/api/graphs/{gid_a}").json()
    assert got_a["generation"] == 1
    assert got_a["review"]["blocked"] is True

    # B is a different database: still empty at generation 0.
    assert cb.get("/api/sensitive-columns").json() == {
        "generation": 0, "entries": []}
    # A graph registered now in B is reviewed under B's empty registry.
    gid_b = register(cb)
    got_b = cb.get(f"/api/graphs/{gid_b}").json()
    assert got_b["generation"] == 0
    assert got_b["review"]["blocked"] is False

    # A's graph gets a revision (version 2); B never sees that id at all.
    resp = ca.put(
        f"/api/graphs/{gid_a}",
        json={"nodes": GRAPH["nodes"], "export": GRAPH["export"],
              "expected_version": 1},
    )
    assert resp.status_code == 200
    assert resp.json()["graph_version"] == 2
    assert cb.put(
        f"/api/graphs/{gid_a}",
        json={"nodes": GRAPH["nodes"], "export": GRAPH["export"],
              "expected_version": 1},
    ).status_code == 404
    assert ca.get(f"/api/graphs/{gid_a}").json()["graph_version"] == 2
    assert cb.get(f"/api/graphs/{gid_b}").json()["graph_version"] == 1


def test_packages_are_unknown_to_the_other_instance(instances):
    ca, cb, _aa, _ab, _da, _db = instances
    gid_a = register(ca)
    pid_a = seal_ok(ca, gid_a)
    assert ca.post(
        f"/api/packages/{pid_a}/release",
        json={"expected_generation": 0},
    ).status_code == 200

    # Every package route on B answers 404 for A's package id.
    resp = cb.get(f"/api/packages/{pid_a}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "PACKAGE_NOT_FOUND"
    resp = cb.post(
        f"/api/packages/{pid_a}/release",
        json={"expected_generation": 0},
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "PACKAGE_NOT_FOUND"

    # Claim on A still serves A's sealed rows.
    body = ca.get(f"/api/packages/{pid_a}").json()
    assert body["rows"] == [[1, "a@x.example", "ALL"]]


def test_registry_generations_advance_independently(instances):
    ca, cb, _aa, _ab, _da, _db = instances
    register(ca)
    # A advances twice, B once: the counters share no state.
    assert put_registry(ca, ENTRIES, 0).status_code == 200
    assert put_registry(ca, [], 1).status_code == 200
    assert put_registry(cb, ENTRIES, 0).status_code == 200
    assert ca.get("/api/sensitive-columns").json()["generation"] == 2
    assert cb.get("/api/sensitive-columns").json()["generation"] == 1
    # A stale expectation on A is unaffected by B's generation.
    resp = put_registry(ca, [], 0)
    assert resp.status_code == 409
    assert resp.json()["error"]["current_generation"] == 2


def test_audit_logs_are_per_instance(instances):
    ca, cb, _aa, _ab, _da, _db = instances
    gid_a = register(ca)
    pid_a = seal_ok(ca, gid_a)
    assert put_registry(ca, ENTRIES, 0).status_code == 200
    assert ca.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "email"}]},
    ).json() == {"purged": [pid_a]}

    records_a = ca.get("/api/audit-log").json()["records"]
    assert [r["type"] for r in records_a] == ["replace", "purge"]
    # B logged nothing: its seq space starts and stays at 0.
    assert cb.get("/api/audit-log").json() == {
        "until_seq": 0, "records": [], "next_cursor": 0}


def test_interleaved_operations_keep_independent_states(instances):
    ca, cb, _aa, _ab, _da, _db = instances
    register(ca)
    register(cb)

    outcomes = {}
    barrier = threading.Barrier(2)

    def churn(client, name, times):
        barrier.wait()
        generation = 0
        for _ in range(times):
            entries = [] if generation % 2 else ENTRIES
            resp = put_registry(client, entries, generation)
            assert resp.status_code == 200, resp.text
            generation += 1
        outcomes[name] = generation

    ta = threading.Thread(target=churn, args=(ca, "a", 7))
    tb = threading.Thread(target=churn, args=(cb, "b", 4))
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)

    # Every replacement on each instance serialised against its own gen only.
    assert outcomes == {"a": 7, "b": 4}
    assert ca.get("/api/sensitive-columns").json()["generation"] == 7
    assert cb.get("/api/sensitive-columns").json()["generation"] == 4
    # Each instance carries exactly its own audit history.
    assert len(ca.get("/api/audit-log").json()["records"]) == 7
    assert len(cb.get("/api/audit-log").json()["records"]) == 4


# ------------------------------------------------- shutdown independence --


def test_closing_one_instance_keeps_the_other_readable_and_writable(tmp_path):
    db_a = str(tmp_path / "a.db")
    db_b = str(tmp_path / "b.db")
    app_a = create_app(db_a)
    app_b = create_app(db_b)

    with TestClient(app_a) as ca:
        gid_a = register(ca)
        assert put_registry(ca, ENTRIES, 0).status_code == 200

    # App A has shut down and its own connection is closed.
    with pytest.raises(sqlite3.ProgrammingError):
        app_a.state.store._conn.execute("SELECT 1")

    with TestClient(app_b) as cb:
        # B never noticed A's shutdown: writes and reads keep working.
        gid_b = register(cb)
        assert put_registry(cb, ENTRIES, 0).status_code == 200
        body = cb.get(f"/api/graphs/{gid_b}").json()
        assert body["review"]["blocked"] is True
        assert cb.get(f"/api/graphs/{gid_a}").status_code == 404

        # And a brand-new instance reopened on A's file sees A's persisted
        # state (restart persistence), while B stays open the whole time.
        app_a2 = create_app(db_a)
        with TestClient(app_a2) as ca2:
            got = ca2.get(f"/api/graphs/{gid_a}").json()
            assert got["generation"] == 1
            assert got["review"]["blocked"] is True
            # Operations on the restarted A and the still-running B interleave.
            assert put_registry(ca2, [], 1).status_code == 200
            assert put_registry(cb, [], 1).status_code == 200
            assert ca2.get("/api/sensitive-columns").json()["generation"] == 2
            assert cb.get("/api/sensitive-columns").json()["generation"] == 2
            # Same numeric generation, unrelated databases: A's graph/package
            # ids are still unknown to B.
            assert cb.get(f"/api/graphs/{gid_a}").status_code == 404


# ------------------------------------------------------- path precedence --


def test_explicit_path_wins_over_environment(tmp_path, monkeypatch):
    env_db = tmp_path / "from-env.db"
    explicit_db = tmp_path / "explicit.db"
    monkeypatch.setenv("RISK_API_DB", str(env_db))

    with TestClient(create_app(str(explicit_db))) as client:
        register(client)
        assert put_registry(client, ENTRIES, 0).status_code == 200

    # The explicit file holds the write; the environment-named file was never
    # created or touched.
    assert explicit_db.exists()
    assert not env_db.exists()


def test_omitted_path_reads_environment_at_startup(tmp_path, monkeypatch):
    env_db = tmp_path / "env-at-startup.db"
    monkeypatch.setenv("RISK_API_DB", str(env_db))

    with TestClient(create_app()) as client:
        gid = register(client)
        assert put_registry(client, ENTRIES, 0).status_code == 200
        assert client.app.state.store._path == str(env_db)

    assert env_db.exists()
    # A later instance with a different env value resolves the new value; the
    # database is chosen per instance at startup, never via a global switch.
    other_db = tmp_path / "other.db"
    monkeypatch.setenv("RISK_API_DB", str(other_db))
    with TestClient(create_app()) as client:
        assert client.app.state.store._path == str(other_db)
        assert client.get(f"/api/graphs/{gid}").status_code == 404
        assert client.get("/api/sensitive-columns").json()["generation"] == 0


def test_default_path_when_environment_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("RISK_API_DB", raising=False)
    monkeypatch.chdir(tmp_path)
    with TestClient(create_app()) as client:
        # The default is kept verbatim (relative); sqlite resolves it against
        # the process working directory.
        assert client.app.state.store._path == "data/export-risk.db"
        register(client)
    assert (tmp_path / "data" / "export-risk.db").exists()


# ------------------------------------------- rollback stays per instance --


class FaultyConnection:
    """Delegates to a real connection, failing an armed statement once."""

    def __init__(self, real):
        self._real = real
        self._predicate = None

    def arm(self, predicate):
        self._predicate = predicate

    def _check(self, sql):
        if self._predicate is not None and self._predicate(sql):
            self._predicate = None
            raise sqlite3.OperationalError("injected storage fault")

    def execute(self, sql, parameters=()):
        self._check(sql)
        return self._real.execute(sql, parameters)

    def executemany(self, sql, seq):
        self._check(sql)
        return self._real.executemany(sql, seq)

    def commit(self):
        self._check("COMMIT")
        return self._real.commit()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        if exc_type is None:
            self._real.commit()
        else:
            self._real.rollback()
        return False

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_failed_commit_on_one_instance_leaves_other_instance_committed(
    instances,
):
    ca, cb, app_a, _app_b, _da, _db = instances
    register(ca)
    register(cb)

    store_a = app_a.state.store
    real = store_a._conn
    proxy = FaultyConnection(real)
    store_a._conn = proxy
    try:
        proxy.arm(lambda sql: sql == "COMMIT")
        # A's replacement fails at COMMIT: 500, A stays at generation 0.
        resp = ca.put(
            "/api/sensitive-columns",
            json={"entries": ENTRIES, "expected_generation": 0},
        )
        assert resp.status_code == 500
        assert ca.get("/api/sensitive-columns").json()["generation"] == 0
        assert ca.get("/api/audit-log").json()["records"] == []
    finally:
        store_a._conn = real

    # B is a different connection: the fault injected into A never reaches it,
    # and B's own replacement commits normally while A is rolled back.
    assert put_registry(cb, ENTRIES, 0).status_code == 200
    assert cb.get("/api/sensitive-columns").json()["generation"] == 1
    assert len(cb.get("/api/audit-log").json()["records"]) == 1
    # A can retry the very same replacement afterwards.
    assert put_registry(ca, ENTRIES, 0).status_code == 200
    assert ca.get("/api/sensitive-columns").json()["generation"] == 1


def test_rollback_and_commit_on_two_instances_can_interleave(instances):
    ca, cb, app_a, _app_b, _da, _db = instances
    register(ca)
    register(cb)

    started = threading.Event()
    gate = threading.Event()
    store_a = app_a.state.store
    real = store_a._conn
    proxy = FaultyConnection(real)
    store_a._conn = proxy

    def fail_behind_gate(sql):
        if sql == "COMMIT":
            started.set()
            assert gate.wait(timeout=5)
            return True
        return False

    def faulty_replace():
        proxy.arm(fail_behind_gate)
        # A's COMMIT waits on the gate until B has fully committed.
        resp = ca.put(
            "/api/sensitive-columns",
            json={"entries": ENTRIES, "expected_generation": 0},
        )
        outcomes["a"] = resp.status_code

    outcomes: dict[str, int] = {}
    worker = threading.Thread(target=faulty_replace)
    worker.start()
    assert started.wait(timeout=5)
    # B commits on its own connection and lock while A's commit is pending.
    outcomes["b"] = put_registry(cb, ENTRIES, 0).status_code
    gate.set()
    worker.join(timeout=5)
    store_a._conn = real

    assert outcomes == {"a": 500, "b": 200}
    assert ca.get("/api/sensitive-columns").json()["generation"] == 0
    assert cb.get("/api/sensitive-columns").json()["generation"] == 1
    assert ca.get("/api/audit-log").json()["records"] == []
    assert len(cb.get("/api/audit-log").json()["records"]) == 1
