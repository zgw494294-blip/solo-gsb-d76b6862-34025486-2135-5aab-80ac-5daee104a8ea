"""Two application instances with different databases in one process.

Every :func:`app.main.create_app` instance owns its store (connection +
lock): graphs, graph versions, the sensitive registry and its generation,
sealed packages and the audit log of one instance can never be observed from
another — cross-instance graph/package reads answer 404. These tests cover:

* interleaved HTTP operations against two instances with separate files;
* independent registry generations, re-reviews, package release/purge and
  audit logs (including the historical-graph release decision);
* shutting one instance down (its connection closes) while the other keeps
  reading and writing;
* restart persistence by opening a second instance on a closed instance's
  file;
* explicit ``db_path`` precedence over ``RISK_API_DB``;
* storage-fault rollback staying inside the failing instance.
"""

import os
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


# Non-sensitive source -> filter(country) -> projection export: the registry
# is the only thing that can block it, mirroring the other persistence tests.
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
ENTRIES_PAIRS = [("users", "email")]


@pytest.fixture
def instances(tmp_path):
    """Two running app instances on two separate database files."""
    db_a = str(tmp_path / "a.db")
    db_b = str(tmp_path / "b.db")
    app_a = create_app(db_a)
    app_b = create_app(db_b)
    with TestClient(app_a) as client_a, TestClient(app_b) as client_b:
        yield client_a, client_b, db_a, db_b


def register(client):
    resp = client.post("/api/graphs", json=GRAPH)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def seal(client, gid):
    resp = client.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def put_registry(client, entries, generation):
    return client.put(
        "/api/sensitive-columns",
        json={"entries": entries, "expected_generation": generation},
    )


# ------------------------------------------------------------- isolation --


def test_graph_registered_in_one_instance_is_404_in_the_other(instances):
    client_a, client_b, _, _ = instances
    gid = register(client_a)

    assert client_a.get(f"/api/graphs/{gid}").status_code == 200
    # The other instance has its own database: the id does not exist there.
    resp = client_b.get(f"/api/graphs/{gid}")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "GRAPH_NOT_FOUND"
    # Registered preview and sealing against the foreign id 404 too.
    resp = client_b.post(
        f"/api/graphs/{gid}/preview",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 404
    resp = client_b.post(
        f"/api/graphs/{gid}/packages",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 404
    # And the other instance's own graph space starts empty.
    assert client_b.get("/api/sensitive-columns").json() == {
        "generation": 0, "entries": []}


def test_each_instance_keeps_its_own_graph_and_graph_versions(instances):
    client_a, client_b, _, _ = instances
    gid_a = register(client_a)
    gid_b = register(client_b)
    assert gid_a != gid_b

    # Revise the graph in A twice; B's independent graph stays at version 1.
    resp = client_a.put(
        f"/api/graphs/{gid_a}",
        json={"nodes": GRAPH["nodes"], "export": GRAPH["export"],
              "expected_version": 1},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["graph_version"] == 2
    assert client_a.get(f"/api/graphs/{gid_a}").json()["graph_version"] == 2
    assert client_b.get(f"/api/graphs/{gid_b}").json()["graph_version"] == 1

    # A cannot read B's id and B cannot read A's id at any endpoint.
    assert client_a.get(f"/api/graphs/{gid_b}").status_code == 404
    assert client_b.get(f"/api/graphs/{gid_a}").status_code == 404
    # A stale-version conflict is scoped to the instance owning the graph.
    resp = client_a.put(
        f"/api/graphs/{gid_a}",
        json={"nodes": GRAPH["nodes"], "export": GRAPH["export"],
              "expected_version": 1},
    )
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "VERSION_CONFLICT"


def test_registry_generations_and_rereviews_are_independent(instances):
    client_a, client_b, _, _ = instances
    gid_a = register(client_a)
    gid_b = register(client_b)

    assert put_registry(client_a, ENTRIES, 0).status_code == 200
    # Only A advanced: its graph is blocked, B stays on generation 0.
    assert client_a.get(f"/api/graphs/{gid_a}").json()["generation"] == 1
    assert client_a.get(f"/api/graphs/{gid_a}").json()["review"][
        "blocked"] is True
    assert client_b.get(f"/api/graphs/{gid_b}").json()["generation"] == 0
    assert client_b.get(f"/api/graphs/{gid_b}").json()["review"][
        "blocked"] is False
    assert client_b.get("/api/sensitive-columns").json() == {
        "generation": 0, "entries": []}

    # B now advances its own generation once; the two generations coexist.
    assert put_registry(client_b, [], 0).status_code == 200
    assert client_b.get("/api/sensitive-columns").json()["generation"] == 1
    assert client_a.get("/api/sensitive-columns").json()["generation"] == 1
    # ... but the entries differ, and A's stale expectation 0 is now rejected
    # only against A's own generation.
    assert client_a.get("/api/sensitive-columns").json()["entries"] == ENTRIES
    assert client_b.get("/api/sensitive-columns").json()["entries"] == []
    assert put_registry(client_a, [], 0).status_code == 409
    assert put_registry(client_b, ENTRIES, 1).status_code == 200


def test_packages_cross_instances_404_and_release_decisions_are_independent(
    instances,
):
    client_a, client_b, _, _ = instances
    gid_a = register(client_a)
    gid_b = register(client_b)
    pid_a = seal(client_a, gid_a)
    pid_b = seal(client_b, gid_b)
    assert pid_a != pid_b

    # A package id is unknown to the other instance's storage.
    assert client_b.get(f"/api/packages/{pid_a}").status_code == 404
    assert client_a.post(
        f"/api/packages/{pid_b}/release",
        json={"expected_generation": 0},
    ).status_code == 404

    # Both packages release cleanly while their registries are at gen 0.
    assert client_a.post(
        f"/api/packages/{pid_a}/release",
        json={"expected_generation": 0},
    ).status_code == 200
    assert client_b.post(
        f"/api/packages/{pid_b}/release",
        json={"expected_generation": 0},
    ).status_code == 200

    # A marks the source column: re-releasing the *bound historical version*
    # is blocked in A, while B's identical package still releases fine.
    assert put_registry(client_a, ENTRIES, 0).status_code == 200
    resp = client_a.post(
        f"/api/packages/{pid_a}/release",
        json={"expected_generation": 1},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "RELEASE_BLOCKED"
    resp = client_b.post(
        f"/api/packages/{pid_b}/release",
        json={"expected_generation": 0},
    )
    assert resp.status_code == 200
    assert client_b.get(f"/api/packages/{pid_b}").status_code == 200


def test_purge_and_audit_log_are_scoped_to_one_instance(instances):
    client_a, client_b, _, _ = instances
    gid_a = register(client_a)
    gid_b = register(client_b)
    pid_a = seal(client_a, gid_a)
    pid_b = seal(client_b, gid_b)
    client_a.post(f"/api/packages/{pid_a}/release",
                  json={"expected_generation": 0})
    client_b.post(f"/api/packages/{pid_b}/release",
                  json={"expected_generation": 0})

    # Purging in A touches only A's package; B's sealed sample stays live.
    resp = client_a.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "email"}]},
    )
    assert resp.status_code == 200
    assert resp.json() == {"purged": [pid_a]}
    assert client_a.get(f"/api/packages/{pid_a}").status_code == 410
    assert client_b.get(f"/api/packages/{pid_b}").status_code == 200
    assert client_b.get(f"/api/packages/{pid_b}").json()[
        "rows"] == [[1, "a@x.example", "ALL"]]

    # A's purge is still claimable/purgeable nowhere else, and B's later purge
    # lists only B's package with an audit seq of its own.
    resp = client_b.post(
        "/api/packages/purge",
        json={"columns": [{"source": "users", "column": "email"}]},
    )
    assert resp.json() == {"purged": [pid_b]}
    assert client_b.get(f"/api/packages/{pid_b}").status_code == 410

    log_a = client_a.get("/api/audit-log").json()
    log_b = client_b.get("/api/audit-log").json()
    assert [r["type"] for r in log_a["records"]] == ["purge"]
    assert [r["type"] for r in log_b["records"]] == ["purge"]
    # Each database numbers its own log from seq 1 and records only its own
    # package ids.
    assert log_a["records"][0]["seq"] == 1
    assert log_b["records"][0]["seq"] == 1
    assert log_a["records"][0]["package_ids"] == [pid_a]
    assert log_b["records"][0]["package_ids"] == [pid_b]

    # A registry replacement appends to A's log only.
    assert put_registry(client_a, [], 0).status_code == 200
    types_a = [r["type"] for r in
               client_a.get("/api/audit-log").json()["records"]]
    types_b = [r["type"] for r in
               client_b.get("/api/audit-log").json()["records"]]
    assert types_a == ["purge", "replace"]
    assert types_b == ["purge"]


def test_interleaved_operations_never_cross_databases(instances):
    client_a, client_b, _, _ = instances

    # Interleave graph creation, registry changes and packages step by step.
    gid_a = register(client_a)
    assert put_registry(client_b, ENTRIES, 0).status_code == 200
    # B's registry blocks a fresh graph in B but nothing in A changes.
    gid_b = register(client_b)
    assert client_b.get(f"/api/graphs/{gid_b}").json()["review"][
        "blocked"] is True
    assert client_a.get(f"/api/graphs/{gid_a}").json()["generation"] == 0

    # Sealing is allowed in A (safe) and refused in B (blocked).
    pid_a = seal(client_a, gid_a)
    resp = client_b.post(
        f"/api/graphs/{gid_b}/packages",
        json={"rows": ROWS, "conditions": CONDITIONS},
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "PREVIEW_BLOCKED"

    # A revises while B stays put; the package A sealed stays on version 1.
    resp = client_a.put(
        f"/api/graphs/{gid_a}",
        json={"nodes": GRAPH["nodes"], "export": GRAPH["export"],
              "expected_version": 1},
    )
    assert resp.status_code == 200
    assert client_a.post(
        f"/api/packages/{pid_a}/release",
        json={"expected_generation": 0},
    ).status_code == 200
    claimed = client_a.get(f"/api/packages/{pid_a}").json()
    assert claimed["graph_version"] == 1
    assert client_b.get(f"/api/graphs/{gid_b}").json()["graph_version"] == 1
    # The package itself is invisible to B.
    assert client_b.get(f"/api/packages/{pid_a}").status_code == 404


def test_concurrent_operations_on_two_instances_run_independently(instances):
    client_a, client_b, _, _ = instances
    gid_a = register(client_a)
    gid_b = register(client_b)

    outcome = {}

    def workflow(client, name, entries, gen_before):
        # Replace, register-preview-gated operations on one instance.
        assert put_registry(client, entries, gen_before).status_code == 200
        reg = client.get("/api/sensitive-columns").json()
        outcome[name] = reg["generation"]
        resp = client.post(
            f"/api/graphs/{gid_a if name == 'a' else gid_b}/preview",
            json={"rows": ROWS, "conditions": CONDITIONS},
        )
        outcome[name + "_blocked"] = resp.status_code == 400

    t_a = threading.Thread(
        target=workflow, args=(client_a, "a", ENTRIES, 0))
    t_b = threading.Thread(
        target=workflow, args=(client_b, "b", [], 0))
    t_a.start()
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)

    # Each instance advanced its own unrelated generation exactly once; A's
    # preview is blocked by its own registry while B's still returns data.
    assert outcome["a"] == 1 and outcome["b"] == 1
    assert outcome["a_blocked"] is True
    assert outcome["b_blocked"] is False
    assert client_a.get("/api/sensitive-columns").json()["entries"] == ENTRIES
    assert client_b.get("/api/sensitive-columns").json()["entries"] == []


# ------------------------------------------------------- close / restart --


def test_closing_one_instance_keeps_the_other_reading_and_writing(tmp_path):
    db_a = str(tmp_path / "close-a.db")
    db_b = str(tmp_path / "close-b.db")
    app_a = create_app(db_a)
    app_b = create_app(db_b)

    with TestClient(app_b) as client_b:
        with TestClient(app_a) as client_a:
            gid_a = register(client_a)
            gid_b = register(client_b)
            assert put_registry(client_a, ENTRIES, 0).status_code == 200
            pid_b = seal(client_b, gid_b)

        # Leaving A's lifespan shuts A down: it closed its own connection.
        assert app_a.state.store_holder._store is None
        # Closing again is harmless and still does not touch B.
        app_a.state.store_holder.close()

        # B is untouched by A's shutdown and keeps serving reads and writes.
        body = client_b.get(f"/api/graphs/{gid_b}").json()
        assert body["graph_version"] == 1
        assert body["generation"] == 0
        assert put_registry(client_b, ENTRIES, 0).status_code == 200
        assert client_b.get(f"/api/graphs/{gid_b}").json()[
            "generation"] == 1
        resp = client_b.post(
            f"/api/graphs/{gid_b}/preview",
            json={"rows": ROWS, "conditions": CONDITIONS},
        )
        assert resp.status_code == 400
        # A's graph and B's package are scoped to their own databases.
        assert client_b.get(f"/api/graphs/{gid_a}").status_code == 404
        assert client_b.get(f"/api/packages/{pid_b}").status_code == 409


def test_closed_instance_database_reopens_with_all_state(tmp_path):
    db_path = str(tmp_path / "restart.db")
    app_one = create_app(db_path)
    with TestClient(app_one) as client_one:
        gid = register(client_one)
        pid = seal(client_one, gid)
        assert put_registry(client_one, ENTRIES, 0).status_code == 200

    # Simulate a process restart: a brand-new instance on the same file.
    app_two = create_app(db_path)
    with TestClient(app_two) as client_two:
        assert client_two.get("/api/sensitive-columns").json() == {
            "generation": 1, "entries": ENTRIES}
        assert client_two.get(f"/api/graphs/{gid}").status_code == 200
        # The package sealed before the restart is bound and visible again.
        resp = client_two.get(f"/api/packages/{pid}")
        assert resp.status_code == 409  # never released under gen 1
        assert resp.json()["error"]["code"] == "PACKAGE_NOT_RELEASED"
        log = client_two.get("/api/audit-log").json()
        assert [r["type"] for r in log["records"]] == ["replace"]
        # Normal writes continue against the reopened instance.
        assert put_registry(client_two, [], 1).status_code == 200
        assert client_two.get("/api/sensitive-columns").json()[
            "generation"] == 2


# ------------------------------------------------ path/env configuration --


def test_explicit_path_beats_environment_variable(tmp_path, monkeypatch):
    env_db = str(tmp_path / "from-env.db")
    explicit_db = str(tmp_path / "explicit.db")
    monkeypatch.setenv("RISK_API_DB", env_db)

    application = create_app(explicit_db)
    with TestClient(application) as client:
        register(client)

    # The explicit file was created and used; the env-named file untouched.
    assert os.path.exists(explicit_db)
    assert not os.path.exists(env_db)


def test_environment_variable_is_used_without_explicit_path(tmp_path,
                                                            monkeypatch):
    env_db = str(tmp_path / "env-only.db")
    monkeypatch.setenv("RISK_API_DB", env_db)

    application = create_app()
    with TestClient(application) as client:
        gid = register(client)
        assert client.get(f"/api/graphs/{gid}").status_code == 200
    assert os.path.exists(env_db)

    # A later instance pointing at the env file sees the persisted state.
    application_two = create_app()
    with TestClient(application_two) as client_two:
        assert client_two.get(f"/api/graphs/{gid}").status_code == 200


def test_startup_eagerly_opens_an_unwritable_database(tmp_path):
    # A path whose parent is an ordinary file cannot host a database; the
    # failure surfaces at lifespan startup, not on the first request.
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    application = create_app(str(blocker / "risk.db"))
    with pytest.raises(Exception):
        with TestClient(application):
            pytest.fail("startup should have failed to open the database")


# ------------------------------------------------- fault stays in instance --


class FaultyConnection:
    """Delegates to the real connection, failing an armed statement once."""

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


def test_rollback_fault_in_one_instance_leaves_other_instance_intact(
    instances,
):
    client_a, client_b, _, _ = instances
    gid_a = register(client_a)
    gid_b = register(client_b)

    store_a = client_a.app.state.store_holder.get()
    real = store_a._conn
    proxy = FaultyConnection(real)
    store_a._conn = proxy
    try:
        proxy.arm(lambda sql: sql == "COMMIT")
        resp = TestClient(client_a.app, raise_server_exceptions=False).put(
            "/api/sensitive-columns",
            json={"entries": ENTRIES, "expected_generation": 0},
        )
        assert resp.status_code == 500
        assert resp.json()["error"]["code"] == "INTERNAL"
    finally:
        store_a._conn = real

    # A's failed replacement rolled back completely: same expected generation
    # still works on the retry.
    assert client_a.get("/api/sensitive-columns").json()["generation"] == 0
    assert client_a.get(f"/api/graphs/{gid_a}").json()["review"][
        "blocked"] is False
    assert put_registry(client_a, ENTRIES, 0).status_code == 200

    # B was never involved: still generation 0, clean and fully writable.
    assert client_b.get("/api/sensitive-columns").json()["generation"] == 0
    assert client_b.get(f"/api/graphs/{gid_b}").json()["review"][
        "blocked"] is False
    pid_b = seal(client_b, gid_b)
    assert client_b.post(
        f"/api/packages/{pid_b}/release",
        json={"expected_generation": 0},
    ).status_code == 200
    assert client_b.get(f"/api/packages/{pid_b}").status_code == 200
    # A's post-fault state is not visible in B and vice versa.
    assert client_b.get(f"/api/graphs/{gid_a}").status_code == 404
