"""Test isolation: every test gets a fresh, on-disk throwaway database.

The module-level ``app`` (the ``uvicorn app.main:app`` entry point) is the
client every test uses; the autouse fixture simply rebinds its instance-owned
store before each test — exactly what a fresh ``create_app(path)`` process
startup would do, but without rebuilding the whole app. Direct registry calls
receive ``app.state.store`` explicitly.
"""

import pytest

from app.main import app
from app.storage import Store


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    db_path = str(tmp_path / "test-risk.db")
    previous = getattr(app.state, "store", None)
    store = Store(db_path)
    app.state.store = store
    yield store
    # Tests may have simulated a restart (a different store on the same or
    # another file): close whichever store the app currently holds, then hand
    # the entry-point app back to the next test untouched.
    current = app.state.store
    if current is not None and current is not store:
        current.close()
    store.close()
    if previous is not None:
        app.state.store = previous
    else:
        del app.state.store


def restart_store(db_path):
    """Simulate a process restart of the shared test app against ``db_path``.

    Replaces only that application instance's store: its current connection is
    closed and the same file reopened, mirroring lifespan shutdown/startup.
    Returns the fresh store (also reachable as ``app.state.store``).
    """
    old = app.state.store
    new = Store(db_path)
    app.state.store = new
    old.close()
    return new


def replace_store(application, db_path):
    """Point any application instance at a fresh store on ``db_path``."""
    old = getattr(application.state, "store", None)
    new = Store(db_path)
    application.state.store = new
    if old is not None:
        old.close()
    return new

