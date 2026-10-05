"""Test isolation: every test gets a fresh, on-disk throwaway database.

The module-level ``app`` (and every ``TestClient`` built from it) reads its
store through its own :class:`~app.storage.StoreHolder`; the fixture resets
that holder per test. Helpers kept under the old names let tests inspect or
reopen the app's store without touching any process-global state (there is
none): second instances created with :func:`app.main.create_app` are wholly
independent and are exercised separately in ``test_multi_instance.py``.
"""

import pytest

from app.main import app
from app.storage import Store, StoreHolder


def app_holder() -> StoreHolder:
    """The store holder owned by the module-level test application."""
    return app.state.store_holder


def get_store() -> Store:
    """The module-level app's currently open store."""
    return app_holder().get()


def reset_store(db_path):
    """Point the module-level app at a fresh store (throwaway or a reopen)."""
    return app_holder().reset(db_path)


@pytest.fixture(autouse=True)
def fresh_store(tmp_path):
    db_path = str(tmp_path / "test-risk.db")
    reset_store(db_path)
    yield
    app_holder().close()
