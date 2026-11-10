"""An SGET edit, its log entry and its version bump go together; without a database.

A fake stands in for Neo4j (it has no driver, so map_edit uses it as it is).
The Neo4j-backed version of these is test_scene_graph_edit_db.
"""

import time

import pytest

pytest.importorskip("heracles_ros.map_state")

from heracles_ros import scene_change  # noqa: E402

from heracles_agents import scene_graph_edit as sge  # noqa: E402


class _Point:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Db:
    def __init__(self, rebuilding=False):
        self.since = time.time() if rebuilding else None
        self.version = 4
        self.console = 0
        self.logged = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def query(self, query):
        if "rebuilding_since AS t" in query:
            return [{"t": self.since, "f": None}]
        if "SET m.version" in query:
            self.version += 1
            return [{"v": self.version}]
        raise AssertionError(query)

    def execute(self, query, **kw):
        if "_State" in query and "MATCH (s:" in query:
            self.console += 1
        elif "CREATE CONSTRAINT" in query:
            pass
        elif "CREATE (n:_SceneChange" in query:
            self.logged.append(kw["op"])
            return [{"seq": len(self.logged)}], None, None
        elif "RETURN p.center" in query:
            return [{"c": _Point(3.0, 4.0, 0.0)}], None, None
        else:
            raise AssertionError(query)
        return [], None, None


@pytest.fixture
def chat(monkeypatch):
    reports = []
    sge.gate.allowed = True
    sge.gate.report = lambda changes, version: reports.append((changes, version))
    monkeypatch.setattr(sge, "_map_problem", lambda conf: "")
    yield reports
    sge.gate.allowed = sge.gate.wrote = False
    sge.gate.report = None


def _use(monkeypatch, db):
    monkeypatch.setattr(sge, "_connect", lambda conf: db)


def test_a_place_removal_is_logged_bumped_and_announced(chat, monkeypatch):
    db = _Db()
    _use(monkeypatch, db)
    monkeypatch.setattr(
        scene_change,
        "remove_place",
        lambda d, s: {"symbol": "t5571", "edges": 2, "objects": []},
    )
    out = sge.remove_scene_place("t5571")
    assert "map_version 5" in out
    assert db.logged == ["remove_place"]
    assert db.console == 1  # the console's counter moved too
    ((changes, version),) = chat
    assert version == 5
    assert changes == [
        {
            "kind": sge.REMOVED,
            "symbol": "t5571",
            "label": "place",
            "old": (3.0, 4.0, 0.0),
            "new": None,
        }
    ]


def test_a_connection_edit_is_announced_as_a_new_version(chat, monkeypatch):
    db = _Db()
    _use(monkeypatch, db)
    monkeypatch.setattr(
        scene_change, "remove_connection", lambda d, a, b: {"a": a, "b": b}
    )
    sge.disconnect_scene_places("t1", "t2")
    assert chat == [([], 5)] and db.logged == ["remove_connection"]


def test_no_edit_while_the_map_is_rebuilt(chat, monkeypatch):
    db = _Db(rebuilding=True)
    _use(monkeypatch, db)
    monkeypatch.setattr(
        scene_change, "remove_place", lambda d, s: pytest.fail("must not edit")
    )
    assert sge.remove_scene_place("t5571") == sge.REBUILDING
    assert db.version == 4 and db.logged == [] and chat == []


def test_a_refused_edit_is_not_bumped_or_announced(chat, monkeypatch):
    db = _Db()
    _use(monkeypatch, db)
    monkeypatch.setattr(scene_change, "remove_place", lambda d, s: None)
    assert sge.remove_scene_place("t1").startswith("Refused")
    assert db.version == 4 and chat == [] and not sge.gate.wrote
