"""Map edits run only for an SGET request, through the edit tools; Cypher only reads."""

import pytest

from heracles_agents.scene_graph_edit import (
    EDIT_TOOLS,
    NOT_CONFIRMED,
    READ_ONLY,
    REFUSED,
    add_scene_object,
    clear_scene_edits,
    confirming,
    gate,
    gated,
    goal_context_note,
    is_edit_request,
    is_write_query,
    remove_scene_place,
    resend_offer,
    reset_scene_graph,
    turn_allows_edits,
    undo_scene_edit,
)


@pytest.fixture
def query():
    ran = []
    run = gated(
        lambda cypher_string, dsgdb_conf=None: ran.append(cypher_string) or "[]"
    )
    _clear()
    yield run, ran
    _clear()


def _clear():
    gate.allowed = gate.wrote = gate.user_turn = False
    gate.last_goal = gate.pending = gate.report = gate.call_writer = None
    gate.check_goal = None
    gate.context = gate.last_context = None
    gate._proposed = {}


def test_edit_requests():
    assert is_edit_request("SGET: delete t444")
    assert is_edit_request("  sget: delete t444")
    assert not is_edit_request("delete t444, SGET: later")
    assert turn_allows_edits(["[Scene change report] ...", "SGET: delete t444"])
    assert not turn_allows_edits(["Go to t444", "[Scene change report] ..."])


def test_write_queries():
    assert is_write_query("MATCH (p:MeshPlace {nodeSymbol: 't444'}) DETACH DELETE p")
    assert is_write_query("MATCH ()-[r]-() DELETE r")
    assert is_write_query("MATCH (o:Object) SET o.class = 'cone'")
    assert is_write_query("CALL apoc.refactor.rename.label('A', 'B')")
    assert not is_write_query("MATCH (o:Object) RETURN o.nodeSymbol, o.center")
    # Keywords inside names and comments are not writes.
    assert not is_write_query("MATCH (o:Object {class: 'reset button'}) RETURN o")
    assert not is_write_query("MATCH (o) // delete later\nRETURN o")


def test_reads_always_run(query):
    run, ran = query
    run("MATCH (o:Object) RETURN o")
    assert ran == ["MATCH (o:Object) RETURN o"]


def test_cypher_never_writes(query):
    run, ran = query
    delete = "MATCH (p:MeshPlace {nodeSymbol: 't444'}) DETACH DELETE p"
    assert run(delete) == REFUSED  # outside SGET: resend with SGET
    gate.allowed = True
    assert run(delete) == READ_ONLY  # in SGET: use the edit tools
    assert ran == [] and not gate.wrote


def test_after_an_edit_the_last_goal_is_offered_again(query):
    send = confirming(lambda goal, **kw: f"Sent goal {goal}")
    goal, constraints = "(visited-place t500)", '[["forbidden-poi", "o3"]]'
    gate.start_turn(user_turn=True)
    send(goal, constraints_json=constraints)  # proposed
    gate.start_turn(user_turn=True)
    send(goal, constraints_json=constraints)  # confirmed
    offer = resend_offer(gate.last_goal)
    assert "(visited-place t500)" in offer and "forbidden-poi" in offer
    assert "send this goal again" in offer
    gone = resend_offer(gate.last_goal, missing=["o3"])
    assert "o3 is no longer in the map" in gone and "send this goal again" not in gone


def test_no_offer_without_a_goal_sent(query):
    assert resend_offer(gate.last_goal) == ""


def test_adding_an_object_needs_sget():
    gate.allowed = False
    assert add_scene_object("chair", 5.0, -8.0) == REFUSED


def _sender():
    sent = []
    return confirming(lambda goal, **kw: sent.append(goal) or f"Sent goal {goal}"), sent


def test_a_goal_is_sent_only_after_the_user_confirms_it(query):
    send, sent = _sender()
    gate.start_turn(user_turn=True)  # "bring the trash can to t4"
    assert send("(object-in-place o4 t4)") == NOT_CONFIRMED
    assert send("(object-in-place o4 t4)") == NOT_CONFIRMED  # same turn: still no
    gate.start_turn(user_turn=True)  # "yes"
    assert (
        send("(object-in-place  O4 t4)", constraints_json="")
        == "Sent goal (object-in-place  O4 t4)"
    )
    assert sent == ["(object-in-place  O4 t4)"] and gate.pending is None


def test_a_changed_goal_needs_its_own_confirmation(query):
    send, sent = _sender()
    gate.start_turn(user_turn=True)
    send("(visited-place t5)")
    gate.start_turn(user_turn=True)  # "yes, but avoid o3"
    assert (
        send("(visited-place t5)", constraints_json='[["forbidden-poi", "o3"]]')
        == NOT_CONFIRMED
    )
    assert sent == []


def test_a_scene_change_report_never_sends(query):
    send, sent = _sender()
    gate.start_turn(user_turn=True)
    send("(visited-object o8)")
    gate.start_turn(user_turn=False)  # a report, not the user, began this turn
    assert send("(visited-object o8)") == NOT_CONFIRMED and sent == []


def test_adding_a_place_needs_sget():
    from heracles_agents.scene_graph_edit import add_scene_place

    gate.allowed = False
    assert add_scene_place(29.4, -14.9, "t1104, t1405", "t999") == REFUSED


def test_every_edit_tool_needs_sget(query):
    assert remove_scene_place("t444") == REFUSED
    assert undo_scene_edit() == REFUSED
    assert reset_scene_graph() == REFUSED


def test_undo_and_reset_ask_the_writer(query):
    calls = []
    gate.call_writer = lambda service: calls.append(service) or (
        True,
        f"{service} done",
    )
    gate.allowed = True
    assert undo_scene_edit() == "undo_last_edit done" and gate.wrote
    clear_scene_edits()
    reset_scene_graph()
    assert calls == ["undo_last_edit", "clear_edits", "reset"]
    gate.call_writer = lambda service: (False, "the writer is not running")
    assert reset_scene_graph() == "Failed: the writer is not running"


def test_undo_without_a_writer_says_so(query):
    gate.allowed = True
    assert "not connected" in undo_scene_edit()


def test_the_chat_config_lists_every_edit_tool():
    import os

    import yaml

    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "chatdsg_repair",
        "agent_config.yaml",
    )
    with open(path) as fo:
        names = {t["name"] for t in yaml.safe_load(fo)["agent_info"]["tools"]}
    assert {t.name for t in EDIT_TOOLS} <= names


def test_the_repair_chat_config_binds_args_the_tools_accept():
    """Every bound arg in the chat config must be one its gated tool takes.

    The confirm wrapper once listed its parameters by hand, so the config's
    robot_name_env_var made every send raise inside the agent's thread: the
    chat just stopped answering.
    """
    import inspect
    import os

    import yaml

    from heracles_agents.scene_graph_edit import install
    from heracles_agents.tool_registry import ToolRegistry

    install()
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "examples",
        "chatdsg_repair",
        "agent_config.yaml",
    )
    with open(path) as fo:
        tools = yaml.safe_load(fo)["agent_info"]["tools"]
    for t in tools:
        fn = ToolRegistry.tools[t["name"]].function
        bound = dict.fromkeys(t.get("bound_args", {}), None)
        # follow_wrapped=False: check the wrapper itself, not what it wraps.
        sig = inspect.signature(fn, follow_wrapped=False)
        sig.bind_partial(**bound)  # raises TypeError if not


def test_a_goal_remembers_what_was_asked_and_from_where(query):
    send = confirming(lambda goal, **kw: f"Sent goal {goal}")
    here = {
        "request": "send hamilton to the nearest cone",
        "robots": {"hamilton": (-25.5, 2.3)},
    }
    gate.context = lambda: here
    gate.start_turn(user_turn=True)
    send("(visited-object-by hamilton o7)")  # proposed
    gate.context = lambda: {"request": "yes", "robots": {"hamilton": (-16.0, 2.0)}}
    gate.start_turn(user_turn=True)
    send("(visited-object-by hamilton o7)")  # confirmed
    assert gate.last_context == here

    # o7 is removed; the agent re-proposes from a scene change report ...
    gate.start_turn(user_turn=False)
    send("(visited-object-by hamilton o16)")
    gate.start_turn(user_turn=True)
    send("(visited-object-by hamilton o16)")  # ... and the user confirms
    assert gate.last_context == here  # still the original request and position
    note = goal_context_note(gate.last_context)
    assert "nearest cone" in note and "hamilton at (-25.5, 2.3)" in note


def test_no_note_before_any_goal():
    assert goal_context_note(None) == ""


# ---------- goals naming robots, regions or ids the map does not have ----------


class _MapDb:
    def __init__(self, robots=("testbot_b", "testbot_a"), rooms=("R1", "R3")):
        self.robots, self.rooms = robots, rooms
        self.symbols = {"o58", "t2249", "t1253"}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def query(self, cypher):
        if "Robot" in cypher:
            return [{"n": r} for r in self.robots]
        if "Room" in cypher:
            return [{"s": r} for r in self.rooms]
        return [{"s": s} for s in self.symbols if f"'{s}'" in cypher]


@pytest.fixture
def map_db(monkeypatch):
    import heracles_agents.scene_graph_edit as sge

    db = _MapDb()
    monkeypatch.setattr(sge, "_connect", lambda conf: db)
    return db


def test_a_misspelled_robot_is_caught_with_the_known_names(map_db):
    from heracles_agents.scene_graph_edit import goal_problems

    problem = goal_problems("(visited-object-by testbot_am o58)", "", None)
    assert "unknown robot testbot_am" in problem and "testbot_a, testbot_b" in problem
    problem = goal_problems(
        "(object-in-place o58 t2249)", '[["use-robots", "testbot_x"]]', None
    )
    assert "unknown robot testbot_x" in problem
    assert goal_problems("(object-placed-by testbot_a o58 t2249)", "", None) == ""


def test_regions_are_checked_against_the_rooms(map_db):
    from heracles_agents.scene_graph_edit import goal_problems

    assert goal_problems("(visited-region r3)", "", None) == ""
    assert goal_problems("(explored-region-by testbot_b r(1))", "", None) == ""
    assert "unknown region r7" in goal_problems("(visited-region r7)", "", None)
    map_db.rooms = ()
    assert "no regions" in goal_problems("(visited-region r3)", "", None)


def test_ids_the_map_lacks_are_caught(map_db):
    from heracles_agents.scene_graph_edit import goal_problems

    assert "o99 not in the scene graph" in goal_problems(
        "(object-in-place o99 t2249)", "", None
    )


@pytest.mark.parametrize("goal", ["(visited-object O999)", "(visited-place T999)"])
def test_uppercase_goal_ids_the_map_lacks_are_caught(map_db, goal):
    from heracles_agents.scene_graph_edit import goal_problems

    assert "not in the scene graph" in goal_problems(goal, "", None)


def test_valid_uppercase_ids_pass_validation(map_db):
    from heracles_agents.scene_graph_edit import goal_problems

    assert (
        goal_problems(
            "(object-in-place O58 T2249)",
            '[["forbidden-edge", "T2249", "T1253"]]',
            None,
        )
        == ""
    )


@pytest.mark.parametrize(
    "constraints, missing",
    [
        ('[["forbidden-poi", "o999"]]', "o999"),
        ('[["forbidden-edge", "t999", "t1253"]]', "t999"),
        ('[["before", "o999", "t1253"]]', "o999"),
        ('[["forbidden-poi", "O999"]]', "o999"),
        ('[["forbidden-edge", "T999", "T1253"]]', "t999"),
        ('[["before", "O999", "T1253"]]', "o999"),
    ],
)
def test_ids_a_constraint_names_are_checked_too(map_db, constraints, missing):
    from heracles_agents.scene_graph_edit import goal_problems

    problem = goal_problems("(visited-place t1253)", constraints, None)
    assert f"{missing} not in the scene graph" in problem
    assert (
        goal_problems(
            "(visited-place t1253)", '[["forbidden-edge", "t2249", "t1253"]]', None
        )
        == ""
    )


def test_no_robot_nodes_means_no_robot_check(map_db):
    from heracles_agents.scene_graph_edit import goal_problems

    map_db.robots = ()
    assert goal_problems("(visited-object-by anyone o58)", "", None) == ""


def test_a_goal_with_a_problem_is_never_held_or_sent(query, map_db):
    from heracles_agents.scene_graph_edit import goal_problems

    send, sent = _sender()
    gate.check_goal = lambda g, c: goal_problems(g, c, None)
    gate.start_turn(user_turn=True)
    out = send("(visited-object-by testbot_am o58)")
    assert out.startswith("NOT SENT: unknown robot testbot_am")
    assert gate.pending is None and sent == []
