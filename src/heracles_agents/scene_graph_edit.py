"""Scene graph edits by language, only when the user asks with "SGET:".

A user message starting with "SGET:" (Scene Graph EdiT) lets the agent edit
the map in that turn, with the edit tools below: add, move or remove an
object; add or remove a place; connect or disconnect two places; undo the
last edit, undo every chat edit, or reset the map to the prior scene graph.
Outside an SGET turn every edit tool refuses, and run_cypher_query only ever
reads: an edit made with free Cypher could not be undone, and a node it made
without the map's attributes could stop the map being published.

Every edit is logged (heracles_ros.change_log, source "sget/chat"), so the
scene change writer can rebuild the map from the prior scene graph without
it: undo and reset are its services, which the chat calls (gate.call_writer).

An edit changes the same map change detection does, so it is announced the
same way: the objects and places it removed, added or moved go out as a scene
change (source "sget/chat") through gate.report, and the planner judges them
against the running plan -- a deleted object the plan picks, or a deleted
place its route crosses, pauses the robots. A connection edit is announced
with no changes, as a new map version.

The edit, its change log entry and the version bump are one transaction
(heracles_ros.map_state.map_edit), refused while the map is being rebuilt or
reloaded. The bump is notify_map_change, so the heracles publisher republishes
the edited map at once and the console's viewers see it too. A plan already running
is not changed by the edit, so after an edit the chat offers to send the last
goal again (resend_offer); the goal manager replans it on the edited map.
"""

import functools
import json
import re
from typing import Iterable

from heracles_agents.dsg_interfaces import HeraclesDsgInterface
from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import register_tool

EDIT_PREFIX = "SGET:"
SGET_SOURCE = "sget/chat"
ADDED, REMOVED, MOVED = 0, 1, 2  # heracles_ros.scene_change

_WRITE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV)\b"
    r"|\bCALL\s+(apoc|db)\.",
    re.IGNORECASE,
)
_LITERAL = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|//[^\n]*")

REFUSED = (
    "Refused: this query would change the scene graph. Map edits run only for a "
    "user message that starts with 'SGET:'. Ask the user to resend the request "
    "as 'SGET: <what to change>'."
)
READ_ONLY = (
    "Refused: run_cypher_query only reads the map. Edit it with the edit tools: "
    "add_scene_object, move_scene_object, remove_scene_object, add_scene_place, "
    "remove_scene_place, connect_scene_places, disconnect_scene_places; undo with "
    "undo_scene_edit. If no tool does the edit, tell the user it is not supported."
)
REBUILDING = (
    "Not done: the map is being rebuilt (an undo or reset is running). Try again "
    "in a few seconds."
)


def is_edit_request(text: str) -> bool:
    return text.lstrip().upper().startswith(EDIT_PREFIX)


def turn_allows_edits(contents: Iterable[str]) -> bool:
    """Whether the user messages starting this agent turn include an SGET request."""
    return any(is_edit_request(c) for c in contents)


def is_write_query(cypher: str) -> bool:
    # Quoted names ('reset area', "Set") and comments are not keywords.
    return bool(_WRITE.search(_LITERAL.sub("''", cypher)))


class EditGate:
    def __init__(self):
        self.allowed = False
        self.wrote = False  # a write ran in this turn
        self.last_goal = None  # (pddl_goal, constraints_json) last sent
        # report(changes, map_version): announce what an edit changed.
        self.report = None
        # call_writer(service) -> (ok, message): a scene change writer service
        # ("undo_last_edit", "clear_edits", "reset").
        self.call_writer = None
        # check_goal(goal, constraints_json) -> why it cannot be planned, or
        # "": robots, regions and ids the map does not have (goal_problems).
        self.check_goal = None
        # Goals are sent only once the user confirmed them: turn counts agent
        # turns, user_turn says whether a user message (not only a scene change
        # report) began this one, and pending is (goal key, turn shown).
        self.turn = 0
        self.user_turn = False
        self.pending = None
        # What a goal was asked for: context() returns {"request": the user's
        # words, "robots": {name: (x, y)}} when a user turn proposes a goal, and
        # the context of the goal last sent is kept, so a later replan can redo
        # the request from where the robots were when it was asked.
        self.context = None
        self._proposed = {}  # goal key -> context
        self.last_context = None

    def start_turn(self, user_turn: bool) -> None:
        self.turn += 1
        self.user_turn = user_turn
        self.allowed = self.wrote = False

    def show(self, goal, constraints) -> None:
        """The user has been shown this goal: their next message may confirm it."""
        key = goal_key(goal, constraints)
        self.pending = (key, self.turn)
        resent = self.last_goal is not None and goal_key(*self.last_goal) == key
        if self.user_turn and not resent and self.context is not None:
            self._proposed[key] = self.context()
        else:
            # Re-proposed after a scene change, or offered again after an edit:
            # still the request the user first made.
            self._proposed[key] = self.last_context


NOT_CONFIRMED = (
    "NOT SENT: the user has not confirmed this goal yet. Show the user this "
    "exact goal and constraints and ask them to confirm. Call the tool again "
    "with the same goal once they say yes."
)


def goal_key(goal: str, constraints) -> tuple:
    """A goal and its constraints, written the same way however they were typed."""
    try:
        facts = json.loads(constraints) if constraints else []
    except ValueError:
        facts = [str(constraints)]
    facts = sorted(json.dumps([str(x).lower() for x in f]) for f in facts)
    return " ".join((goal or "").lower().split()), tuple(facts)


def robot_positions(dsgdb_conf) -> dict:
    """{robot: (x, y)} from the Robot nodes the state updater keeps in Neo4j."""
    with _connect(dsgdb_conf) as db:
        rows = db.query("MATCH (r:Robot) RETURN r.name AS n, r.position AS p")
    return {r["n"]: (r["p"].x, r["p"].y) for r in rows if r["p"] is not None}


def goal_context_note(context) -> str:
    """For a scene change report: what the current goal was asked for, and from where."""
    if not context:
        return ""
    where = ", ".join(
        f"{name} at ({x:.1f}, {y:.1f})"
        for name, (x, y) in sorted(context["robots"].items())
    )
    note = f'The current goal was asked for as: "{context["request"].strip()}".'
    if where:
        note += (
            " If you redo it, measure distances (nearest, closest, ...) from where the "
            f"robots were when it was asked: {where}; not from where they are now."
        )
    return note


def goal_symbols(goal: str) -> list:
    """The object and place ids a PDDL goal names, normalized to lowercase."""
    return sorted(set(re.findall(r"\b[ot]\d+\b", (goal or "").lower())))


# Predicates whose first argument is a robot, and those naming a region last.
_ROBOT_FIRST = re.compile(
    r"\((?:visited-(?:object|place|poi|region)-by|explored-region-by|inspected-by"
    r"|at-place|at-object|holding|object-placed-by)\s+([^\s()]+)",
    re.IGNORECASE,
)
_REGION = re.compile(
    r"\((?:visited|explored)-region(?:-by\s+[^\s()]+)?\s+([^\s()]+(?:\(\d+\))?)",
    re.IGNORECASE,
)


def _region_key(symbol: str) -> str:
    """'R3', 'r3' and 'r(3)' are the same region."""
    return re.sub(r"[()\s]", "", symbol).lower()


def goal_problems(goal: str, constraints_json, dsgdb_conf) -> str:
    """Why the planner would reject this goal on the current map, or "".

    Checks what it can cheaply: robot names against the Robot nodes (one per
    fleet robot, kept by the state updater), region symbols against the Room
    nodes, and object and place ids. A part with nothing to check against (no
    Robot nodes yet) is not checked.
    """
    robots = {r.lower() for r in _ROBOT_FIRST.findall(goal or "")}
    try:
        facts = json.loads(constraints_json) if constraints_json else []
    except ValueError:
        facts = []
    # Ids a constraint names (forbidden-poi, forbidden-edge, before, ...): one
    # the map lacks matches nothing in the planner, so the constraint is lost.
    constraint_ids = set()
    for fact in facts:
        if not isinstance(fact, list) or not fact:
            continue
        if str(fact[0]).lower() == "use-robots":
            robots.update(str(r).lower() for r in fact[1:])
        else:
            constraint_ids.update(goal_symbols(" ".join(str(a) for a in fact[1:])))
    regions = {_region_key(g): g for g in _REGION.findall(goal or "")}
    problems = []
    with _connect(dsgdb_conf) as db:
        if robots:
            rows = db.query("MATCH (r:Robot) RETURN r.name AS n")
            known = sorted({str(r["n"]).lower() for r in rows if r["n"]})
            unknown = sorted(robots - set(known))
            if known and unknown:
                problems.append(
                    f"unknown robot {', '.join(unknown)} (known robots: "
                    f"{', '.join(known)}). Ask the user which robot they mean."
                )
        if regions:
            rows = db.query("MATCH (g:Room) RETURN g.nodeSymbol AS s")
            known = sorted({_region_key(str(r["s"])) for r in rows if r["s"]})
            unknown = sorted(g for k, g in regions.items() if k not in known)
            if unknown and not known:
                problems.append(
                    f"region {', '.join(unknown)} named, but this map has no regions "
                    "(no Room nodes)."
                )
            elif unknown:
                problems.append(
                    f"unknown region {', '.join(unknown)} (regions in the map: "
                    f"{', '.join(known)})."
                )
    missing = missing_symbols(
        sorted(set(goal_symbols(goal)) | constraint_ids), dsgdb_conf
    )
    if missing:
        problems.append(
            f"{', '.join(missing)} not in the scene graph: look the ids up again."
        )
    return " ".join(problems)


def resend_offer(last_goal, missing=()) -> str:
    """What the chat says after an edit: offer to replan the last goal."""
    if last_goal is None:
        return ""
    goal, constraints = last_goal
    with_constraints = f" with constraints {constraints}" if constraints else ""
    if missing:
        return (
            f"The map changed. The last goal sent was {goal}{with_constraints}, "
            f"but {', '.join(missing)} is no longer in the map, so it cannot be "
            "sent again as it is. What should the robot do instead?"
        )
    return (
        f"The map changed. The last goal sent was {goal}{with_constraints}. "
        "A plan already running does not change with the map: shall I send "
        "this goal again so the robot replans on the edited map?"
    )


gate = EditGate()


def _connect(dsgdb_conf):
    from heracles.query_interface import Neo4jWrapper

    auth = (
        dsgdb_conf.username.get_secret_value(),
        dsgdb_conf.password.get_secret_value(),
    )
    return Neo4jWrapper(dsgdb_conf.uri, auth, atomic_queries=True)


def missing_symbols(symbols, dsgdb_conf) -> list:
    """Which of these goal ids (o16, t4) the map no longer has."""
    if not symbols:
        return []
    with _connect(dsgdb_conf) as db:
        rows = db.query(
            "MATCH (n) WHERE (n:Object OR n:MeshPlace) AND toLower(n.nodeSymbol) IN "
            f"{[s.lower() for s in symbols]} RETURN toLower(n.nodeSymbol) AS s"
        )
    have = {r["s"] for r in rows}
    return [s for s in symbols if s.lower() not in have]


def _map_problem(dsgdb_conf) -> str:
    """Why the scene graph cannot be built from the database, or ""."""
    from heracles.graph_interface import db_to_spark_dsg

    with _connect(dsgdb_conf) as db:
        try:
            db_to_spark_dsg(db)
        except Exception as exc:
            return str(exc)
    return ""


def broken_map_warning(problem: str) -> str:
    return (
        "WARNING: after this edit the scene graph cannot be built, so planners "
        f"keep the old map: {problem} Fix or undo the edit now, then check again."
    )


def gated(query_fn):
    """run_cypher_query's function, read-only: map edits go through the edit tools.

    Edits made with free Cypher could not be logged, so they could not be
    undone, and a missing attribute could stop the map being published.
    """

    @functools.wraps(query_fn)
    def run(cypher_string, dsgdb_conf=None):
        if is_write_query(cypher_string):
            return READ_ONLY if gate.allowed else REFUSED
        return query_fn(cypher_string, dsgdb_conf=dsgdb_conf)

    return run


def _editing(dsgdb_conf, edit) -> str:
    """Run one map edit as SGET: refuse outside SGET, log it, bump, announce, check.

    edit(db) returns (message, changes); changes None means nothing was
    changed and message says why. changes are applied records (objects or
    places) to announce; [] announces just the new version.
    """
    if not gate.allowed:
        return REFUSED
    from heracles_ros.map_state import map_edit, notify_map_change

    with _connect(dsgdb_conf) as db:
        with map_edit(db) as tx:
            if tx is None:
                return REBUILDING
            message, changes = edit(tx)
            if changes is None:
                return message
            version = notify_map_change(tx)
    gate.wrote = True
    if gate.report:
        gate.report(changes, version)
    problem = _map_problem(dsgdb_conf)
    done = f"{message} (map_version {version}). It can be undone with undo_scene_edit."
    return f"{done}\n{broken_map_warning(problem)}" if problem else done


def _place_center(db, symbol):
    rows = db.execute(
        "MATCH (p:MeshPlace) WHERE toLower(p.nodeSymbol) = $s RETURN p.center AS c",
        s=(symbol or "").strip().lower(),
    )[0]
    c = rows[0]["c"] if rows else None
    return None if c is None else (c.x, c.y, c.z)


def _place_change(kind, symbol, old=None, new=None) -> dict:
    """A place edit as an applied record, announced like an object change."""
    return {"kind": kind, "symbol": symbol, "label": "place", "old": old, "new": new}


def _record(db, op, data):
    from heracles_ros.change_log import record

    record(db, SGET_SOURCE, op, data)


def _object_symbol(symbol: str) -> str:
    s = (symbol or "").strip()
    return s[:1].upper() + s[1:]


def add_scene_object(
    object_class: str,
    x: float,
    y: float,
    z: float = 0.0,
    dsgdb_conf: HeraclesDsgInterface = None,
) -> str:
    """Add an object to the scene graph, as change detection would (SGET only)."""

    def edit(db):
        from heracles_ros.change_log import record_applied
        from heracles_ros.scene_change import add_object

        added = add_object(db, "", object_class, (x, y, z))
        record_applied(db, SGET_SOURCE, [added])
        sym = added["symbol"]
        rows = db.query(
            f"MATCH (p:MeshPlace)-[:CONTAINS]->(o:Object {{nodeSymbol: '{sym}'}}) "
            "RETURN p.nodeSymbol AS p"
        )
        place = f", in place {rows[0]['p']}" if rows else ""
        return f"Added {sym} ({object_class}) at ({x}, {y}, {z}){place}", [added]

    return _editing(dsgdb_conf, edit)


add_object_tool = ToolDescription(
    name="add_scene_object",
    description=(
        "Add an object to the scene graph at a map position. Only for a user "
        "message starting with 'SGET:'. The object gets the next free id and is "
        "contained by its nearest MeshPlace."
    ),
    parameters=[
        FunctionParameter("object_class", str, "Class of the object, e.g. 'chair'"),
        FunctionParameter("x", float, "x position in the map frame (m)"),
        FunctionParameter("y", float, "y position in the map frame (m)"),
        FunctionParameter("z", float, "z position (m), default 0", required=False),
    ],
    function=add_scene_object,
)


def move_scene_object(
    symbol: str,
    x: float,
    y: float,
    z: float = None,
    dsgdb_conf: HeraclesDsgInterface = None,
) -> str:
    """Move an object to a map position (SGET only)."""

    def edit(db):
        from heracles_ros.change_log import record_applied
        from heracles_ros.scene_change import holder_of, move_object, object_center

        sym = _object_symbol(symbol)
        old = object_center(db, sym)
        if old is None:
            return f"Refused: there is no object {sym}.", None
        if holder_of(db, sym):
            return f"Refused: {sym} is held by {holder_of(db, sym)}.", None
        moved = move_object(db, sym, (x, y, old[2] if z is None else z))
        record_applied(db, SGET_SOURCE, [moved])
        return f"Moved {sym} from ({old[0]:.1f}, {old[1]:.1f}) to ({x}, {y})", [moved]

    return _editing(dsgdb_conf, edit)


move_object_tool = ToolDescription(
    name="move_scene_object",
    description=(
        "Move an object in the scene graph to a map position. Only for a user "
        "message starting with 'SGET:'."
    ),
    parameters=[
        FunctionParameter("symbol", str, "The object's id, e.g. 'O15'"),
        FunctionParameter("x", float, "new x position in the map frame (m)"),
        FunctionParameter("y", float, "new y position in the map frame (m)"),
        FunctionParameter(
            "z", float, "new z (m); default: keep its height", required=False
        ),
    ],
    function=move_scene_object,
)


def remove_scene_object(symbol: str, dsgdb_conf: HeraclesDsgInterface = None) -> str:
    """Remove an object from the scene graph (SGET only)."""

    def edit(db):
        from heracles_ros.change_log import record_applied
        from heracles_ros.scene_change import holder_of, object_center, remove_object

        sym = _object_symbol(symbol)
        if object_center(db, sym) is None:
            return f"Refused: there is no object {sym}.", None
        if holder_of(db, sym):
            return f"Refused: {sym} is held by {holder_of(db, sym)}.", None
        removed = remove_object(db, sym)
        record_applied(db, SGET_SOURCE, [removed])
        return f"Removed {sym} ({removed['label'] or 'unknown class'})", [removed]

    return _editing(dsgdb_conf, edit)


remove_object_tool = ToolDescription(
    name="remove_scene_object",
    description=(
        "Remove an object from the scene graph. Only for a user message starting "
        "with 'SGET:'."
    ),
    parameters=[FunctionParameter("symbol", str, "The object's id, e.g. 'O15'")],
    function=remove_scene_object,
)


def add_scene_place(
    x: float,
    y: float,
    connect_to: str,
    symbol: str = "",
    dsgdb_conf: HeraclesDsgInterface = None,
) -> str:
    """Add a traversable place, connected both ways to the places named (SGET only)."""
    neighbours = [n.strip().lower() for n in connect_to.split(",") if n.strip()]
    symbol = symbol.strip().lower()
    if gate.allowed and symbol and not re.fullmatch(r"t\d+", symbol):
        return f"Refused: a place id is t followed by a number, not '{symbol}'."

    def edit(db):
        from heracles_ros.scene_change import add_place, place_exists

        if symbol and place_exists(db, symbol):
            return f"Refused: place {symbol} already exists.", None
        missing = [n for n in neighbours if not place_exists(db, n)]
        if not neighbours or missing:
            return (
                "Refused: connect_to must name existing places; missing: "
                f"{missing or 'none given'}.",
                None,
            )
        added = add_place(db, symbol, x, y, neighbours)
        _record(db, "add_place", added)
        return (
            f"Added place {added['symbol']} at ({x}, {y}), connected to "
            f"{', '.join(neighbours)}",
            [_place_change(ADDED, added["symbol"], new=(float(x), float(y), 0.0))],
        )

    return _editing(dsgdb_conf, edit)


add_place_tool = ToolDescription(
    name="add_scene_place",
    description=(
        "Add a traversable place (MeshPlace) at a map position, connected both "
        "ways to existing places. Only for a user message starting with 'SGET:'."
    ),
    parameters=[
        FunctionParameter("x", float, "x position in the map frame (m)"),
        FunctionParameter("y", float, "y position in the map frame (m)"),
        FunctionParameter(
            "connect_to",
            str,
            "Comma-separated places to connect it to, e.g. 't1104, t1405'",
        ),
        FunctionParameter(
            "symbol",
            str,
            "Id for the new place, e.g. 't999'; empty for the next free one",
            required=False,
        ),
    ],
    function=add_scene_place,
)


def remove_scene_place(symbol: str, dsgdb_conf: HeraclesDsgInterface = None) -> str:
    """Delete a place and all its connections (SGET only)."""

    def edit(db):
        from heracles_ros.scene_change import remove_place

        was_at = _place_center(db, symbol)
        removed = remove_place(db, symbol)
        if removed is None:
            return f"Refused: there is no place {symbol.strip().lower()}.", None
        _record(db, "remove_place", {"symbol": removed["symbol"]})
        moved = (
            f"; its objects {', '.join(removed['objects'])} now belong to the nearest place"
            if removed["objects"]
            else ""
        )
        return (
            f"Deleted place {removed['symbol']} and its {removed['edges']} edges{moved}",
            [_place_change(REMOVED, removed["symbol"], old=was_at)],
        )

    return _editing(dsgdb_conf, edit)


remove_place_tool = ToolDescription(
    name="remove_scene_place",
    description=(
        "Delete a place (MeshPlace) with all its connections, e.g. where the "
        "robot can no longer go. Only for a user message starting with 'SGET:'."
    ),
    parameters=[FunctionParameter("symbol", str, "The place's id, e.g. 't444'")],
    function=remove_scene_place,
)


def connect_scene_places(
    place_a: str, place_b: str, dsgdb_conf: HeraclesDsgInterface = None
) -> str:
    """Connect two places both ways (SGET only)."""

    def edit(db):
        from heracles_ros.scene_change import add_connection

        done = add_connection(db, place_a, place_b)
        if done is None:
            return (
                f"Refused: {place_a} and {place_b} must be two existing places that "
                "are not connected yet.",
                None,
            )
        _record(db, "add_connection", done)
        return f"Connected {done['a']} and {done['b']}", []

    return _editing(dsgdb_conf, edit)


connect_places_tool = ToolDescription(
    name="connect_scene_places",
    description=(
        "Connect two existing places, both ways. Only for a user message starting "
        "with 'SGET:'."
    ),
    parameters=[
        FunctionParameter("place_a", str, "A place id, e.g. 't444'"),
        FunctionParameter("place_b", str, "The other place id"),
    ],
    function=connect_scene_places,
)


def disconnect_scene_places(
    place_a: str, place_b: str, dsgdb_conf: HeraclesDsgInterface = None
) -> str:
    """Remove the connection between two places, both ways (SGET only)."""

    def edit(db):
        from heracles_ros.scene_change import remove_connection

        done = remove_connection(db, place_a, place_b)
        if done is None:
            return f"Refused: {place_a} and {place_b} are not connected.", None
        _record(db, "remove_connection", done)
        return f"Disconnected {done['a']} and {done['b']}", []

    return _editing(dsgdb_conf, edit)


disconnect_places_tool = ToolDescription(
    name="disconnect_scene_places",
    description=(
        "Remove the connection between two places, both ways, e.g. a blocked "
        "passage. Only for a user message starting with 'SGET:'."
    ),
    parameters=[
        FunctionParameter("place_a", str, "A place id, e.g. 't444'"),
        FunctionParameter("place_b", str, "The other place id"),
    ],
    function=disconnect_scene_places,
)


def _call_writer(service: str) -> str:
    """Ask the scene change writer to undo or reset (it rebuilds the map)."""
    if not gate.allowed:
        return REFUSED
    if gate.call_writer is None:
        return "Not available: the chat is not connected to the scene change writer."
    ok, message = gate.call_writer(service)
    if ok:
        gate.wrote = True
    return message if ok else f"Failed: {message}"


def undo_scene_edit() -> str:
    """Undo the latest map edit made from the chat (SGET only)."""
    return _call_writer("undo_last_edit")


def clear_scene_edits() -> str:
    """Undo every map edit made from the chat (SGET only)."""
    return _call_writer("clear_edits")


def reset_scene_graph() -> str:
    """Return the map to the prior scene graph: every change dropped (SGET only)."""
    return _call_writer("reset")


undo_edit_tool = ToolDescription(
    name="undo_scene_edit",
    description=(
        "Undo the latest scene graph edit made from this chat. Only for a user "
        "message starting with 'SGET:'."
    ),
    parameters=[],
    function=undo_scene_edit,
)
clear_edits_tool = ToolDescription(
    name="clear_scene_edits",
    description=(
        "Undo every scene graph edit made from this chat; changes found by change "
        "detection stay. Only for a user message starting with 'SGET:'."
    ),
    parameters=[],
    function=clear_scene_edits,
)
reset_tool = ToolDescription(
    name="reset_scene_graph",
    description=(
        "Return the scene graph to the original prior map: every edit and every "
        "change detection result is dropped, and the change detection base "
        "station is cleared. Only for a user message starting with 'SGET:'."
    ),
    parameters=[],
    function=reset_scene_graph,
)

EDIT_TOOLS = [
    add_object_tool,
    move_object_tool,
    remove_object_tool,
    add_place_tool,
    remove_place_tool,
    connect_places_tool,
    disconnect_places_tool,
    undo_edit_tool,
    clear_edits_tool,
    reset_tool,
]


def confirming(send_fn):
    """send_pddl_goal_with_constraints's function, sending only confirmed goals.

    A goal goes out only if it was shown to the user in an earlier turn and a
    user message began this one; otherwise it is held as shown and the agent
    is told to ask. So nothing is sent from a scene change report, or in the
    turn that first proposes a goal.
    """

    @functools.wraps(send_fn)
    def send(pddl_goal_string, constraints_json="", **kwargs):
        # kwargs are the tool's bound args (robot_name, planner_topic,
        # robot_name_env_var, ...), passed through as the config gives them.
        key = goal_key(pddl_goal_string, constraints_json)
        problem = ""
        if gate.check_goal is not None:
            try:
                problem = gate.check_goal(pddl_goal_string, constraints_json)
            except Exception:
                problem = ""  # Neo4j unavailable: the planner still checks
        if problem:
            return f"NOT SENT: {problem}"
        confirmed = (
            gate.user_turn
            and gate.pending is not None
            and gate.pending[0] == key
            and gate.pending[1] < gate.turn
        )
        if not confirmed:
            gate.show(pddl_goal_string, constraints_json)
            return NOT_CONFIRMED
        gate.pending = None
        gate.last_context = gate._proposed.get(key, gate.last_context)
        result = send_fn(pddl_goal_string, constraints_json=constraints_json, **kwargs)
        gate.last_goal = (pddl_goal_string, constraints_json)
        return result

    return send


def install() -> None:
    """Make Cypher read-only, add the edit tools, send only confirmed goals.

    Call before the LlmAgent is built.
    """
    import heracles_agents.tools.cypher_query_tool  # noqa: F401  registers it
    import heracles_agents.tools.pddl_repair_tool  # noqa: F401
    from heracles_agents.tool_registry import ToolRegistry

    tool = ToolRegistry.tools["run_cypher_query"]
    if not getattr(tool.function, "_sget_gated", False):
        tool.function = gated(tool.function)
        tool.function._sget_gated = True
    for edit_tool in EDIT_TOOLS:
        register_tool(edit_tool)
    tool = ToolRegistry.tools["send_pddl_goal_with_constraints"]
    if not getattr(tool.function, "_sget_confirms", False):
        tool.function = confirming(tool.function)
        tool.function._sget_confirms = True
