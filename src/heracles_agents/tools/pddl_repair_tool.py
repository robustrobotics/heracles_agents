"""PDDL goal + constraints tool for execution-time plan repair.

Publishes a ConstrainedPddlGoalMsg to the goal_manager's input topic so the
omniplanner pipeline can ground the goal with runtime constraints applied.
"""

import json
import os
import subprocess

from heracles_agents.tool_interface import FunctionParameter, ToolDescription
from heracles_agents.tool_registry import register_tool


def _default_planner_topic() -> str:
    """Topic the goal_manager listens on, derived from the running session.

    goal_manager remaps ~/commanded_goal to /$ADT4_ROBOT_NAME/commanded_goal, so
    the topic namespace is the namespace of the session the planner runs in --
    on a base station that is the base station's name, not the robot's.
    """
    session_name = os.environ.get("ADT4_ROBOT_NAME")
    if not session_name:
        raise ValueError(
            "planner_topic was not bound and ADT4_ROBOT_NAME is unset, so the "
            "goal_manager topic cannot be derived. Set ADT4_ROBOT_NAME, or bind "
            "planner_topic explicitly in the agent config."
        )
    return f"/{session_name}/commanded_goal"


def _default_robot_name() -> str:
    """Robot the goal is issued to, i.e. goal.robot_id.

    This is not always the session name: on a base station the planner session is
    named for the base station while the goal targets an actual robot, which is
    what ADT4_EXECUTOR_ROBOT_NAME carries. In a single-session sim run the two
    coincide, so fall back to ADT4_ROBOT_NAME.
    """
    robot = os.environ.get("ADT4_EXECUTOR_ROBOT_NAME") or os.environ.get(
        "ADT4_ROBOT_NAME"
    )
    if not robot:
        raise ValueError(
            "robot_name was not bound and neither ADT4_EXECUTOR_ROBOT_NAME nor "
            "ADT4_ROBOT_NAME is set, so the target robot is unknown. Set one of "
            "them, or bind robot_name explicitly in the agent config."
        )
    return robot


def _constraints_from_json(constraints_json: str) -> str:
    """Turn a JSON list like [["forbidden-poi","o1"],["forbidden-edge","p1","p2"]]
    into a YAML fragment for ros2 topic pub:

        [{predicate: 'forbidden-poi', symbols: ['o1']},
         {predicate: 'forbidden-edge', symbols: ['p1','p2']}]
    """
    if not constraints_json:
        return "[]"
    try:
        facts = json.loads(constraints_json)
    except Exception as exc:
        raise ValueError(f"constraints_json is not valid JSON: {exc}") from exc

    yaml_parts = []
    for fact in facts:
        if not isinstance(fact, (list, tuple)) or len(fact) < 1:
            continue
        predicate = fact[0]
        symbols = [str(s) for s in fact[1:]]
        symbols_yaml = "[" + ",".join(f"'{s}'" for s in symbols) + "]"
        yaml_parts.append(f"{{predicate: '{predicate}', symbols: {symbols_yaml}}}")
    return "[" + ",".join(yaml_parts) + "]"


def send_pddl_with_constraints(
    pddl_goal_string: str,
    constraints_json: str = "",
    robot_name: str = None,
    planner_topic: str = None,
):
    # Both stay overridable from the agent config; unbound they follow the
    # environment, so a config is not pinned to whichever robot it was written
    # against.
    if robot_name is None:
        robot_name = _default_robot_name()
    if planner_topic is None:
        planner_topic = _default_planner_topic()

    constraints_yaml = _constraints_from_json(constraints_json)
    msg_yaml = (
        f"{{goal: {{robot_id: '{robot_name}', pddl_goal: '{pddl_goal_string}'}}, "
        f"constraints: {constraints_yaml}}}"
    )

    cmd = [
        "ros2",
        "topic",
        "pub",
        planner_topic,
        "omniplanner_msgs/msg/ConstrainedPddlGoalMsg",
        msg_yaml,
        "-1",
    ]
    print(f"[pddl_repair_tool] cmd: {' '.join(cmd)}")
    subprocess.run(cmd)

    result = f"Sent goal {pddl_goal_string} to robot {robot_name}"
    if constraints_json:
        result += f" with constraints {constraints_json}"
    return result


pddl_repair_tool = ToolDescription(
    name="send_pddl_goal_with_constraints",
    description=(
        "Send a PDDL goal to a robot, optionally with runtime constraints. "
        "Use the constraints parameter to forbid the robot from visiting "
        "specific locations or traversing specific edges. Constraints are "
        "persistent — they accumulate across calls until cleared. "
        "Please ask the user for confirmation before sending."
    ),
    parameters=[
        FunctionParameter(
            "pddl_goal_string", str, "A PDDL goal string, e.g. '(visited-place p23778)'"
        ),
        FunctionParameter(
            "constraints_json",
            str,
            (
                "JSON list of constraint facts. Each fact is a list like "
                '[["forbidden-poi", "o188"]] to forbid a location, or '
                '[["forbidden-edge", "p1", "p2"]] to forbid an edge. '
                "Empty string means no new constraints."
            ),
            required=False,
        ),
    ],
    function=send_pddl_with_constraints,
)

register_tool(pddl_repair_tool)
