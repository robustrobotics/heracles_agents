"""The repair prompts' example goals use the predicates with the right shape."""

import os
import re

import yaml

COMMON = os.path.join(os.path.dirname(__file__), "..", "examples", "prompts", "common")
EXAMPLES = [
    "pddl_in_context_examples_with_constraints.yaml",
    "pddl_in_context_examples_grstaps.yaml",
]
_GOAL = re.compile(r"pddl_goal_string=(\(.*?\))(?: and constraints_json|$)")


def _goals(name):
    with open(os.path.join(COMMON, name)) as f:
        examples = yaml.safe_load(f)["in_context_examples"]
    for ex in examples:
        for goal in _GOAL.findall(ex["assistant"].replace("\\n", "\n")):
            yield ex["user"], goal


def _uses(goal, predicate):
    return re.findall(rf"\({predicate}\s+([^()]*)\)", goal)


def test_object_placed_by_takes_a_robot_then_an_object_then_a_place():
    for name in EXAMPLES:
        found = 0
        for _, goal in _goals(name):
            for args in _uses(goal, "object-placed-by"):
                robot, obj, place = args.split()
                assert not re.fullmatch(r"[ot]\d+", robot), goal
                assert re.fullmatch(r"o\d+", obj) and re.fullmatch(r"t\d+", place)
                found += 1
        assert found, f"{name} has no object-placed-by example"


def test_object_placed_by_only_when_the_user_names_the_mover():
    for name in EXAMPLES:
        for user, goal in _goals(name):
            for args in _uses(goal, "object-placed-by"):
                assert args.split()[0] in user.lower(), (user, goal)
            # A robot named only for a visit does not make the move its own.
            if "look at" in user.lower():
                assert "object-placed-by" not in goal and "object-in-place" in goal


def test_no_example_uses_the_base_station_as_a_robot():
    for name in EXAMPLES:
        for _, goal in _goals(name):
            assert "willow" not in goal
