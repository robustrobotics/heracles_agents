"""heracles_agents.prompts: the installed, single source of the plan-repair chat's
planning knowledge (the repair chats in examples/ and the ADT4 web console)."""

import json
import os
import re

import pytest

from heracles_agents import prompts
from heracles_agents.prompt import PromptSettings

ROOT = os.path.join(os.path.dirname(__file__), "..")
CHATS = {
    "fast_downward": os.path.join(
        ROOT, "examples", "chatdsg_repair", "agent_prompt.yaml"
    ),
}
_ROBOT_FIRST = re.compile(
    r"\((?:visited-object-by|visited-place-by|inspected-by|visited-region-by|"
    r"explored-region-by|object-placed-by|holding|at-place|at-object)\s+([a-z]\w*)"
)


def _rules_in(text, engine):
    """The required goal rules, as phrases of the rule text."""
    assert "the goal MUST name that robot" in text
    for form in (
        "visited-object-by",
        "visited-place-by",
        "inspected-by",
        "object-placed-by",
    ):
        assert form in text, form
    assert "use the robot-free forms" in text
    for form in ("(visited-object o)", "(visited-place t)", "(object-in-place o t)"):
        assert form in text, form
    assert "let the planner allocate" in text
    assert "Never pick a robot yourself" in text
    assert "not the nearest robot" in text and "not the selected robot" in text
    assert "Never split the targets" in text
    assert '"Visit all cones" (cones O1 ... O15) is ONE goal' in text
    assert (
        "(and (visited-object o1) (visited-object o2) ... (visited-object o15))" in text
    )
    assert "only when the user explicitly asks for it" in text  # the nearest robot
    assert "One goal per request covers the whole team" in text
    assert "A new goal replaces the old one" in text
    assert "re-send the tasks of the last goal that are still wanted" in text
    # the only ask left about robots: a name that is not in the fleet
    assert "that is not in it" in text and "ask which robot they mean" in text
    assert "without naming a robot you know, ask" not in text
    assert "then give the task to that robot with a -by predicate" not in text
    assert "plus a use-robots constraint naming exactly" in text
    assert '"Visit all the chairs with both euclid and hamilton' in text
    assert "no use-robots constraint" not in text
    if engine == "fast_downward":
        assert "holding / at-place / at-object" in text


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_team_goal_rules(engine):
    _rules_in(prompts.team_goal_rules(engine), engine)


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_repair_goal_prompt_has_rules_domain_and_examples(engine):
    text = prompts.repair_goal_prompt(engine)
    _rules_in(text, engine)
    assert "<PDDL Domain>" in text and "<Runtime Constraints>" in text
    assert "<Examples>" in text and "User: Instruction:" in text
    # "Put X at Y" (heracles_agents #8): the place nearest Y, never X's own place
    assert "MeshPlace closest to Y" in text
    assert "X is already at Y: send no goal" in text


def test_repair_goal_prompt_renames_tools_for_another_chat():
    text = prompts.repair_goal_prompt(
        "fast_downward", names={"send_pddl_goal_with_constraints": "propose_goal"}
    )
    assert "send_pddl_goal_with_constraints" not in text and "propose_goal" in text


def test_unknown_engine_is_an_error():
    with pytest.raises(ValueError):
        prompts.repair_goal_prompt("fd")


def test_descriptions():
    assert "MESH PLACES" in prompts.scene_graph_description()
    assert "ROOMS" in prompts.scene_graph_description("hierarchical")
    assert "point.distance" in prompts.cypher_interface_description()
    assert os.path.isfile(prompts.path("team_goal_rules.yaml"))


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_chatdsg_repair_prompt_comes_from_the_package(engine, monkeypatch):
    monkeypatch.setenv("HERACLES_EVALUATION_PATH", ROOT)
    prompt = PromptSettings(base_prompt=CHATS[engine]).base_prompt
    _rules_in(prompt.system, engine)
    assert prompt.system == prompts.repair_chat_prompt(engine)["system"]
    assert "<<" not in prompt.system  # every marker filled
    assert prompt.domain_description == prompts.domain_description(engine)
    assert prompt.interface_description == prompts.cypher_interface_description()
    assert prompt.scene_graph_description == prompts.scene_graph_description()
    assert len(prompt.in_context_examples) == len(prompts.in_context_examples(engine))
    assert prompt.labelspace_description  # the example's own labelspace
    assert prompt.novel_instruction_template == "{question}"
    # the repair chat's own parts stay
    assert "SCENE CHANGE REPORTS" in prompt.system and "SGET" in prompt.system
    assert 'returns "NOT SENT"' in prompt.system
    if engine == "fast_downward":
        assert '["use-robots", "<name>", ...]' in prompt.system
        assert "This graph has no Room nodes" in prompt.system


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_examples_only_name_robots_the_user_named(engine):
    """No example picks a robot itself (the old "hilbert picks it up" one did). A
    follow-up ("Also have ...") re-sends robots of the examples before it."""
    earlier = ""
    for ex in prompts.in_context_examples(engine):
        user = ex["user"].lower()
        if user.strip().startswith("instruction: also"):
            user += earlier
        earlier += ex["user"].lower()
        for goal in re.findall(
            r"pddl_goal_string=(\(.*?\))(?: and|$)", ex["assistant"]
        ):
            for robot in _ROBOT_FIRST.findall(goal):
                assert robot in user, (ex["user"], goal)


_STAY_WORDS = ('"be at"', '"stay at"', '"wait at"', '"park at"', '"stop at"', '"block"')
_AT_EXAMPLES = (
    '- "Euclid, be at O20": (at-object euclid o20)',
    '- "Hamilton, wait at t7": (at-place hamilton t7)',
    '- "Euclid visits O8 then stays at t12": '
    "(and (visited-object-by euclid o8) (at-place euclid t12))",
)
_GO_TO_O21 = (
    '- "Hamilton, go to O21. Gauss, inspect O26": '
    "(and (visited-object-by hamilton o21) (inspected-by gauss o26))"
)


def _fast_downward_at_rules_in(text):
    """Fast Downward: "be / stay / wait at" is at-place / at-object; "go to" is not."""
    assert "use (at-place <robot> <t>) / (at-object <robot> <o>)" in text
    for word in _STAY_WORDS:
        assert word in text, word
    assert "If the user explicitly asks for the at predicate, use it." in text
    assert "at-place / at-object always need a robot" in text
    assert "ask which robot should end there" in text
    assert '"Go to", "visit", "check" or "look at"' in text
    for example in _AT_EXAMPLES:
        assert example in text, example
    assert _GO_TO_O21 in text  # "go to" stays visited-object-by


def test_fast_downward_rules_map_end_or_stay_to_at_predicates():
    _fast_downward_at_rules_in(prompts.team_goal_rules("fast_downward"))


def test_chatdsg_repair_has_the_at_rules():
    _fast_downward_at_rules_in(prompts.repair_chat_prompt("fast_downward")["system"])


def test_fast_downward_domain_uses_at_for_end_or_stay():
    text = prompts.domain_description("fast_downward")
    assert "(at-place ?r ?t): Robot ?r must END at place ?t" in text
    assert "(at-object ?r ?o): Robot ?r must END at object ?o" in text
    assert 'use "at-" predicates for the final destination' in text
    assert "when the user wants a robot to end or stay somewhere" in text


# ------------------------------------- Fast Downward text against 095df69
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures",
                       "fast_downward_prompts_095df69.json")  # fmt: skip


# The two team goal rules the 2026-11-12 follow-up rewrote (use-robots; robot
# names and the fleet). Every other Fast Downward line is as it was at 095df69.
_REWRITTEN_RULES = ('- "Both robots", "hamilton and euclid"', "- Robot names are lowercase")


def test_fast_downward_text_is_as_at_095df69_but_for_the_rewritten_rules():
    """Every Fast Downward text was as at 095df69 when the fixture was added
    (the fixture is that commit's rendered text). Since then
    only the team goal rules changed: two rules were rewritten and rules and
    examples were added; the domain and the in-context examples are untouched."""
    with open(FIXTURE, encoding="utf-8") as f:
        before = json.load(f)
    e = "fast_downward"
    assert prompts.domain_description(e) == before["domain_description"]
    assert prompts.in_context_examples(e) == before["in_context_examples"]
    now = prompts.team_goal_rules(e).splitlines()
    kept = [
        line
        for line in before["team_goal_rules"].splitlines()
        if not line.startswith(_REWRITTEN_RULES)
    ]
    assert len(kept) == len(before["team_goal_rules"].splitlines()) - 2
    assert [line for line in kept if line not in now] == []
    # and in the same order
    assert [line for line in now if line in kept] == kept
    for text in (prompts.repair_goal_prompt(e), prompts.repair_chat_prompt(e)["system"]):
        assert prompts.team_goal_rules(e) in text


# ---------- 2026-11-11 rig test: wrong or odd constraints and robot lists ----------


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_rules_for_several_robots_one_target_and_for_all_robots(engine):
    text = prompts.team_goal_rules(engine)
    # (a) the same target for several robots: one -by atom per robot
    assert "one -by atom per robot, in one goal" in text
    assert (
        '- "Hamilton and euclid, go to O3": '
        "(and (visited-object-by hamilton o3) (visited-object-by euclid o3))"
    ) in text
    assert "is done as soon as ONE robot gets there" in text
    # use-robots stays for work shared out over several targets and for "only X"
    assert "for work to share out over several targets" in text
    assert '"only X" / "just X"' in text
    assert '- "Visit all the chairs with both euclid and hamilton' in text
    # (b) "all robots" does not limit the team; no husky unless named
    assert "does not limit the robots: leave use-robots out" in text
    assert "Never put a husky in use-robots" in text
    assert "unless the user names it" in text
    assert "Tell the user which robots you used" in text
    assert (
        '- "All robots visit all the trash cans (trash cans O4, O8, O15)": '
        "(and (visited-object o4) (visited-object o8) (visited-object o15))"
    ) in text
    # (c) one robot list; an offline robot is not an unknown one
    assert "The fleet is ONE list: the fleet roster in this prompt when it has one" in text
    assert 'never call it unknown or "not in the fleet"' in text
    assert "are the robots of the fleet (MATCH (r:Robot)" not in text
    # (d) a concrete choice per robot, not an enumeration
    assert "one specific target for each named robot, the nearest first" in text
    assert (
        '- "Send hamilton to a trash can and euclid to a different one '
        '(trash cans O4, O8, O15)": '
        "(and (visited-object-by hamilton o4) (visited-object-by euclid o8))"
    ) in text
    # (e) the console's constraint form, stated once
    assert text.count("{predicate, symbols}") == 1
    assert "{predicate: 'use-robots', symbols: ['euclid', 'hamilton']}" in text


def test_only_fast_downward_is_told_about_its_one_or():
    assert "the only or is the two-robots-two-objects case" in prompts.team_goal_rules(
        "fast_downward"
    )


@pytest.mark.parametrize("engine", prompts.ENGINES)
def test_renaming_the_tools_keeps_the_constraint_forms_apart(engine):
    """The console renames constraints_json to constraints: the rule must not
    then say the console's parameter takes the JSON-list form."""
    text = prompts.repair_goal_prompt(
        engine,
        names={
            "send_pddl_goal_with_constraints": "propose_goal",
            "constraints_json": "constraints",
        },
    )
    rule = next(line for line in text.splitlines() if "{predicate, symbols}" in line)
    assert "constraints takes" not in rule and "constraints_json" not in rule
