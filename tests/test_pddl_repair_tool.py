"""What the goal tool tells the agent after sending a goal."""

from heracles_agents.tools import pddl_repair_tool


def test_the_reply_does_not_call_the_planner_host_a_robot(monkeypatch):
    sent = []
    monkeypatch.setattr(pddl_repair_tool.subprocess, "run", sent.append)
    monkeypatch.setenv("ADT4_ROBOT_NAME", "willow")
    reply = pddl_repair_tool.send_pddl_with_constraints(
        "(visited-object-by hamilton o91)",
        planner_topic="/willow/commanded_goal",
        robot_name_env_var="ADT4_ROBOT_NAME",
    )
    assert "robot_id: 'willow'" in " ".join(sent[0])
    assert reply == (
        "Sent goal (visited-object-by hamilton o91) to the planner on willow; "
        "the goal decides which robots act"
    )
