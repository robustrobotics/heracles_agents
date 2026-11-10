#!/usr/bin/env python3
"""ChatDSG with plan repair tool support.

Also listens for scene changes (heracles_agents.scene_change_listener): each
one is put to the agent as a report, so it can recheck whether the change
affects what the user asked for. Reports and user messages take turns: one
arriving while the agent is busy waits, and waiting ones go in together.

The scene graph is edited only for a message starting with "SGET:", with the
edit tools; Cypher queries only read (heracles_agents.scene_graph_edit). Edits
are logged, so they can be undone, or the map reset to the prior scene graph.
After an edit the chat always offers to send the last goal again, so the robot
replans.
"""

# Register the repair tool before the agent loads
# Import the app class from the original chatdsg
import importlib.util
import os
import threading

import yaml
from rich.markup import escape
from textual.widgets import RichLog

import heracles_agents.tools.pddl_repair_tool  # noqa: F401
from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext
from heracles_agents.scene_change_listener import REPORT_PREFIX, SceneChangeListener
from heracles_agents.scene_graph_edit import (
    gate,
    goal_context_note,
    goal_symbols,
    install,
    missing_symbols,
    resend_offer,
    robot_positions,
    turn_allows_edits,
)

original = os.path.join(os.path.dirname(__file__), "..", "chatdsg", "chatdsg.py")
spec = importlib.util.spec_from_file_location("chatdsg_original", original)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

install()  # gate map writes on SGET, before the agent binds its tools


class RepairChatApp(mod.InputDisplayApp):
    def __init__(self, agent):
        super().__init__(agent)
        self._busy = False
        self._waiting = []
        self._listener = SceneChangeListener(
            lambda text: self.call_from_thread(self.post_scene_change, text)
        )

    def on_mount(self) -> None:
        self._listener.start()
        # Map edits go out as scene changes, so the planner checks the plan.
        gate.report = self._listener.publish_applied
        # Undo and reset are rebuilt by the scene change writer.
        gate.call_writer = self._listener.call_writer

    def on_unmount(self) -> None:
        self._listener.stop()

    def action_submit(self) -> None:
        """Called when ctrl+b is pressed."""
        box = self.query_one("#text_area", mod.MyTextArea)
        text, box.text = box.text, ""
        self._take_turn(f"[bold black on white]User:[/] {escape(text)}", text)

    def post_scene_change(self, report: str) -> None:
        # What the running goal was asked for and from where, so the agent
        # redoes "the nearest cone" from where the robot was, not where it is.
        note = goal_context_note(gate.last_context)
        self._take_turn(
            f"[bold black on yellow]Scene change:[/] {escape(report)}",
            f"{report}\n{note}" if note else report,
        )

    def _take_turn(self, shown: str, content: str) -> None:
        log = self.query_one(RichLog)
        log.write(shown)
        log.write("")
        self._waiting.append(content)
        if not self._busy:
            self._run_agent()

    def _run_agent(self) -> None:
        if not self._waiting:
            self._busy = False
            return
        self._busy = True
        contents, self._waiting = self._waiting, []
        content = "\n\n".join(contents)
        gate.start_turn(any(not c.startswith(REPORT_PREFIX) for c in contents))
        gate.allowed = turn_allows_edits(contents)
        request = "\n".join(c for c in contents if not c.startswith(REPORT_PREFIX))
        gate.context = lambda: {"request": request, "robots": self._robot_positions()}
        self.messages += mod.new_user_message(content)
        initial_length = len(self.messages)
        cxt = AgentContext(self.agent)
        cxt.history = self.messages
        log = self.query_one(RichLog)

        def run():
            try:
                cxt.run()
                for r in cxt.get_agent_responses()[initial_length:]:
                    self.call_from_thread(log.write, r.parsed_response)
                    self.call_from_thread(log.write, "")
                offer = ""
                if gate.wrote and gate.last_goal is not None:
                    conf = self.agent.agent_info.tools["run_cypher_query"]._bound_args
                    missing = missing_symbols(
                        goal_symbols(gate.last_goal[0]), conf["dsgdb_conf"]
                    )
                    offer = resend_offer(gate.last_goal, missing)
                    if not missing:
                        gate.show(*gate.last_goal)  # a "yes" next turn resends it
                if offer:
                    # In the history too, so a "yes" next turn is understood.
                    self.messages.append({"role": "assistant", "content": offer})
                    self.call_from_thread(log.write, offer)
                    self.call_from_thread(log.write, "")
            finally:
                gate.allowed = False
                self.call_from_thread(self._run_agent)  # next waiting turn, if any

        threading.Thread(target=run, daemon=True).start()

    def _robot_positions(self) -> dict:
        conf = self.agent.agent_info.tools["run_cypher_query"]._bound_args
        try:
            return robot_positions(conf["dsgdb_conf"])
        except Exception:
            return {}  # Neo4j unavailable: the request alone still helps


if __name__ == "__main__":
    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    agent = LlmAgent(**yml)
    app = RepairChatApp(agent)
    app.run()
