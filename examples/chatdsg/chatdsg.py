#!/usr/bin/env python3
import argparse
import logging
import os
import threading
from urllib.parse import urlsplit, urlunsplit

import spark_dsg
import yaml
from heracles.dsg_utils import summarize_dsg
from heracles.utils import extract_labelspaces_from_dsg, load_dsg_to_db
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Input, Label, RichLog, Rule, Static, TextArea

from heracles_agents.llm_agent import LlmAgent
from heracles_agents.llm_interface import AgentContext

logger = logging.getLogger(__name__)


def resolve_neo4j_uri(neo4j_uri, db_ip, db_port):
    """Apply the --db_ip/--db_port overrides on top of the configured URI."""
    if db_ip is None and db_port is None:
        return neo4j_uri

    parts = urlsplit(neo4j_uri or "neo4j://")
    host = db_ip or parts.hostname or ""
    port = db_port or parts.port
    return urlunsplit(
        (
            parts.scheme or "neo4j",
            f"{host}:{port}" if port else host,
            parts.path,
            parts.query,
            parts.fragment,
        )
    )


def load_prior_dsg(dsg_filepath, neo4j_uri):
    """Load a DSG from file into Neo4j using the labelspaces embedded in the graph.

    Returns True if the graph was loaded, False if it was skipped.
    """
    if not os.path.isfile(dsg_filepath):
        logger.warning(
            f"No DSG at '{dsg_filepath}'; skipping load. "
            "The database will only contain whatever is already in it."
        )
        return False

    if not neo4j_uri:
        logger.warning(
            'No Neo4j URI: pass --neo4j-uri or set "$HERACLES_NEO4J_URI"; '
            "skipping DSG load."
        )
        return False

    neo4j_creds = (
        os.getenv("HERACLES_NEO4J_USERNAME"),
        os.getenv("HERACLES_NEO4J_PASSWORD"),
    )
    if not all(neo4j_creds):
        logger.warning(
            'Neo4j credentials are not set ("$HERACLES_NEO4J_USERNAME" and '
            '"$HERACLES_NEO4J_PASSWORD"); skipping DSG load.'
        )
        return False

    logger.info(f"Loading DSG into database from filepath: {dsg_filepath}")
    scene_graph = spark_dsg.DynamicSceneGraph.load(dsg_filepath)
    summarize_dsg(scene_graph)

    # load_dsg_to_db reads the labelspaces out of the graph metadata, so a graph
    # saved without them will come back with unlabeled objects and rooms.
    object_labelspace, room_labelspace = extract_labelspaces_from_dsg(scene_graph)
    missing = [
        name
        for name, labelspace in (
            ("object", object_labelspace),
            ("room", room_labelspace),
        )
        if not labelspace
    ]
    if missing:
        logger.warning(
            f"DSG '{dsg_filepath}' has no embedded {' or '.join(missing)} labelspace; "
            "those nodes will be loaded without semantic labels."
        )

    load_dsg_to_db(neo4j_uri, neo4j_creds, scene_graph)
    logger.info("DSG loaded!")
    return True


def new_user_message(text):
    return [{"role": "user", "content": text}]


def generate_initial_prompt(agent: LlmAgent):
    prompt = agent.agent_info.prompt_settings.base_prompt
    return prompt


class MyTextArea(TextArea):
    BINDINGS = [
        Binding("ctrl+b", "submit", "Submit text"),
    ]

    def action_submit(self) -> None:
        self.app.action_submit()


class InputDisplayApp(App):
    def __init__(self, agent):
        self.agent = agent
        self.messages = generate_initial_prompt(agent).to_openai_json(
            "Now you will interact with the user:"
        )
        super().__init__()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield VerticalScroll(
            Label("Agent Chat:"),
            Rule(),
            RichLog(highlight=True, markup=True, wrap=True),
            Rule(line_style="thick"),
            Label("Enter text below:"),
            Rule(),
            MyTextArea("text", id="text_area"),
            Footer(id="footer"),
        )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""

        input_widget = self.query_one("#input_box", Input)
        # Clear the input box
        input_widget.value = ""

        self.display_text = f"You entered: {event.value}"
        self.query_one("#display_panel", Static).update(self.display_text)
        text_log = self.query_one(RichLog)
        text_log.write(event.value)

    def action_submit(self) -> None:
        """Called when ctrl+b is pressed."""
        input_text_box = self.query_one("#text_area", MyTextArea)
        input_text = input_text_box.text
        text_log = self.query_one(RichLog)
        formatted_text = f"[bold black on white]User:[/] {input_text}"
        text_log.write(formatted_text)
        text_log.write("")
        input_text_box.text = ""

        self.messages += new_user_message(input_text)
        initial_length = len(self.messages)

        cxt = AgentContext(self.agent)
        cxt.history = self.messages

        def run_agent():
            success, answer = cxt.run()
            responses = cxt.get_agent_responses()
            for r in responses[initial_length:]:
                text_log.write(r.parsed_response)
                text_log.write("")

        thread = threading.Thread(target=run_agent)
        thread.start()

        # success, answer = cxt.run()
        # responses = cxt.get_agent_responses()
        # for r in responses[initial_length:]:
        #    text_log.write(r.parsed_response)
        #    text_log.write("")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser("ChatDSG agent")
    parser.add_argument(
        "--scene-graph",
        type=str,
        default=None,
        help="DSG filepath to load into the database on startup",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default=os.getenv("HERACLES_NEO4J_URI"),
        help='Neo4j URI (defaults to "$HERACLES_NEO4J_URI")',
    )
    parser.add_argument(
        "--db_ip", type=str, help="Heracles database ip (overrides the URI host)"
    )
    parser.add_argument(
        "--db_port", type=int, help="Heracles database port (overrides the URI port)"
    )
    args = parser.parse_args()

    # Passing a scene graph is what asks for the database to be loaded.
    if args.scene_graph:
        load_prior_dsg(
            args.scene_graph,
            resolve_neo4j_uri(args.neo4j_uri, args.db_ip, args.db_port),
        )

    with open("agent_config.yaml", "r") as fo:
        yml = yaml.safe_load(fo)
    agent = LlmAgent(**yml)
    app = InputDisplayApp(agent)
    app.run()
