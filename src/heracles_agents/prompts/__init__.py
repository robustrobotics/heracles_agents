"""The plan-repair chat's planning knowledge, installed with the package.

One source for every chat that sends PDDL goals to the plan-repair goal
manager: examples/chatdsg_repair, examples/chatdsg_repair_grstaps, and the ADT4
web console (adt4_console), which imports this module instead of keeping its
own copy.

    team_goal_rules(engine)        who does what: -by forms only for robots the
                                   user names, robot-free forms otherwise
    domain_description(engine)     the PDDL predicates and runtime constraints
    in_context_examples(engine)    user / assistant example pairs
    repair_goal_prompt(engine)     the three above as one text
    scene_graph_description(kind)  the scene graph's layers
    cypher_interface_description() the Neo4j labels and relationships
    repair_chat_prompt(engine)     the full Prompt fields of chatdsg_repair

``engine`` is "fast_downward" (multi-robot Fast Downward) or "grstaps"
(GRSTAPS-X). The texts are the YAML files next to this module; nothing here
needs ROS or a database.
"""

from __future__ import annotations

import functools
from importlib import resources

import yaml

ENGINES = ("fast_downward", "grstaps")

_DOMAIN = {
    "fast_downward": "pddl_domain_description_with_constraints.yaml",
    "grstaps": "pddl_domain_description_grstaps.yaml",
}
_EXAMPLES = {
    "fast_downward": "pddl_in_context_examples_with_constraints.yaml",
    "grstaps": "pddl_in_context_examples_grstaps.yaml",
}
_SCENE_GRAPH = {
    # traversability places, no semantic class (the repair chats' maps)
    "traversability": "scene_graph_description_traversability.yaml",
    # mesh places with a class, objects and rooms (Hydra's hierarchy)
    "hierarchical": "scene_graph_description.yaml",
}


def _engine(engine: str) -> str:
    if engine not in ENGINES:
        raise ValueError(f"unknown planner engine {engine!r}: use one of {ENGINES}")
    return engine


@functools.lru_cache(maxsize=None)
def _load(name: str):
    return yaml.safe_load(resources.files(__name__).joinpath(name).read_text())


def path(name: str) -> str:
    """The installed path of one of the prompt files (for configs that take a path)."""
    return str(resources.files(__name__).joinpath(name))


def team_goal_rules(engine: str) -> str:
    """The team goal rules and their examples, as one text."""
    engine = _engine(engine)
    data = _load("team_goal_rules.yaml")

    def mine(item):
        return engine in item.get("engines", ENGINES)

    lines = [data["title"]]
    lines += [f"- {r['text'].strip()}" for r in data["rules"] if mine(r)]
    lines.append("Examples (object ids as the user sees them; in goals lower case):")
    for ex in data["examples"]:
        if not mine(ex):
            continue
        line = f'- "{ex["user"]}": {ex["goal"]}'
        if ex.get("note"):
            line += f" ({ex['note']})"
        lines.append(line)
    return "\n".join(lines)


def domain_description(engine: str) -> str:
    return _load(_DOMAIN[_engine(engine)])["domain_description"]


def in_context_examples(engine: str) -> list[dict]:
    """[{user, assistant}] pairs; copies, so callers may change them."""
    return [dict(e) for e in _load(_EXAMPLES[_engine(engine)])["in_context_examples"]]


def scene_graph_description(kind: str = "traversability") -> str:
    if kind not in _SCENE_GRAPH:
        raise ValueError(
            f"unknown scene graph kind {kind!r}: use one of {tuple(_SCENE_GRAPH)}"
        )
    return _load(_SCENE_GRAPH[kind])["scene_graph_description"]


def cypher_interface_description() -> str:
    return _load("cypher_interface_description.yaml")["interface_description"]


def _rename(text: str, names: dict | None) -> str:
    for old, new in (names or {}).items():
        text = text.replace(old, new)
    return text


def repair_goal_prompt(engine: str, names: dict | None = None) -> str:
    """Everything a chat needs to write a team goal for ``engine``: the team goal
    rules, the PDDL domain with its constraints, and the in-context examples.

    ``names`` renames the repair chat's tool and parameter names for a chat whose
    tools are called differently, e.g. {"send_pddl_goal_with_constraints":
    "propose_goal"}.
    """
    examples = "\n".join(
        f"User: {e['user'].strip()}\nAssistant: {e['assistant'].strip()}"
        for e in in_context_examples(engine)
    )
    text = "\n\n".join(
        [
            team_goal_rules(engine),
            domain_description(engine).strip(),
            "<Examples>\n" + examples + "\n</Examples>",
        ]
    )
    return _rename(text, names)


def repair_chat_prompt(engine: str) -> dict:
    """The plan-repair chat's Prompt fields (heracles_agents.prompt.Prompt) for
    ``engine``: its system text with the team goal rules, and the scene graph,
    Cypher interface, PDDL domain and in-context examples."""
    engine = _engine(engine)
    data = _load("repair_chat.yaml")
    system = data["system"].replace("<<TEAM_GOAL_RULES>>", team_goal_rules(engine))
    for key, value in data["engine_lines"][engine].items():
        system = system.replace(f"<<{key}>>", value)
    return {
        "system": system,
        "scene_graph_description": scene_graph_description("traversability"),
        "interface_description": cypher_interface_description(),
        "domain_description": domain_description(engine),
        "in_context_examples": in_context_examples(engine),
    }
