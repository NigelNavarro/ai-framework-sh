"""
Author: Nigel J. Navarro
Governance / Safety Tests.

Two guarantees for MVP:
  1. The built-in tool set is exactly the known safe set (adding a tool forces a conscious update here as well)
  2. `draft_document` is the only writing tool. It cannot escape data/drafts/ even when handed a path-traversal filename.

Run:  .venv/bin/pytest tests/test_safety.py -v
"""
from __future__ import annotations

from framework.config import load_config
from framework.tools.builtin import (
    make_search_knowledge_tool,
    make_remember_tool,
    make_recall_tool,
    make_draft_document_tool,
)

# These are the complete set of tools the agent is allowed in the MVP. 
# All Read-Only, or Write-Only in data/.
# Changing this set must be a deliberate and reviewed act. 
SAFE_TOOL_NAMES = {"search_knowledge", "remember", "recall", "draft_document"}

def build_all_builtin_tools(config):
    """
    Build all the built-in tools for the agent.
    Builders capture their dependents see in a closure, but never call it at build time (only when the Handler runs). 
    The use of None is a safe stand-in just for constructing the tool
    """
    return [
        make_search_knowledge_tool(None),
        make_remember_tool(None),
        make_recall_tool(None),
        make_draft_document_tool(config),
    ]

def test_builtin_tools_are_known_safe_set():
    """
    The built-in tools are exactly the known safe allow list.
    """
    config = load_config()
    names = {tool.name for tool in build_all_builtin_tools(config)}
    assert names == SAFE_TOOL_NAMES, (
        "Built-in toolset has changed. If you added or removed the tool, update `SAFE_TOOL_NAMES` "
        "and confirm the new set is still safe (Read-Only or Write-Only in data/)."
    )

def test_draft_document_cannot_escape_drafts_dir(tmp_path):
    """
    The writing tool that we have must stay inside data/drafts/ even when presented a malicious filename.
    """
    config = load_config()
    config.data_dir = tmp_path                             # Use a temporary directory for testing to avoid writing to the real data directory.

    tool = make_draft_document_tool(config)
    tool.handler(filename="../../escaped.txt", content="I should stay contained")

    drafts_dir = tmp_path / "drafts"
    assert (drafts_dir / "escaped.txt").exists()            # The file landed inside the drafts directory, not outside it, and as a bare filename.
    assert not (tmp_path.parent / "escaped.txt").exists()  # The file did not escape to the parent directory.

