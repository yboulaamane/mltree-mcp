import pytest

from mltree_mcp import Store


@pytest.fixture
def project(tmp_path, monkeypatch):
    """An empty project directory that the MCP tools resolve via MLTREE_PROJECT."""
    monkeypatch.setenv("MLTREE_PROJECT", str(tmp_path))
    monkeypatch.delenv("CLAUDE_PROJECT_DIR", raising=False)
    return tmp_path


@pytest.fixture
def store(project):
    s = Store(project)
    s.init("test")
    return s
