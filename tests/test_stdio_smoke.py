"""Start the real `mltree-mcp` stdio server as a subprocess and talk to it with the MCP client SDK."""
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED = {"init_project", "add_step", "connect_steps", "get_pipeline", "log_run", "list_runs", "best_run", "open_map", "scan_project", "export_map", "start_sweep", "finish_sweep", "run_step", "run_sweep", "run_log", "cancel_run", "data_status", "compare_runs", "add_note", "tag_run", "gnn_summary"}


async def test_stdio_server_lists_and_calls_tools(tmp_path):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mltree_mcp"],
        env={"MLTREE_PROJECT": str(tmp_path)},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == EXPECTED

            result = await session.call_tool("init_project", {"name": "smoke"})
            assert not (getattr(result, "is_error", False) or getattr(result, "isError", False))
            assert (tmp_path / ".mltree" / "mltree.db").exists()

            resources = await session.list_resources()
            assert "mltree://pipeline" in {str(r.uri) for r in resources.resources}
            read_res = await session.read_resource("mltree://pipeline")
            assert json.loads(read_res.contents[0].text)["project"] == "smoke"

            bad = await session.call_tool("connect_steps", {"source": "x", "target": "y"})
            assert getattr(bad, "is_error", False) or getattr(bad, "isError", False)
            assert "Unknown step" in bad.content[0].text
