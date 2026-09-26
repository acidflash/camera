"""Tester för keep_alive i Ollama-anropen och att cooldownen inte förbrukas
när analysen misslyckas (annars tappas nästa rörelse tyst)."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from app.main import CameraAnalyzer
from app.ollama_client import OllamaClient


class _FakeResponse:
    status = 200

    async def json(self) -> dict:
        return {"response": "En person syns."}

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class _FakeSession:
    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.closed = False

    def post(self, url: str, json: dict, **kwargs: object) -> _FakeResponse:
        self.payloads.append(json)
        return _FakeResponse()


class OllamaKeepAliveTest(unittest.TestCase):
    def test_describe_and_warm_up_send_keep_alive(self) -> None:
        client = OllamaClient("http://ollama", "m", 10, keep_alive=-1)
        session = _FakeSession()
        client._session = session  # type: ignore[assignment]

        async def run() -> None:
            await client.describe_image(b"img", "p")
            await client.warm_up()

        asyncio.run(run())
        self.assertEqual([p["keep_alive"] for p in session.payloads], [-1, -1])
        self.assertNotIn("prompt", session.payloads[1])


class CooldownResetTest(unittest.TestCase):
    def _analyzer(self) -> CameraAnalyzer:
        analyzer = CameraAnalyzer.__new__(CameraAnalyzer)
        analyzer.config = SimpleNamespace(
            event_cooldown_seconds=60.0,
            analysis_prompt="p",
            roi_masks={},
        )
        analyzer.client = mock.Mock()
        analyzer.client.get_camera_snapshot = mock.AsyncMock(return_value=b"snap")
        analyzer.ollama = mock.Mock()
        analyzer.discord = mock.Mock()
        analyzer.discord.send_snapshot = mock.AsyncMock()
        analyzer._last_sent = {}
        analyzer._baseline_snapshots = {}
        analyzer._roi_diff_masks = {}
        return analyzer

    def test_failed_analysis_does_not_consume_cooldown(self) -> None:
        analyzer = self._analyzer()
        analyzer.ollama.describe_image = mock.AsyncMock(return_value=None)
        camera = SimpleNamespace(id="c1", name="Carport")
        asyncio.run(analyzer.process_motion(camera))
        self.assertEqual(analyzer._last_sent["c1"], 0.0)

    def test_successful_analysis_consumes_cooldown(self) -> None:
        analyzer = self._analyzer()
        analyzer.ollama.describe_image = mock.AsyncMock(return_value="En person.")
        camera = SimpleNamespace(id="c1", name="Carport")
        asyncio.run(analyzer.process_motion(camera))
        self.assertGreater(analyzer._last_sent["c1"], 0.0)
        analyzer.discord.send_snapshot.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
