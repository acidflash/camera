from __future__ import annotations

import base64
import logging

import aiohttp

log = logging.getLogger("camera-analyzer.ollama")


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int,
        keep_alive: str | int = -1,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = timeout_seconds
        # Hur länge Ollama ska hålla modellen i VRAM efter ett anrop. Ollamas
        # default (5m) gör att nästan varje rörelse efter en lugn stund
        # träffar en kallstart som tar längre än timeouten - -1 = för alltid.
        self._keep_alive = keep_alive
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def warm_up(self, timeout_seconds: int = 300) -> None:
        """Laddar modellen i Ollama utan att generera något (ett anrop utan
        prompt), så att nästa riktiga analys slipper kallstarten. Kastar
        aldrig - misslyckas uppvärmningen tar nästa analys helt enkelt
        längre tid."""
        payload = {"model": self._model, "keep_alive": self._keep_alive}
        url = f"{self._base_url}/api/generate"
        try:
            session = await self._get_session()
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_seconds)
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.warning("Uppvärmning av Ollama svarade %s: %s", resp.status, body[:500])
        except TimeoutError:
            log.warning("Uppvärmning av Ollama tog längre än %ss", timeout_seconds)
        except aiohttp.ClientError as exc:
            log.warning("Kunde inte värma upp Ollama på %s: %r", url, exc)

    async def describe_image(self, image_bytes: bytes, prompt: str) -> str | None:
        return await self.describe_images([image_bytes], prompt)

    async def describe_images(self, images: list[bytes], prompt: str) -> str | None:
        payload = {
            "model": self._model,
            "prompt": prompt,
            "images": [base64.b64encode(img).decode("ascii") for img in images],
            "stream": False,
            "keep_alive": self._keep_alive,
        }
        url = f"{self._base_url}/api/generate"
        try:
            session = await self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    log.error("Ollama svarade %s: %s", resp.status, body[:500])
                    return None
                data = await resp.json()
                return data.get("response", "").strip() or None
        except TimeoutError:
            log.error(
                "Ollama svarade inte inom %ss (%s)", self._timeout_seconds, url
            )
            return None
        except aiohttp.ClientError as exc:
            log.error("Kunde inte nå Ollama på %s: %r", url, exc)
            return None
