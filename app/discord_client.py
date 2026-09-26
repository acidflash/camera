from __future__ import annotations

import json
import logging

import aiohttp

log = logging.getLogger("camera-analyzer.discord")


class DiscordNotifier:
    def __init__(self, webhook_url: str) -> None:
        self._webhook_url = webhook_url
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def send_snapshot(self, camera_name: str, comment: str, image_bytes: bytes) -> None:
        payload = {
            "embeds": [
                {
                    "title": camera_name,
                    "description": comment,
                    "image": {"url": "attachment://snapshot.jpg"},
                    "color": 0x2ECC71,
                }
            ]
        }

        form = aiohttp.FormData()
        form.add_field("payload_json", json.dumps(payload), content_type="application/json")
        form.add_field(
            "file",
            image_bytes,
            filename="snapshot.jpg",
            content_type="image/jpeg",
        )

        try:
            session = await self._get_session()
            async with session.post(self._webhook_url, data=form) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    log.error("Discord-webhook svarade %s: %s", resp.status, body[:500])
        except aiohttp.ClientError as exc:
            log.error("Kunde inte skicka till Discord: %s", exc)
