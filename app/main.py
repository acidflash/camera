from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from dotenv import load_dotenv
from uiprotect import ProtectApiClient
from uiprotect.data import Camera
from uiprotect.data.websocket import WSAction, WSSubscriptionMessage

from app.discord_client import DiscordNotifier
from app.image_diff import diff_ratio
from app.ollama_client import OllamaClient
from app.roi import (
    ROI_MASK_PROMPT_NOTE,
    Polygon,
    build_bool_mask,
    mask_image_inside_polygon,
    parse_roi_masks,
)

load_dotenv()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("camera-analyzer")

# Ett omöjligt-att-böja token istället för en svensk fras eller ett engelskt
# ord: modellen har setts rätta "INGET_SYNLIGT" till grammatiskt "INGEN_SYNLIGT"
# (ingen/inget-kongruens), och senare tappa "-ER" i "NOTHING_MARKER_9F3K" och
# skriva "NOTHING_MARK_9F3K" istället - "marker" är böjningsbart även på
# svenska (plural av "mark"), så modellen böjde om det till singular "mark"
# mitt i sitt svenska svar. Det missar substrängsmatchningen och läcker
# markören rakt till Discord. Token måste alltså sakna varje uttolkningsbar
# ordstam på både svenska och engelska - ren slumpad bokstavs/siffersoppa.
NOTHING_MARKER = "ZQX9WKM73VT5"

_DEFAULT_ANALYSIS_PROMPT = (
    "Beskriv kort på svenska vad du ser i denna bild från en säkerhetskamera. "
    "Fokusera på människor, fordon, djur eller annat ovanligt. Max två meningar. "
    f"Om bilden inte visar något av intresse (tom scen, inga människor/fordon/djur, "
    f"bara t.ex. vind i grenar eller ljusförändring): svara med exakt ordet {NOTHING_MARKER} "
    "och inget annat."
)

_DEFAULT_COMPARISON_PROMPT = (
    "Du får två bilder från samma säkerhetskamera. Bild 1 är en referensbild "
    "tagen tidigare (bakgrund/normalläge). Bild 2 är en ny bild tagen just nu "
    "när rörelse upptäcktes. Jämför bilderna. "
    "VIKTIGT: nämn ALDRIG, inte ens i förbigående, sådant som redan syns "
    "oförändrat i bild 1 - t.ex. en bil som redan stod parkerad, möbler eller "
    "andra föremål som inte flyttat sig. Skriv INTE meningar som 'bilen står "
    "kvar' eller 'som tidigare'. Beskriv ENDAST det som är NYTT eller har "
    "FÖRÄNDRATS/RÖR SIG i bild 2 jämfört med bild 1 (t.ex. en person, ett "
    "fordon som kommer/kör iväg/rör sig, ett djur). Om ett redan känt objekt "
    "(t.ex. den parkerade bilen) bara står stilla ska det inte nämnas alls, "
    "även om något annat nytt händer i närheten - beskriv då bara det nya. "
    "Max två meningar, och bara om det nya/förändrade. "
    f"Om inget nytt eller förändrat syns (allt matchar referensbilden, eller "
    f"bara vind/ljusförändring): svara med exakt ordet {NOTHING_MARKER} och "
    "inget annat."
)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _keep_alive_env(name: str, default: str) -> str | int:
    """Ollama tar keep_alive som antingen ett tal (sekunder, -1 = för
    alltid) eller en duration-sträng som "60m"."""
    val = os.environ.get(name, default).strip()
    try:
        return int(val)
    except ValueError:
        return val


def _required_env(name: str) -> str:
    """Läser en obligatorisk env-variabel med ett vänligt felmeddelande."""
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(
            f"Miljövariabel '{name}' saknas eller är tom i .env. Se .env.example."
        )
    return val


@dataclass(frozen=True)
class Config:
    protect_host: str
    protect_port: int
    protect_username: str
    protect_password: str
    protect_verify_ssl: bool
    camera_ids: set[str]
    event_cooldown_seconds: float
    ollama_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    ollama_keep_alive: str | int
    analysis_prompt: str
    comparison_prompt: str
    baseline_refresh_seconds: float
    diff_ratio_threshold: float
    discord_webhook_url: str
    roi_masks: dict[str, Polygon]


def load_config() -> Config:
    return Config(
        protect_host=_required_env("PROTECT_HOST"),
        protect_port=int(os.environ.get("PROTECT_PORT", "443")),
        protect_username=_required_env("PROTECT_USERNAME"),
        protect_password=_required_env("PROTECT_PASSWORD"),
        protect_verify_ssl=_env_bool("PROTECT_VERIFY_SSL", False),
        camera_ids={
            c.strip() for c in os.environ.get("CAMERA_IDS", "").split(",") if c.strip()
        },
        event_cooldown_seconds=float(os.environ.get("EVENT_COOLDOWN_SECONDS", "60")),
        ollama_url=_required_env("OLLAMA_URL"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "gemma4:e4b-it-qat"),
        ollama_timeout_seconds=int(os.environ.get("OLLAMA_TIMEOUT_SECONDS", "120")),
        ollama_keep_alive=_keep_alive_env("OLLAMA_KEEP_ALIVE", "-1"),
        analysis_prompt=os.environ.get("ANALYSIS_PROMPT", _DEFAULT_ANALYSIS_PROMPT),
        comparison_prompt=os.environ.get(
            "COMPARISON_PROMPT", _DEFAULT_COMPARISON_PROMPT
        ),
        baseline_refresh_seconds=float(
            os.environ.get("BASELINE_REFRESH_SECONDS", "900")
        ),
        diff_ratio_threshold=float(os.environ.get("DIFF_RATIO_THRESHOLD", "0.02")),
        discord_webhook_url=_required_env("DISCORD_WEBHOOK_URL"),
        roi_masks=parse_roi_masks(os.environ.get("ROI_MASKS", "")),
    )

class CameraAnalyzer:
    """Innehåller allt tillstånd och logik, byggd utifrån en Config så att den
    går att konstruera/testa utan modul-globaler."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.client = ProtectApiClient(
            host=config.protect_host,
            port=config.protect_port,
            username=config.protect_username,
            password=config.protect_password,
            verify_ssl=config.protect_verify_ssl,
        )
        self.ollama = OllamaClient(
            config.ollama_url,
            config.ollama_model,
            config.ollama_timeout_seconds,
            config.ollama_keep_alive,
        )
        self.discord = DiscordNotifier(config.discord_webhook_url)
        self._last_sent: dict[str, float] = {}
        self._baseline_snapshots: dict[str, bytes] = {}
        self._roi_diff_masks: dict[str, np.ndarray] = {}

    def _roi_for(self, camera: Camera) -> Polygon | None:
        """Slår upp ett bevakat område (ROI-polygon) för kameran, via namn
        i första hand (så ROI_MASKS kan skrivas med t.ex. "Carport"), annars
        kamera-ID."""
        return self.config.roi_masks.get(camera.name) or self.config.roi_masks.get(
            camera.id
        )

    def _roi_diff_mask_for(
        self, camera: Camera, roi: Polygon, resize: tuple[int, int]
    ) -> np.ndarray:
        """Bygger (och cachar) boolean-masken som används i pixel-diffen, i
        samma upplösning som diff_ratio jobbar med. ROI-polygonen anger det
        ignorerade området, så masken som skickas till diff_ratio är dess
        invers (dvs True för allt UTOM polygonen, det som ska bevakas)."""
        cached = self._roi_diff_masks.get(camera.id)
        if cached is not None:
            return cached
        excluded = build_bool_mask(roi, resize[0], resize[1])
        mask = ~excluded
        self._roi_diff_masks[camera.id] = mask
        return mask

    async def refresh_baseline(self, camera: Camera) -> None:
        """Tar en ny referens-snapshot av kamerans normalläge (t.ex. parkerade
        bilar som stått där ett tag) att jämföra mot vid framtida rörelse."""
        if camera.is_motion_detected:
            log.debug("Hoppar över referensuppdatering för %s, rörelse pågår", camera.name)
            return
        snapshot = await self.client.get_camera_snapshot(camera.id)
        if snapshot is None:
            log.warning("Kunde inte hämta referens-snapshot för %s", camera.name)
            return
        self._baseline_snapshots[camera.id] = snapshot
        log.debug("Uppdaterade referensbild för %s", camera.name)

    async def baseline_refresh_loop(self, cameras: list[Camera]) -> None:
        while True:
            # Laddar om modellen om Ollama-servern startats om eller någon
            # annan klient fått den avlastad, så nästa rörelse slipper kallstart.
            await self.ollama.warm_up()
            for camera in cameras:
                try:
                    await self.refresh_baseline(camera)
                except Exception:
                    log.exception("Fel vid uppdatering av referensbild för %s", camera.name)
            await asyncio.sleep(self.config.baseline_refresh_seconds)

    async def process_motion(self, camera: Camera) -> None:
        cfg = self.config
        now = time.monotonic()
        last = self._last_sent.get(camera.id, 0.0)
        if now - last < cfg.event_cooldown_seconds:
            log.debug("Cooldown aktiv för %s, hoppar över", camera.name)
            return
        # Sätts direkt så att parallella triggers inte startar dubbla analyser,
        # men återställs om analysen misslyckas - annars tappas nästa rörelse
        # inom cooldownen trots att ingen notis/bedömning gjordes.
        self._last_sent[camera.id] = now

        try:
            log.info("Rörelse upptäckt: %s", camera.name)
            snapshot = await self.client.get_camera_snapshot(camera.id)
            if snapshot is None:
                log.warning("Kunde inte hämta snapshot för %s", camera.name)
                self._last_sent[camera.id] = last
                return

            roi = self._roi_for(camera)
            diff_resize = (320, 240)
            roi_diff_mask = (
                self._roi_diff_mask_for(camera, roi, diff_resize)
                if roi is not None
                else None
            )

            baseline = self._baseline_snapshots.get(camera.id)
            if baseline is not None:
                ratio = diff_ratio(
                    baseline, snapshot, resize=diff_resize, mask=roi_diff_mask
                )
                if ratio < cfg.diff_ratio_threshold:
                    log.info(
                        "[%s] endast %.1f%% pixelskillnad mot referens%s, hoppar över (ingen Ollama-analys)",
                        camera.name,
                        ratio * 100,
                        " (inom bevakat område)" if roi is not None else "",
                    )
                    return
                baseline_img, snapshot_img = baseline, snapshot
                prompt = cfg.comparison_prompt
                if roi is not None:
                    baseline_img = mask_image_inside_polygon(baseline, roi)
                    snapshot_img = mask_image_inside_polygon(snapshot, roi)
                    prompt = f"{prompt} {ROI_MASK_PROMPT_NOTE}"
                comment = await self.ollama.describe_images(
                    [baseline_img, snapshot_img], prompt
                )
            else:
                snapshot_img = snapshot
                prompt = cfg.analysis_prompt
                if roi is not None:
                    snapshot_img = mask_image_inside_polygon(snapshot, roi)
                    prompt = f"{prompt} {ROI_MASK_PROMPT_NOTE}"
                comment = await self.ollama.describe_image(snapshot_img, prompt)
            if comment is None:
                log.warning("Fick ingen analys från Ollama för %s", camera.name)
                self._last_sent[camera.id] = last
                return

            if NOTHING_MARKER in comment.upper():
                log.info("[%s] inget synligt av intresse, hoppar över Discord", camera.name)
                return

            log.info("[%s] %s", camera.name, comment)
            await self.discord.send_snapshot(camera.name, comment, snapshot)
        except Exception:
            log.exception("Fel vid bearbetning av rörelse för %s", camera.name)


    def _make_ws_callback(
        self, loop: asyncio.AbstractEventLoop
    ) -> Callable[[WSSubscriptionMessage], None]:
        def on_ws_message(msg: WSSubscriptionMessage) -> None:
            if msg.action != WSAction.UPDATE:
                return
            new_obj = msg.new_obj
            old_obj = msg.old_obj
            if not isinstance(new_obj, Camera):
                return
            if self.config.camera_ids and new_obj.id not in self.config.camera_ids:
                return

            was_detecting = (
                bool(old_obj.is_motion_detected) if isinstance(old_obj, Camera) else False
            )
            if new_obj.is_motion_detected and not was_detecting:
                fut = asyncio.run_coroutine_threadsafe(self.process_motion(new_obj), loop)
                fut.add_done_callback(
                    lambda f: (
                        log.error("Fel i process_motion", exc_info=f.exception())
                        if not f.cancelled() and f.exception()
                        else None
                    )
                )

        return on_ws_message

    async def shutdown(self, baseline_task: asyncio.Task[None]) -> None:
        log.info("Stänger ner ...")
        self.client.stop()
        baseline_task.cancel()
        try:
            await baseline_task
        except asyncio.CancelledError:
            pass
        await self.ollama.close()
        await self.discord.close()

    async def run(self) -> None:
        cfg = self.config
        log.info(
            "Ansluter till UniFi Protect på %s:%s ...", cfg.protect_host, cfg.protect_port
        )
        await self.client.update()

        cameras = self.client.bootstrap.cameras.values()
        watched = [c for c in cameras if not cfg.camera_ids or c.id in cfg.camera_ids]
        log.info(
            "Bevakar %d kamera(or): %s", len(watched), ", ".join(c.name for c in watched)
        )

        log.info("Tar initiala referensbilder ...")
        for camera in watched:
            await self.refresh_baseline(camera)

        log.info("Laddar Ollama-modellen %s ...", cfg.ollama_model)
        await self.ollama.warm_up()

        loop = asyncio.get_running_loop()
        self.client.subscribe_websocket(self._make_ws_callback(loop))
        baseline_task = asyncio.create_task(self.baseline_refresh_loop(watched))

        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _signal_handler)

        log.info("Redo. Väntar på rörelsehändelser ...")
        await stop_event.wait()
        await self.shutdown(baseline_task)


async def main() -> None:
    config = load_config()
    await CameraAnalyzer(config).run()


if __name__ == "__main__":
    asyncio.run(main())

