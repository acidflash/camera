# camera-analyzer

Bevakar rörelsehändelser på UniFi Protect-kameror, skickar snapshot till en
Ollama vision-modell (t.ex. `llava`) för beskrivning, och postar bild +
kommentar till Discord via webhook.

## Arkitektur

```
UniFi Protect (websocket, motion) -> app/main.py -> snapshot
                                            |
                                            v
                                   Ollama /api/generate (extern server)
                                            |
                                            v
                                   Discord webhook (bild + kommentar)
```

En enda långlivad Python-process, ingen lokal Ollama-container behövs
eftersom du kör Ollama på extern server.

## Setup

1. **UniFi Protect-konto**: skapa ett lokalt "viewer"-konto i
   UniFi OS (Settings > Users) — använd inte ditt Ubiquiti-cloud-konto.
2. **Ollama**: se till att modellen finns på den externa servern:
   ```
   ollama pull llava
   ```
   och att Ollama lyssnar på nätverket (inte bara localhost):
   `OLLAMA_HOST=0.0.0.0` i dess miljö.
3. **Discord**: skapa en webhook i kanalens inställningar
   (Integrations > Webhooks > New Webhook), kopiera URL:en.
4. Kopiera `.env.example` till `.env` och fyll i värden:
   ```
   cp .env.example .env
   ```
5. Starta:
   ```
   docker compose up -d --build
   ```
6. Följ loggar:
   ```
   docker compose logs -f
   ```

## Konfiguration (.env)

| Variabel | Beskrivning |
|---|---|
| `PROTECT_HOST` / `PROTECT_PORT` | IP/port till UniFi OS-konsolen |
| `PROTECT_USERNAME` / `PROTECT_PASSWORD` | Lokalt Protect-konto |
| `PROTECT_VERIFY_SSL` | `false` för självsignerat cert (standard) |
| `CAMERA_IDS` | Kommaseparerad lista, tomt = alla kameror |
| `EVENT_COOLDOWN_SECONDS` | Minsta tid mellan analyser per kamera |
| `OLLAMA_URL` | t.ex. `http://192.168.1.50:11434` |
| `OLLAMA_MODEL` | vision-modell, standard `gemma4:e4b-it-qat` |
| `OLLAMA_TIMEOUT_SECONDS` | max väntetid per analys, standard `120` |
| `OLLAMA_KEEP_ALIVE` | hur länge modellen hålls laddad i Ollama, standard `-1` (för alltid, ~5,5 GB VRAM). Modellen värms också upp vid start och var `BASELINE_REFRESH_SECONDS` |
| `ANALYSIS_PROMPT` | prompt som används när ingen referensbild finns ännu |
| `COMPARISON_PROMPT` | prompt som används tillsammans med referensbild, för att ignorera saker som stått där hela tiden |
| `BASELINE_REFRESH_SECONDS` | hur ofta (sek) en ny referensbild tas i vila, standard `900` |
| `DIFF_RATIO_THRESHOLD` | andel pixlar (0.0-1.0) som måste skilja sig mot referensbilden för att skicka till Ollama, standard `0.02` |
| `ROI_MASKS` | JSON-dict kameranamn -> ignorerad polygon, för att strunta i delar av bilden (t.ex. vägen bakom en carport) - se nedan |
| `DISCORD_WEBHOOK_URL` | webhook-URL för notiser |

## Hur det fungerar

- `app/main.py` öppnar en websocket mot Protect och lyssnar på
  `Camera.is_motion_detected`-övergångar (false -> true).
- I bakgrunden tas en "referensbild" per kamera var `BASELINE_REFRESH_SECONDS`
  sekund (bara när kameran är i vila, dvs ingen rörelse pågår just då). Denna
  bild fungerar som facit för vad som är normalt i bilden — t.ex. en bil som
  stått parkerad hela tiden.
- Vid ny rörelse hämtas en snapshot via Protects REST-API, med cooldown per
  kamera för att undvika spam vid ihållande rörelse.
- Finns en referensbild jämförs den pixel-för-pixel mot den nya snapshoten
  (nedskalad, gråskala). Om färre än `DIFF_RATIO_THRESHOLD` av pixlarna
  skiljer sig anses ingenting nytt hänt och Ollama anropas inte alls — detta
  är en mekanisk spärr som garanterar att t.ex. en bil som stått parkerad
  hela tiden aldrig triggar en notis, oavsett hur vision-modellen skulle
  tolka bilden. Har kameran ett ignorerat område satt via `ROI_MASKS` (se
  nedan) räknas inte pixlar innanför det området med, t.ex. vägen bakom
  grusplanen.
- Snapshoten skickas till Ollamas `/api/generate` (base64, `stream: false`).
  Finns en referensbild skickas båda bilderna med `COMPARISON_PROMPT`, som
  ber modellen ignorera allt som redan syns oförändrat i referensbilden
  (parkerade bilar, möbler etc.) och bara rapportera det som är nytt eller
  rör sig. Finns ingen referensbild ännu (t.ex. precis vid uppstart) används
  `ANALYSIS_PROMPT` på enbart den nya bilden.
- Svaret postas till Discord som en embed med bilden bifogad — men bara om
  Ollama faktiskt såg något nytt av intresse. Om inget nytt syns (tom scen,
  bara det som redan fanns i referensbilden, eller rörelse triggad av
  vind/ljus) svarar modellen med markören `INGET_SYNLIGT` och inget skickas
  till Discord.

## Ignorera en del av bilden (ROI)

Om kameran ser mer än vad som ska bevakas - t.ex. en carport-kamera som
även har vägen i bild och du bara vill ha larm för grusplanen, inte
förbipasserande bilar på vägen - kan du sätta `ROI_MASKS`. Det är en
JSON-dict från kameranamn till en polygon av `[x,y]`-punkter i andelar
(0.0-1.0) av bildens bredd/höjd, som pekar ut ytan som ska IGNORERAS
(t.ex. en ruta runt vägen). Allt innanför polygonen:

- räknas inte med i pixel-diffen mot referensbilden (`DIFF_RATIO_THRESHOLD`
  jämförs bara mot pixlar utanför polygonen), och
- svärtas bort i bilden som skickas till Ollama, så modellen aldrig ens ser
  eller kommenterar det (t.ex. en bil som kör förbi på vägen).

Discord-notisen visar ändå hela originalbilden (för sammanhang) - det är
bara vad som triggar ett larm och vad texten beskriver som utesluter det
ignorerade området.

Ta fram rätt punkter med kalibreringsscriptet mot en sparad exempelbild
(t.ex. en snapshot från Protects webbgränssnitt):

```
python scripts/roi_preview.py carport.jpg \
    --polygon "0.42,0.0 0.86,0.0 0.86,0.14 0.43,0.15" \
    --out carport_preview.jpg
```

Öppna `carport_preview.jpg`: rödtonad yta = maskeras bort (ignoreras, t.ex.
vägen), gul linje = polygonens kant. Justera punkterna och kör om tills den
röda ytan täcker exakt det du vill strunta i. Klistra sedan in samma
punkter i `.env`:

```
ROI_MASKS={"Carport": [[0.42,0.0],[0.86,0.0],[0.86,0.14],[0.43,0.15]]}
```

Kameranamnet måste matcha `camera.name` i Protect (samma namn som visas i
loggen vid uppstart, "Bevakar N kamera(or): ..."). Kameror som saknas i
`ROI_MASKS` fortsätter att bevaka hela bilden som vanligt.

## Byta modell / notis-kanal

- Annan vision-modell: ändra `OLLAMA_MODEL` (kräver att modellen är pull:ad
  på Ollama-servern, t.ex. `llama3.2-vision` eller `moondream`).
- Vill du hellre logga eller skicka till t.ex. ntfy/MQTT istället för
  Discord: byt ut `app/discord_client.py` mot en egen notifier med samma
  `send_snapshot(camera_name, comment, image_bytes)`-signatur.
