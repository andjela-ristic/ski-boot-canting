# Web App (PWA)

`web_app/` je novi browser klijent za repo. Zamenjuje stari Flutter folder.

## Struktura

- `index.html`: mali shell sa slotovima za panele
- `panels/`: HTML fragmenti po delu interfejsa
- `js/app/`: JS moduli podeljeni na state, services, ui i layout
- `styles/`: baza, layout i panel-specifik CSS

## Sta radi

- otvara se direktno iz browsera ili kao Home Screen app
- salje video na `POST /frames`
- prikazuje vraceni `overlay_data_url`
- ima dva toka snimanja:
  - live 2s browser recording kada su `getUserMedia` i `MediaRecorder` dostupni
  - fallback `Snimi/izaberi video` upload koji je cesto pouzdaniji na iPhone-u

## Zasto Web PWA

Bez Mac-a je to najprakticniji put do iPhone-a:

- nema Xcode ni native iOS build
- isti UI radi na iPhone-u, Androidu i desktopu
- kada isti Python backend servira i frontend, `/frames` radi sa istog origin-a

## Pokretanje

Najprostiji nacin:

```powershell
python -m api --host 0.0.0.0 --port 8000
```

Ili bez lokalnog Python setup-a:

```powershell
docker compose up --build
```

Onda otvori:

- desktop: `http://127.0.0.1:8000`
- drugi uredjaj u mrezi: `http://<LAN-IP-tvog-racunara>:8000`

## iPhone napomena

Live browser kamera trazi secure context. Prakticno to znaci:

- za puni live preview i 2s browser recording otvori aplikaciju kroz HTTPS tunnel
- bez HTTPS-a koristi fallback `Snimi/izaberi video`

## HTTPS tunnel primer

Sa bilo kojim alatom koji daje javni HTTPS URL za lokalni port `8000`, npr:

```powershell
cloudflared tunnel --url http://localhost:8000
```

U ovom repou mozes i direktno:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_cloudflare_tunnel.ps1
```

Ako hoces da jednim pozivom dignes i lokalni API i Cloudflare tunnel:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_web_app_https.ps1
```

Podrazumevano koristi Docker backend. Ako vec imas lokalno podignut servis na `8000`, koristi:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_web_app_https.ps1 -Backend Existing
```

Ako hoces direktno preko sistemskog Python-a sa laptopa:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_web_app_https.ps1 -Backend SystemPython -PythonPath python
```

Zatim HTTPS URL otvoris u Safari-ju na iPhone-u i po zelji dodas na Home Screen.
