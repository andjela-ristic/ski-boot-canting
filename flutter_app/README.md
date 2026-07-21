# Flutter Skeleton

Minimal Flutter klijent za tvoj lokalni canting API sa jednom stranicom i kamerom.

## Sta radi

- prikazuje live camera preview
- klik na `Slikaj` snima video od 2 sekunde
- salje ceo snimljeni clip na `POST /frames`
- ocekuje jedan overlay nazad kroz `overlay_data_url`
- prikazuje overlay i vreme obrade na istoj stranici

## Pokretanje

```powershell
cd flutter_app
flutter create .
flutter pub get
flutter run
```

`flutter create .` ce dopuniti Android/iOS/desktop platform foldere preko ovog kostura.

## Base URL

- desktop Flutter + Docker/local API: `http://127.0.0.1:8000`
- Android emulator + Docker/local API: `http://10.0.2.2:8000`
- iOS simulator + Docker/local API: `http://127.0.0.1:8000`
- fizicki Android/iPhone uredjaj + Docker/local API: `http://<LAN-IP-tvog-racunara>:8000`

## Ocekivani API

Flutter salje `multipart/form-data` na:

```text
POST /frames
```

Polja:

- `video`: snimljeni 2s video
- `clip_duration_ms`: `2000`
- `frame_count`: `6`
- `response_mode`: `json`

Trenutno backend za `multipart /frames` vraca stub overlay, ali u istom top-level formatu koji Flutter vec ume da iscrta:

```json
{
  "processing_time_ms": 1234.5,
  "overlay_data_url": "data:image/png;base64,..."
}
```
