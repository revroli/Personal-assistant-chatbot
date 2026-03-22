# Personal Assistant Chatbot (CLI)

Ez a projekt egy parancssorból futtatható, profilalapú Gemini chatbot.
A fókusz a profilkezelésen és a folyamatos, kontextust tartó beszélgetésen van.


## Fájlstruktúra

- `cli_chatbot.py`: a fő program, ezt kell futtatni.
- `template_profiles.json`: verziózott profilforrás (Bill + Template).
- `user_profiles.json`: felhasználó által létrehozott profilok (helyi adatok).
- `requirements.txt`: függőségek.
- `.env`: API kulcs helyi tárolására.
- `dev_codes/`: korábbi notebookos/fejlesztői anyagok.

## Előfeltételek

- Python 3.13
- Gemini API kulcs

## Telepítés és környezet

1. Virtuális környezet létrehozása (ha még nem létezik):

```powershell
python -m venv .venv
```

2. Virtuális környezet aktiválása:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Függőségek telepítése:

```powershell
pip install -r requirements.txt
```

4. `.env` létrehozása API kulccsal:

```powershell
Set-Content -Path .env -Value 'GEMINI_API_KEY="IDE_A_SAJAT_KULCSOD"'
```

## A chatbot indítása

```powershell
python cli_chatbot.py
```

## A chatbot működése (részletesen)

### 1. Induláskor profilforrás betöltés

A program két helyről olvas:

- `template_profiles.json`: tartalmazza a `Bill` és `Template` profilokat.
- `user_profiles.json`: tartalmazza a felhasználó által létrehozott profilokat.

Fontos:

- A `Template` profil rejtett a választólistában.
- A `Template` csak új profil létrehozásához szolgál mintaként.

### 2. Profil kiválasztás vagy új profil létrehozás

Induláskor a program felkínálja:

- meglévő profil kiválasztása
- új profil létrehozása

Új profil létrehozás esetén:

- az új profil automatikusan a `Template` mély másolata lesz
- azonnal mentésre kerül `user_profiles.json` fájlba

### 3. Profil szerkesztés indítás előtt

A program rákérdez, hogy szeretnél-e további tulajdonságokat beállítani.

Szerkesztési szabályok:

- Új kulcs/típus nem adható hozzá.
- Lista típusú mezőkhöz bármennyi új elem adható.
- Nem lista típusú mező egyszer állítható be, ha még üres.
- Nem lista mező, ha már be van állítva, nem módosítható újra.

### 4. Chat mód

A chat folyamatos, körkörös input-output módban fut:

1. Beírod a kérdést.
2. A program hozzáfűzi az előzményekhez.
3. A teljes beszélgetést (`history`) elküldi kontextusként.
4. A rendszerprompt a teljes kiválasztott profilból készül.
5. A modell streamelve válaszol.
6. A válasz bekerül az előzményekbe.
7. Újra kérdez, amíg ki nem lépsz.

Kilépés parancsok:

- `/exit`
- `/quit`

## Mi kerül a modellhez pontosan

- `system_instruction`: a kiválasztott profil teljes tartalma.
- `contents`: a teljes beszélgetési előzmény + az aktuális felhasználói üzenet.

## Ajánlott git kezelés

- `template_profiles.json` maradjon verziókövetett (Bill + Template).
- `user_profiles.json` legyen ignorálva (helyi, személyes profiladatok).

## Gyors indítás

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Set-Content -Path .env -Value 'GEMINI_API_KEY="IDE_A_SAJAT_KULCSOD"'
python cli_chatbot.py
```