import json
import os
from pathlib import Path
from copy import deepcopy
from typing import Literal

from google import genai
from google.genai import types

import json

def generate_system_prompt(profile, alap_talalatok = None, talalatok = 5):
    
    if alap_talalatok is None:
        prompted_profile = profile
    else:
        # Csak az első 5 találat kulcsai alapján szűrjük a profilt.
        top_keys = [doc.page_content for doc in alap_talalatok[:talalatok]]
        prompted_profile = {k: profile[k] for k in top_keys if k in profile}
        
    # A JSON formátum segít a modellnek az adatok hierarchiájának megértésében
    profile_json = json.dumps(prompted_profile, indent=2, ensure_ascii=False)

    #print(profile_json)
    prompt = f"""
    Te egy személyre szabott életviteli tanácsadó AI vagy. A feladatod, hogy tanácsokat adj a felhasználónak az alábbi profilja alapján. 
    Válaszaidban kerüld az egészségre káros dolgok (pl. öngyilkosság, cigarette, stb.) ajánlását, mégha ez a felhasználó profiljához illene is. 
    Amennyiben a felhasználó olyan témában kérdez, amihez a profilja nem tartalmaz releváns adatot, kérdezd meg őt a preferenciáiról vagy fejezd ki tudatlanságodat, ahelyett, hogy egy általános választ adnál.
    Mindig válaszolj maximum 3 mondatban, hogy a felhasználó gyorsan tovább folytathassa életét.
    Minden válaszban vedd figyelembe a preferenciáit, a világnézetét és a jelenlegi élethelyzetét. 
    Ne adj olyan tanácsot, ami ellentétes az értékeivel vagy az étrendi igényeivel.

    FELHASZNÁLÓI PROFIL:
    ---
    {profile_json}
    ---

    """
    return prompt

BILL_PROFILES_FILE = Path("template_profiles.json")
USER_PROFILES_FILE = Path("user_profiles.json")
LEGACY_PROFILES_FILE = Path("profiles.json")
ENV_FILE = Path(".env")
MODEL_NAME = "gemini-3.1-flash-lite-preview"


DEFAULT_BILL_PROFILES = {
    "Bill": {
        "eletkor": "69 eves",
        "magassag": "177 cm",
        "testtomeg": "~70 kg",
        "testfelepites": ["atlagos", "idosebb, aktiv"],
        "MBTI tipus": "INTJ",
        "DISC tipus": ["D", "C"],
        "tarsadalmi nezetek": "technologia- es innovaciokozpontu, filantrop fokusz",
        "politikai allaspont": ["centrista", "globalis egyuttmukodes parti"],
        "etrendi preferencia": ["egyszeru, kiegyensulyozott etrend", "mertekletes adagok"],
        "kedvelt_etelek": ["hamburger", "sajtburger", "diet kola", "egyszeru amerikai fogasok"],
        "nem_kedvelt_etelek": ["tul fuszeres etelek", "cukros uditok nagy mennyisegben"],
        "utazasi_preferencia": ["uzleti utak", "nagyvarosok", "konferenciak", "hatekony idobeosztas"],
        "sportolasi szokasok": ["rendszeres seta", "tenisz alkalmankent"],
        "kapcsolati_statusz": "Elvalt",
        "munka": "uzletember, befekteto, filantrop",
        "baratok": "szeles szakmai es nemzetkozi kapcsolati halo",
        "csaladi_helyzet": ["3 gyermek", "aktiv csaladi kapcsolatok"],
        "lakcim": "Seattle, Washington, USA",
        "lakhely": ["kulvarosi villa", "csaladi kornyezet"],
    },
    "Template": {
        "eletkor": [],
        "magassag": [],
        "testtomeg": [],
        "testfelepites": [],
        "MBTI tipus": [],
        "DISC tipus": [],
        "tarsadalmi nezetek": [],
        "politikai allaspont": [],
        "etrendi preferencia": [],
        "kedvelt_etelek": [],
        "nem_kedvelt_etelek": [],
        "utazasi_preferencia": [],
        "sportolasi szokasok": [],
        "kapcsolati_statusz": [],
        "munka": [],
        "baratok": [],
        "csaladi_helyzet": [],
        "lakcim": [],
        "lakhely": [],
    },
}


def load_env_file() -> None:
    if not ENV_FILE.exists():
        return

    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _save_profiles(path: Path, profiles: dict) -> None:
    path.write_text(
        json.dumps(profiles, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_profiles(path: Path, default: dict) -> dict:
    if not path.exists():
        _save_profiles(path, default)
        return dict(default)

    try:
        profiles = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(profiles, dict):
            raise ValueError("Profile file is not a JSON object")
        return profiles
    except (json.JSONDecodeError, OSError, ValueError):
        _save_profiles(path, default)
        return dict(default)


def load_profile_sources() -> tuple[dict, dict]:
    bill_profiles = _load_profiles(BILL_PROFILES_FILE, DEFAULT_BILL_PROFILES)
    user_profiles = _load_profiles(USER_PROFILES_FILE, {})

    if "Bill" not in bill_profiles:
        bill_profiles["Bill"] = DEFAULT_BILL_PROFILES["Bill"]
        _save_profiles(BILL_PROFILES_FILE, bill_profiles)

    if "Template" not in bill_profiles:
        bill_profiles["Template"] = DEFAULT_BILL_PROFILES["Template"]
        _save_profiles(BILL_PROFILES_FILE, bill_profiles)

    if LEGACY_PROFILES_FILE.exists():
        legacy_profiles = _load_profiles(LEGACY_PROFILES_FILE, {})
        changed = False

        if "Bill" in legacy_profiles and "Bill" not in bill_profiles:
            bill_profiles["Bill"] = legacy_profiles["Bill"]
            changed = True

        for name, profile in legacy_profiles.items():
            if name == "Bill":
                continue
            if name not in user_profiles:
                user_profiles[name] = profile
                changed = True

        if changed:
            _save_profiles(BILL_PROFILES_FILE, bill_profiles)
            _save_profiles(USER_PROFILES_FILE, user_profiles)

    return bill_profiles, user_profiles


def prompt_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} (i/n): ").strip().lower()
        if answer in {"i", "igen", "y", "yes"}:
            return True
        if answer in {"n", "nem", "no"}:
            return False
        print("Kérlek i vagy n választ adj.")


def choose_or_create_profile(
    bill_profiles: dict,
    user_profiles: dict,
) -> tuple[str, Literal["bill", "user"]]:
    while True:
        print("\nElérhető profilok:")
        names = sorted((set(bill_profiles.keys()) | set(user_profiles.keys())) - {"Template"})
        for idx, name in enumerate(names, start=1):
            print(f"{idx}. {name}")
        print("0. Új profil létrehozása")

        choice = input("Válassz egy profilt sorszám alapján: ").strip()
        if not choice.isdigit():
            print("Kérlek számot adj meg.")
            continue

        choice_num = int(choice)
        if choice_num == 0:
            name = input("Új profil neve: ").strip()
            if not name:
                print("A profil neve nem lehet üres.")
                continue
            if name in bill_profiles or name in user_profiles:
                print("Ilyen nevű profil már létezik.")
                continue
            template_profile = bill_profiles.get("Template")
            if not isinstance(template_profile, dict):
                print("A Template profil hiányzik vagy hibás. Hozd létre újra a temp_profiles.json fájlban.")
                continue
            user_profiles[name] = deepcopy(template_profile)
            _save_profiles(USER_PROFILES_FILE, user_profiles)
            return name, "user"

        if 1 <= choice_num <= len(names):
            name = names[choice_num - 1]
            source: Literal["bill", "user"] = "bill" if name in bill_profiles else "user"
            return name, source

        print("Nincs ilyen sorszám.")


def _get_profile(
    profile_name: str,
    source: Literal["bill", "user"],
    bill_profiles: dict,
    user_profiles: dict,
) -> dict:
    return bill_profiles[profile_name] if source == "bill" else user_profiles[profile_name]


def _save_profile_source(
    source: Literal["bill", "user"],
    bill_profiles: dict,
    user_profiles: dict,
) -> None:
    if source == "bill":
        _save_profiles(BILL_PROFILES_FILE, bill_profiles)
    else:
        _save_profiles(USER_PROFILES_FILE, user_profiles)


def edit_profile(
    profile_name: str,
    source: Literal["bill", "user"],
    bill_profiles: dict,
    user_profiles: dict,
) -> None:
    profile = _get_profile(profile_name, source, bill_profiles, user_profiles)
    keys = list(profile.keys())

    if keys:
        print("\nTulajdonságok szerkesztése:")
        print("- Lista mezők: bármennyi új elemet adhatsz hozzá.")
        print("- Nem lista mezők: egyszer állíthatók be, ha még üresek.")
    for key in keys:
        print(f"\n- {key}")
        print(f"Jelenlegi érték: {profile[key]}")

        if not isinstance(profile[key], list):
            current_value = profile[key]
            if current_value not in (None, ""):
                print("Ez a nem lista típusú mező már be lett állítva, ezért nem módosítható.")
                continue

            if not prompt_yes_no("Szeretnéd ezt az értéket most beállítani?"):
                continue

            new_value = input("Új érték (Enter = kihagyás): ").strip()
            if not new_value:
                print("Nem történt módosítás.")
                continue

            profile[key] = new_value
            _save_profile_source(source, bill_profiles, user_profiles)
            continue

        if not prompt_yes_no("Szeretnél ehhez új érték(ek)et hozzáadni?"):
            continue

        additions: list[str] = []
        while True:
            value = input("Új érték (Enter = továbblépés): ").strip()
            if not value:
                break
            additions.append(value)

        if additions:
            profile[key].extend(additions)
        _save_profile_source(source, bill_profiles, user_profiles)


def format_history(history: list[dict]) -> str:
    lines = []
    for item in history:
        role = "Felhasználó" if item["role"] == "user" else "Asszisztens"
        lines.append(f"{role}: {item['text']}")
    return "\n".join(lines)


def chat_loop(client: genai.Client, profile: dict) -> None:
    history: list[dict] = []

    print("\nChat indul. Kilépés: /exit vagy /quit")
    while True:
        user_input = input("\nTe: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"/exit", "/quit"}:
            print("Kilépés.")
            return

        history.append({"role": "user", "text": user_input})

        history_text = format_history(history)
        contents = (
            "Beszélgetés előzménye:\n"
            f"{history_text}\n\n"
            f"A felhasználó legutóbbi üzenete: {user_input}"
        )

        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=generate_system_prompt(profile),
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )

        print("Asszisztens: ", end="", flush=True)
        response_parts = []
        for chunk in stream:
            if chunk.text:
                response_parts.append(chunk.text)
                print(chunk.text, end="", flush=True)
        print()

        assistant_text = "".join(response_parts).strip()
        history.append({"role": "assistant", "text": assistant_text})


def main() -> None:
    load_env_file()

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("Hiba: A GEMINI_API_KEY nincs beállítva környezeti változóként vagy a .env fájlban.")
        return

    bill_profiles, user_profiles = load_profile_sources()
    profile_name, source = choose_or_create_profile(bill_profiles, user_profiles)
    print(f"\nKiválasztott profil: {profile_name}")

    if prompt_yes_no("Szeretnél további tulajdonságokat beállítani?"):
        edit_profile(profile_name, source, bill_profiles, user_profiles)

    profile = _get_profile(profile_name, source, bill_profiles, user_profiles)
    client = genai.Client(api_key=api_key)
    chat_loop(client, profile)


if __name__ == "__main__":
    main()