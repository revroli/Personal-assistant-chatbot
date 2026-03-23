import json
import os
import logging
from pathlib import Path
from copy import deepcopy
from typing import Literal

# Environment beállítások a verbose output letiltásához
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from google import genai
from google.genai import types

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Logging szintek csökkentése a külső függőségek verbose kimenetének letiltásához
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('langchain').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

# RAG profiladatok
PROFILE_FIELDS = [
    "eletkor",
    "magassag",
    "testtomeg",
    "testfelepites",
    "MBTI tipus",
    "DISC tipus",
    "tarsadalmi nezetek",
    "politikai allaspont",
    "etrendi preferencia",
    "kedvelt_etelek",
    "nem_kedvelt_etelek",
    "utazasi_preferencia",
    "sportolasi szokasok",
    "kapcsolati_statusz",
    "munka",
    "baratok",
    "csaladi_helyzet",
    "lakcim",
    "lakhely",
]

PROFILE_FIELD_DESCRIPTIONS = [
    "Kategória: eletkor. Jelentés: a felhasználó életkora vagy korcsoportja. Használat: élethelyzet, napi rutin, energia- és terhelhetőségi javaslatok finomhangolása.",
    "Kategória: magassag. Jelentés: testmagasságra vonatkozó alapadat. Használat: fizikai aktivitás, ergonómia, felszerelés- és méretválasztás kontextusa.",
    "Kategória: testtomeg. Jelentés: testsúlyra vonatkozó adat vagy tartomány. Használat: terhelési, mozgás- és életmód-ajánlások személyre szabásának egyik bemenete.",
    "Kategória: testfelepites. Jelentés: általános testalkat és fizikai alkatleírás. Használat: mozgásforma, edzéstípus és terhelés jellegének megválasztása.",
    "Kategória: MBTI tipus. Jelentés: személyiségdimenziókat leíró önjellemző kategória. Használat: kommunikációs stílus és tanácsadás hangnemének igazítása.",
    "Kategória: DISC tipus. Jelentés: viselkedési és kommunikációs preferenciákat jelző profil. Használat: konfliktuskezelés, visszajelzés és együttműködési ajánlások testreszabása.",
    "Kategória: tarsadalmi nezetek. Jelentés: társadalmi kérdésekhez kapcsolódó értékpreferenciák. Használat: közéleti vagy értékalapú témákban érzékeny, konzisztens válaszadás.",
    "Kategória: politikai allaspont. Jelentés: politikai orientáció vagy világnézeti beállítódás. Használat: közéleti kontextusban neutrális, de preferenciát tiszteletben tartó megfogalmazás.",
    "Kategória: etrendi preferencia. Jelentés: étkezési elvek, korlátozások és szokások összessége. Használat: recept- és ételajánlások szűrése, kompatibilis opciók előnyben részesítése.",
    "Kategória: kedvelt_etelek. Jelentés: preferált alapanyagok, fogások és ízvilágok. Használat: gyorsan elfogadható, motiváló ételötletek ajánlása.",
    "Kategória: nem_kedvelt_etelek. Jelentés: kerülendő ételek, italok vagy összetevők listája. Használat: kizárási szabályok alkalmazása javaslatgeneráláskor.",
    "Kategória: utazasi_preferencia. Jelentés: utazási stílus, célterület és költségkeret preferenciák. Használat: úticél- és időzítésajánlások személyre szabása.",
    "Kategória: sportolasi szokasok. Jelentés: mozgásgyakoriság, aktivitástípus és sportos rutin. Használat: fenntartható edzés- és regenerációs javaslatok kialakítása.",
    "Kategória: kapcsolati_statusz. Jelentés: aktuális párkapcsolati helyzet leírása. Használat: érzelmi, kommunikációs és élethelyzeti tanácsok kontextusba helyezése.",
    "Kategória: munka. Jelentés: tanulmányi vagy szakmai státusz és terhelési környezet. Használat: időgazdálkodási, produktivitási és stresszkezelési ajánlásokhoz.",
    "Kategória: baratok. Jelentés: társas kapcsolati háló erőssége és elérhetősége. Használat: közösségi támogatásra építő javaslatok vagy visszakérdezés szükségességének jelzése.",
    "Kategória: csaladi_helyzet. Jelentés: családszerkezet és családi kapcsolati dinamika. Használat: érzékeny, családi kontextushoz illesztett tanácsadás.",
    "Kategória: lakcim. Jelentés: földrajzi elhelyezkedésre utaló információ. Használat: helyhez kötött opciók, szolgáltatások vagy logisztikai javaslatok szűrése.",
    "Kategória: lakhely. Jelentés: lakhatási környezet típusa és együttélési feltételek. Használat: napi rutinra, térhasználatra és életmódra szabott ajánlások.",
]


def setup_rag() -> tuple:
    """Initialize RAG components: embeddings, reranker, and vector database."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
    
    metadatas = [{"orig_word": PROFILE_FIELDS[i]} for i in range(len(PROFILE_FIELD_DESCRIPTIONS))]
    vector_db = Chroma.from_texts(PROFILE_FIELD_DESCRIPTIONS, embeddings, metadatas=metadatas)
    
    return embeddings, reranker, vector_db


def generate_system_prompt(profile, alap_talalatok = None, talalatok = 5):
    
    if alap_talalatok is None:
        prompted_profile = profile
    else:
        # Az első talalatok számú találat metadatájából nyerjük ki az eredeti mezőneveket
        top_keys = [doc.metadata['orig_word'] for doc in alap_talalatok[:talalatok]]
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


def chat_loop(client: genai.Client, profile: dict, use_rag: bool = False, vector_db = None) -> None:
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

        # RAG módban releváns profil mezők szűrése
        alap_talalatok = None
        if use_rag and vector_db is not None:
            alap_talalatok = vector_db.similarity_search(user_input, k=10)

        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=generate_system_prompt(profile, alap_talalatok, talalatok=5),
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
    
    # RAG mód választása
    use_rag = prompt_yes_no("\nRAG módot szeretnél használni? (Relevánciafilteres profil)")
    vector_db = None
    
    if use_rag:
        print("RAG komponensek inicializálása...")
        try:
            _, _, vector_db = setup_rag()
            print("RAG rendszer sikeresen inicializálva.")
        except Exception as e:
            print(f"Hiba az RAG inicializálásában: {e}")
            print("RAG mód kikapcsolása.")
            use_rag = False
    
    client = genai.Client(api_key=api_key)
    chat_loop(client, profile, use_rag=use_rag, vector_db=vector_db)


if __name__ == "__main__":
    main()