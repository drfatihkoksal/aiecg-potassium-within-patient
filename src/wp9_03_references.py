"""Manuscript reference list in AMA style, built from Crossref/DataCite metadata.

Every reference is resolved from its DOI so that authors, volume, issue and pages
come from the registry rather than from memory. Order = order of first citation.
Writes outputs/manuscript/references_ama.json.
"""
import json, time, urllib.parse, urllib.request
from paths import OUT

DOIS = ["10.1001/jamacardio.2019.0640", "10.2196/15931", "10.1038/s41746-021-00550-0",
        "10.1038/s41598-024-71562-5", "10.1038/s41598-024-65223-w", "10.7759/cureus.113226",
        "10.1016/j.jacep.2024.07.023", "10.1038/s41598-025-85722-8", "10.1093/ckj/sfaf092",
        "10.1053/j.ajkd.2026.05.020", "10.1093/ehjdh/ztac072", "10.1093/ehjdh/ztaf011",
        "10.1093/gigascience/gix019", "10.1038/s41746-019-0178-x", "10.1109/access.2023.3284458",
        "10.1097/hjh.0000000000003224", "10.1038/srep42492", "10.1038/s41598-021-82935-5",
        "10.1016/j.compbiomed.2022.105304", "10.22489/cinc.2020.461", "10.1038/s41597-022-01899-x",
        "10.13026/kpb9-mt58", "10.13026/4nqg-sb35", "10.1038/s44360-026-00096-z", "10.1136/bmj-2023-078378",
        "10.1056/aioa2401033", "10.1093/clinchem/25.3.432", "10.1177/096228029900800204",
        "10.3389/fpsyg.2017.00456", "10.1161/circulationaha.125.076279", "10.1161/jaha.115.002746"]
UA = {"User-Agent": "deltaK-manuscript/1.0"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def initials(given):
    parts = given.replace("-", " ").replace(".", " ").split()
    return "".join(p[0].upper() for p in parts if p)


def authors_ama(people):
    names = [f"{p['family']} {initials(p.get('given', ''))}".strip() for p in people if "family" in p]
    return ", ".join(names) if len(names) <= 6 else ", ".join(names[:3]) + ", et al"


def from_crossref(doi):
    m = get("https://api.crossref.org/works/" + urllib.parse.quote(doi))["message"]
    title = (m.get("title") or [""])[0].rstrip(".")
    journal = (m.get("short-container-title") or m.get("container-title") or [""])[0]
    year = (m.get("published-print") or m.get("published-online") or m["issued"])["date-parts"][0][0]
    loc = f"{year}"
    if m.get("volume"):
        loc += f";{m['volume']}"
        if m.get("issue"):
            loc += f"({m['issue']})"
        if m.get("page"):
            loc += f":{m['page']}"
        elif m.get("article-number"):
            loc += f":{m['article-number']}"
    return f"{authors_ama(m.get('author', []))}. {title}. {journal}. {loc}. doi:{doi}"


def from_datacite(doi):
    a = get("https://api.datacite.org/dois/" + doi)["data"]["attributes"]
    people = [{"family": c.get("familyName", c["name"]), "given": c.get("givenName", "")} for c in a["creators"]]
    ver = f" (version {a['version']})" if a.get("version") else ""
    return f"{authors_ama(people)}. {a['titles'][0]['title']}{ver}. {a['publisher'] if isinstance(a['publisher'], str) else a['publisher']['name']}. {a['publicationYear']}. doi:{doi}"


# NLM journal abbreviations, and registry records whose author strings are malformed
ABBREV = {"npj Digit. Med.": "NPJ Digit Med", "JACC: Clinical Electrophysiology": "JACC Clin Electrophysiol",
          "Clinical Kidney Journal": "Clin Kidney J", "American Journal of Kidney Diseases": "Am J Kidney Dis",
          "European Heart Journal - Digital Health": "Eur Heart J Digit Health",
          "Journal of Hypertension": "J Hypertens", "Computers in Biology and Medicine": "Comput Biol Med",
          "Computing in Cardiology Conference (CinC)": "Comput Cardiol", "Clinical Chemistry": "Clin Chem",
          "Front. Psychol.": "Front Psychol", "JAHA": "J Am Heart Assoc", "Nat. Health": "Nat Health"}
FIXES = {"Costa TBDS, ": "da Silva Costa TB, ",
         "SRINIVASAN S, Hassaan Ahmed Bukhari S, ": "Srinivasan S, Bukhari HA, ",
         "NEJM AI. 2025;2(7).": "NEJM AI. 2025;2(7):AIoa2401033."}

refs = []
for doi in DOIS:
    r = from_datacite(doi) if doi.startswith("10.13026/") else from_crossref(doi)
    for a, b in ABBREV.items():
        r = r.replace(f". {a}. ", f". {b}. ")
    for a, b in FIXES.items():
        r = r.replace(a, b)
    refs.append(r)
    print(len(refs), refs[-1])
    time.sleep(0.3)
json.dump(refs, open(OUT / "manuscript" / "references_ama.json", "w"), indent=1, ensure_ascii=False)
