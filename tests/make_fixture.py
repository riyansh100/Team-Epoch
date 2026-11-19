"""Build a tiny synthetic NTCIR Data Search fixture.

It mimics the layout of the real Hugging Face repo `mpkato/ntcir_data_search` (English part)
closely enough to exercise every code path of notebooks/01_build_ntcir_ddg.ipynb and
notebooks/02_eval_retrieval.ipynb without touching the 478 GB archive:

    <out>/hf/data_search_e_collection.jsonl.bz2      collection, one dataset per line
    <out>/hf/data_search_e_data.tar.bz2              data_search_e_data/<data_filename> members
    <out>/hf/data_search_{e,2_e}_*_topics.tsv         topics
    <out>/hf/data_search_{e,2_e}_*_qrels.txt          qrels (QID DOCID L0/L1/L2)
    <out>/web/files/<name>                            files served as fake "source URLs"

Edge cases built in (each is checked by tests/test_pipeline.py):
    * latin-1 encoded CSV, semicolon-delimited CSV, malformed row, UTF-8 BOM, quoted newline
    * a "large" CSV (exercises the streaming row-count + reservoir-sample path)
    * an HTML error page saved as .text.csv, an empty file
    * a dataset with several CSV files (multi-file policy) and junk non-CSV members
    * a needed file missing from the archive AND dead at its URL (acquisition failure)
    * URLs that are alive / 404 / return HTML / are slow / serve drifted content
    * data_format says "csv" but the MIME suffix says text.html (CSV-mode differences)

Usage:  python tests/make_fixture.py <out_dir>
The fixture URLs point at http://127.0.0.1:18765/ ; notebook 01 starts that server itself
when FIXTURE_MODE is on.
"""

from __future__ import annotations

import bz2
import hashlib
import io
import json
import random
import shutil
import sys
import tarfile
import uuid
from pathlib import Path

PORT = 18765
BASE = f"http://127.0.0.1:{PORT}"
NS = uuid.UUID("12345678-1234-5678-1234-567812345678")

rng = random.Random(42)

STATES = ["Alabama", "Alaska", "Arizona", "California", "Colorado", "Florida", "Georgia",
          "Hawaii", "Illinois", "Kansas", "Maine", "Nevada", "New York", "Ohio", "Texas"]


def did(name: str) -> str:
    return str(uuid.uuid5(NS, name))


def csv_text(header: list[str], rows: list[list], sep: str = ",") -> str:
    out = io.StringIO()
    out.write(sep.join(header) + "\n")
    for r in rows:
        out.write(sep.join(str(x) for x in r) + "\n")
    return out.getvalue()


def table(kind: str, n: int) -> str:
    """Generic small tables per theme."""
    if kind == "school":
        return csv_text(["state", "year", "enrollment", "graduation_rate"],
                        [[rng.choice(STATES), rng.randint(2005, 2018), rng.randint(1000, 90000),
                          round(rng.uniform(0.6, 0.95), 3)] for _ in range(n)])
    if kind == "crime":
        return csv_text(["incident_id", "date", "offense", "district", "latitude", "longitude"],
                        [[i, f"2017-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}",
                          rng.choice(["Burglary", "Assault", "Theft", "Robbery"]),
                          rng.choice(["North", "South", "Central"]),
                          round(rng.uniform(39.2, 39.4), 5), round(rng.uniform(-76.7, -76.5), 5)]
                         for i in range(n)])
    if kind == "hospital":
        return csv_text(["provider_id", "hospital_name", "state", "measure", "readmission_rate"],
                        [[10000 + i, f"General Hospital {i}", rng.choice(STATES),
                          rng.choice(["READM-30-AMI", "READM-30-HF"]),
                          round(rng.uniform(10, 25), 1)] for i in range(n)])
    if kind == "air":
        return csv_text(["site", "date", "ozone_ppm", "pm25"],
                        [[f"S{i % 7}", f"2016-05-{1 + i % 28:02d}", round(rng.uniform(0.01, 0.09), 3),
                          round(rng.uniform(2, 40), 1)] for i in range(n)])
    if kind == "traffic":
        return csv_text(["crash_id", "county", "year", "fatalities", "weather"],
                        [[i, f"County {i % 12}", rng.randint(2010, 2018), rng.randint(0, 3),
                          rng.choice(["Clear", "Rain", "Snow"])] for i in range(n)])
    if kind == "budget":
        return csv_text(["department", "fiscal_year", "amount_usd"],
                        [[rng.choice(["Police", "Fire", "Parks", "Water"]), rng.randint(2012, 2019),
                          rng.randint(10 ** 5, 10 ** 8)] for _ in range(n)])
    raise ValueError(kind)


HTML_PAGE = ("<!DOCTYPE html>\n<html><head><title>Page not found</title></head>"
             "<body><h1>404</h1><p>The dataset you requested has moved.</p></body></html>\n")

PDF_JUNK = b"%PDF-1.4\n" + bytes(rng.randrange(256) for _ in range(3000))


def build(out: Path) -> dict:
    if out.exists():
        shutil.rmtree(out)
    hf = out / "hf"
    web = out / "web" / "files"
    hf.mkdir(parents=True)
    web.mkdir(parents=True)

    datasets: list[dict] = []
    files: dict[str, bytes] = {}         # data_filename -> archive content
    web_files: dict[str, bytes] = {}     # web name -> served content
    not_in_archive: set[str] = set()
    expected: dict[str, str] = {}        # doc name -> expected outcome (for tests)

    def add(name: str, title: str, desc: str, file_specs: list[dict]) -> str:
        """file_specs: dicts with content(bytes), fmt, mime, url_mode(alive|dead|html|slow|drift),
        in_archive(bool)."""
        data = []
        for k, fs in enumerate(file_specs):
            webname = f"{name}_{k}.{'csv' if fs['mime'] == 'text.csv' else fs['mime'].split('.')[-1]}"
            mode = fs.get("url_mode", "alive")
            url = f"{BASE}/{'slow' if mode == 'slow' else 'files'}/{webname}"
            fname = hashlib.sha256(url.encode()).hexdigest() + "." + fs["mime"]
            content = fs["content"]
            if fs.get("in_archive", True):
                files[fname] = content
            else:
                not_in_archive.add(fname)
            if mode in ("alive", "slow"):
                web_files[webname] = content
            elif mode == "drift":   # the live portal has been updated since 2019: one extra row
                web_files[webname] = content + content.splitlines(keepends=True)[-1]
            elif mode == "html":
                web_files[webname] = HTML_PAGE.encode()
            # dead -> not served (404)
            data.append({"data_format": fs["fmt"], "data_organization": "Fixture Agency",
                         "data_url": url, "data_filename": fname})
        rec = {
            "id": did(name), "title": title, "description": desc, "data": data,
            "data_fields": {"Publisher": "Fixture Agency", "Metadata Created Date": "May 1, 2019",
                            "Landing Page": f"{BASE}/landing/{name}"},
            "url": f"https://catalog.data.gov/dataset/{did(name)}",
            "attribution": (f"{title} (https://catalog.data.gov/dataset/{did(name)}) is licensed "
                            "under " + rng.choice(["CC0 1.0 (https://creativecommons.org/publicdomain/zero/1.0/)",
                                                   "U.S. Government Work (http://www.usa.gov/publicdomain/label/1.0/)"])),
        }
        datasets.append(rec)
        return rec["id"]

    def plain(kind: str, n: int = 30, **kw) -> dict:
        return {"content": table(kind, n).encode(), "fmt": "csv", "mime": "text.csv", **kw}

    ids: dict[str, list[str]] = {k: [] for k in ["school", "crime", "hospital", "air", "traffic", "budget"]}
    words = {
        "school": ("School enrollment", "Public school enrollment and graduation rates by state and year."),
        "crime": ("Crime incidents", "Police reported crime incidents with offense type, district and location."),
        "hospital": ("Hospital readmission", "Hospital 30-day readmission rates by provider and measure."),
        "air": ("Air quality ozone", "Daily ozone and PM2.5 air quality measurements at monitoring sites."),
        "traffic": ("Traffic crashes", "Traffic accident and crash records with fatalities and weather."),
        "budget": ("City budget", "Municipal budget appropriations by department and fiscal year."),
    }
    for kind, count in [("school", 8), ("crime", 8), ("hospital", 8), ("air", 3), ("traffic", 8), ("budget", 6)]:
        for i in range(count):
            t, d = words[kind]
            name = f"{kind}{i}"
            mode = ["alive", "dead", "alive", "html", "alive", "dead", "drift", "alive"][i % 8]
            spec = plain(kind, rng.randint(15, 60), url_mode=mode)
            ids[kind].append(add(name, f"{t} {2010 + i}", f"{d} Edition {i}.", [spec]))

    # ---- edge cases (all themed "school" or "crime" so they land in query pools) ----
    latin = csv_text(["city", "province", "population"],
                     [["Montréal", "Québec", 1704694], ["Trois-Rivières", "Québec", 134413],
                      ["Saint-Jérôme", "Québec", 80213]] * 5).encode("latin-1")
    ids["school"].append(add("latin1", "School districts Québec",
                         "School district populations, encoded in latin-1.",
                         [{"content": latin, "fmt": "csv", "mime": "text.csv", "url_mode": "dead"}]))
    expected["latin1"] = "ok:latin-1"

    semi = table("school", 25).replace(",", ";").encode()
    ids["school"].append(add("semicolon", "School enrollment semicolon export",
                             "Enrollment export using semicolon separators.",
                             [{"content": semi, "fmt": "csv", "mime": "text.csv", "url_mode": "alive"}]))
    expected["semicolon"] = "ok:delimiter=;"

    bad = table("crime", 20).splitlines()
    bad.insert(5, "999,2017-01-01,Theft,North,39.3,-76.6,EXTRA,FIELDS")
    ids["crime"].append(add("malformed", "Crime incidents with a malformed row",
                            "Crime incidents export containing one malformed row.",
                            [{"content": ("\n".join(bad) + "\n").encode(), "fmt": "csv",
                              "mime": "text.csv", "url_mode": "dead"}]))
    expected["malformed"] = "ok:bad_lines=1"

    bom = ("﻿" + table("crime", 20)).encode("utf-8")
    ids["crime"].append(add("bom", "Crime incidents BOM", "Crime incidents with a UTF-8 byte order mark.",
                            [{"content": bom, "fmt": "csv", "mime": "text.csv"}]))
    expected["bom"] = "ok:utf-8-sig"

    qn = 'id,comment,score\n1,"line one\nline two",5\n2,"plain",3\n' + "".join(
        f'{i},"c{i}",{i % 5}\n' for i in range(3, 30))
    ids["crime"].append(add("quotednl", "Crime survey comments", "Survey free-text comments about crime.",
                            [{"content": qn.encode(), "fmt": "csv", "mime": "text.csv"}]))
    expected["quotednl"] = "ok:rows=29"

    large = table("school", 12000).encode()   # ~0.4 MB; fixture LARGE_FILE_BYTES is 200 KB
    ids["school"].append(add("large", "School enrollment large panel",
                             "A large enrollment panel used to exercise streaming profiling.",
                             [{"content": large, "fmt": "csv", "mime": "text.csv", "url_mode": "dead"}]))
    expected["large"] = "ok:rows=12000:streamed"

    ids["crime"].append(add("htmlpage", "Crime incidents (moved)", "Crime incidents; the file is an HTML error page.",
                            [{"content": HTML_PAGE.encode(), "fmt": "csv", "mime": "text.csv", "url_mode": "dead"}]))
    expected["htmlpage"] = "error:html"

    ids["school"].append(add("empty", "School enrollment empty file", "An empty enrollment file.",
                             [{"content": b"", "fmt": "csv", "mime": "text.csv", "url_mode": "dead"}]))
    expected["empty"] = "error:empty"

    multi = [plain("crime", 10, url_mode="dead"), plain("crime", 40, url_mode="dead"),
             {"content": PDF_JUNK, "fmt": "pdf", "mime": "application.pdf", "url_mode": "dead"}]
    ids["crime"].append(add("multi", "Crime incidents by quarter (multi-file)",
                            "Quarterly crime incident files.", multi))
    expected["multi"] = "ok:multi"

    ids["school"].append(add("missing", "School enrollment missing file",
                             "Enrollment file that is missing from the archive and offline.",
                             [plain("school", 10, url_mode="dead", in_archive=False)]))
    expected["missing"] = "error:not_acquired"

    ids["school"].append(add("slowurl", "School enrollment slow server",
                             "Enrollment file whose server is slow.",
                             [plain("school", 12, url_mode="slow")]))
    expected["slowurl"] = "ok"

    # data_format says csv, but the stored file is an HTML page (mime text.html)
    ids["crime"].append(add("fmtonly", "Crime incidents portal page", "Portal page for crime data.",
                            [{"content": HTML_PAGE.encode(), "fmt": "csv", "mime": "text.html",
                              "url_mode": "dead"}]))

    # non-CSV datasets (judged but not CSV)
    pdfs = [add(f"pdf{i}", f"Annual report {i}", "A PDF report.",
                [{"content": PDF_JUNK, "fmt": "pdf", "mime": "application.pdf", "url_mode": "dead"}])
            for i in range(6)]
    # unrelated junk members in the archive
    for i in range(20):
        files[hashlib.sha256(f"junk{i}".encode()).hexdigest() + ".application.pdf"] = PDF_JUNK

    # ---- topics & qrels ----
    def qrels_for(qid: str, rel_kind: str, nonrel: list[str], l2: int = 2) -> list[str]:
        lines = []
        rel = ids[rel_kind]
        for k, d in enumerate(rel):
            lines.append(f"{qid} {d} {'L2' if k < l2 else 'L1'}")
        for d in nonrel:
            lines.append(f"{qid} {d} L0")
        for d in pdfs[:2]:
            lines.append(f"{qid} {d} L1")   # relevant but not CSV -> filtered out
        return lines

    topics = {
        "e_train": [("DS1-E-0001", "school enrollment graduation by state"),
                    ("DS1-E-0002", "crime incidents police district")],
        "e_test": [("DS1-E-1001", "hospital readmission rates"),
                   ("DS1-E-1002", "ozone air quality")],
        "2_e_test": [("DS2-E-0001", "traffic crash fatalities")],
    }
    q_train = (qrels_for("DS1-E-0001", "school", ids["budget"][:4] + ids["crime"][:2])
               + qrels_for("DS1-E-0002", "crime", ids["budget"][2:6] + ids["school"][:3]))
    q_test = (qrels_for("DS1-E-1001", "hospital", ids["budget"][:3] + ids["traffic"][:2])
              + qrels_for("DS1-E-1002", "air", ids["budget"][:5] + ids["hospital"][:3]))
    q_extra = [f"DS1-E-1001 {ids['traffic'][3]} L0", f"DS1-E-1001 {ids['school'][1]} L1"]
    q_2test = qrels_for("DS2-E-0001", "traffic", ids["budget"][:4] + ids["air"][:2])

    def wtopics(fn: str, rows: list[tuple[str, str]]) -> None:
        (hf / fn).write_text("".join(f"{a}\t{b}\n" for a, b in rows))

    def wqrels(fn: str, rows: list[str]) -> None:
        (hf / fn).write_text("\n".join(rows) + "\n")

    wtopics("data_search_e_train_topics.tsv", topics["e_train"])
    wtopics("data_search_e_test_topics.tsv", topics["e_test"])
    wtopics("data_search_2_e_train_topics.tsv", topics["e_train"] + topics["e_test"])
    wtopics("data_search_2_e_test_topics.tsv", topics["2_e_test"])
    wqrels("data_search_e_train_qrels.txt", q_train)
    wqrels("data_search_e_test_qrels.txt", q_test)
    wqrels("data_search_e_extra_qrels.txt", q_extra)
    wqrels("data_search_e_all_qrels.txt", q_test + q_extra)
    wqrels("data_search_2_e_train_qrels.txt", q_train + q_test + q_extra)
    wqrels("data_search_2_e_test_qrels.txt", q_2test)

    with bz2.open(hf / "data_search_e_collection.jsonl.bz2", "wt", encoding="utf-8") as f:
        for rec in datasets:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # archive: directory entry first, members in shuffled order like the real one
    names = sorted(files)
    rng.shuffle(names)
    with tarfile.open(hf / "data_search_e_data.tar.bz2", "w:bz2") as tar:
        d = tarfile.TarInfo("data_search_e_data/")
        d.type = tarfile.DIRTYPE
        tar.addfile(d)
        for n in names:
            ti = tarfile.TarInfo(f"data_search_e_data/{n}")
            ti.size = len(files[n])
            tar.addfile(ti, io.BytesIO(files[n]))

    for n, c in web_files.items():
        (web / n).write_bytes(c)
    (hf / "README.md").write_text("# Fixture of NTCIR Data Search\n")

    meta = {"expected": {k: v for k, v in expected.items()}, "ids": {k: did(k) for k in expected},
            "n_datasets": len(datasets), "n_archive_members": len(files)}
    (out / "fixture_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "fixture")
    m = build(out_dir)
    print(f"fixture written to {out_dir}: {m['n_datasets']} datasets, {m['n_archive_members']} archive members")
