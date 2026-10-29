#!/usr/bin/env python3
"""
Query the FCA Handbook API. Built to close one specific open item: the UK
commencement date for ESEF structured filing, in DTR 4.1.14, which carries the
cost argument for extending the corpus forward rather than back.

    export FCA_TOKEN='your bearer token'
    python3 fca_handbook.py tree                  # find the chapter_id for DTR
    python3 fca_handbook.py tree --grep DTR       # just the DTR branch
    python3 fca_handbook.py provisions DTR4.1     # the rule text
    python3 fca_handbook.py esef                  # tree + DTR 4.1, and grep for dates

Notes from the API documentation:
  timeline_date rejects PAST dates, so this always asks for today or later.
  Protected routes are rate limited and return 429 with Retry-After.
Standard library only.
"""
import argparse, json, os, re, sys, time, urllib.error, urllib.request

BASE = os.environ.get("FCA_HANDBOOK_BASE", "https://handbook.fca.org.uk/handbook-api")
TOKEN = os.environ.get("FCA_TOKEN", "")


def call(path, body, retries=3):
    if not TOKEN:
        sys.exit("set FCA_TOKEN first:  export FCA_TOKEN='...'")
    url = f"{BASE.rstrip('/')}/{path.lstrip('/')}"
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method="POST", headers={
        "Authorization": f"Bearer {TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                wait = int(e.headers.get("Retry-After") or 10)
                print(f"  rate limited, waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
                continue
            detail = e.read().decode(errors="replace")[:400]
            hint = {401: "token missing, wrong or expired",
                    400: "bad chapter_id, or a past timeline_date",
                    404: "no data for that chapter or date"}.get(e.code, "")
            sys.exit(f"HTTP {e.code} on {url}\n{hint}\n{detail}")
        except urllib.error.URLError as e:
            sys.exit(f"could not reach {url}: {e.reason}\n"
                     f"if the base path is different, set FCA_HANDBOOK_BASE")


def walk(nodes, depth=0):
    """Yield (depth, name, entityId, type) over the nested tree."""
    for n in nodes or []:
        yield depth, n.get("name", ""), n.get("entityId") or n.get("id"), n.get("type", "")
        yield from walk(n.get("parts") or n.get("contains") or [], depth + 1)


def cmd_tree(a):
    res = call("get_all_handbook_tree", {})
    tree = res.get("tree") or res.get("handbook") or []
    rows = list(walk(tree))
    print(f"{len(rows):,} nodes\n")
    pat = re.compile(a.grep, re.I) if a.grep else None
    shown = 0
    for depth, name, eid, typ in rows:
        if pat and not pat.search(f"{name} {eid}"):
            continue
        print(f"{'  '*min(depth,6)}{name}   [{typ}]  id={eid}")
        shown += 1
        if shown >= a.limit:
            print(f"... stopped at {a.limit}, raise --limit")
            break
    if pat and shown == 0:
        print(f"nothing matched {a.grep!r}. Run without --grep and look for the "
              f"sourcebook name the Handbook uses.")


def show_provisions(chapter_id, grep=None):
    res = call("get_latest_handbook_provisions",
               {"chapter_id": chapter_id, "is_metadata_required": False})
    provs = res.get("provision") or []
    print(f"{len(provs)} provisions under {chapter_id}\n")
    pat = re.compile(grep, re.I) if grep else None
    for p in provs:
        name = p.get("ProvisionName", "")
        content = re.sub(r"<[^>]+>", " ", p.get("Content") or "")
        content = re.sub(r"\s+", " ", content).strip()
        if pat and not (pat.search(name) or pat.search(content)):
            continue
        print(f"--- {name}  [{p.get('ProvisionType','')}]  {p.get('ProvisionDate','')}")
        print(f"    {p.get('HandbookTreeHierarchy', p.get('SectionName',''))}")
        print(f"    {content[:1500]}")
        print()
    return provs


def cmd_provisions(a):
    show_provisions(a.chapter_id, a.grep)


DATEPAT = (r"financial years? (beginning|commencing|starting) on or after|"
           r"1 January 20\d\d|"
           r"European Single Electronic Format|ESEF|"
           r"structured (electronic )?format|XHTML|inline XBRL|iXBRL")


def cmd_esef(a):
    print("Looking for the DTR branch of the Handbook tree\n")
    res = call("get_all_handbook_tree", {})
    rows = list(walk(res.get("tree") or []))
    hits = [(d, n, e, t) for d, n, e, t in rows
            if re.search(r"\bDTR\b|disclosure (guidance|and transparency)", f"{n} {e}", re.I)]
    if not hits:
        sys.exit("no DTR node found. Run 'tree' with no --grep and look for the "
                 "sourcebook name, then pass its id to 'provisions'.")
    for d, n, e, t in hits[:25]:
        print(f"  {n}   [{t}]  id={e}")

    cand = [e for d, n, e, t in hits
            if re.search(r"4\.1|^dtr4", f"{n} {e}", re.I)] or [hits[0][2]]
    print(f"\nTrying chapter_id {cand[0]!r} and filtering for commencement wording\n")
    show_provisions(str(cand[0]), DATEPAT)
    print("If nothing came back, pass the right id explicitly:")
    print("  python3 fca_handbook.py provisions <id> --grep 'on or after'")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tree"); t.add_argument("--grep"); t.add_argument("--limit", type=int, default=400)
    t.set_defaults(func=cmd_tree)
    p = sub.add_parser("provisions"); p.add_argument("chapter_id"); p.add_argument("--grep")
    p.set_defaults(func=cmd_provisions)
    e = sub.add_parser("esef"); e.set_defaults(func=cmd_esef)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
