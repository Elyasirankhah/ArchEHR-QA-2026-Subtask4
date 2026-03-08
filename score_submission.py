import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


Link = Tuple[str, str, str]  # (case_id, answer_id, evidence_id)


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def load_gold_links_from_dev_key(key_path: Path) -> Set[Link]:
    key = json.loads(key_path.read_text(encoding="utf-8"))
    gold: Set[Link] = set()
    for case in key:
        cid = str(case["case_id"])
        for a in case.get("clinician_answer_sentences", []):
            aid = str(a.get("id"))
            cits = a.get("citations", [])
            if cits is None:
                cits = []
            if isinstance(cits, (int, float)):
                cits = [str(int(cits))]
            elif isinstance(cits, str):
                cits = re.findall(r"\d+", cits)
            elif isinstance(cits, list):
                cits = [str(x) for x in cits]
            else:
                cits = []
            for eid in cits:
                gold.add((cid, aid, str(eid)))
    return gold


def load_pred_links(sub_path: Path) -> Set[Link]:
    sub = json.loads(sub_path.read_text(encoding="utf-8"))
    pred: Set[Link] = set()
    for case in sub:
        cid = str(case["case_id"])
        for item in case.get("prediction", []) or []:
            aid = str(item.get("answer_id"))
            for eid in item.get("evidence_id", []) or []:
                pred.add((cid, aid, str(eid)))
    return pred


def per_case_sets(links: Set[Link]) -> Dict[str, Set[Tuple[str, str]]]:
    by_case: Dict[str, Set[Tuple[str, str]]] = {}
    for cid, aid, eid in links:
        by_case.setdefault(cid, set()).add((aid, eid))
    return by_case


def score(gold_links: Set[Link], pred_links: Set[Link]) -> dict:
    tp = len(gold_links & pred_links)
    pred_n = len(pred_links)
    gold_n = len(gold_links)
    micro_p = tp / pred_n if pred_n else 0.0
    micro_r = tp / gold_n if gold_n else 0.0
    micro_f1 = _f1(micro_p, micro_r)

    gold_by_case = per_case_sets(gold_links)
    pred_by_case = per_case_sets(pred_links)
    case_ids = sorted(set(gold_by_case.keys()) | set(pred_by_case.keys()), key=lambda x: int(x) if x.isdigit() else x)
    per_case = []
    ps, rs, fs = [], [], []
    for cid in case_ids:
        g = gold_by_case.get(cid, set())
        p = pred_by_case.get(cid, set())
        tp_c = len(g & p)
        p_c = tp_c / len(p) if p else 0.0
        r_c = tp_c / len(g) if g else 0.0
        f_c = _f1(p_c, r_c)
        ps.append(p_c)
        rs.append(r_c)
        fs.append(f_c)
        per_case.append(
            {
                "case_id": cid,
                "precision": p_c,
                "recall": r_c,
                "f1": f_c,
                "tp": tp_c,
                "predicted": len(p),
                "gold": len(g),
            }
        )

    macro_p = sum(ps) / len(ps) if ps else 0.0
    macro_r = sum(rs) / len(rs) if rs else 0.0
    macro_f1 = sum(fs) / len(fs) if fs else 0.0

    return {
        "micro_precision": micro_p * 100.0,
        "micro_recall": micro_r * 100.0,
        "micro_f1": micro_f1 * 100.0,
        "macro_precision": macro_p * 100.0,
        "macro_recall": macro_r * 100.0,
        "macro_f1": macro_f1 * 100.0,
        "overall_score": micro_f1 * 100.0,
        "per_case": per_case,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=str, default=str(Path(__file__).resolve().parent / "v1.5" / "v1.5" / "dev" / "archehr-qa_key.json"))
    ap.add_argument("--submission", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    key_path = Path(args.key)
    sub_path = Path(args.submission)

    gold = load_gold_links_from_dev_key(key_path)
    pred = load_pred_links(sub_path)
    result = score(gold, pred)

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

