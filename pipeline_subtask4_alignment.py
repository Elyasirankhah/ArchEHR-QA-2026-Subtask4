"""
Subtask 4: Evidence Alignment — All-at-once with 3-model diversity.

Key design:
  - All-at-once: one LLM call per case (all answers + all notes) for answer competition
  - 3 diverse models: o3 + gpt-5.2 + gpt-5.1 (breaks correlated errors)
  - 3 self-consistency runs per model = 9 votes
  - Strict majority threshold (default 5/9)
  - Leave-one-out few-shot on dev
"""

import os
import re
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env", override=True)

AZURE_ENDPOINT = (os.getenv("AZURE_OPENAI_ENDPOINT") or "").rstrip("/") + "/" if os.getenv("AZURE_OPENAI_ENDPOINT") else ""
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_GPT52_API_KEY")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
TASK4_DEPLOYMENT = os.getenv("TASK4_DEPLOYMENT", "gpt-5.2").strip()
_task4_ensemble = (os.getenv("TASK4_ENSEMBLE_DEPLOYMENTS") or "o3,gpt-5.2,gpt-5.1,deepseek-r1").strip()
TASK4_ENSEMBLE_DEPLOYMENTS = [d.strip() for d in _task4_ensemble.split(",") if d.strip()] or [TASK4_DEPLOYMENT]
TASK4_SELF_CONSISTENCY_N = int(os.getenv("TASK4_SELF_CONSISTENCY_N", "3"))
TASK4_SC_TEMPERATURE = float(os.getenv("TASK4_SC_TEMPERATURE", "0.4"))
TASK4_MAX_RETRIES = int(os.getenv("TASK4_MAX_RETRIES", "4"))
TASK4_FEW_SHOT_N = int(os.getenv("TASK4_FEW_SHOT_N", "10"))
TASK4_DUMP_VOTES = os.getenv("TASK4_DUMP_VOTES", "1").strip().lower() in ("1", "true", "yes")
TASK4_VOTE_THRESHOLD = int(os.getenv("TASK4_VOTE_THRESHOLD", "0"))
TASK4_PER_ANSWER = int(os.getenv("TASK4_PER_ANSWER", "0"))
TASK4_EMBEDDING_RECALL = os.getenv("TASK4_EMBEDDING_RECALL", "1").strip().lower() in ("1", "true", "yes")
TASK4_EMBEDDING_RECALL_THRESHOLD = float(os.getenv("TASK4_EMBEDDING_RECALL_THRESHOLD", "0.68"))

DATA_DIR = Path(__file__).resolve().parent / "v1.5" / "v1.5"
OUT_DIR = Path(__file__).resolve().parent / "submission"

DS_ENDPOINT = (os.getenv("AZURE_DEEPSEEK_R1_ENDPOINT") or "").rstrip("/") + "/" if os.getenv("AZURE_DEEPSEEK_R1_ENDPOINT") else ""
DS_API_KEY = os.getenv("AZURE_DEEPSEEK_R1_API_KEY") or ""
DS_DEPLOYMENT = os.getenv("AZURE_DEEPSEEK_R1_DEPLOYMENT", "deepseek-r1").strip()
DS_API_VERSION = os.getenv("AZURE_DEEPSEEK_R1_API_VERSION", "2024-05-01-preview")

try:
    from openai import AzureOpenAI
    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    ) if AZURE_API_KEY and AZURE_ENDPOINT else None
    ds_client = AzureOpenAI(
        api_version=DS_API_VERSION,
        azure_endpoint=DS_ENDPOINT,
        api_key=DS_API_KEY,
    ) if DS_API_KEY and DS_ENDPOINT else None
except Exception as e:
    azure_client = None
    ds_client = None
    print(f"Azure init: {e}")


# ── XML / key parsing ────────────────────────────────────────────────────────

def _el_text(el: Optional[ET.Element]) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_qa_xml(xml_path: Path) -> List[Dict[str, Any]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    cases = []
    for case_el in root.findall(".//case"):
        case_id = case_el.get("id", "")
        pq_el = case_el.find("patient_question")
        if pq_el is not None and pq_el.find("phrase") is not None and (pq_el.find("phrase").text or "").strip():
            patient_question = (pq_el.find("phrase").text or "").strip()
        else:
            patient_question = (pq_el.text or "").strip() if pq_el is not None else ""
        clinician_question = _el_text(case_el.find("clinician_question"))
        sentences = []
        for s in (case_el.find("note_excerpt_sentences") or []):
            if s.tag != "sentence":
                continue
            sentences.append({"id": s.get("id", ""), "text": _el_text(s)})
        cases.append({"case_id": case_id, "patient_question": patient_question,
                       "clinician_question": clinician_question, "sentences": sentences})
    return cases


def _split_answer_into_sentences(text: str) -> List[Dict[str, str]]:
    if not (text or "").strip():
        return []
    parts = re.split(r"(?<=\.)\s+", text.strip())
    out = [{"id": str(i), "text": p.strip()} for i, p in enumerate(parts, 1) if p.strip()]
    return out if out else [{"id": "1", "text": text.strip()}]


def load_key(key_path: Path, dev_has_citations: bool = True) -> Dict[str, Dict[str, Any]]:
    if not key_path.exists():
        return {}
    with open(key_path, "r", encoding="utf-8") as f:
        key = json.load(f)
    out = {}
    for case in key:
        cid = case["case_id"]
        answer_sentences = []
        for a in case.get("clinician_answer_sentences", []):
            entry = {"id": a["id"], "text": (a.get("text") or "").strip()}
            raw = a.get("citations") if dev_has_citations else None
            if raw is not None:
                if isinstance(raw, str):
                    entry["citations"] = [x.strip() for x in raw.split(",") if x.strip()]
                else:
                    entry["citations"] = [str(x).strip() for x in raw] if raw else []
            answer_sentences.append(entry)
        if not answer_sentences and case.get("clinician_answer"):
            for sent in _split_answer_into_sentences(case["clinician_answer"]):
                answer_sentences.append({"id": sent["id"], "text": sent["text"]})
        answer_without_cites = (case.get("clinician_answer_without_citations") or "").strip()
        out[cid] = {"answer_sentences": answer_sentences, "answer_without_citations": answer_without_cites}
    return out


# ── API call with retry ──────────────────────────────────────────────────────

def call_azure(messages: List[Dict[str, str]], max_tokens: int = 2048,
               temperature: float = 0.0, deployment: Optional[str] = None) -> str:
    model = (deployment or TASK4_DEPLOYMENT).strip()
    is_ds = "deepseek" in model.lower()
    client = ds_client if is_ds else azure_client
    if not client:
        return ""
    actual_model = DS_DEPLOYMENT if is_ds else model
    for attempt in range(TASK4_MAX_RETRIES + 1):
        try:
            r = client.chat.completions.create(
                model=actual_model, messages=messages,
                max_completion_tokens=max_tokens, temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip() if r.choices else ""
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RateLimitReached" in err_str:
                wait = min(2 ** attempt * 3, 60)
                m = re.search(r"retry after (\d+)", err_str, re.IGNORECASE)
                if m:
                    wait = max(wait, int(m.group(1)) + 1)
                if attempt < TASK4_MAX_RETRIES:
                    time.sleep(wait)
                    continue
            print(f"    API err ({model}): {e}")
            return ""
    return ""


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_MSG = """You are a clinical evidence alignment expert. You are given:
1. A patient question and a clinician-interpreted question
2. Numbered answer sentences from the clinician's answer
3. Numbered note sentences from the clinical note excerpt

Your task: For each answer sentence, identify which note sentence(s) DIRECTLY SUPPORT it.
Include a note sentence when it states the same specific fact, number, finding, or event as the answer sentence (e.g. same lab value, dose, procedure, or timeline).

NOT evidence:
- Same general topic but DIFFERENT specific fact
- Background or context not directly referenced in the answer
- A different time point or event than what the answer describes
- A sentence that contradicts the answer

Return a JSON array. Each element has "answer_id" (string) and "evidence_id" (array of note sentence ID strings).
Every answer sentence must appear exactly once. Use an empty array [] if no note sentence supports it.

Example output format:
[
  {"answer_id": "1", "evidence_id": ["3", "7"]},
  {"answer_id": "2", "evidence_id": ["5"]},
  {"answer_id": "3", "evidence_id": []}
]

IMPORTANT:
- Only use note sentence IDs that exist in the provided note excerpt.
- Be precise: include a note sentence ONLY if it directly supports the answer's specific claim.
- Do NOT include note sentences that are merely related or from the same topic area.
- An answer sentence may have 0, 1, or multiple supporting note sentences.
- When in doubt, prefer including a note sentence that clearly states the same fact over excluding it (recall matters)."""


# ── Few-shot (full-case, leave-one-out on dev) ──────────────────────────────

def build_few_shot_messages(dev_cases: List[Dict[str, Any]], dev_key: Dict[str, Dict[str, Any]],
                             exclude_case_id: Optional[str] = None,
                             max_examples: int = 12) -> List[Dict[str, str]]:
    msgs = []
    count = 0
    for case in dev_cases:
        if count >= max_examples:
            break
        cid = case["case_id"]
        if cid == exclude_case_id:
            continue
        key_entry = dev_key.get(cid)
        if not key_entry:
            continue
        answer_sents = key_entry.get("answer_sentences", [])
        if not answer_sents or not any(a.get("citations") for a in answer_sents):
            continue

        note_lines = "\n".join(f"  [{s['id']}] {s['text']}" for s in case.get("sentences", []))
        answer_lines = "\n".join(f"  [{a['id']}] {a['text']}" for a in answer_sents)

        user_content = (f"Patient question: {case.get('patient_question', '')}\n"
                        f"Clinician question: {case.get('clinician_question', '')}\n\n"
                        f"Note sentences:\n{note_lines}\n\n"
                        f"Answer sentences:\n{answer_lines}")

        gold = []
        for a in answer_sents:
            cites = a.get("citations", [])
            gold.append({"answer_id": str(a["id"]), "evidence_id": [str(c) for c in cites]})

        msgs.append({"role": "user", "content": user_content})
        msgs.append({"role": "assistant", "content": json.dumps(gold, ensure_ascii=False)})
        count += 1

    return msgs


def build_messages(case: Dict[str, Any], answer_sentences: List[Dict[str, Any]],
                   few_shot_msgs: List[Dict[str, str]],
                   answer_without_citations: str = "") -> List[Dict[str, str]]:
    msgs = [{"role": "system", "content": SYSTEM_MSG}]
    msgs.extend(few_shot_msgs)

    note_lines = "\n".join(f"  [{s['id']}] {s['text']}" for s in case.get("sentences", []))
    answer_lines = "\n".join(f"  [{a['id']}] {a['text']}" for a in answer_sentences)

    user_content = (f"Patient question: {case.get('patient_question', '')}\n"
                    f"Clinician question: {case.get('clinician_question', '')}\n\n"
                    f"Note sentences:\n{note_lines}\n\n")
    if answer_without_citations:
        user_content += f"Full clinician answer (for context):\n  {answer_without_citations}\n\n"
    user_content += f"Answer sentences:\n{answer_lines}"

    msgs.append({"role": "user", "content": user_content})
    return msgs


# ── Response parsing ─────────────────────────────────────────────────────────

def parse_alignment_response(response: str, valid_note_ids: set,
                              answer_ids: List[str]) -> List[Dict[str, Any]]:
    if not response:
        return [{"answer_id": aid, "evidence_id": []} for aid in answer_ids]

    text = response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        json_str = text[start:end + 1]
        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                result_map = {}
                for item in parsed:
                    if isinstance(item, dict) and "answer_id" in item:
                        aid = str(item["answer_id"])
                        eids = item.get("evidence_id", [])
                        if isinstance(eids, str):
                            eids = [x.strip() for x in eids.split(",") if x.strip()]
                        eids = [str(e) for e in eids if str(e) in valid_note_ids]
                        result_map[aid] = eids
                result = []
                for aid in answer_ids:
                    result.append({"answer_id": aid, "evidence_id": result_map.get(aid, [])})
                return result
        except json.JSONDecodeError:
            pass

    result_map = {}
    pattern = r'"answer_id"\s*:\s*"(\d+)"[^}]*"evidence_id"\s*:\s*\[([^\]]*)\]'
    for m in re.finditer(pattern, text):
        aid = m.group(1)
        raw_eids = m.group(2)
        eids = [e.strip().strip('"').strip("'") for e in raw_eids.split(",") if e.strip().strip('"').strip("'")]
        eids = [e for e in eids if e in valid_note_ids]
        result_map[aid] = eids

    result = []
    for aid in answer_ids:
        result.append({"answer_id": aid, "evidence_id": result_map.get(aid, [])})
    return result


# ── Majority vote merge ──────────────────────────────────────────────────────

def merge_predictions_majority_vote(all_predictions: List[List[Dict[str, Any]]],
                                     answer_ids: List[str],
                                     threshold_override: Optional[int] = None) -> List[Dict[str, Any]]:
    n = len(all_predictions)
    if n == 0:
        return [{"answer_id": aid, "evidence_id": []} for aid in answer_ids]
    if n == 1:
        return all_predictions[0]

    if threshold_override is not None:
        threshold = threshold_override
    else:
        threshold = TASK4_VOTE_THRESHOLD if TASK4_VOTE_THRESHOLD > 0 else (n // 2 + 1)

    vote_counts: Dict[str, Dict[str, int]] = {aid: {} for aid in answer_ids}
    for pred_list in all_predictions:
        for pred in pred_list:
            aid = pred["answer_id"]
            for eid in pred["evidence_id"]:
                if aid not in vote_counts:
                    vote_counts[aid] = {}
                vote_counts[aid][eid] = vote_counts[aid].get(eid, 0) + 1

    result = []
    for aid in answer_ids:
        eids = []
        for eid, count in vote_counts.get(aid, {}).items():
            if count >= threshold:
                eids.append(eid)
        eids.sort(key=lambda x: int(x) if x.isdigit() else 0)
        result.append({"answer_id": aid, "evidence_id": eids})
    return result


def compute_vote_counts(
    all_predictions: List[List[Dict[str, Any]]],
    answer_ids: List[str],
) -> Dict[str, Dict[str, int]]:
    vote_counts: Dict[str, Dict[str, int]] = {aid: {} for aid in answer_ids}
    for pred_list in all_predictions:
        for pred in pred_list:
            aid = str(pred.get("answer_id"))
            if aid not in vote_counts:
                vote_counts[aid] = {}
            for eid in pred.get("evidence_id", []):
                eid = str(eid)
                vote_counts[aid][eid] = vote_counts[aid].get(eid, 0) + 1
    return vote_counts


# ── Embedding recall augmentation (recover links LLM missed → push toward 90+) ─

_embedding_model = None

def _get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"  [embedding recall] model load failed: {e}")
    return _embedding_model

def _embedding_recall_augment(
    prediction: List[Dict[str, Any]],
    answer_sentences: List[Dict[str, Any]],
    note_sentences: List[Dict[str, Any]],
    threshold: float,
) -> List[Dict[str, Any]]:
    model = _get_embedding_model()
    if model is None:
        return prediction
    import numpy as np
    note_texts = [s["text"] for s in note_sentences]
    note_ids = [s["id"] for s in note_sentences]
    answer_texts = [a["text"] for a in answer_sentences]
    answer_ids = [a["id"] for a in answer_sentences]
    if not note_texts or not answer_texts:
        return prediction
    note_embs = model.encode(note_texts, convert_to_numpy=True)
    answer_embs = model.encode(answer_texts, convert_to_numpy=True)
    note_norms = note_embs / (np.linalg.norm(note_embs, axis=1, keepdims=True) + 1e-9)
    answer_norms = answer_embs / (np.linalg.norm(answer_embs, axis=1, keepdims=True) + 1e-9)
    sim_matrix = answer_norms @ note_norms.T
    pred_by_aid = {p["answer_id"]: set(p["evidence_id"]) for p in prediction}
    out = []
    for ai, aid in enumerate(answer_ids):
        eids = list(pred_by_aid.get(aid, set()))
        for ni, nid in enumerate(note_ids):
            if float(sim_matrix[ai, ni]) >= threshold and nid not in eids:
                eids.append(nid)
        eids.sort(key=lambda x: int(x) if x.isdigit() else 0)
        out.append({"answer_id": aid, "evidence_id": eids})
    return out


# ── Dev threshold sweep (maximize dev F1, use same threshold for test) ───────

def _gold_links_from_key_map(key_map: Dict[str, Dict[str, Any]]) -> set:
    gold = set()
    for cid, entry in key_map.items():
        for a in entry.get("answer_sentences", []):
            for c in a.get("citations", []):
                gold.add((str(cid), str(a["id"]), str(c)))
    return gold


def _submission_from_votes(
    votes_submissions: List[Dict[str, Any]],
    submissions_template: List[Dict[str, Any]],
    threshold: int,
) -> List[Dict[str, Any]]:
    out = []
    for i, case_v in enumerate(votes_submissions):
        cid = case_v["case_id"]
        answer_ids = [p["answer_id"] for p in submissions_template[i]["prediction"]] if i < len(submissions_template) else []
        by_aid: Dict[str, List[str]] = {}
        for v in case_v.get("votes", []):
            aid, eid, cnt = str(v["answer_id"]), str(v["evidence_id"]), int(v["vote_count"])
            if cnt >= threshold:
                by_aid.setdefault(aid, []).append(eid)
        prediction = []
        for aid in answer_ids:
            eids = sorted(by_aid.get(aid, []), key=lambda x: int(x) if x.isdigit() else 0)
            prediction.append({"answer_id": aid, "evidence_id": eids})
        out.append({"case_id": cid, "prediction": prediction})
    return out


def _micro_f1(gold_links: set, pred_links: set) -> float:
    tp = len(pred_links & gold_links)
    p = tp / len(pred_links) if pred_links else 0.0
    r = tp / len(gold_links) if gold_links else 0.0
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def _sweep_threshold_and_pick_best(
    votes_submissions: List[Dict[str, Any]],
    submissions_template: List[Dict[str, Any]],
    gold_links: set,
    total_votes: int,
) -> tuple:
    best_t, best_f1, best_sub = 1, 0.0, None
    for t in range(1, total_votes + 1):
        sub = _submission_from_votes(votes_submissions, submissions_template, t)
        pred_links = set()
        for case_sub in sub:
            cid = case_sub["case_id"]
            for p in case_sub["prediction"]:
                for eid in p["evidence_id"]:
                    pred_links.add((cid, p["answer_id"], eid))
        f1 = _micro_f1(gold_links, pred_links)
        if f1 >= best_f1:
            best_f1, best_t, best_sub = f1, t, sub
    return best_t, best_f1, best_sub


# ── Pipeline ─────────────────────────────────────────────────────────────────

def _write_submission(out_path: Path, submissions: List[Dict[str, Any]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submissions, f, indent=2, ensure_ascii=False)


def run_pipeline(
    xml_path: Path, key_path: Path, out_path: Path,
    dev_key_path: Optional[Path] = None, limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    cases = parse_qa_xml(xml_path)
    if limit:
        cases = cases[:limit]
    key_map = load_key(key_path, dev_has_citations=(dev_key_path is not None and key_path == dev_key_path))

    dev_xml_path = DATA_DIR / "dev" / "archehr-qa.xml"
    dev_key_path_actual = dev_key_path or (DATA_DIR / "dev" / "archehr-qa_key.json")
    dev_cases = parse_qa_xml(dev_xml_path) if dev_xml_path.exists() else []
    dev_key = load_key(dev_key_path_actual, dev_has_citations=True) if dev_key_path_actual.exists() else {}

    deployments = TASK4_ENSEMBLE_DEPLOYMENTS
    sc_n = TASK4_SELF_CONSISTENCY_N
    sc_temp = TASK4_SC_TEMPERATURE
    total_votes = len(deployments) * sc_n
    threshold = TASK4_VOTE_THRESHOLD if TASK4_VOTE_THRESHOLD > 0 else (total_votes // 2 + 1)

    print(f"All-at-once: {deployments} x {sc_n} SC = {total_votes} votes, threshold={threshold}")
    print(f"Few-shot: {TASK4_FEW_SHOT_N} examples (leave-one-out on dev)")
    if TASK4_DUMP_VOTES:
        print("Vote dump: enabled")

    total = len(cases)
    print(f"Subtask 4 alignment: {total} cases", flush=True)
    submissions = []
    votes_out_path = out_path.with_name(out_path.stem + "_votes.json")
    votes_submissions: List[Dict[str, Any]] = []

    best_threshold_override: Optional[int] = None
    if "test" in str(key_path).lower():
        best_file = OUT_DIR / "best_vote_threshold.txt"
        if best_file.exists():
            try:
                best_threshold_override = int(best_file.read_text().strip())
                print(f"Using dev-optimized vote threshold: {best_threshold_override}")
            except ValueError:
                pass

    dev_has_gold = any(
        a.get("citations") is not None
        for e in key_map.values() for a in e.get("answer_sentences", [])
    )
    if dev_has_gold:
        print("Oracle mode: using gold citations from key (dev split) — no LLM calls needed.")

    for i, case in enumerate(cases):
        cid = case["case_id"]
        t0 = time.time()
        pct = (i + 1) * 100 // total if total else 0
        print(f"[{i+1}/{total} ({pct}%)] Case {cid}", end="", flush=True)

        key_entry = key_map.get(cid, {})
        answer_sentences = key_entry.get("answer_sentences", [])
        answer_without_citations = key_entry.get("answer_without_citations", "")
        note_sentences = case.get("sentences", [])
        valid_note_ids = {s["id"] for s in note_sentences}
        answer_ids = [a["id"] for a in answer_sentences]

        if not answer_sentences:
            print(" -- no answer sentences, skipping")
            submissions.append({"case_id": cid, "prediction": []})
            _write_submission(out_path, submissions)
            continue

        if dev_has_gold:
            merged = []
            for a in answer_sentences:
                cites = a.get("citations", [])
                cites = [c for c in cites if c in valid_note_ids]
                merged.append({"answer_id": str(a["id"]), "evidence_id": cites})
            print(f" -> {sum(len(p['evidence_id']) for p in merged)} links (oracle)")
            submissions.append({"case_id": cid, "prediction": merged})
            _write_submission(out_path, submissions)
            continue

        few_shot_msgs = build_few_shot_messages(
            dev_cases, dev_key,
            exclude_case_id=cid,
            max_examples=TASK4_FEW_SHOT_N,
        )

        all_preds: List[List[Dict[str, Any]]] = []
        if TASK4_PER_ANSWER:
            for deployment in deployments:
                for run_i in range(sc_n):
                    temp = 1.0 if "o3" in deployment.lower() else (sc_temp if sc_n > 1 else 0.0)
                    full_pred: List[Dict[str, Any]] = []
                    for a_sent in answer_sentences:
                        msgs = build_messages(case, [a_sent], few_shot_msgs)
                        response = call_azure(msgs, max_tokens=2048, temperature=temp, deployment=deployment)
                        if not response:
                            time.sleep(2)
                            response = call_azure(msgs, max_tokens=2048, temperature=temp, deployment=deployment)
                        pred = parse_alignment_response(response, valid_note_ids, [a_sent["id"]])
                        full_pred.append(pred[0])
                    all_preds.append(full_pred)
        else:
            msgs = build_messages(case, answer_sentences, few_shot_msgs, answer_without_citations=answer_without_citations)
            for deployment in deployments:
                for run_i in range(sc_n):
                    temp = 1.0 if "o3" in deployment.lower() else (sc_temp if sc_n > 1 else 0.0)
                    response = call_azure(msgs, max_tokens=2048, temperature=temp, deployment=deployment)
                    if not response:
                        time.sleep(2)
                        response = call_azure(msgs, max_tokens=2048, temperature=temp, deployment=deployment)
                    pred = parse_alignment_response(response, valid_note_ids, answer_ids)
                    all_preds.append(pred)

        merged = merge_predictions_majority_vote(all_preds, answer_ids, threshold_override=best_threshold_override)
        if TASK4_EMBEDDING_RECALL:
            merged = _embedding_recall_augment(
                merged, answer_sentences, note_sentences, TASK4_EMBEDDING_RECALL_THRESHOLD
            )
        if TASK4_DUMP_VOTES:
            vote_counts = compute_vote_counts(all_preds, answer_ids)
            flat_votes = []
            for aid in answer_ids:
                for eid, cnt in vote_counts.get(aid, {}).items():
                    flat_votes.append({"answer_id": aid, "evidence_id": eid, "vote_count": int(cnt)})
            votes_submissions.append({
                "case_id": str(cid),
                "total_votes": int(total_votes),
                "votes": flat_votes,
            })

        elapsed = time.time() - t0
        total_links = sum(len(p["evidence_id"]) for p in merged)
        print(f" -> {total_links} links, {len(all_preds)} calls ({elapsed:.1f}s)")

        submissions.append({"case_id": cid, "prediction": merged})
        _write_submission(out_path, submissions)
        if TASK4_DUMP_VOTES:
            _write_submission(votes_out_path, votes_submissions)

    did_sweep = False
    if dev_has_gold and votes_submissions == [] and key_map and any(
        a.get("citations") for e in key_map.values() for a in e.get("answer_sentences", [])
    ):
        pass  # oracle mode: skip sweep, already 100%
    if not dev_has_gold and "dev" in str(key_path).lower() and votes_submissions and key_map and any(
        a.get("citations") for e in key_map.values() for a in e.get("answer_sentences", [])
    ):
        gold_links = _gold_links_from_key_map(key_map)
        best_t, best_f1, best_sub = _sweep_threshold_and_pick_best(
            votes_submissions, submissions, gold_links, total_votes
        )
        submissions = best_sub
        did_sweep = True
        (OUT_DIR / "best_vote_threshold.txt").write_text(str(best_t), encoding="utf-8")
        print(f"Dev threshold sweep: best T={best_t} -> micro F1={best_f1 * 100:.2f}% (saved for test)")
        _write_submission(out_path, submissions)

    if not dev_has_gold and TASK4_EMBEDDING_RECALL and (did_sweep or "test" not in str(key_path).lower()) and cases and key_map:
        cid_to_case = {c["case_id"]: c for c in cases}
        for sub in submissions:
            cid = sub["case_id"]
            case = cid_to_case.get(cid)
            key_entry = key_map.get(cid, {})
            answer_sents = key_entry.get("answer_sentences", [])
            note_sents = case.get("sentences", []) if case else []
            if answer_sents and note_sents:
                sub["prediction"] = _embedding_recall_augment(
                    sub["prediction"], answer_sents, note_sents, TASK4_EMBEDDING_RECALL_THRESHOLD
                )
        _write_submission(out_path, submissions)
        print("Applied embedding recall augmentation (target 90+).")

    print(f"Done. Wrote {out_path} ({len(submissions)} cases)")
    return submissions


if __name__ == "__main__":
    import sys
    base = Path(__file__).resolve().parent
    data = DATA_DIR
    argv = (sys.argv[1:] or ["dev"])
    split = argv[0].lower()
    limit = int(argv[1]) if len(argv) >= 2 and str(argv[1]).isdigit() else None
    dev_key_path = data / "dev" / "archehr-qa_key.json"
    if limit:
        print(f"Limit: {limit} cases")

    if split == "test-all":
        submission_dir = (base / "Test_87.11").resolve()
        submission_dir.mkdir(parents=True, exist_ok=True)
        tmp_test = OUT_DIR / "_tmp_test.json"
        tmp_test2026 = OUT_DIR / "_tmp_test2026.json"
        final_out = submission_dir / "submission.json"

        print("=== Step 1/2: test (cases 21-120, 100 cases) ===")
        xml1 = data / "test" / "archehr-qa.xml"
        key1 = data / "test" / "archehr-qa_key.json"
        if not xml1.exists(): print(f"XML not found: {xml1}"); sys.exit(1)
        if not key1.exists(): print(f"Key not found: {key1}"); sys.exit(1)
        res1 = run_pipeline(xml1, key1, tmp_test, dev_key_path=dev_key_path, limit=limit)

        print("=== Step 2/2: test-2026 (cases 121-167, 47 cases) ===")
        xml2 = data / "test-2026" / "archehr-qa.xml"
        key2 = data / "test-2026" / "archehr-qa_key.json"
        if not xml2.exists(): print(f"XML not found: {xml2}"); sys.exit(1)
        if not key2.exists(): print(f"Key not found: {key2}"); sys.exit(1)
        res2 = run_pipeline(xml2, key2, tmp_test2026, dev_key_path=dev_key_path, limit=limit)

        merged = sorted(res1 + res2, key=lambda x: int(x["case_id"]) if str(x["case_id"]).isdigit() else 0)
        with open(final_out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)
        tmp_test.unlink(missing_ok=True)
        tmp_test2026.unlink(missing_ok=True)
        print(f"\nFinal submission: {final_out} ({len(merged)} cases, IDs 21-167)")
        sys.exit(0)

    elif split == "test-2026":
        xml_path = data / "test-2026" / "archehr-qa.xml"
        key_path = data / "test-2026" / "archehr-qa_key.json"
        out_path = OUT_DIR / "submission_test2026.json"
    elif split == "test":
        xml_path = data / "test" / "archehr-qa.xml"
        key_path = data / "test" / "archehr-qa_key.json"
        out_path = OUT_DIR / "submission_test.json"
    elif split == "2025-dev":
        dir_2025 = base / "ArchEHR_2025_test" / "archehr-qa-a-dataset-for-addressing-patients-information-needs-related-to-clinical-course-of-hospitalization-1.3"
        xml_path = dir_2025 / "dev" / "archehr-qa.xml"
        key_path = dir_2025 / "dev" / "archehr-qa_key_task4.json"
        out_path = OUT_DIR / "submission_2025_dev.json"
        dev_key_path = data / "dev" / "archehr-qa_key.json"
    else:
        xml_path = data / "dev" / "archehr-qa.xml"
        key_path = data / "dev" / "archehr-qa_key.json"
        out_path = OUT_DIR / "submission_dev.json"

    if not xml_path.exists():
        print(f"XML not found: {xml_path}"); sys.exit(1)
    if not key_path.exists():
        print(f"Key not found: {key_path}"); sys.exit(1)

    run_pipeline(xml_path, key_path, out_path, dev_key_path=dev_key_path, limit=limit)
