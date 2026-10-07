"""General-capability evaluation: MMLU-Pro, MMLU-Redux, IFEval, GPQA-Diamond, ZebraLogic.

Each benchmark is loaded from a pinned public revision, rendered with the
model's chat template, sampled through the vLLM client in ``generate.py``, and
scored per completion. ``--summarize`` reports the change from a Base
checkpoint in percentage points and the mean change over the four
prior-capability benchmarks.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import sys
from typing import Callable, Dict, List, Optional
from urllib.request import Request, urlopen
import zipfile

from .generate import batched_generate, parse_server_urls

BENCHMARKS = ("mmlu_pro", "mmlu_redux", "ifeval", "gpqa_diamond", "zebralogic")
PRIOR_BENCHMARKS = ("mmlu_pro", "mmlu_redux", "ifeval", "gpqa_diamond")
DEFAULT_NUM_SAMPLES = {"mmlu_pro": 1, "mmlu_redux": 64, "ifeval": 64,
                       "gpqa_diamond": 64, "zebralogic": 64}
POPULATION = {"mmlu_pro": 12032, "mmlu_redux": 2778, "ifeval": 541,
              "gpqa_diamond": 198, "zebralogic": 1000}
# GPQA, IFEval and ZebraLogic also reject a response that reopens <think>.
REJECT_REOPENED_THINK = ("ifeval", "gpqa_diamond", "zebralogic")
DEFAULT_DATA_DIR = Path(os.environ.get("EXPDIS_GENERAL_EVAL_DATA_DIR",
                                       Path.home() / ".cache" / "expdis" / "general_eval"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pinned_url(url: str, digest: str, cache: Path) -> bytes:
    """Download once into ``cache``; refuse files whose SHA-256 differs."""
    if cache.exists():
        data = cache.read_bytes()
    else:
        with urlopen(Request(url, headers={"User-Agent": "expdis-general-eval"}), timeout=120) as response:
            data = response.read()
    if _sha256(data) != digest:
        raise ValueError(f"SHA-256 mismatch for {url}")
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    return data


def _hf_file(repo: str, filename: str, revision: str, digest: str | None = None) -> bytes:
    from huggingface_hub import hf_hub_download

    data = Path(hf_hub_download(repo, filename, revision=revision, repo_type="dataset")).read_bytes()
    if digest is not None and _sha256(data) != digest:
        raise ValueError(f"SHA-256 mismatch for {repo}/{filename}@{revision}")
    return data


def _parquet_rows(data: bytes) -> list[dict]:
    import pyarrow.parquet as parquet

    return parquet.read_table(io.BytesIO(data)).to_pylist()


def _check_population(benchmark: str, records: list[dict]) -> list[dict]:
    if len(records) != POPULATION[benchmark] or len({r["id"] for r in records}) != len(records):
        raise ValueError(f"{benchmark}: expected {POPULATION[benchmark]} unique questions, got {len(records)}")
    return records


def final_response(text: str, thinking: bool, *, reject_reopened: bool = False) -> Optional[str]:
    """Text after the last ``</think>``; None for an unfinished thinking trace."""
    if not thinking:
        return text
    if "</think>" not in text:
        return None
    response = text.rsplit("</think>", 1)[1]
    if reject_reopened and "<think>" in response:
        return None
    return response


# MMLU-Pro: five category-matched validation CoT examples, one draw per question.

MMLU_PRO_REPO = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
MMLU_PRO_INITIAL = ("The following are multiple choice questions (with answers) about {$}. "
                    'Think step by step and then finish your answer with "the answer is (X)" '
                    "where X is the correct letter choice.\n\n\n")
_MMLU_PRO_PATTERNS = (
    r"\b(?:the\s+)?(?:final\s+)?answer\s+is\s*[:=]?\s*\(?\s*([A-J])\s*\)?\b",
    r"\b(?:final\s+)?answer\s*[:=]\s*\(?\s*([A-J])\s*\)?\b",
    r"\\boxed\s*\{\s*\(?([A-J])\)?\s*\}",
)


def _mmlu_pro_options(row: dict) -> list[str]:
    options = list(row["options"])
    if "N/A" in options:
        first = options.index("N/A")
        if any(o != "N/A" for o in options[first:]):
            raise ValueError("non-terminal N/A would change option labels")
        options = options[:first]
    if not 2 <= len(options) <= 10:
        raise ValueError("unexpected number of answer options")
    return options


def _mmlu_pro_question(row: dict, *, example: bool = False) -> str:
    text = "Question:\n" + row["question"] + "\nOptions:\n"
    text += "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(_mmlu_pro_options(row))) + "\n"
    if example:
        cot = row["cot_content"].replace("A: Let's think step by step.", "Answer: Let's think step by step.")
        if not cot.strip():
            raise ValueError("few-shot rationale missing")
        return text + cot + "\n\n"
    return text + "Answer: Let's think step by step."


def mmlu_pro_prompt(row: dict, validation: list[dict]) -> str:
    shots = [r for r in validation if r["category"] == row["category"]][:5]
    if len(shots) != 5:
        raise ValueError(f"need five validation examples for category {row['category']!r}")
    return (MMLU_PRO_INITIAL.replace("{$}", row["category"])
            + "".join(_mmlu_pro_question(r, example=True) for r in shots)
            + _mmlu_pro_question(row))


def mmlu_pro_records(test: list[dict], validation: list[dict]) -> list[dict]:
    records = []
    for row in test:
        if row["answer"] != chr(65 + row["answer_index"]):
            raise ValueError("MMLU-Pro gold label mismatch")
        records.append({"id": str(row["question_id"]), "prompt": mmlu_pro_prompt(row, validation),
                        "answer": row["answer"], "category": row["category"],
                        "option_count": len(_mmlu_pro_options(row))})
    return records


def mmlu_pro_answer(response: str, option_count: int) -> Optional[str]:
    """Last explicit answer declaration or box; else a bare final letter line."""
    matches = [(m.start(), m.group(1).upper()) for p in _MMLU_PRO_PATTERNS
               for m in re.finditer(p, response, re.IGNORECASE)]
    if matches:
        answer = max(matches)[1]
    else:
        final = response.strip().splitlines()[-1] if response.strip() else ""
        match = re.fullmatch(r"\(?([A-J])\)?[.]?", final.strip())
        answer = match.group(1) if match else None
    return answer if answer and ord(answer) - ord("A") < option_count else None


def grade_mmlu_pro(record: dict, response: str) -> dict:
    pred = mmlu_pro_answer(response, record["option_count"])
    return {"correct": pred == record["answer"], "invalid": pred is None, "extracted": pred}


def load_mmlu_pro(data_dir: Path) -> list[dict]:
    splits = {split: _parquet_rows(_hf_file(MMLU_PRO_REPO, f"data/{split}-00000-of-00001.parquet",
                                            MMLU_PRO_REVISION))
              for split in ("test", "validation")}
    return _check_population("mmlu_pro", mmlu_pro_records(splits["test"], splits["validation"]))


# MMLU-Redux: curated 2,778-question population with publisher gold corrections,
# five subject-matched MMLU dev examples in delimited blocks, JSON answer field.

REDUX_CURATED_REPO = "WildEval/ZeroEval"
REDUX_CURATED_REVISION = "a100ff81ecec0ed983dd3f0038d6d6bb2ab3dc1c"
REDUX_CURATED_SHA256 = "1a6fccbd962dcabb8fb2b6ca88ac37d7e3d494e1e772ce505aea72b51d866e8f"
REDUX_PUBLISHER_REPO = "edinburgh-dawg/mmlu-redux"
REDUX_PUBLISHER_REVISION = "3720db6aeb3d019de48bf37916c1a54074ff4997"
MMLU_DEV_REPO = "cais/mmlu"
MMLU_DEV_REVISION = "c30699e8356da336a370243923dbaf21066bb9fe"
REDUX_JSON_INSTRUCTION = ('Please show your choice in the answer field with only the choice letter, '
                          'e.g., {"answer": "C"}.')
_REDUX_ID = re.compile(r"^mmlu-redux-([a-z_]+)-#([0-9]+)$")
ABCD = "ABCD"


def _redux_gold(row: dict) -> int:
    if len(row["choices"]) != 4 or not all(isinstance(c, str) and c for c in row["choices"]):
        raise ValueError("invalid four-option choices")
    answer = row["answer"]
    if type(answer) is not int or answer not in range(4):
        raise ValueError("invalid publisher gold index")
    if row["error_type"] == "ok":
        return answer
    if row["error_type"] != "wrong_groundtruth":
        raise ValueError(f"unsupported publisher annotation: {row['error_type']}")
    corrected = row["correct_answer"]
    if not isinstance(corrected, str) or corrected not in ("0", "1", "2", "3", *ABCD):
        raise ValueError("invalid corrected gold index")
    return int(corrected) if corrected.isdigit() else ABCD.index(corrected)


def redux_join(curated: list[dict], publisher: dict[str, list[dict]]) -> list[dict]:
    """Map curated IDs to publisher rows by subject and row index; apply gold corrections."""
    rows, seen = [], set()
    for old in curated:
        qid = old["id"]
        match = _REDUX_ID.fullmatch(qid)
        if qid in seen or not match:
            raise ValueError(f"duplicate or invalid curated ID: {qid}")
        seen.add(qid)
        subject, index = match.group(1), int(match.group(2))
        if old["config"] != subject or subject not in publisher or index >= len(publisher[subject]):
            raise ValueError(f"missing publisher row: {qid}")
        current = publisher[subject][index]
        gold = _redux_gold(current)
        if len(old["choices"]) != 4 or old["choices"].count(old["correct_answer"]) != 1:
            raise ValueError(f"invalid curated gold: {qid}")
        if old["choices"].index(old["correct_answer"]) != gold:
            raise ValueError(f"curated/publisher gold mismatch: {qid}")
        rows.append({"id": qid, "question": current["question"], "choices": list(current["choices"]),
                     "answer": ABCD[gold], "subject": subject})
    return rows


def _redux_block(row: dict) -> str:
    return "\n".join([f"Question:\n{row['question']}",
                      *(f"{letter}. {choice}" for letter, choice in zip(ABCD, row["choices"]))])


def redux_prompt(row: dict, dev: list[dict]) -> str:
    examples = [e for e in dev if e["subject"] == row["subject"]][:5]
    if len(examples) != 5:
        raise ValueError("exactly five subject-matched examples required")
    blocks = [f"The following are multiple choice questions about {row['subject'].replace('_', ' ')}.",
              "The five examples below are already answered. Do not answer them again. "
              "Answer only the target question after the examples."]
    blocks += ["<example>\n" + _redux_block(e) + "\nAnswer: " + e["answer"] + "\n</example>" for e in examples]
    blocks.append("<target_question>\n" + _redux_block(row) + "\n</target_question>")
    blocks.append(REDUX_JSON_INSTRUCTION)
    return "\n\n".join(blocks)


def redux_records(test: list[dict], dev: list[dict]) -> list[dict]:
    dev_questions = {(d["subject"], " ".join(d["question"].casefold().split())) for d in dev}
    records = []
    for row in test:
        if (row["subject"], " ".join(row["question"].casefold().split())) in dev_questions:
            raise ValueError(f"test question appears among its own subject's examples: {row['id']}")
        records.append({"id": row["id"], "prompt": redux_prompt(row, dev),
                        "answer": row["answer"], "subject": row["subject"]})
    return records


def _json_spans(text: str):
    """Top-level brace spans, ignoring braces inside quoted strings."""
    depth, start, quoted, escaped = 0, None, False, False
    for i, char in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            continue
        if char == '"':
            quoted = True
        elif char == "{":
            if depth == 0:
                start = i
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                yield text[start:i + 1]
                start = None
    if depth:
        yield text[start:]


def _has_answer_key(span: str) -> bool:
    """True if ``answer`` appears in key position at depth 1, even in malformed JSON."""
    depth, key_position, i = 0, False, 0
    while i < len(span):
        char = span[i]
        if char == '"':
            end, escaped = i + 1, False
            while end < len(span):
                if escaped:
                    escaped = False
                elif span[end] == "\\":
                    escaped = True
                elif span[end] == '"':
                    break
                end += 1
            if depth == 1 and key_position and end < len(span):
                if span[i + 1:end] == "answer":
                    return True
                key_position = False
            i = end + 1
            continue
        if char == "{":
            depth += 1
            if depth == 1:
                key_position = True
        elif char == "}":
            depth -= 1
        elif depth == 1 and char == ",":
            key_position = True
        elif depth == 1 and key_position and not char.isspace():
            if span.startswith("answer", i) and (
                    i + 6 == len(span) or not (span[i + 6].isalnum() or span[i + 6] == "_")):
                return True
            key_position = False
        i += 1
    return False


def redux_answer(response: str) -> Optional[str]:
    """The last object with an ``answer`` key decides; it must parse to one of A-D."""
    result = None
    for span in _json_spans(response):
        if not _has_answer_key(span):
            continue
        result = None
        try:
            def unique(pairs):
                if sum(key == "answer" for key, _ in pairs) > 1:
                    raise ValueError("duplicate answer keys")
                return dict(pairs)
            obj = json.loads(span, object_pairs_hook=unique)
            value = obj.get("answer") if isinstance(obj, dict) else None
            if type(value) is str and value in ABCD:
                result = value
        except (ValueError, TypeError):
            pass
    return result


def grade_mmlu_redux(record: dict, response: str) -> dict:
    pred = redux_answer(response)
    return {"correct": pred == record["answer"], "invalid": pred is None, "extracted": pred}


def load_mmlu_redux(data_dir: Path) -> list[dict]:
    import pyarrow.ipc as ipc

    curated = _parquet_rows(_hf_file(REDUX_CURATED_REPO, "mmlu-redux/test-00000-of-00001.parquet",
                                     REDUX_CURATED_REVISION, REDUX_CURATED_SHA256))
    subjects = sorted({row["config"] for row in curated})
    publisher = {}
    for subject in subjects:
        data = _hf_file(REDUX_PUBLISHER_REPO, f"{subject}/data-00000-of-00001.arrow", REDUX_PUBLISHER_REVISION)
        publisher[subject] = ipc.open_stream(io.BytesIO(data)).read_all().to_pylist()
    test = redux_join(curated, publisher)
    dev = [{"question": r["question"], "choices": list(r["choices"]),
            "answer": ABCD[r["answer"]], "subject": r["subject"]}
           for r in _parquet_rows(_hf_file(MMLU_DEV_REPO, "all/dev-00000-of-00001.parquet", MMLU_DEV_REVISION))
           if r["subject"] in subjects]
    return _check_population("mmlu_redux", redux_records(test, dev))


# GPQA-Diamond: author archive, seed-0 option shuffle, boxed letter.

GPQA_REVISION = "56686c06f5e19865c153de0fdb11be3890014df7"
GPQA_ZIP_SHA256 = "461ae7329f15a3e35f8184d2dac24b990f34fdf12f366ca4062d8e6638cd08dc"
GPQA_DIAMOND_SHA256 = "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305"
GPQA_ARCHIVE_PASSWORD = b"deserted-untie-orchid"  # published in the author README
GPQA_OPTION_FIELDS = ("Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3", "Correct Answer")
GPQA_SUFFIX = ("Reason through the question and finish with exactly one answer "
               r"letter (A, B, C, or D) in \boxed{X}.")


def gpqa_prompt(record: dict) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(ABCD, record["choices"]))
    return ("Answer the following graduate-level multiple-choice question.\n\n"
            f"Question:\n{record['question']}\n\nOptions:\n{options}\n\n" + GPQA_SUFFIX)


def gpqa_records(rows: list[dict], expected: int = POPULATION["gpqa_diamond"]) -> list[dict]:
    """All CSV rows in order; one ``random.Random(0)`` shuffle stream across rows."""
    if len(rows) != expected:
        raise ValueError(f"GPQA-Diamond requires {expected} rows")
    rng = random.Random(0)
    records, seen = [], set()
    for index, row in enumerate(rows):
        if any(not isinstance(row.get(f), str) or not row[f].strip()
               for f in ("Question", "Record ID", *GPQA_OPTION_FIELDS)):
            raise ValueError(f"missing GPQA field at row {index}")
        if row["Record ID"] in seen:
            raise ValueError(f"duplicate GPQA record at row {index}")
        seen.add(row["Record ID"])
        choices = [row[f] for f in GPQA_OPTION_FIELDS]
        if choices.count(row["Correct Answer"]) != 1:
            raise ValueError(f"ambiguous GPQA gold at row {index}")
        rng.shuffle(choices)
        record = {"id": f"gpqa-diamond-#{index}", "question": row["Question"], "choices": choices,
                  "answer": ABCD[choices.index(row["Correct Answer"])]}
        record["prompt"] = gpqa_prompt(record)
        records.append(record)
    return records


_DECLARATION = re.compile(r"\b(?:the[ \t]+)?(?:final[ \t]+)?answer(?:[ \t]+is|[ \t]*[:=])[ \t]*", re.I)
_BOX = re.compile(r"\\boxed\s*\{[^}\n]*\}?", re.I)
_LETTER = re.compile(r"\(?([A-Z])\)?\s*[.!?]?\s*", re.I)
_BOX_START = re.compile(r"\\boxed\s*\{", re.I)
_WRAPPED = re.compile(r"\s*\\(?:text|textbf|textnormal|mathrm|mathbf|mathit|mathsf)\s*\{\s*([A-Z])\s*\}\s*", re.I)
_OR = re.compile(r"\bor\b", re.I)


def letter_answer(response: str) -> Optional[str]:
    """Last explicit declaration or box decides; a line containing "or" is ambiguous."""
    marks = [(m.start(), "answer", m.end()) for m in _DECLARATION.finditer(response)]
    marks += [(m.start(), "box", m.group()) for m in _BOX.finditer(response)]
    if marks:
        start, kind, value = max(marks, key=lambda item: item[0])
        line_start = response.rfind("\n", 0, start) + 1
        line_end = response.find("\n", start)
        line_end = line_end if line_end >= 0 else len(response)
        if _OR.search(response[line_start:line_end]):
            return None
        if kind == "box":
            candidate = re.fullmatch(r"\\boxed\s*\{\s*([A-Z])\s*\}", value, re.I)
        else:
            candidate = _LETTER.fullmatch(response[value:line_end].strip())
    else:
        last = response.strip().splitlines()[-1] if response.strip() else ""
        candidate = _LETTER.fullmatch(last.strip())
    letter = candidate.group(1).upper() if candidate else None
    return letter if letter is not None and letter in ABCD else None


def _box_end(text: str, open_brace: int) -> Optional[int]:
    depth = 0
    for i in range(open_brace, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return None


def normalize_boxed_letter(response: str) -> str:
    """Rewrite a lone top-level ``\\boxed{\\text{C}}`` (and similar wrappers) to ``\\boxed{C}``."""
    starts = {m.start() for m in _BOX_START.finditer(response)}
    top_level, depth = set(), 0
    for i, char in enumerate(response):
        if i in starts and depth == 0:
            top_level.add(i)
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return response
    if depth != 0:
        return response
    replacements, position = [], 0
    while match := _BOX_START.search(response, position):
        end = _box_end(response, match.end() - 1)
        if end is None:
            return response
        if match.start() not in top_level:
            position = end
            continue
        wrapped = _WRAPPED.fullmatch(response[match.end():end - 1])
        if wrapped:
            line_start = response.rfind("\n", 0, match.start()) + 1
            line_end = response.find("\n", end)
            line = response[line_start:line_end if line_end >= 0 else len(response)]
            prior = list(_DECLARATION.finditer(response, 0, match.start()))
            competing = bool(prior) and (len(prior) != 1 or bool(response[prior[0].end():match.start()].strip()))
            if not _OR.search(line) and len(_BOX_START.findall(line)) == 1 and not competing:
                replacements.append((match.start(), end, r"\boxed{" + wrapped.group(1) + "}"))
        position = end
    for start, end, canonical in reversed(replacements):
        response = response[:start] + canonical + response[end:]
    return response


def grade_gpqa_diamond(record: dict, response: str) -> dict:
    if record.get("answer") not in tuple(ABCD):
        raise ValueError("GPQA gold answer must be one of A-D")
    pred = letter_answer(normalize_boxed_letter(response))
    return {"correct": pred == record["answer"], "invalid": pred is None, "extracted": pred}


def load_gpqa_diamond(data_dir: Path) -> list[dict]:
    url = f"https://raw.githubusercontent.com/idavidrein/gpqa/{GPQA_REVISION}/dataset.zip"
    archive = _pinned_url(url, GPQA_ZIP_SHA256, data_dir / "gpqa" / "dataset.zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as handle:
        data = handle.read("dataset/gpqa_diamond.csv", pwd=GPQA_ARCHIVE_PASSWORD)
    if _sha256(data) != GPQA_DIAMOND_SHA256:
        raise ValueError("SHA-256 mismatch for GPQA Diamond CSV")
    return gpqa_records(list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))))


# IFEval: verbatim author prompts and the author strict scorer
# (google-research instruction_following_eval), downloaded at a pinned revision.

IFEVAL_REVISION = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
IFEVAL_FILES = {
    "evaluation_lib.py": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    "instructions.py": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
    "instructions_registry.py": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "data/input_data.jsonl": "67ffeee0fcb87c317c5b08a2de85557b4a7e96ada6178aa645b4954fe4b53d49",
}
IFEVAL_REQUIREMENTS = ("absl-py", "langdetect", "nltk", "immutabledict")
NLTK_DATA_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
NLTK_PUNKT_TAB_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"


def _ifeval_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "ifeval"


def fetch_ifeval(data_dir: Path) -> Path:
    root = _ifeval_dir(data_dir)
    base = f"https://raw.githubusercontent.com/google-research/google-research/{IFEVAL_REVISION}/"
    for name, digest in IFEVAL_FILES.items():
        path = f"instruction_following_eval/{name}"
        _pinned_url(base + path, digest, root / path)
    (root / "instruction_following_eval" / "__init__.py").touch()
    punkt = root / "nltk_data" / "tokenizers" / "punkt_tab"
    if not (punkt / "english").is_dir():
        url = (f"https://raw.githubusercontent.com/nltk/nltk_data/{NLTK_DATA_REVISION}"
               "/packages/tokenizers/punkt_tab.zip")
        archive = _pinned_url(url, NLTK_PUNKT_TAB_SHA256, root / "punkt_tab.zip")
        with zipfile.ZipFile(io.BytesIO(archive)) as handle:
            for member in handle.namelist():
                if member.startswith("punkt_tab/english/") and not member.endswith("/"):
                    target = punkt / "english" / Path(member).name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(handle.read(member))
    return root


def ifeval_records(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        ids, kwargs = row.get("instruction_id_list"), row.get("kwargs")
        if (not isinstance(ids, list) or not ids or not isinstance(kwargs, list) or len(ids) != len(kwargs)
                or not isinstance(row.get("prompt"), str) or type(row.get("key")) is not int):
            raise ValueError(f"invalid IFEval row: {row.get('key')}")
        records.append(dict(row, id=f"ifeval-{row['key']}"))
    return records


def load_ifeval(data_dir: Path) -> list[dict]:
    root = fetch_ifeval(data_dir)
    lines = (root / "instruction_following_eval/data/input_data.jsonl").read_text(encoding="utf-8").splitlines()
    return _check_population("ifeval", ifeval_records([json.loads(line) for line in lines if line.strip()]))


_IFEVAL_LIB = None


def ifeval_scorer(data_dir: Path = DEFAULT_DATA_DIR):
    """Import the author scorer from ``data_dir`` with its own NLTK data and seeded langdetect."""
    global _IFEVAL_LIB
    if _IFEVAL_LIB is None:
        root = fetch_ifeval(data_dir)
        import nltk
        from langdetect import DetectorFactory

        nltk.data.path[:] = [str(root / "nltk_data")]
        DetectorFactory.seed = 0
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        _IFEVAL_LIB = importlib.import_module("instruction_following_eval.evaluation_lib")
    return _IFEVAL_LIB


def grade_ifeval(record: dict, response: str, *, scorer=None) -> dict:
    """Primary: every instruction satisfied under the strict scorer. Loose results are diagnostics."""
    n = len(record["instruction_id_list"])
    if not isinstance(response, str) or not response.strip():
        return {"correct": False, "invalid": True, "strict_instruction_correct": [False] * n,
                "loose_prompt_correct": False}
    lib = scorer or ifeval_scorer()
    example = lib.InputExample(**{k: copy.deepcopy(record[k])
                                  for k in ("key", "instruction_id_list", "prompt", "kwargs")})
    responses = {example.prompt: response}
    state = random.getstate()
    try:
        random.seed(0)
        strict = lib.test_instruction_following_strict(example, responses)
        random.seed(0)
        loose = lib.test_instruction_following_loose(example, responses)
    finally:
        random.setstate(state)
    return {"correct": bool(strict.follow_all_instructions), "invalid": False,
            "strict_instruction_correct": [bool(v) for v in strict.follow_instruction_list],
            "loose_prompt_correct": bool(loose.follow_all_instructions)}


# ZebraLogic: grid-mode puzzles, ZeroEval ZEBRA_GRID prompt, whole-puzzle accuracy.

ZEBRA_REPO = "WildEval/ZebraLogic"
ZEBRA_REVISION = "0a473f5a0054835754ed156d5a79c6ce27178bb1"
ZEBRA_PARQUET_SHA256 = "545cd5cce9c54575c7f74769b8f1250b8bb94f5745519c83b99b8db01f95924d"
ZEROEVAL_REVISION = "8c1485edf12c6efb5f69135a562927c5ad484059"
ZEBRA_TEMPLATE_SHA256 = "bbeed4d971d19dc04b84e888255598ab1d35c3de6701aad240d7d26abfd4bd08"


def zebra_prompt(item: dict, template: str) -> str:
    prompt = template.replace("{puzzle}", item["puzzle"])
    columns = item["solution"]["header"]
    if columns[0] != "House":
        raise ValueError("ZebraLogic header must start with House")
    skeleton = {"reasoning": "___", "solution": {
        f"House {i + 1}": {columns[j]: "___" for j in range(1, len(columns))}
        for i in range(len(item["solution"]["rows"]))}}
    return prompt.replace("{json_template}", json.dumps(skeleton, indent=4))


def zebra_records(rows: list[dict], template: str, expected: int = POPULATION["zebralogic"]) -> list[dict]:
    ids = [row["id"] for row in rows]
    if len(rows) != expected or len(set(ids)) != len(ids):
        raise ValueError(f"expected {expected} puzzles with unique IDs")
    records = []
    for row in rows:
        houses, attributes = map(int, row["size"].split("*"))
        header, cells = row["solution"]["header"], row["solution"]["rows"]
        if len(header) != attributes + 1 or header[0] != "House" or len(cells) != houses:
            raise ValueError(f"invalid gold grid: {row['id']}")
        for i, values in enumerate(cells):
            if (len(values) != len(header) or values[0] != str(i + 1)
                    or any(not isinstance(v, str) or not v.strip() or v.strip() in {"___", "???", "[REDACTED]"}
                           for v in values)):
                raise ValueError(f"malformed gold cells: {row['id']}")
        gold = {f"House {i + 1}": {header[j]: values[j] for j in range(1, len(header))}
                for i, values in enumerate(cells)}
        records.append({"id": row["id"], "prompt": zebra_prompt(row, template), "gold": gold, "size": row["size"]})
    return records


def zebra_last_json(text: str):
    """ZeroEval extractor: last balanced top-level ``{...}`` (brace scan is not string-aware)."""
    stack, start, last = [], None, None
    for i, char in enumerate(text):
        if char == "{":
            stack.append(i)
            if start is None:
                start = i
        elif char == "}" and stack:
            stack.pop()
            if not stack:
                last = text[start:i + 1]
                start = None
    if last:
        try:
            return json.loads(last.replace("\n", ""))
        except json.JSONDecodeError:
            pass
    return None


def grade_zebralogic(record: dict, response: str) -> dict:
    gold = record["gold"]
    total = sum(len(columns) for columns in gold.values())
    prediction = zebra_last_json(response)
    table = prediction.get("solution") if isinstance(prediction, dict) else None
    invalid = not isinstance(table, dict)
    hits = 0
    if not invalid:
        for house, columns in gold.items():
            if house not in table:
                continue
            row = table[house]
            if not isinstance(row, dict):
                invalid = True
                continue
            for column, truth in columns.items():
                if column not in row:
                    continue
                value = row[column]
                if value is None or isinstance(value, (str, list, dict)) and len(value) == 0:
                    continue
                if isinstance(value, list):
                    value = value[0]
                if not isinstance(value, str):
                    invalid = True
                    continue
                hits += int(truth.lower().strip() == value.lower().strip())
    return {"correct": not invalid and hits == total, "invalid": invalid, "cell_accuracy": hits / total}


def load_zebralogic(data_dir: Path) -> list[dict]:
    rows = _parquet_rows(_hf_file(ZEBRA_REPO, "grid_mode/test-00000-of-00001.parquet",
                                  ZEBRA_REVISION, ZEBRA_PARQUET_SHA256))
    url = f"https://raw.githubusercontent.com/WildEval/ZeroEval/{ZEROEVAL_REVISION}/src/templates/ZEBRA_GRID.py"
    source = _pinned_url(url, ZEBRA_TEMPLATE_SHA256, Path(data_dir) / "zebralogic" / "ZEBRA_GRID.py")
    tree = ast.parse(source.decode("utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "ZEBRA_GRID" for t in n.targets))
    return zebra_records(rows, ast.literal_eval(node.value))


LOADERS: Dict[str, Callable[[Path], list]] = {
    "mmlu_pro": load_mmlu_pro, "mmlu_redux": load_mmlu_redux, "ifeval": load_ifeval,
    "gpqa_diamond": load_gpqa_diamond, "zebralogic": load_zebralogic,
}
GRADERS: Dict[str, Callable[[dict, str], dict]] = {
    "mmlu_pro": grade_mmlu_pro, "mmlu_redux": grade_mmlu_redux, "ifeval": grade_ifeval,
    "gpqa_diamond": grade_gpqa_diamond, "zebralogic": grade_zebralogic,
}
PRIMARY_METRIC = {"mmlu_pro": "accuracy", "mmlu_redux": "accuracy", "ifeval": "strict_prompt_accuracy",
                  "gpqa_diamond": "accuracy", "zebralogic": "puzzle_accuracy"}


def render_prompt_ids(tokenizer, content: str, thinking: bool) -> List[int]:
    messages = [{"role": "user", "content": content}]
    kwargs = {"tokenize": True, "add_generation_prompt": True, "enable_thinking": bool(thinking)}
    try:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    ids = list(ids)
    if not ids or not all(type(i) is int for i in ids):
        raise ValueError("chat template did not return token IDs")
    return ids


def thinking_enabled(tokenizer, mode: str) -> bool:
    """``auto``: on when the chat template exposes ``enable_thinking`` (Qwen3), off otherwise."""
    if mode in ("always", "never"):
        return mode == "always"
    if mode != "auto":
        raise ValueError(f"unknown --enable-thinking {mode!r}")
    return "enable_thinking" in str(getattr(tokenizer, "chat_template", "") or "")


def pass_at_k(n: int, c: int, k: int) -> float:
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def score_completions(records: list[dict], benchmark: str, completions: list, *, num_samples: int,
                      thinking: bool, grade: Optional[Callable[[dict, str], dict]] = None) -> dict:
    """Grade every completion; an unfinished thinking trace or transport error is incorrect."""
    if len(completions) != len(records) or any(len(group) != num_samples for group in completions):
        raise RuntimeError("evaluation requires exactly n completions for every problem")
    grade = grade or GRADERS[benchmark]
    reject = benchmark in REJECT_REOPENED_THINK
    counts, rows, finish = [], [], Counter()
    invalid = errors = 0
    loose = []
    for record, group in zip(records, completions):
        hits = 0
        for sample_idx, completion in enumerate(group):
            finish[str(completion.finish_reason)] += 1
            if completion.finish_reason == "error":
                errors += 1
                result = {"correct": False, "invalid": True, "invalid_reason": "error"}
            else:
                response = final_response(completion.text, thinking, reject_reopened=reject)
                result = ({"correct": False, "invalid": True, "invalid_reason": "unclosed_thinking"}
                          if response is None else grade(record, response))
            if type(result["correct"]) is not bool or (result["invalid"] and result["correct"]):
                raise ValueError("grader returned an inconsistent result")
            hits += int(result["correct"])
            invalid += int(result["invalid"])
            if "loose_prompt_correct" in result:
                loose.append(bool(result["loose_prompt_correct"]))
            rows.append({"problem_id": record["id"], "sample_idx": sample_idx,
                         "finish_reason": completion.finish_reason, "completion_text": completion.text,
                         **result})
        counts.append(hits)
    total = len(records) * num_samples
    accuracy = sum(counts) / total
    out = {
        "benchmark": benchmark, "primary_metric": PRIMARY_METRIC[benchmark],
        "accuracy": accuracy, "accuracy_percent": 100 * accuracy,
        "num_problems": len(records), "num_samples": num_samples,
        "num_correct": sum(counts), "invalid_answers": invalid, "error_total": errors,
        "finish_reasons": dict(finish),
        "pass_at_k": {str(k): sum(pass_at_k(num_samples, c, k) for c in counts) / len(counts)
                      for k in (1, 2, 4, 8, 16, 32, 64) if k <= num_samples},
        "problem_counts": {record["id"]: c for record, c in zip(records, counts)},
        "rollouts": rows,
    }
    if benchmark == "ifeval" and loose:
        out["loose_prompt_accuracy"] = sum(loose) / len(loose)
    return out


def run_general_eval(records: list[dict], benchmark: str, tokenizer, server_urls: List[str], *,
                     num_samples: int, thinking: bool, max_tokens: int = 32768, temperature: float = 0.6,
                     top_p: float = 0.95, top_k: int = 20, model: str = "Qwen/Qwen3-1.7B",
                     concurrency: int = 32, fanout_per_prompt: int = 0, timeout: int = 3600,
                     max_retries: int = 3, seed: int = 1234, max_error_completions: int = 0) -> dict:
    prompts = [render_prompt_ids(tokenizer, r["prompt"], thinking) for r in records]
    print(f"[general-eval] {benchmark}: {len(records)} problems x {num_samples} samples", flush=True)
    completions = batched_generate(
        prompts, server_urls, n_per_prompt=num_samples, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, top_k=top_k, model=model, concurrency=concurrency,
        fanout_per_prompt=fanout_per_prompt or num_samples, timeout=timeout, max_retries=max_retries,
        allow_error_completions=True, seed_base=seed)
    result = score_completions(records, benchmark, completions, num_samples=num_samples, thinking=thinking)
    print(f"[general-eval] {benchmark}: accuracy={result['accuracy_percent']:.2f}% "
          f"invalid={result['invalid_answers']} errors={result['error_total']}", flush=True)
    if 0 <= max_error_completions < result["error_total"]:
        raise RuntimeError(f"error completions {result['error_total']} exceed {max_error_completions}")
    return result


_PROTOCOL_KEYS = ("benchmark", "num_samples", "max_tokens", "temperature", "top_p", "top_k", "min_p",
                  "prompt_version", "grader_version", "dataset")


def summarize_change(base_paths: List[str], method_paths: List[str]) -> dict:
    """Per-benchmark change from Base in percentage points and the mean over PRIOR_BENCHMARKS."""
    def load(paths):
        out = {}
        for path in paths:
            result = json.loads(Path(path).read_text())
            benchmark = result["protocol"]["benchmark"]
            if benchmark in out:
                raise ValueError(f"duplicate {benchmark} result")
            if result.get("error_total", 0) or result["protocol"].get("max_problems", 0):
                raise ValueError(f"{path}: failed or partial benchmark pool")
            out[benchmark] = result
        return out

    base, method = load(base_paths), load(method_paths)
    if set(base) != set(method):
        raise ValueError(f"benchmarks differ: base {sorted(base)} vs method {sorted(method)}")
    rows = {}
    for benchmark in BENCHMARKS:
        if benchmark not in base:
            continue
        for key in _PROTOCOL_KEYS:
            if base[benchmark]["protocol"].get(key) != method[benchmark]["protocol"].get(key):
                raise ValueError(f"{benchmark}: protocol field {key} differs between Base and method")
        b, m = base[benchmark]["accuracy_percent"], method[benchmark]["accuracy_percent"]
        rows[benchmark] = {"base_percent": b, "method_percent": m, "change_percent": m - b}
    missing = [b for b in PRIOR_BENCHMARKS if b not in rows]
    if missing:
        raise ValueError(f"missing prior-capability benchmarks: {missing}")
    return {"benchmarks": rows,
            "prior_capabilities_change_percent":
                sum(rows[b]["change_percent"] for b in PRIOR_BENCHMARKS) / len(PRIOR_BENCHMARKS),
            "prior_benchmarks": list(PRIOR_BENCHMARKS),
            "base_model": next(iter(base.values()))["protocol"].get("model_name"),
            "method_model": next(iter(method.values()))["protocol"].get("model_name")}


PROMPT_VERSION = {"mmlu_pro": "mmlu_pro_cot_5shot", "mmlu_redux": "delimited_json_5shot",
                  "ifeval": "author_verbatim", "gpqa_diamond": "zero_shot_boxed_letter",
                  "zebralogic": "zeroeval_zebra_grid"}
GRADER_VERSION = {"mmlu_pro": "explicit-final-letter-v1", "mmlu_redux": "strict-json-answer-v1",
                  "ifeval": "author-strict-prompt", "gpqa_diamond": "explicit-final-letter-format-v3",
                  "zebralogic": "zeroeval-whole-puzzle"}
DATASET = {
    "mmlu_pro": {"repo": MMLU_PRO_REPO, "revision": MMLU_PRO_REVISION, "split": "test", "shots": "validation"},
    "mmlu_redux": {"repo": REDUX_CURATED_REPO, "revision": REDUX_CURATED_REVISION,
                   "publisher": [REDUX_PUBLISHER_REPO, REDUX_PUBLISHER_REVISION],
                   "shots": [MMLU_DEV_REPO, MMLU_DEV_REVISION, "dev"]},
    "ifeval": {"repo": "google-research/google-research", "revision": IFEVAL_REVISION},
    "gpqa_diamond": {"repo": "idavidrein/gpqa", "revision": GPQA_REVISION, "sha256": GPQA_DIAMOND_SHA256},
    "zebralogic": {"repo": ZEBRA_REPO, "revision": ZEBRA_REVISION, "config": "grid_mode", "split": "test",
                   "template": ["WildEval/ZeroEval", ZEROEVAL_REVISION]},
}


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="General-capability evaluation and change from Base.")
    p.add_argument("--benchmark", choices=BENCHMARKS)
    p.add_argument("--model-name", default="Qwen/Qwen3-1.7B")
    p.add_argument("--served-model-name", default="")
    p.add_argument("--tokenizer-name", default="")
    p.add_argument("--server-urls", default=os.environ.get("EXPDIS_VLLM_SERVER_URLS", ""))
    p.add_argument("--num-samples", type=int, default=None,
                   help="Default: 1 for mmlu_pro, 64 otherwise")
    p.add_argument("--max-tokens", type=int, default=32768)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--enable-thinking", default="auto", choices=["auto", "always", "never"])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--fanout-per-prompt", type=int, default=0, help="0: one request per sample")
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--max-error-completions", type=int, default=0)
    p.add_argument("--max-problems", type=int, default=0, help="Smoke tests only; --summarize rejects partial pools")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-path", default="")
    p.add_argument("--summarize", action="store_true", help="Compute change from Base; no server needed")
    p.add_argument("--base-results", nargs="+", default=[])
    p.add_argument("--results", nargs="+", default=[])
    args = p.parse_args(argv)

    if args.summarize:
        if not args.base_results or not args.results:
            p.error("--summarize needs --base-results and --results")
        out = summarize_change(args.base_results, args.results)
    else:
        if not args.benchmark or not args.server_urls:
            p.error("--benchmark and --server-urls are required for generation")
        from transformers import AutoTokenizer

        tokenizer_name = args.tokenizer_name or args.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, fix_mistral_regex=True)
        thinking = thinking_enabled(tokenizer, args.enable_thinking)
        num_samples = args.num_samples or DEFAULT_NUM_SAMPLES[args.benchmark]
        data_dir = Path(args.data_dir)
        if args.benchmark == "ifeval":
            ifeval_scorer(data_dir)
        records = LOADERS[args.benchmark](data_dir)
        if args.max_problems > 0:
            records = records[:args.max_problems]
        out = run_general_eval(
            records, args.benchmark, tokenizer, parse_server_urls(args.server_urls),
            num_samples=num_samples, thinking=thinking, max_tokens=args.max_tokens,
            temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
            model=args.served_model_name or args.model_name, concurrency=args.concurrency,
            fanout_per_prompt=args.fanout_per_prompt, timeout=args.timeout, max_retries=args.max_retries,
            seed=args.seed, max_error_completions=args.max_error_completions)
        out["protocol"] = {
            "benchmark": args.benchmark, "model_name": args.model_name, "tokenizer_name": tokenizer_name,
            "num_samples": num_samples, "max_tokens": args.max_tokens, "temperature": args.temperature,
            "top_p": args.top_p, "top_k": args.top_k, "min_p": 0.0, "enable_thinking": thinking,
            "seed": args.seed, "max_problems": args.max_problems,
            "prompt_version": PROMPT_VERSION[args.benchmark], "grader_version": GRADER_VERSION[args.benchmark],
            "dataset": DATASET[args.benchmark],
        }
    text = json.dumps(out, indent=2) + "\n"
    if args.output_path:
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_path).write_text(text, encoding="utf-8")
        print(f"[general-eval] wrote {args.output_path}", flush=True)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
