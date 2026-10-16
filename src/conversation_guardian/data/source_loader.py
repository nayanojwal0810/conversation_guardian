"""Source loader and integrity validator for the Cornell ConvoKit CGA-WIKI corpus."""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
import importlib.metadata

from .schemas import UtteranceRecord, ConversationRecord

VALID_SPLITS: Set[str] = {"train", "val", "test"}


def get_default_corpus_path() -> Path:
    """Return the default local cache path for conversations-gone-awry-corpus."""
    return Path(os.path.expanduser("~")) / ".convokit" / "saved-corpora" / "conversations-gone-awry-corpus"


def ensure_corpus_available(corpus_dir: Optional[str] = None) -> Path:
    """
    Ensure the ConvoKit corpus is available locally.
    
    If corpus_dir is explicitly provided:
      - Validates existence and presence of required files (conversations.json, utterances.jsonl).
      - If invalid, raises FileNotFoundError immediately. Never falls back silently!
    
    If corpus_dir is None:
      - Checks default cache path.
      - If absent, invokes convokit.download('conversations-gone-awry-corpus').
    """
    if corpus_dir is not None:
        cpath = Path(corpus_dir)
        if not cpath.exists() or not cpath.is_dir():
            raise FileNotFoundError(
                f"Explicitly specified --source-dir does not exist or is not a directory: '{corpus_dir}'"
            )
        conv_file = cpath / "conversations.json"
        utt_file = cpath / "utterances.jsonl"
        if not conv_file.exists():
            raise FileNotFoundError(
                f"Specified source directory '{corpus_dir}' lacks required file: conversations.json"
            )
        if not utt_file.exists():
            raise FileNotFoundError(
                f"Specified source directory '{corpus_dir}' lacks required file: utterances.jsonl"
            )
        return cpath

    default_path = get_default_corpus_path()
    if default_path.exists() and (default_path / "conversations.json").exists() and (default_path / "utterances.jsonl").exists():
        return default_path

    # Download through official ConvoKit API
    from convokit import download
    downloaded_str = download("conversations-gone-awry-corpus")
    return Path(downloaded_str)


def compute_file_sha256(filepath: Path) -> Tuple[int, str]:
    """Compute file size in bytes and SHA-256 hex digest."""
    sha256 = hashlib.sha256()
    size = filepath.stat().st_size
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return size, sha256.hexdigest()


def generate_source_manifest(corpus_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate portable source manifest recording source identity, checksums, and package versions
    WITHOUT exposing machine-specific absolute file system paths or user directories.
    """
    cpath = ensure_corpus_available(corpus_dir)
    manifest = {
        "dataset_name": "conversations-gone-awry-corpus",
        "dataset_source": "Cornell ConvoKit (Zhang et al. ACL 2018)",
        "acquisition_method": "convokit.download('conversations-gone-awry-corpus')",
        "canonical_relative_path": ".convokit/saved-corpora/conversations-gone-awry-corpus",
        "environment": {
            "python_version": sys.version.split()[0],
            "convokit_version": importlib.metadata.version("convokit"),
            "pandas_version": importlib.metadata.version("pandas"),
            "pyarrow_version": importlib.metadata.version("pyarrow"),
            "numpy_version": importlib.metadata.version("numpy"),
        },
        "files": {},
    }

    for fname in sorted(os.listdir(cpath)):
        fpath = cpath / fname
        if fpath.is_file():
            size, hsh = compute_file_sha256(fpath)
            manifest["files"][fname] = {
                "size_bytes": size,
                "sha256": hsh,
            }

    return manifest


def load_cga_source_records(corpus_dir: Optional[str] = None) -> Dict[str, ConversationRecord]:
    """
    Load raw conversation and utterance records with strict validation.
    Fails fast if conversation IDs are missing, split names are invalid, or orphaned utterances exist.
    """
    cpath = ensure_corpus_available(corpus_dir)
    conv_file = cpath / "conversations.json"
    utt_file = cpath / "utterances.jsonl"

    with open(conv_file, "r", encoding="utf-8") as f:
        raw_convs: Dict[str, Dict[str, Any]] = json.load(f)

    conversations: Dict[str, ConversationRecord] = {}
    for cid, cmeta in raw_convs.items():
        cid_str = str(cid).strip()
        if not cid_str:
            raise ValueError("Corpus contains an empty or null conversation_id in conversations.json")

        split_str = str(cmeta.get("split", "")).lower().strip()
        if split_str not in VALID_SPLITS:
            raise ValueError(
                f"Conversation '{cid_str}' has invalid or missing split '{split_str}'. Expected one of {VALID_SPLITS}."
            )

        pair_id = str(cmeta.get("pair_id", "")).strip()
        if not pair_id:
            raise ValueError(f"Conversation '{cid_str}' is missing required pair_id.")

        conversations[cid_str] = ConversationRecord(
            conversation_id=cid_str,
            pair_id=pair_id,
            split=split_str,
            conversation_has_personal_attack=bool(cmeta.get("conversation_has_personal_attack", False)),
            page_title=str(cmeta.get("page_title", "")),
            page_id=int(cmeta.get("page_id", 0)),
            utterances=[],
        )

    # Load utterances and fail fast on unknown conversation references
    unknown_conv_utterances: List[str] = []
    with open(utt_file, "r", encoding="utf-8") as f:
        for orig_idx, line in enumerate(f):
            if not line.strip():
                continue
            udata = json.loads(line)
            cid = str(udata.get("conversation_id", "")).strip()
            uid = str(udata.get("id", "")).strip()

            if not uid:
                raise ValueError(f"Corpus contains utterance with empty or null ID at line {orig_idx + 1}")

            if cid not in conversations:
                unknown_conv_utterances.append(f"utt_id={uid}, invalid_conv_id='{cid}' (line {orig_idx + 1})")
                continue

            ts = udata.get("timestamp")
            if ts is None:
                raise ValueError(f"Utterance '{uid}' has null timestamp.")

            meta = udata.get("meta", {})
            utt_rec = UtteranceRecord(
                id=uid,
                conversation_id=cid,
                speaker=str(udata.get("speaker", "unknown")),
                reply_to=str(udata.get("reply-to")) if udata.get("reply-to") is not None else None,
                timestamp=float(ts),
                text=str(udata.get("text") or ""),
                is_section_header=bool(meta.get("is_section_header", False)),
                comment_has_personal_attack=bool(meta.get("comment_has_personal_attack", False)),
                toxicity=float(meta.get("toxicity", 0.0) or 0.0),
                original_index=orig_idx,
            )
            conversations[cid].utterances.append(utt_rec)

    if unknown_conv_utterances:
        sample_errs = "\n".join(unknown_conv_utterances[:10])
        raise ValueError(
            f"Corpus corruption: found {len(unknown_conv_utterances)} utterances referencing missing or unknown conversation IDs:\n{sample_errs}"
        )

    # Perform structural source integrity checks
    validate_pair_integrity(conversations)
    validate_attack_annotation_consistency(conversations)
    validate_section_headers(conversations)

    return conversations


def validate_pair_integrity(conversations: Dict[str, ConversationRecord]) -> None:
    """
    Validate that all pair_ids resolve, pairs are strictly reciprocal,
    pair partners share the exact same partition split, and have opposite personal attack labels.
    """
    for cid, c in conversations.items():
        pid = c.pair_id
        if pid not in conversations:
            raise ValueError(f"Pair integrity failure: conversation '{cid}' references non-existent pair_id '{pid}'")
        partner = conversations[pid]
        if partner.pair_id != cid:
            raise ValueError(
                f"Pair reciprocity failure: '{cid}' points to '{pid}', but '{pid}' points to '{partner.pair_id}'"
            )
        if partner.split != c.split:
            raise ValueError(
                f"Cross-partition pair leakage: conversation '{cid}' ({c.split}) and partner '{pid}' ({partner.split}) cross splits"
            )
        if c.conversation_has_personal_attack == partner.conversation_has_personal_attack:
            raise ValueError(
                f"Pair label concordance violation: paired conversations '{cid}' and '{pid}' have identical attack label ({c.conversation_has_personal_attack})"
            )


def validate_attack_annotation_consistency(conversations: Dict[str, ConversationRecord]) -> None:
    """
    Validate 100% concordance between conversation-level attack flags and utterance-level annotations:
    - 0 attack annotations in healthy conversations.
    - Exactly 1 attack annotation in attacked conversations.
    """
    mismatches = []
    for cid, c in conversations.items():
        attack_utts = [u for u in c.utterances if u.comment_has_personal_attack]
        n_attacks = len(attack_utts)
        if not c.conversation_has_personal_attack and n_attacks > 0:
            mismatches.append(f"Conversation '{cid}' is healthy (has_attack=False) but contains {n_attacks} attack-annotated utterances")
        elif c.conversation_has_personal_attack and n_attacks != 1:
            mismatches.append(f"Conversation '{cid}' has_attack=True but contains {n_attacks} attack-annotated utterances (expected exactly 1)")

    if mismatches:
        sample = "\n".join(mismatches[:10])
        raise ValueError(f"Attack annotation concordance violation in {len(mismatches)} conversations:\n{sample}")


def validate_section_headers(conversations: Dict[str, ConversationRecord]) -> Dict[str, int]:
    """
    Validate that section headers are never annotated as attacks,
    and that every conversation contains at least one authentic dialogue turn after header filtering.
    """
    total_headers = 0
    total_dialogue = 0
    header_attacks = []
    empty_dialogues = []

    for cid, c in conversations.items():
        headers = [u for u in c.utterances if u.is_section_header]
        dialogue = [u for u in c.utterances if not u.is_section_header]

        total_headers += len(headers)
        total_dialogue += len(dialogue)

        for h in headers:
            if h.comment_has_personal_attack:
                header_attacks.append(h.id)

        if len(dialogue) == 0:
            empty_dialogues.append(cid)

    if header_attacks:
        raise ValueError(f"Section header attack violation: headers {header_attacks[:10]} marked as attacks")

    if empty_dialogues:
        raise ValueError(f"Section header filtering violation: conversations {empty_dialogues[:10]} have 0 dialogue messages")

    return {
        "total_conversations": len(conversations),
        "total_section_headers": total_headers,
        "total_filtered_dialogue": total_dialogue,
    }
