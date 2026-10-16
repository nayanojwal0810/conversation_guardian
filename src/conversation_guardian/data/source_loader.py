"""Source loader for the Cornell ConvoKit CGA-WIKI corpus."""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import importlib.metadata

from .schemas import UtteranceRecord, ConversationRecord


def get_default_corpus_path() -> Path:
    """Return the default local path for conversations-gone-awry-corpus."""
    home_dir = Path(os.path.expanduser("~"))
    return home_dir / ".convokit" / "saved-corpora" / "conversations-gone-awry-corpus"


def ensure_corpus_available(corpus_dir: Optional[str] = None) -> Path:
    """Ensure the ConvoKit corpus is available locally, downloading via convokit if absent."""
    if corpus_dir:
        cpath = Path(corpus_dir)
        if cpath.exists():
            return cpath

    default_path = get_default_corpus_path()
    if default_path.exists():
        return default_path

    # Download through official ConvoKit API
    from convokit import download
    downloaded_str = download("conversations-gone-awry-corpus")
    return Path(downloaded_str)


def compute_file_sha256(filepath: Path) -> Tuple[int, str]:
    """Compute file size in bytes and SHA256 hex digest."""
    sha256 = hashlib.sha256()
    size = filepath.stat().st_size
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return size, sha256.hexdigest()


def generate_source_manifest(corpus_dir: Optional[str] = None) -> Dict[str, Any]:
    """Generate reproducible source manifest with checksums, versions, and paths."""
    cpath = ensure_corpus_available(corpus_dir)
    manifest = {
        "dataset_name": "conversations-gone-awry-corpus",
        "dataset_source": "Cornell ConvoKit (Zhang et al. ACL 2018)",
        "acquisition_method": "convokit.download('conversations-gone-awry-corpus')",
        "corpus_directory": str(cpath.resolve()),
        "environment": {
            "python_version": sys.version,
            "convokit_version": importlib.metadata.version("convokit"),
            "pandas_version": importlib.metadata.version("pandas"),
            "pyarrow_version": importlib.metadata.version("pyarrow"),
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
    Load raw conversation and utterance records directly from the official corpus JSON files.
    Preserves all original fields and official split assignments without truncation or modification.
    """
    cpath = ensure_corpus_available(corpus_dir)
    conv_file = cpath / "conversations.json"
    utt_file = cpath / "utterances.jsonl"

    if not conv_file.exists():
        raise FileNotFoundError(f"Missing conversations.json in {cpath}")
    if not utt_file.exists():
        raise FileNotFoundError(f"Missing utterances.jsonl in {cpath}")

    # Load conversation metadata
    with open(conv_file, "r", encoding="utf-8") as f:
        raw_convs: Dict[str, Dict[str, Any]] = json.load(f)

    conversations: Dict[str, ConversationRecord] = {}
    for cid, cmeta in raw_convs.items():
        conversations[cid] = ConversationRecord(
            conversation_id=str(cid),
            pair_id=str(cmeta.get("pair_id", "")),
            split=str(cmeta.get("split", "unknown")).lower().strip(),
            conversation_has_personal_attack=bool(cmeta.get("conversation_has_personal_attack", False)),
            page_title=str(cmeta.get("page_title", "")),
            page_id=int(cmeta.get("page_id", 0)),
            utterances=[],
        )

    # Load utterances
    with open(utt_file, "r", encoding="utf-8") as f:
        for orig_idx, line in enumerate(f):
            if not line.strip():
                continue
            udata = json.loads(line)
            cid = str(udata.get("conversation_id", ""))
            if cid not in conversations:
                continue

            meta = udata.get("meta", {})
            utt_rec = UtteranceRecord(
                id=str(udata.get("id", "")),
                conversation_id=cid,
                speaker=str(udata.get("speaker", "unknown")),
                reply_to=str(udata.get("reply-to")) if udata.get("reply-to") is not None else None,
                timestamp=float(udata.get("timestamp", 0.0) or 0.0),
                text=str(udata.get("text") or ""),
                is_section_header=bool(meta.get("is_section_header", False)),
                comment_has_personal_attack=bool(meta.get("comment_has_personal_attack", False)),
                toxicity=float(meta.get("toxicity", 0.0) or 0.0),
                original_index=orig_idx,
            )
            conversations[cid].utterances.append(utt_rec)

    return conversations
