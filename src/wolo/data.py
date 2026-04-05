from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import torch
from libzim.reader import Archive


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ignored_tag_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._ignored_tag_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._ignored_tag_depth > 0:
            self._ignored_tag_depth -= 1

    def handle_data(self, data: str) -> None:
        if data and self._ignored_tag_depth == 0:
            self._parts.append(data)

    def extract(self) -> str:
        return " ".join(self._parts)


def _strip_html(text: str) -> str:
    if "<" not in text or ">" not in text:
        return text

    parser = _HTMLTextExtractor()
    try:
        parser.feed(text)
        parser.close()
        return parser.extract()
    except Exception:
        return text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_likely_article_entry(entry, item) -> bool:
    namespace = getattr(entry, "namespace", None)
    if namespace is not None and str(namespace) != "A":
        return False

    path = getattr(entry, "path", None) or getattr(entry, "url", None)
    if isinstance(path, str) and "/" in path:
        prefix = path.split("/", 1)[0]
        if len(prefix) == 1 and prefix.isalpha() and prefix != "A":
            return False

    mimetype = getattr(item, "mimetype", None)
    if mimetype is None:
        mimetype = getattr(item, "mime_type", None)
    if callable(mimetype):
        mimetype = mimetype()

    if isinstance(mimetype, str):
        lowered = mimetype.lower()
        if "html" not in lowered and "text/plain" not in lowered:
            return False

    return True


def _looks_like_boilerplate(text: str) -> bool:
    lowered = text.lower()
    noisy_markers = (
        "mw-parser-output",
        "vector-sticky-header",
        "ext.cite",
        "navbox",
        "wikitable",
    )
    hit_count = sum(marker in lowered for marker in noisy_markers)
    return hit_count >= 2


@dataclass
class ZimByteSampler:
    zim_path: Path
    min_entry_id: int | None = None
    max_entry_id: int | None = None
    seed: int = 0
    max_attempts: int = 128
    min_text_chars: int = 256

    def __post_init__(self) -> None:
        self._archive = Archive(self.zim_path)
        
        # Sequential reading: iterate through entries by ID in order
        self._entry_id = 0
        # Determine the range of entry IDs to iterate through
        max_id = getattr(self._archive, "all_entry_count", None) or getattr(self._archive, "entry_count", None)
        if max_id is None:
            raise RuntimeError("Archive does not expose entry count")
        
        if self.min_entry_id is not None:
            self._entry_id = self.min_entry_id
        if self.max_entry_id is not None:
            self._max_entry_id = self.max_entry_id + 1
        else:
            self._max_entry_id = int(max_id)
        
        self._current_buffer = b""
        self._advance_to_next_entry()

    def _advance_to_next_entry(self) -> None:
        """Load the next entry's text into the buffer."""
        while self._entry_id < self._max_entry_id:
            try:
                entry = self._archive._get_entry_by_id(self._entry_id)
                self._entry_id += 1
                
                text = self._entry_to_text(entry)
                self._current_buffer = text.encode("utf-8", errors="ignore")
                
                if len(self._current_buffer) > 0:
                    return
            except Exception:
                continue
        
        # If we exhaust entries, wrap around to the beginning
        self._entry_id = self.min_entry_id if self.min_entry_id is not None else 0
        self._current_buffer = b""

    def close(self) -> None:
        if hasattr(self, "_archive"):
            del self._archive

    def _entry_to_text(self, entry) -> str:
        item = entry.get_item()
        if not _is_likely_article_entry(entry, item):
            return ""

        content = item.content
        if isinstance(content, (bytes, bytearray, memoryview)):
            text = bytes(content).decode("utf-8", errors="ignore")
        elif hasattr(content, "tobytes"):
            text = content.tobytes().decode("utf-8", errors="ignore")
        else:
            text = str(content)

        normalized = _normalize_text(_strip_html(text))
        if len(normalized) < self.min_text_chars:
            return ""
        if _looks_like_boilerplate(normalized):
            return ""
        return normalized

    def sample_sequence(self, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a single sequence by reading sequentially through the archive."""
        for _ in range(self.max_attempts):
            # If current buffer has enough data, sample from it
            if len(self._current_buffer) > sequence_length:
                start = 0  # Always take from the beginning of buffer
                window = self._current_buffer[start : start + sequence_length + 1]
                # Remove the used portion from buffer
                self._current_buffer = self._current_buffer[start + sequence_length :]
                
                inputs = torch.tensor(list(window[:-1]), dtype=torch.long)
                targets = torch.tensor(list(window[1:]), dtype=torch.long)
                return inputs, targets
            
            # Buffer exhausted, move to next entry
            self._advance_to_next_entry()
            # Check if we wrapped around (no more valid entries found)
            if len(self._current_buffer) == 0:
                raise RuntimeError("Could not sample a usable Wikipedia sequence from the ZIM archive")

        raise RuntimeError("Could not sample a usable Wikipedia sequence from the ZIM archive")

    def sample_batch(self, batch_size: int, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_inputs = []
        batch_targets = []

        for _ in range(batch_size):
            inputs, targets = self.sample_sequence(sequence_length)
            batch_inputs.append(inputs)
            batch_targets.append(targets)

        return torch.stack(batch_inputs, dim=0), torch.stack(batch_targets, dim=0)