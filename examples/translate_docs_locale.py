#!/usr/bin/env python3
"""Batch audit/translate Sphinx locale catalogs using SimpleLLMFunc.

This script targets docs/source/locale/<lang>/LC_MESSAGES/**/*.po and supports:
1) Audit mode: coverage, fuzzy, metadata consistency.
2) Translation mode: fill empty/fuzzy msgstr entries with SimpleLLMFunc.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import polib


LANG_METADATA: Dict[str, Dict[str, str]] = {
    "en": {
        "Language": "en",
        "Language-Team": "en <LL@li.org>",
        "Plural-Forms": "nplurals=2; plural=(n != 1);",
    },
    "zh_CN": {
        "Language": "zh_CN",
        "Language-Team": "zh_CN <LL@li.org>",
        "Plural-Forms": "nplurals=1; plural=0;",
    },
}

IDENTIFIER_ONLY_RE = re.compile(r"^[`{}\w\-./:+|()]+$")
HAS_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


@dataclass
class EntryTask:
    po_path: Path
    entry: polib.POEntry


def has_cjk(text: str) -> bool:
    return bool(HAS_CJK_RE.search(text))


def should_copy_as_is(msgid: str) -> bool:
    """Detect technical tokens that should remain unchanged."""
    text = msgid.strip()
    if not text:
        return True
    if has_cjk(text):
        return False
    if "\n" in text:
        return False
    if IDENTIFIER_ONLY_RE.fullmatch(text) is None:
        return False
    return True


def discover_po_files(locale_root: Path, target_lang: str) -> List[Path]:
    base = locale_root / target_lang / "LC_MESSAGES"
    if not base.exists():
        raise FileNotFoundError(f"Locale path not found: {base}")
    return sorted(base.rglob("*.po"))


def normalize_metadata(po: polib.POFile, target_lang: str) -> None:
    if target_lang not in LANG_METADATA:
        return
    for key, value in LANG_METADATA[target_lang].items():
        po.metadata[key] = value


def gather_stats(
    po_files: List[Path], target_lang: str
) -> Tuple[int, int, int, int, int]:
    total = translated = fuzzy = obsolete = metadata_bad = 0
    expected = LANG_METADATA.get(target_lang)

    for po_path in po_files:
        po = polib.pofile(str(po_path))
        if expected:
            for key, value in expected.items():
                if po.metadata.get(key) != value:
                    metadata_bad += 1
                    break

        for entry in po:
            if entry.obsolete:
                obsolete += 1
                continue
            if not entry.msgid:
                continue
            total += 1
            if entry.msgstr.strip():
                translated += 1
            if "fuzzy" in entry.flags:
                fuzzy += 1

    return total, translated, fuzzy, obsolete, metadata_bad


def build_tasks(
    po_files: List[Path],
    target_lang: str,
    include_fuzzy: bool,
    normalize_meta: bool,
) -> Tuple[Dict[Path, polib.POFile], List[EntryTask]]:
    po_map: Dict[Path, polib.POFile] = {}
    tasks: List[EntryTask] = []

    for po_path in po_files:
        po = polib.pofile(str(po_path))
        if normalize_meta:
            normalize_metadata(po, target_lang)
        po_map[po_path] = po

        for entry in po:
            if entry.obsolete or not entry.msgid:
                continue

            needs_translation = not entry.msgstr.strip()
            if include_fuzzy and "fuzzy" in entry.flags:
                needs_translation = True

            if needs_translation:
                tasks.append(EntryTask(po_path=po_path, entry=entry))

    return po_map, tasks


def pick_llm_interface(
    provider_json: Path,
    provider_id: str | None,
    model_name: str | None,
):
    from SimpleLLMFunc.interface.openai_compatible import OpenAICompatible

    providers = OpenAICompatible.load_from_json_file(str(provider_json))
    if not providers:
        raise ValueError("No provider loaded from JSON.")

    chosen_provider = provider_id or sorted(providers.keys())[0]
    if chosen_provider not in providers:
        available = ", ".join(sorted(providers.keys()))
        raise ValueError(
            f"Provider '{chosen_provider}' not found. Available: {available}"
        )

    models = providers[chosen_provider]
    if not models:
        raise ValueError(f"Provider '{chosen_provider}' has no models.")

    chosen_model = model_name or sorted(models.keys())[0]
    if chosen_model not in models:
        available = ", ".join(sorted(models.keys()))
        raise ValueError(
            f"Model '{chosen_model}' not found under '{chosen_provider}'. Available: {available}"
        )

    print(f"Using provider/model: {chosen_provider}/{chosen_model}")
    return models[chosen_model]


def build_translator(llm_interface) -> Callable[..., asyncio.Future]:
    from SimpleLLMFunc.llm_decorator.llm_function_decorator import llm_function

    @llm_function(llm_interface=llm_interface)  # type: ignore
    async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
        """
        You are a professional software documentation translator.

        Translate `text` from {source_lang} to {target_lang}.

        Hard rules:
        1) Keep Markdown structure, punctuation style, and line breaks as much as possible.
        2) Keep URLs, file paths, code identifiers, and inline code unchanged.
        3) Keep placeholders and symbols unchanged (for example: `{text: str}`, `%s`, `{name}`).
        4) Do not add explanations or notes.
        5) Return only translated text.
        """

        return ""

    return translate_text


async def translate_one(
    task: EntryTask,
    translator: Callable[..., Any] | None,
    source_lang: str,
    target_lang: str,
    max_retries: int,
) -> Tuple[bool, str | None]:
    msgid = task.entry.msgid

    if source_lang == target_lang:
        return True, msgid

    if should_copy_as_is(msgid):
        return True, msgid

    if translator is None:
        return False, "Translator is not initialized"

    for attempt in range(max_retries + 1):
        try:
            output = await translator(
                text=msgid,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            if not isinstance(output, str):
                output = str(output)
            output = output.strip()
            if not output:
                raise ValueError("Translator returned empty output.")
            return True, output
        except Exception as exc:
            if attempt >= max_retries:
                return False, str(exc)
            await asyncio.sleep(min(2**attempt, 5))

    return False, "Unknown translation error"


async def run_translation(
    tasks: List[EntryTask],
    translator: Callable[..., Any] | None,
    source_lang: str,
    target_lang: str,
    max_concurrent: int,
    max_retries: int,
) -> Tuple[int, int]:
    semaphore = asyncio.Semaphore(max_concurrent)
    success_count = 0
    failure_count = 0

    async def worker(idx: int, entry_task: EntryTask) -> None:
        nonlocal success_count, failure_count

        async with semaphore:
            ok, text_or_err = await translate_one(
                task=entry_task,
                translator=translator,
                source_lang=source_lang,
                target_lang=target_lang,
                max_retries=max_retries,
            )

        if ok and text_or_err is not None:
            entry_task.entry.msgstr = text_or_err
            entry_task.entry.flags = [f for f in entry_task.entry.flags if f != "fuzzy"]
            success_count += 1
        else:
            failure_count += 1
            print(
                f"[FAIL {idx}/{len(tasks)}] {entry_task.po_path.name}: "
                f"{entry_task.entry.msgid[:80]} :: {text_or_err}"
            )

        if idx % 10 == 0 or idx == len(tasks):
            print(
                f"Progress: {idx}/{len(tasks)} (ok={success_count}, fail={failure_count})"
            )

    await asyncio.gather(
        *[worker(i + 1, task) for i, task in enumerate(tasks)],
        return_exceptions=False,
    )

    return success_count, failure_count


def save_po_files(po_map: Dict[Path, polib.POFile]) -> None:
    for po_path, po in po_map.items():
        po.save(str(po_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and batch-translate docs locale .po catalogs with SimpleLLMFunc"
    )
    parser.add_argument(
        "--locale-root",
        default="docs/source/locale",
        help="Locale root path (default: docs/source/locale)",
    )
    parser.add_argument(
        "--target-lang",
        default="en",
        help="Target locale language key (default: en)",
    )
    parser.add_argument(
        "--source-lang",
        default="zh_CN",
        help="Source language label for LLM prompt (default: zh_CN)",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable translation mode (default is audit-only)",
    )
    parser.add_argument(
        "--include-fuzzy",
        action="store_true",
        help="Also translate fuzzy entries",
    )
    parser.add_argument(
        "--normalize-metadata",
        action="store_true",
        help="Normalize Language/Language-Team/Plural-Forms metadata",
    )
    parser.add_argument(
        "--provider-json",
        default="provider.json",
        help="Path to provider.json (default: provider.json)",
    )
    parser.add_argument("--provider", default=None, help="Provider id in provider.json")
    parser.add_argument("--model", default=None, help="Model name under provider id")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum concurrent translation calls (default: 8)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries for one translation (default: 2)",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=0,
        help="Limit number of entries to translate (0 means no limit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, only print planned actions",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    locale_root = Path(args.locale_root).resolve()
    provider_json = Path(args.provider_json).resolve()

    po_files = discover_po_files(locale_root, args.target_lang)
    total, translated, fuzzy, obsolete, metadata_bad = gather_stats(
        po_files, args.target_lang
    )

    print(f"Locale root: {locale_root}")
    print(f"Target lang: {args.target_lang}")
    print(f"PO files: {len(po_files)}")
    print(
        f"Entries: total={total}, translated={translated}, "
        f"untranslated={total - translated}, fuzzy={fuzzy}, obsolete={obsolete}, "
        f"metadata_mismatch_files={metadata_bad}"
    )

    po_map, tasks = build_tasks(
        po_files=po_files,
        target_lang=args.target_lang,
        include_fuzzy=args.include_fuzzy,
        normalize_meta=args.normalize_metadata,
    )

    if args.max_entries > 0:
        tasks = tasks[: args.max_entries]

    print(f"Candidate entries: {len(tasks)}")

    if not args.translate:
        print("Audit complete (translation disabled).")
        return

    if not tasks:
        print("No entries need translation.")
        if args.normalize_metadata and not args.dry_run:
            save_po_files(po_map)
            print("Saved metadata normalization updates.")
        return

    translator: Callable[..., Any] | None = None
    if args.source_lang != args.target_lang:
        llm_interface = pick_llm_interface(
            provider_json=provider_json,
            provider_id=args.provider,
            model_name=args.model,
        )
        translator = build_translator(llm_interface)
    else:
        print(
            "Source and target language are identical; backfilling msgstr from msgid."
        )

    success_count, failure_count = await run_translation(
        tasks=tasks,
        translator=translator,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
    )

    print(f"Translation done: success={success_count}, failure={failure_count}")

    if args.dry_run:
        print("Dry run enabled; changes were not saved.")
        return

    save_po_files(po_map)
    print("PO files saved.")


if __name__ == "__main__":
    asyncio.run(main())
