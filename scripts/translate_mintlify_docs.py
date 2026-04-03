#!/usr/bin/env python3
"""Audit and sync Mintlify locale pages using override memory + SimpleLLMFunc.

This script is designed for the Mintlify docs tree under ``mintlify_docs``.
It intentionally does not require direct manual translation work.

Capabilities:
1) Audit source pages referenced by ``docs.json`` and compare them with a target locale.
2) Sync locale pages under ``mintlify_docs/<lang>/...`` while preserving MDX structure.
3) Reuse existing locale pages and local override dictionaries as translation memory.
4) Optionally fall back to SimpleLLMFunc for untranslated segments.
5) Optionally rewrite ``docs.json`` to Mintlify ``navigation.languages`` format.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple


IDENTIFIER_ONLY_RE = re.compile(r"^[`{}\w\-./:+|()\[\]<>]+$")
HAS_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
CODE_FENCE_RE = re.compile(r"^\s*(```|~~~)")
TAG_ONLY_RE = re.compile(r"^\s*</?[A-Za-z][^>]*?/?>\s*$")
INLINE_TAG_WITH_TEXT_RE = re.compile(r"^(\s*<[^>]+>)(.*?)(</[A-Za-z][^>]*>\s*)$")
HEADING_RE = re.compile(r"^(\s*#{1,6}\s+)(.+?)\s*$")
LIST_RE = re.compile(r"^(\s*(?:[-*+]|\d+\.)\s+)(.+?)\s*$")
TITLE_ATTR_RE = re.compile(r'(\b(?:title|label|header)\s*=\s*")([^"]+)(")')
FRONTMATTER_FIELD_RE = re.compile(r"^(\s*title\s*:\s*)(.+?)\s*$")
PLACEHOLDER_RE = re.compile(r"__SLMF_I18N_(\d+)__")

DEFAULT_ASSET_PREFIXES = (
    "/img/",
    "/images/",
    "/favicon",
    "/logo",
    "/.well-known/",
    "/llms.txt",
    "/llms-full.txt",
)


@dataclass(frozen=True)
class NavGroup:
    group: str
    pages: List[str]


@dataclass
class PageSpec:
    route: str
    source_path: Path
    target_path: Path
    page_memory: Dict[str, str]


@dataclass
class TranslationTask:
    text: str


def has_cjk(text: str) -> bool:
    return bool(HAS_CJK_RE.search(text))


def should_copy_as_is(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return True
    if has_cjk(raw):
        return False
    if "\n" in raw:
        return False
    return IDENTIFIER_ONLY_RE.fullmatch(raw) is not None


def is_table_separator_line(text: str) -> bool:
    stripped = text.strip()
    if "|" not in stripped:
        return False
    core = stripped.strip("|")
    cells = [cell.strip() for cell in core.split("|")]
    if not cells:
        return False
    for cell in cells:
        if not cell:
            continue
        if any(ch not in "-: " for ch in cell):
            return False
        if "-" not in cell:
            return False
    return True


def load_docs_config(docs_json_path: Path) -> Dict[str, Any]:
    return json.loads(docs_json_path.read_text(encoding="utf-8"))


def dump_docs_config(docs_json_path: Path, config: Dict[str, Any]) -> None:
    docs_json_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def extract_navigation_groups(config: Dict[str, Any]) -> List[NavGroup]:
    navigation = config.get("navigation", {})

    if "languages" in navigation:
        languages = navigation["languages"]
        if not languages:
            raise ValueError("docs.json navigation.languages is empty")
        default_entry = next(
            (entry for entry in languages if entry.get("default")),
            languages[0],
        )
        groups = default_entry.get("groups", [])
        return [
            NavGroup(group=item["group"], pages=list(item["pages"])) for item in groups
        ]

    if "tabs" in navigation:
        tabs = navigation["tabs"]
        if len(tabs) != 1:
            raise ValueError(
                "This script currently expects a single navigation tab before enabling i18n."
            )
        groups = tabs[0].get("groups", [])
        return [
            NavGroup(group=item["group"], pages=list(item["pages"])) for item in groups
        ]

    if "groups" in navigation:
        groups = navigation["groups"]
        return [
            NavGroup(group=item["group"], pages=list(item["pages"])) for item in groups
        ]

    raise ValueError("Unsupported docs.json navigation structure")


def collect_source_routes(
    groups: Sequence[NavGroup], locale_prefixes: Iterable[str]
) -> List[str]:
    prefixes = tuple(f"{prefix}/" for prefix in locale_prefixes)
    seen: set[str] = set()
    routes: List[str] = []
    for group in groups:
        for page in group.pages:
            route = page.strip("/")
            if not route:
                continue
            if route.startswith(prefixes):
                continue
            if route not in seen:
                seen.add(route)
                routes.append(route)
    return routes


def route_to_mdx_path(docs_root: Path, route: str) -> Path:
    return docs_root / f"{route}.mdx"


def load_override_memory(override_path: Path, target_lang: str) -> Dict[str, str]:
    if not override_path.exists():
        return {}

    data = json.loads(override_path.read_text(encoding="utf-8"))
    target_map = data.get(target_lang, {})
    if not isinstance(target_map, dict):
        raise ValueError(f"Override file has invalid shape for language: {target_lang}")
    return {str(key).strip(): str(value).strip() for key, value in target_map.items()}


def build_existing_page_memory(source_text: str, target_text: str) -> Dict[str, str]:
    """Reuse existing localized pages as incremental translation memory."""

    _, source_units = build_placeholder_template(source_text)
    _, target_units = build_placeholder_template(target_text)
    if len(source_units) != len(target_units):
        return {}

    memory: Dict[str, str] = {}
    for source_unit, target_unit in zip(source_units, target_units):
        src = source_unit.strip()
        tgt = target_unit.strip()
        if not src or not tgt or src == tgt:
            continue
        if "__SLMF_I18N_" in tgt:
            continue
        if has_cjk(tgt):
            continue
        memory[src] = tgt
    return memory


def pick_llm_interface(
    provider_json: Path, provider_id: str | None, model_name: str | None
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

    @llm_function(llm_interface=llm_interface)  # type: ignore[misc]
    async def translate_text(text: str, source_lang: str, target_lang: str) -> str:
        """
        You are a professional software documentation translator.

        Translate `text` from {source_lang} to {target_lang}.

        Hard rules:
        1) Preserve MDX/Markdown structure exactly whenever possible.
        2) Keep JSX tag names, YAML keys, URLs, file paths, code identifiers, placeholders, and inline code unchanged.
        3) Keep absolute internal routes that start with `/` unchanged.
        4) Keep punctuation, bullets, and line structure as stable as possible.
        5) Do not add explanations or notes.
        6) Return only translated text.
        """

        return ""

    return translate_text


def make_page_specs(
    docs_root: Path,
    target_lang: str,
    source_routes: Sequence[str],
) -> List[PageSpec]:
    page_specs: List[PageSpec] = []
    for route in source_routes:
        source_path = route_to_mdx_path(docs_root, route)
        if not source_path.exists():
            raise FileNotFoundError(f"Source page not found: {source_path}")
        target_path = docs_root / target_lang / f"{route}.mdx"
        page_memory: Dict[str, str] = {}
        if target_path.exists():
            page_memory.update(
                build_existing_page_memory(
                    source_text=source_path.read_text(encoding="utf-8"),
                    target_text=target_path.read_text(encoding="utf-8"),
                )
            )
        page_specs.append(
            PageSpec(
                route=route,
                source_path=source_path,
                target_path=target_path,
                page_memory=page_memory,
            )
        )
    return page_specs


def build_placeholder_template(text: str) -> Tuple[str, List[str]]:
    units: List[str] = []

    def add_unit(raw: str) -> str:
        idx = len(units)
        units.append(raw)
        return f"__SLMF_I18N_{idx}__"

    def replace_frontmatter_value(line: str) -> str:
        match = FRONTMATTER_FIELD_RE.match(line)
        if not match:
            return line
        value = match.group(2).strip()
        if not value:
            return line
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            quote = value[0]
            inner = value[1:-1]
            return f"{match.group(1)}{quote}{add_unit(inner)}{quote}"
        return f"{match.group(1)}{add_unit(value)}"

    def replace_tag_attrs(line: str) -> str:
        return TITLE_ATTR_RE.sub(
            lambda m: f"{m.group(1)}{add_unit(m.group(2))}{m.group(3)}", line
        )

    def replace_table_cells(line: str) -> str:
        leading = line[: len(line) - len(line.lstrip())]
        row = line.strip()
        starts_pipe = row.startswith("|")
        ends_pipe = row.endswith("|")
        core = row.strip("|")
        cells = [cell.strip() for cell in core.split("|")]
        rebuilt_cells = [add_unit(cell) if cell else "" for cell in cells]
        if starts_pipe or ends_pipe:
            rebuilt = "| " + " | ".join(rebuilt_cells) + " |"
        else:
            rebuilt = " | ".join(rebuilt_cells)
        return leading + rebuilt

    lines = text.splitlines()
    out_lines: List[str] = []
    in_frontmatter = bool(lines) and lines[0].strip() == "---"
    frontmatter_delims = 0
    in_code_fence = False

    for line in lines:
        stripped = line.strip()

        if in_frontmatter:
            out_lines.append(replace_frontmatter_value(line))
            if stripped == "---":
                frontmatter_delims += 1
                if frontmatter_delims >= 2:
                    in_frontmatter = False
            continue

        if CODE_FENCE_RE.match(line):
            out_lines.append(line)
            in_code_fence = not in_code_fence
            continue

        if in_code_fence or not stripped:
            out_lines.append(line)
            continue

        updated = replace_tag_attrs(line)

        inline_tag_match = INLINE_TAG_WITH_TEXT_RE.match(updated)
        if inline_tag_match and inline_tag_match.group(2).strip():
            out_lines.append(
                f"{inline_tag_match.group(1)}{add_unit(inline_tag_match.group(2).strip())}{inline_tag_match.group(3)}"
            )
            continue

        if TAG_ONLY_RE.match(updated):
            out_lines.append(updated)
            continue

        if updated.lstrip().startswith("|"):
            if is_table_separator_line(updated):
                out_lines.append(updated)
            else:
                out_lines.append(replace_table_cells(updated))
            continue

        heading_match = HEADING_RE.match(updated)
        if heading_match:
            out_lines.append(
                f"{heading_match.group(1)}{add_unit(heading_match.group(2))}"
            )
            continue

        list_match = LIST_RE.match(updated)
        if list_match:
            out_lines.append(f"{list_match.group(1)}{add_unit(list_match.group(2))}")
            continue

        leading = updated[: len(updated) - len(updated.lstrip())]
        trailing = updated[len(updated.rstrip()) :]
        core = updated.strip()
        out_lines.append(f"{leading}{add_unit(core)}{trailing}")

    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else ""), units


def build_initial_translation_cache(
    texts: Iterable[str],
    page_memory: Dict[str, str],
    global_memory: Dict[str, str],
    override_memory: Dict[str, str],
    source_lang: str,
    target_lang: str,
) -> Tuple[Dict[str, str], List[str]]:
    cache: Dict[str, str] = {}
    missing: List[str] = []
    seen_missing: set[str] = set()

    for text in texts:
        raw = text.strip()
        if text in cache:
            continue
        if not raw:
            cache[text] = text
            continue
        if source_lang == target_lang:
            cache[text] = text
            continue
        if raw in override_memory:
            cache[text] = override_memory[raw]
            continue
        if raw in page_memory:
            cache[text] = page_memory[raw]
            continue
        if raw in global_memory:
            cache[text] = global_memory[raw]
            continue
        if target_lang == "en" and not has_cjk(raw):
            cache[text] = text
            continue
        if should_copy_as_is(raw):
            cache[text] = text
            continue
        if raw not in seen_missing:
            missing.append(raw)
            seen_missing.add(raw)

    return cache, missing


async def translate_missing_texts(
    missing: Sequence[str],
    translator: Callable[..., Any] | None,
    source_lang: str,
    target_lang: str,
    max_concurrent: int,
    max_retries: int,
) -> Dict[str, str]:
    if not missing:
        return {}

    if translator is None:
        return {text: text for text in missing}

    semaphore = asyncio.Semaphore(max_concurrent)
    results: Dict[str, str] = {}

    async def worker(text: str) -> None:
        async with semaphore:
            for attempt in range(max_retries + 1):
                try:
                    output = await translator(
                        text=text,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                    translated = str(output).strip()
                    if not translated:
                        raise ValueError("Translator returned empty output")
                    results[text] = translated
                    return
                except Exception as exc:
                    if attempt >= max_retries:
                        print(f"[WARN] translation failed: {text[:80]} :: {exc}")
                        results[text] = text
                        return
                    await asyncio.sleep(min(2**attempt, 5))

    await asyncio.gather(*[worker(text) for text in missing], return_exceptions=False)
    return results


def fill_placeholders(
    template: str, units: Sequence[str], cache: Dict[str, str]
) -> str:
    def replace(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        source = units[idx]
        return cache.get(source, source)

    return PLACEHOLDER_RE.sub(replace, template)


def should_prefix_internal_route(path: str, target_lang: str) -> bool:
    if not path.startswith("/"):
        return False
    if path.startswith("//") or path.startswith("/#"):
        return False
    if path == f"/{target_lang}" or path.startswith(f"/{target_lang}/"):
        return False
    if any(path.startswith(prefix) for prefix in DEFAULT_ASSET_PREFIXES):
        return False
    return True


def prefix_internal_routes(text: str, target_lang: str) -> str:
    def rewrite(path: str) -> str:
        if not should_prefix_internal_route(path, target_lang):
            return path
        return f"/{target_lang}{path}"

    text = re.sub(
        r'(href=")(/[^"#][^"]*)(")',
        lambda m: f"{m.group(1)}{rewrite(m.group(2))}{m.group(3)}",
        text,
    )
    text = re.sub(
        r"(\]\()(/[^)\s][^)]*)(\))",
        lambda m: f"{m.group(1)}{rewrite(m.group(2))}{m.group(3)}",
        text,
    )
    return text


async def render_localized_page(
    page_spec: PageSpec,
    global_memory: Dict[str, str],
    override_memory: Dict[str, str],
    translator: Callable[..., Any] | None,
    source_lang: str,
    target_lang: str,
    max_concurrent: int,
    max_retries: int,
) -> Tuple[str, int, int]:
    source_text = page_spec.source_path.read_text(encoding="utf-8")
    template, units = build_placeholder_template(source_text)
    cache, missing = build_initial_translation_cache(
        texts=units,
        page_memory=page_spec.page_memory,
        global_memory=global_memory,
        override_memory=override_memory,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    translated_missing = await translate_missing_texts(
        missing=missing,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
    )
    for source, translated in translated_missing.items():
        cache[source] = translated
    localized = fill_placeholders(template, units, cache)
    localized = prefix_internal_routes(localized, target_lang)
    return localized, len(units), len(missing)


async def sync_locale_pages(
    page_specs: Sequence[PageSpec],
    global_memory: Dict[str, str],
    override_memory: Dict[str, str],
    translator: Callable[..., Any] | None,
    source_lang: str,
    target_lang: str,
    max_concurrent: int,
    max_retries: int,
    dry_run: bool,
) -> None:
    total_units = 0
    total_missing = 0

    for page_spec in page_specs:
        localized, unit_count, missing_count = await render_localized_page(
            page_spec=page_spec,
            global_memory=global_memory,
            override_memory=override_memory,
            translator=translator,
            source_lang=source_lang,
            target_lang=target_lang,
            max_concurrent=max_concurrent,
            max_retries=max_retries,
        )
        total_units += unit_count
        total_missing += missing_count
        if not dry_run:
            page_spec.target_path.parent.mkdir(parents=True, exist_ok=True)
            page_spec.target_path.write_text(localized, encoding="utf-8")
        print(
            f"[SYNC] {page_spec.route} -> {page_spec.target_path.relative_to(page_spec.target_path.parents[2])} "
            f"(segments={unit_count}, missing_memory={missing_count})"
        )

    print(
        f"Synced pages: {len(page_specs)}, segments={total_units}, memory_misses={total_missing}"
    )
    if dry_run:
        print("Dry run enabled; files were not written.")


async def translate_group_labels(
    groups: Sequence[NavGroup],
    global_memory: Dict[str, str],
    override_memory: Dict[str, str],
    translator: Callable[..., Any] | None,
    source_lang: str,
    target_lang: str,
    max_concurrent: int,
    max_retries: int,
) -> List[str]:
    cache, missing = build_initial_translation_cache(
        texts=[group.group for group in groups],
        page_memory={},
        global_memory=global_memory,
        override_memory=override_memory,
        source_lang=source_lang,
        target_lang=target_lang,
    )
    translated_missing = await translate_missing_texts(
        missing=missing,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
    )
    for source, translated in translated_missing.items():
        cache[source] = translated
    return [cache.get(group.group, group.group) for group in groups]


async def write_docs_json_languages(
    docs_json_path: Path,
    groups: Sequence[NavGroup],
    global_memory: Dict[str, str],
    override_memory: Dict[str, str],
    translator: Callable[..., Any] | None,
    source_lang: str,
    target_lang: str,
    default_language_code: str,
    dry_run: bool,
    max_concurrent: int,
    max_retries: int,
) -> None:
    config = load_docs_config(docs_json_path)
    navigation = config.setdefault("navigation", {})

    translated_groups = await translate_group_labels(
        groups=groups,
        global_memory=global_memory,
        override_memory=override_memory,
        translator=translator,
        source_lang=source_lang,
        target_lang=target_lang,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
    )

    default_groups = [
        {"group": group.group, "pages": list(group.pages)} for group in groups
    ]
    target_groups = [
        {
            "group": translated_groups[idx],
            "pages": [f"{target_lang}/{page}" for page in group.pages],
        }
        for idx, group in enumerate(groups)
    ]

    languages = navigation.get("languages", [])
    merged_languages: List[Dict[str, Any]] = []
    preserved = [
        lang
        for lang in languages
        if lang.get("language") not in {default_language_code, target_lang}
    ]
    merged_languages.append(
        {
            "language": default_language_code,
            "default": True,
            "groups": default_groups,
        }
    )
    merged_languages.append(
        {
            "language": target_lang,
            "groups": target_groups,
        }
    )
    merged_languages.extend(preserved)

    navigation.pop("tabs", None)
    navigation.pop("groups", None)
    navigation["languages"] = merged_languages

    if dry_run:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        print("Dry run enabled; docs.json was not written.")
        return

    dump_docs_config(docs_json_path, config)
    print(
        f"Updated docs.json with navigation.languages for {default_language_code} + {target_lang}"
    )


def audit_locale(
    docs_root: Path,
    target_lang: str,
    source_routes: Sequence[str],
) -> None:
    expected = {f"{target_lang}/{route}" for route in source_routes}
    actual_files = (
        sorted((docs_root / target_lang).rglob("*.mdx"))
        if (docs_root / target_lang).exists()
        else []
    )
    actual = {
        str(path.relative_to(docs_root)).removesuffix(".mdx") for path in actual_files
    }

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    print(f"Source pages: {len(source_routes)}")
    print(f"Target locale dir: {docs_root / target_lang}")
    print(f"Existing localized pages: {len(actual)}")
    print(f"Missing localized pages: {len(missing)}")
    for route in missing[:50]:
        print(f"  MISSING {route}")
    print(f"Extra localized pages: {len(extra)}")
    for route in extra[:50]:
        print(f"  EXTRA   {route}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and sync Mintlify i18n pages with translation memory and optional LLM fallback"
    )
    parser.add_argument(
        "--docs-root", default="mintlify_docs", help="Mintlify docs root"
    )
    parser.add_argument(
        "--target-lang", default="en", help="Target Mintlify locale code"
    )
    parser.add_argument(
        "--source-lang",
        default="zh_CN",
        help="Source language label for translation prompt",
    )
    parser.add_argument(
        "--default-language-code",
        default="zh",
        help="Default Mintlify language code for root pages",
    )
    parser.add_argument(
        "--provider-json",
        default="provider.json",
        help="Path to provider.json for LLM fallback",
    )
    parser.add_argument(
        "--override-memory",
        default="scripts/mintlify_i18n_overrides.json",
        help="Path to manual translation override JSON",
    )
    parser.add_argument("--provider", default=None, help="Provider id in provider.json")
    parser.add_argument("--model", default=None, help="Model name under provider id")
    parser.add_argument(
        "--sync-pages",
        action="store_true",
        help="Generate/update locale pages under mintlify_docs/<lang>",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Enable LLM fallback for segments not covered by translation memory",
    )
    parser.add_argument(
        "--write-docs-json",
        action="store_true",
        help="Rewrite docs.json to use navigation.languages",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Maximum concurrent translation calls",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retries for one translation call",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print audit/planned actions without writing files",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    docs_root = Path(args.docs_root).resolve()
    docs_json_path = docs_root / "docs.json"
    provider_json = Path(args.provider_json).resolve()
    override_memory_path = Path(args.override_memory).resolve()

    config = load_docs_config(docs_json_path)
    groups = extract_navigation_groups(config)
    source_routes = collect_source_routes(groups, locale_prefixes=[args.target_lang])

    print(f"Docs root: {docs_root}")
    print(f"Target locale: {args.target_lang}")
    print(f"Source routes in navigation: {len(source_routes)}")

    audit_locale(
        docs_root=docs_root, target_lang=args.target_lang, source_routes=source_routes
    )

    if not args.sync_pages and not args.write_docs_json:
        print("Audit complete (no write actions requested).")
        return

    global_memory: Dict[str, str] = {}
    override_memory = load_override_memory(override_memory_path, args.target_lang)
    print(f"Translation memory entries: {len(global_memory)}")
    print(f"Override memory entries: {len(override_memory)}")

    translator: Callable[..., Any] | None = None
    if args.translate:
        llm_interface = pick_llm_interface(
            provider_json=provider_json,
            provider_id=args.provider,
            model_name=args.model,
        )
        translator = build_translator(llm_interface)
    else:
        print("LLM fallback disabled; only translation memory will be used.")

    page_specs = make_page_specs(
        docs_root=docs_root,
        target_lang=args.target_lang,
        source_routes=source_routes,
    )

    if args.sync_pages:
        await sync_locale_pages(
            page_specs=page_specs,
            global_memory=global_memory,
            override_memory=override_memory,
            translator=translator,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
            dry_run=args.dry_run,
        )

    if args.write_docs_json:
        await write_docs_json_languages(
            docs_json_path=docs_json_path,
            groups=groups,
            global_memory=global_memory,
            override_memory=override_memory,
            translator=translator,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            default_language_code=args.default_language_code,
            dry_run=args.dry_run,
            max_concurrent=args.max_concurrent,
            max_retries=args.max_retries,
        )


if __name__ == "__main__":
    asyncio.run(main())
