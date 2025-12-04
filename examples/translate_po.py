#!/usr/bin/env python3
"""
Translation script for .po files using SimpleLLMFunc.

This script translates Chinese content in .po files to English using LLMs.
Supports structured concurrency for efficient batch translation.
"""

import asyncio
import os
import polib  
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
import logging
from dataclasses import dataclass
import sys
from tqdm.asyncio import tqdm

from SimpleLLMFunc import llm_function, OpenAICompatible


@dataclass
class TranslationTask:
    """Represents a translation task."""
    entry: polib.POEntry
    text: str
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BatchResult:
    """Result of a batch translation operation."""
    successful: List[Tuple[polib.POEntry, str]]
    failed: List[TranslationTask]


# Disable all logging to avoid HTTP request logs
logging.disable(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Disable specific loggers that might show HTTP requests
import urllib3
urllib3.disable_warnings()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configure LLM interface
# Load from provider.json or use default configuration
llm_interface = OpenAICompatible.load_from_json_file("provider.json")["openrouter"]["google/gemini-2.5-flash-lite"]


@llm_function(llm_interface=llm_interface)  # type: ignore
async def translate_text(text: str, source_lang: str = "Chinese", target_lang: str = "English") -> str:
    """
    Translate text from source language to target language.

    Args:
        text: Text to translate
        source_lang: Source language (default: Chinese)
        target_lang: Target language (default: English)

    Returns:
        Translation result with translated text and confidence score
    """
    # This function will be implemented by the LLM decorator
    # The actual implementation is handled by SimpleLLMFunc
    return "Hello, world!"


async def translate_single_text(text: str) -> Optional[str]:
    """
    Translate a single text with error handling.
    
    Args:
        text: Text to translate
        
    Returns:
        Translated text or None if translation failed
    """
    try:
        result: str = await translate_text(text)
        return result
    except Exception as e:
        logger.error(f"Failed to translate text '{text[:50]}...': {e}")
        return None


async def translate_batch_concurrent(texts: List[str], max_concurrent: int = 5) -> List[Optional[str]]:
    """
    Translate a batch of texts using structured concurrency.
    
    Args:
        texts: List of texts to translate
        max_concurrent: Maximum number of concurrent translations
        
    Returns:
        List of translated texts (None for failed translations)
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def translate_with_semaphore(text: str) -> Optional[str]:
        async with semaphore:
            return await translate_single_text(text)
    
    # Create tasks for all translations
    tasks = [translate_with_semaphore(text) for text in texts]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Convert exceptions to None
    processed_results: List[Optional[str]] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Translation task failed: {result}")
            processed_results.append(None)
        elif isinstance(result, str):
            processed_results.append(result)
        else:
            processed_results.append(None)
    
    return processed_results


def load_po_file(file_path: str) -> polib.POFile:
    """
    Load a .po file.

    Args:
        file_path: Path to the .po file

    Returns:
        Loaded PO file object
    """
    return polib.pofile(file_path)


def save_po_file(po_file: polib.POFile, file_path: str) -> None:
    """
    Save a .po file.

    Args:
        po_file: PO file object to save
        file_path: Path to save the .po file
    """
    po_file.save(file_path)


async def translate_po_entries_structured(
    po_file: polib.POFile, 
    batch_size: int = 20, 
    max_concurrent: int = 10,
    max_retries: int = 3,
    show_progress: bool = True
) -> None:
    """
    Translate all entries in a .po file using structured concurrency.

    Args:
        po_file: PO file object to translate
        batch_size: Number of entries to process in each batch
        max_concurrent: Maximum number of concurrent translations
        max_retries: Maximum number of retries for failed translations
        show_progress: Whether to show progress bar
    """
    # Collect entries that need translation
    entries_to_translate = []
    for entry in po_file:
        # Skip entries that are already translated
        if entry.msgstr.strip() != "":
            continue

        # Skip header and metadata
        if entry.msgid == "" or entry.msgid.startswith("#"):
            continue

        # Skip entries that are just formatting placeholders
        if entry.msgid.strip() in ["", "\\n"]:
            continue

        entries_to_translate.append(entry)

    if not entries_to_translate:
        print("✓ No entries need translation")
        return

    print(f"📝 Found {len(entries_to_translate)} entries to translate")

    # Create translation tasks
    tasks = [
        TranslationTask(entry=entry, text=entry.msgid, max_retries=max_retries)
        for entry in entries_to_translate
    ]

    # Process with progress bar
    if show_progress:
        progress_bar = tqdm(total=len(tasks), desc="🔄 Translating", unit="entries", 
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # Process in batches with structured concurrency
    total_batches = (len(tasks) - 1) // batch_size + 1
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(tasks))
        batch_tasks = tasks[start_idx:end_idx]
        
        # Process batch with retry logic
        await process_batch_with_retries(batch_tasks, max_concurrent, show_progress=False)
        
        # Update progress bar
        if show_progress:
            completed = sum(1 for task in tasks[:end_idx] if task.entry.msgstr.strip() != "")
            progress_bar.n = completed
            progress_bar.refresh()
    
    if show_progress:
        progress_bar.close()
        print("✅ Translation completed!")


async def process_batch_with_retries(
    batch_tasks: List[TranslationTask], 
    max_concurrent: int,
    show_progress: bool = False
) -> None:
    """
    Process a batch of translation tasks with retry logic.
    
    Args:
        batch_tasks: List of translation tasks
        max_concurrent: Maximum number of concurrent translations
        show_progress: Whether to show detailed progress
    """
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_task_with_retry(task: TranslationTask) -> None:
        """Process a single task with retry logic."""
        async with semaphore:
            while task.retry_count <= task.max_retries:
                try:
                    result: str = await translate_text(task.text)
                    task.entry.msgstr = result
                    return
                except Exception as e:
                    task.retry_count += 1
                    if task.retry_count > task.max_retries:
                        if show_progress:
                            logger.error(f"Failed to translate '{task.text[:50]}...' after {task.max_retries} retries: {e}")
                        # Set a fallback translation
                        task.entry.msgstr = f"[TRANSLATION_FAILED: {task.text}]"
                    else:
                        if show_progress:
                            logger.warning(f"Retry {task.retry_count}/{task.max_retries} for '{task.text[:50]}...': {e}")
                        # Exponential backoff
                        await asyncio.sleep(2 ** task.retry_count)
    
    # Execute all tasks concurrently
    await asyncio.gather(*[process_task_with_retry(task) for task in batch_tasks], return_exceptions=True)


async def translate_po_file(
    input_path: str, 
    output_path: str, 
    batch_size: int = 20,
    max_concurrent: int = 10,
    max_retries: int = 3,
    show_progress: bool = True
) -> None:
    """
    Translate a .po file from Chinese to English using structured concurrency.

    Args:
        input_path: Path to the input .po file
        output_path: Path to save the translated .po file
        batch_size: Number of entries to process in each batch
        max_concurrent: Maximum number of concurrent translations
        max_retries: Maximum number of retries for failed translations
        show_progress: Whether to show progress bar
    """
    print(f"📂 Loading: {os.path.basename(input_path)}")
    po_file = load_po_file(input_path)

    await translate_po_entries_structured(po_file, batch_size, max_concurrent, max_retries, show_progress)

    print(f"💾 Saving: {os.path.basename(output_path)}")
    save_po_file(po_file, output_path)


async def main():
    """Main function to translate .po files."""
    import argparse

    parser = argparse.ArgumentParser(description="Translate .po files from Chinese to English using LLM with structured concurrency")
    parser.add_argument("input_file", help="Input .po file path")
    parser.add_argument("-o", "--output", help="Output .po file path (default: input file)")
    parser.add_argument("-b", "--batch-size", type=int, default=20, help="Batch size for translations (default: 20)")
    parser.add_argument("-c", "--max-concurrent", type=int, default=10, help="Maximum concurrent translations (default: 10)")
    parser.add_argument("-r", "--max-retries", type=int, default=3, help="Maximum retries for failed translations (default: 3)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.disable(logging.NOTSET)
        logging.getLogger().setLevel(logging.DEBUG)

    input_file = args.input_file
    output_file = args.output or input_file
    batch_size = args.batch_size
    max_concurrent = args.max_concurrent
    max_retries = args.max_retries
    show_progress = not args.no_progress

    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found")
        return

    try:
        await translate_po_file(input_file, output_file, batch_size, max_concurrent, max_retries, show_progress)
    except Exception as e:
        print(f"❌ Error during translation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())