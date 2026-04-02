from __future__ import annotations

from SimpleLLMFunc.builtin import PyRepl


repl = PyRepl()

notes = repl.pack(
    "notes",
    backend={"items": []},
    guidance="notes pack stores short runtime notes for the current REPL.",
)


@notes.primitive("append", description="Append one note to runtime storage.")
def notes_append(ctx, text: str) -> int:
    """
    Use: Append one short note to runtime storage.
    Input: `text: str`.
    Output: `int` count after append.
    Parameters:
    - text: One short note to store.
    Best Practices:
    - Keep notes concise and human-readable.
    - Use for lightweight runtime state only.
    """
    backend = ctx.backend
    if not isinstance(backend, dict):
        raise RuntimeError("notes backend must be a dict")
    items = backend.setdefault("items", [])
    items.append(text)
    return len(items)


@notes.primitive("all", description="Read all stored notes.")
def notes_all(ctx) -> list[str]:
    """
    Use: Read all notes from runtime storage.
    Output: `list[str]`.
    Best Practices:
    - Use when the note list is expected to stay small.
    - Summarize long note collections before returning them to chat.
    """
    backend = ctx.backend
    if not isinstance(backend, dict):
        raise RuntimeError("notes backend must be a dict")
    return list(backend.get("items", []))


repl.install_pack(notes)
