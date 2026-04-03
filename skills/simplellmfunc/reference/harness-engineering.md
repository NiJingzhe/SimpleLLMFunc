# Harness Engineering

## Basic philosophy

Everything starts with context planning.

An agent is not a person. It is a method for constructing the right context for each reasoning step.
Your job is to make sure that, at any moment, the model sees a context that is:

- clean
- short
- necessary
- complete for the current task

And just as importantly, the system should be able to maintain and rebuild that context in a durable closed loop.

## Your role

You are the environment designer.

When the agent fails, the first question is not "why was the model bad?".
The first question is:

- what was missing from the environment, context, tools, checks, or feedback loop?

Then encode that answer into the system itself.

The lesson should live in:

- tools
- constraints
- docs
- checks
- persisted state
- workflow structure

not only in the operator's head.

## Core beliefs

- model quality is often secondary to context quality
- there is a sweet spot for context density; more context is not automatically better
- common agent failures are predictable and should map to harness decisions
- agents do not have cross-session memory unless you rebuild it from external state
- unattended agent-generated code accumulates debt in a different way than human-written code, so cleanup must be part of the system

## Engineering principles

### Encode rules into tools, not only docs

Mechanical constraints beat descriptive instructions.

Prefer:

- linters
- validators
- explicit tool contracts
- error messages that tell the agent how to fix the issue

### Persist progress outside the model

If state matters across sessions, write it to files or structured storage.

Prefer:

- structured JSON or similarly durable machine-readable state
- files that let a new session reconstruct where the work stopped

### Separate thinking from execution

Research, planning, execution, and validation should be distinct stages when the task is complex.

Do not collapse everything into one unreviewed generation step if a cleaner staged flow yields better context.

### Choose fork vs sub-agent by context quality

Sometimes the full current history is the best context, so forking the current conversation is correct.
Sometimes the history is noisy, so a smaller tool-limited sub-agent is better.

The decision rule is simple:

- choose the option that gives this reasoning step the most accurate, compact, task-relevant context

### Design cleanup into the system

Do not only optimize for generation throughput.
Add cleanup passes that scan for redundancy, architecture drift, and low-quality code.

### Treat AGENTS.md as a feedback artifact

`AGENTS.md` should evolve from real mistakes and corrections.
It is not a static instruction pamphlet; it is the visible output of the system's feedback loop.

## What this means in SimpleLLMFunc

- use `@llm_function` when it gives the cleanest context for one typed step
- use `@llm_chat` only when multi-turn history genuinely improves the current reasoning context
- use tools to fetch missing facts or verify work instead of bloating prompts
- use typed outputs and Pydantic models when they help downstream validation and reduce ambiguity
- keep `provider.json` focused on model routing and access concerns, not application logic
- keep orchestration in Python so you can control retries, validation, branching, and persistence explicitly

## One-line heuristic

Always construct the cleanest, most task-relevant, shortest complete context the system can maintain in a closed loop.
