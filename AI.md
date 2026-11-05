# AI Contributions — opencv-engine

*About AI slop.* Yes, a machine wrote nearly all of this, and it was happy to. Call it instant ramen: fast, cheap, genuinely edible, and nobody should pretend it simmered all afternoon. The honest answer isn't a claim about quality; it's the record below: what the maintainer asked for, what the model wrote, what it got wrong, and what the test suite and the conformance run caught rather than luck. Defects in AI-written code go here in the same detail as the wins. Bon appétit, but mind the sodium content.

Tracks what AI helpers contributed to opencv-engine versus the human maintainer.

## Human maintainer

Direction: an OpenCV engine for PlumCraft, as a separate process. The maintainer's decision on 2026-11-01 was to give it a private GitHub repository.

## Contributions

### 2026-11-04 — The SDK pin moved to PlumCraft's Windows support (Claude Opus 5.5)

The maintainer installed this engine from PlumCraft's editor on Windows. It registered and appeared, then died on start: `module 'socket' has no attribute 'AF_UNIX'`. The SDK was pinned to PlumCraft `3995cb7`, from before PlumCraft ran on Windows, and Windows Python has no Unix sockets. The maintainer: "bump the engine pins".

The AI moved the pin to PlumCraft `81dd28f` (`main`, merging PR #19, "PlumCraft on Windows"). There the SDK also reads a `tcp:127.0.0.1:<port>` locator, which the host offers on Windows. `uv.lock` was regenerated. `scripts/check` passes: ruff, format, and 48 tests on macOS.

The same SDK had already been tried on Windows: on `win11-dev`, with the installed OpenCV engine's SDK swapped for the checkout. A fresh install there with this pin hasn't been tried.

### 2026-11-03 — Published, with `register` (Claude Opus 5.5)

The maintainer: "Let's get the python engines (duckdb, opencv, ?) into one or more public repos". They chose one repo per engine, named `plumcraft-<name>-engine`, and the SDK pinned to PlumCraft's pushed `headless-write-files` commit until a release.

The AI:

- pinned the SDK to that commit over https, in place of a path on the maintainer's machine;
- added the MIT and Apache 2.0 licences, PlumCraft's;
- put `examples/` in the wheel, so `register` finds the demo;
- wrote the Install section: `uv tool install git+…`, then `<engine> register`.

The tests and ruff passed against the SDK fetched from GitHub.

### 2026-11-01 — First commit (Claude Opus 5.5)

The code predates this entry: it was built in earlier AI sessions, which recorded it in PlumCraft's own notes, and it had never been committed. This commit took it as it stood, and changed one thing: `pyproject.toml` and `uv.lock` named PlumCraft's Python SDK by the absolute path `/Users/seant/workspace/plumcraft/python`, which works on one machine; it is now `../plumcraft/python`. `.idea/` was left out.

Checked: `uv run pytest`, 48 passed; ruff clean; a scan for keys and passwords found none.
