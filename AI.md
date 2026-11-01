# AI Contributions — opencv-engine

*About AI slop.* Yes, a machine wrote nearly all of this, and it was happy to. Call it instant ramen: fast, cheap, genuinely edible, and nobody should pretend it simmered all afternoon. The honest answer isn't a claim about quality; it's the record below: what the maintainer asked for, what the model wrote, what it got wrong, and what the test suite and the conformance run caught rather than luck. Defects in AI-written code go here in the same detail as the wins. Bon appétit, but mind the sodium content.

Tracks what AI helpers contributed to opencv-engine versus the human maintainer.

## Human maintainer

Direction: an OpenCV engine for PlumCraft, as a separate process. The maintainer's decision on 2026-11-01 was to give it a private GitHub repository.

## Contributions

### 2026-11-01 — First commit (Claude Opus 5.5)

The code predates this entry: it was built in earlier AI sessions, which recorded it in PlumCraft's own notes, and it had never been committed. This commit took it as it stood, and changed one thing: `pyproject.toml` and `uv.lock` named PlumCraft's Python SDK by the absolute path `/Users/seant/workspace/plumcraft/python`, which works on one machine; it is now `../plumcraft/python`. `.idea/` was left out.

Checked: `uv run pytest`, 48 passed; ruff clean; a scan for keys and passwords found none.
