# plan.md — Knowledge Base: Auto-Discovery & MCP Query Refactor

> **Purpose**: This document is the authoritative implementation guide for GitHub Copilot agent(s).  
> Read it fully before writing a single line of code.  
> Subagents should execute tasks in the order given — later tasks depend on earlier ones.

---

## Repository Audit (Current State)

### What Exists

| Layer | Location | Status |
|---|---|---|
| Vector store (Qdrant) | `src/vectorstores/` | ✅ Working |
| Ingestion pipeline | `src/ingestion/pipeline.py` | ✅ Working |
| KB service & repo | `src/application/services/` | ✅ Working |
| MCP server | `src/mcp/server.py` | ✅ Working — tools registered |
| Setup script | `setup_kbs.py` | ❌ Hardcoded to 2 specific folders + absolute paths |
| `data/uploads/` | folder convention | ❌ Not auto-discovered |
| `.vscode/mcp.json` | `cwd` is hardcoded absolute path | ❌ Machine-specific |

### Core Problems to Solve

1. **`setup_kbs.py` is not generic.** It hardcodes `space_autosar` and `space_coding_guidelines` folder names and absolute file paths on the developer's machine. Any new folder under `data/uploads/` requires manual editing.
2. **The MCP server exists and is functional**, but the `.vscode/mcp.json` uses an absolute `cwd` path that breaks on any machine other than the original developer's.
3. **No `ingest_document` MCP tool is exposed**, so a Copilot agent cannot trigger ingestion via MCP — only setup via CLI is possible. (Out of scope for this plan but noted as Task 3.)

---

## Goal Definition

### Goal 1 — Generic KB Setup Script (Single Command)

Running one command must:

1. Scan all **immediate subdirectories** of `data/uploads/`
2. For each subfolder found (e.g., `a`, `b`, `project_x`):
   - Create a KB named after the folder if it does not already exist
   - Ingest **all supported files** inside that folder (PDF, TXT, DOCX, MD) into that KB
   - Skip already-ingested files (idempotent)
3. Print a summary of created KBs and ingested files

### Goal 2 — MCP Query via Copilot Agent

The MCP server (`src/mcp/server.py`) already exposes:
- `list_knowledge_bases`
- `retrieve_from_kb`
- `search_knowledge_bases`
- `list_documents`
- `create_knowledge_base`

The `.vscode/mcp.json` must be updated so any developer can use it without editing absolute paths.

---

## Implementation Plan

### Task 1 — Rewrite `setup_kbs.py` as a Generic Auto-Discovery Script

**File**: `setup_kbs.py` (replace in-place)

**Algorithm**:

```
1. Read settings.paths.uploads_dir  →  default: ./data/uploads
2. List all immediate subdirectories of uploads_dir
3. For each folder `f` in subdirectories:
   a. kb_name = f.name   (folder name becomes KB name)
   b. description = f"Knowledge base for documents in {f.name}"
   c. Create or retrieve KB with kb_name
   d. List all files in f with extensions: .pdf, .txt, .docx, .md, .html
   e. For each file:
      - Check if already ingested: call container.ingestion_service.list_documents(kb.id)
        and compare filenames
      - If not ingested → call container.ingestion_service.ingest_file(file_path, kb.id)
      - Log result (success, skip, or error per file)
4. Print summary table: KB name | # docs | # chunks
```

**Exact changes required in `setup_kbs.py`**:

- Remove all hardcoded folder names (`space_autosar`, `space_coding_guidelines`)
- Remove all hardcoded absolute paths (`/home/sandeep/...`)
- Use `settings.paths.uploads_dir` (already available via `from src.config.settings import settings`)
- Use `Path(settings.paths.uploads_dir).iterdir()` to discover subfolders
- Filter for `p.is_dir()` only (skip loose files at the root of uploads)
- Supported extensions constant: `SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".html"}`
- Wrap each file ingestion in individual `try/except` — one failed file must not abort the loop
- Use the existing `container.kb_repository.get_kb_by_name(name)` for idempotent KB creation (already present in the original, keep this pattern)

**Idempotency check for already-ingested files**:

Call `await container.ingestion_service.list_documents(kb.id)` and collect `{doc.name}`. Before ingesting, check `file_path.name in already_ingested_names`. If yes, log `⏭ Skipping {file_path.name} (already ingested)` and continue.

**Pseudocode skeleton** (agent must produce real Python):

```python
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".html"}
uploads_dir = Path(settings.paths.uploads_dir)

for folder in sorted(uploads_dir.iterdir()):
    if not folder.is_dir():
        continue
    kb_name = folder.name
    # create or get kb ...
    existing_docs = await container.ingestion_service.list_documents(kb.id)
    already_ingested = {d.name for d in existing_docs}
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.name in already_ingested:
            logger.info(f"⏭  Skipping {file_path.name} (already ingested)")
            continue
        try:
            doc = await container.ingestion_service.ingest_file(file_path, kb.id)
            logger.info(f"✓  Ingested {file_path.name} (chunks={doc.chunk_count})")
        except Exception as e:
            logger.error(f"✗  Failed {file_path.name}: {e}")
```

**Add a `Makefile` target** (append to existing `Makefile`):

```makefile
setup-kbs:
	uv run python setup_kbs.py
```

The user's single command becomes: `make setup-kbs`

---

### Task 2 — Fix `.vscode/mcp.json` for Portability

**File**: `.vscode/mcp.json` (replace in-place)

**Problem**: `"cwd": "/home/sandeep/workspace/kb"` is an absolute path that breaks on every other machine.

**Fix**: Use `${workspaceFolder}` — VS Code's built-in variable that resolves to the repo root on any machine.

**New content**:

```json
{
  "servers": {
    "knowledge-base": {
      "command": "uv",
      "args": ["run", "python", "-m", "src.mcp.server"],
      "cwd": "${workspaceFolder}",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    }
  }
}
```

No other changes needed — the MCP server itself works correctly.

---

### Task 3 — Add `ingest_folder` MCP Tool (Optional Enhancement)

> **Status**: Implement only if Tasks 1 and 2 are verified working. This is an enhancement, not a blocker.

This exposes KB setup from within Copilot agent itself, without needing a CLI command.

**File**: `src/mcp/server.py`

Add a new tool schema to `TOOL_SCHEMAS`:

```python
types.Tool(
    name="ingest_folder",
    description=(
        "Ingest all supported documents from a subfolder inside data/uploads/ "
        "into its matching knowledge base. Creates the KB if it does not exist. "
        "Provide the folder name (not the full path). Supported file types: "
        "PDF, TXT, DOCX, MD, HTML."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "folder_name": {
                "type": "string",
                "description": "Name of the subfolder under data/uploads/ to ingest.",
            }
        },
        "required": ["folder_name"],
    },
),
```

Add to `_dispatch` in `KBMCPServer._dispatch()`:

```python
if name == "ingest_folder":
    return await self._kb_tools.ingest_folder(
        folder_name=arguments["folder_name"]
    )
```

Add method to `KBTools` class in `src/tools/tools.py`:

```python
async def ingest_folder(self, folder_name: str) -> dict[str, Any]:
    from pathlib import Path
    from src.config.settings import settings

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".html"}
    folder = Path(settings.paths.uploads_dir) / folder_name
    if not folder.exists() or not folder.is_dir():
        return {"success": False, "error": f"Folder '{folder_name}' not found in uploads."}
    # create or get KB
    kb = await self._kb.get_by_name(folder_name)
    if not kb:
        kb = await self._kb.create(
            name=folder_name,
            description=f"Knowledge base for documents in {folder_name}",
        )
    existing_docs = await self._ingestion.list_documents(kb.id)
    already_ingested = {d.name for d in existing_docs}
    results = []
    for file_path in sorted(folder.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if file_path.name in already_ingested:
            results.append({"file": file_path.name, "status": "skipped"})
            continue
        try:
            doc = await self._ingestion.ingest_file(file_path, kb.id)
            results.append({"file": file_path.name, "status": f"ingested (chunks={doc.chunk_count})"})
        except Exception as e:
            results.append({"file": file_path.name, "status": f"error: {e}"})
    return {"success": True, "kb_name": folder_name, "kb_id": kb.id, "files": results}
```

---

### Task 4 — Update `README.md` Quick-Start Section

**File**: `README.md`

Locate the existing "Quick Start" or "Setup" section and add or replace with:

```markdown
## Quick Start

### 1. Add your documents

Place documents inside `data/uploads/` using one subfolder per knowledge base:

```
data/uploads/
├── project_alpha/
│   ├── spec.pdf
│   └── notes.md
└── project_beta/
    └── manual.pdf
```

Each subfolder name becomes the knowledge base name.

### 2. Build all knowledge bases (single command)

```bash
make setup-kbs
# or directly:
uv run python setup_kbs.py
```

The script is idempotent — re-running it skips already-ingested files.

### 3. Query via GitHub Copilot (MCP)

Open the repo in VS Code with the MCP extension enabled.  
Copilot automatically connects to the `knowledge-base` MCP server via `.vscode/mcp.json`.

Example agent prompts:
- "List all available knowledge bases"
- "Search for authentication flows across all knowledge bases"
- "Retrieve information about signal processing from the project_alpha KB"
```

---

## Verification Checklist

After completing all tasks, verify each item before closing the PR:

- [ ] `setup_kbs.py` contains no hardcoded folder names or absolute paths
- [ ] `setup_kbs.py` uses `settings.paths.uploads_dir` (not a literal string path)
- [ ] Running `make setup-kbs` with 2 folders under `data/uploads/` creates exactly 2 KBs
- [ ] Running `make setup-kbs` a second time produces only `⏭ Skipping` log lines (no duplicate ingestion)
- [ ] `.vscode/mcp.json` uses `${workspaceFolder}` for both `cwd` and `PYTHONPATH`
- [ ] MCP server starts cleanly in VS Code with no path errors in Output panel
- [ ] Copilot agent can call `list_knowledge_bases` and see the newly created KBs
- [ ] Copilot agent can call `retrieve_from_kb` and receive grounded results with citations
- [ ] (If Task 3 implemented) `ingest_folder` appears in `list_tools` MCP response
- [ ] `make lint` passes with zero errors after all changes

---

## Files to Modify (Summary)

| File | Action | Task |
|---|---|---|
| `setup_kbs.py` | Rewrite — generic auto-discovery loop replacing hardcoded names | Task 1 |
| `Makefile` | Add `setup-kbs` target | Task 1 |
| `.vscode/mcp.json` | Replace absolute `cwd`/`PYTHONPATH` with `${workspaceFolder}` | Task 2 |
| `src/mcp/server.py` | Add `ingest_folder` tool schema + dispatch branch | Task 3 (optional) |
| `src/tools/tools.py` | Add `ingest_folder` method to `KBTools` | Task 3 (optional) |
| `README.md` | Add/update Quick Start section | Task 4 |

---

## Constraints & Rules for the Agent

1. **Do not change** `src/config/settings.py`, `src/ingestion/pipeline.py`, or any vectorstore code — the ingestion and storage layers are correct and must not be touched.
2. **Do not change** existing MCP tool schemas (`list_knowledge_bases`, `retrieve_from_kb`, etc.) — only add the new `ingest_folder` tool if doing Task 3.
3. All new code must be **async** — the codebase uses `asyncio` throughout; no blocking I/O.
4. Keep the **per-file `try/except`** pattern — one bad document must never abort the entire setup loop.
5. KB naming: use the **folder name as-is** — do not transform to lowercase, snake_case, or any other format. The folder name is the user-facing contract.
6. The `ingestion_service.list_documents(kb_id)` call returns objects with a `.name` attribute — use this for the idempotency check, not `.id` or `.path`.
7. Run `make lint` after each file change and fix all reported issues before moving to the next task.
