"""A5 (v0.3.1): the documents code sits behind one boundary (decision 21).

- `pypdf`, `html.parser` and `sqlite3` are imported only inside `clarity_chat.documents`
  (`sqlite3`: v0.3.2's decision 18).
- Modules outside import only `clarity_chat.documents`, never a submodule, and only from
  a fixed list.
- The package imports from the rest of Clarity only `profiles`; its embedder (v0.4.1
  decision 21) also reuses llama-server's process, the model lock and the machine facts.
- Only the embedder talks to llama-server's /v1/embeddings.
"""

from __future__ import annotations

import ast
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
PACKAGE = "clarity_chat.documents"
PARSERS = ("pypdf", "html.parser", "sqlite3")
MAY_IMPORT_IT = {"session", "commands", "export", "web"}  # clarity_chat.<module>
EMBEDDER = "clarity_chat.documents.embedder"
EMBEDDER_MAY_IMPORT = ("clarity_chat.backends", "clarity_chat.models", "clarity_chat.logs")


def imported(tree: ast.AST, src: Path) -> list[str]:
    """Every module a file imports, at any depth (a function's lazy import included):
    `from a import b` is `a`, and also `a.b` when b is one of Clarity's modules."""
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.append(node.module)
            folder = src.joinpath(*node.module.split("."))
            # By name, not is_file(): on macOS and Windows `Embedder` would find embedder.py.
            there = {p.name for p in folder.iterdir()} if folder.is_dir() else set()
            for alias in node.names:
                if f"{alias.name}.py" in there or alias.name in there:
                    names.append(f"{node.module}.{alias.name}")
    return list(dict.fromkeys(names))


def within(name: str, module: str) -> bool:
    return name == module or name.startswith(module + ".")


def violations(src: Path) -> list[str]:
    found = []
    for path in sorted((src / "clarity_chat").rglob("*.py")):
        parts = path.relative_to(src).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        inside = within(module, PACKAGE)
        for name in imported(ast.parse(path.read_text(encoding="utf-8")), src):
            if inside:
                if within(name, "clarity_chat") and not within(name, PACKAGE):
                    allowed = ("clarity_chat.profiles",)
                    if module == EMBEDDER:
                        allowed += EMBEDDER_MAY_IMPORT
                    if not any(within(name, m) for m in allowed):
                        found.append(f"{module} imports {name}")
            elif any(within(name, parser) for parser in PARSERS):
                found.append(f"{module} imports {name}")
            elif name != PACKAGE and within(name, PACKAGE):
                found.append(f"{module} imports the submodule {name}")
            elif name == PACKAGE and module.removeprefix("clarity_chat.") not in MAY_IMPORT_IT:
                found.append(f"{module} imports {PACKAGE}, and isn't on the list")
    return found


def test_the_documents_code_sits_behind_its_boundary():
    assert violations(SRC) == []


def test_a_planted_import_breaks_each_rule(tmp_path):
    shutil.copytree(SRC / "clarity_chat", tmp_path / "clarity_chat")
    package = tmp_path / "clarity_chat"
    planted = {
        "planted_parser.py": "import pypdf\n",
        "planted_html.py": "def f():\n    from html.parser import HTMLParser\n",
        "planted_submodule.py": "from clarity_chat.documents.insert import content\n",
        "planted_outsider.py": "from clarity_chat.documents import content\n",
        "planted_package.py": "from clarity_chat import documents\n",
        "documents/planted_session.py": "from clarity_chat.session import Session\n",
        "documents/planted_server.py": "from clarity_chat.backends.llama_cpp import LlamaServer\n",
    }
    for name, code in planted.items():
        (package / name).write_text(code, encoding="utf-8")
    (package / "session.py").write_text(
        (package / "session.py").read_text(encoding="utf-8")
        + "\nfrom clarity_chat.documents import reading\n",
        encoding="utf-8",
    )
    assert violations(tmp_path) == [
        "clarity_chat.documents.planted_server imports clarity_chat.backends.llama_cpp",
        "clarity_chat.documents.planted_session imports clarity_chat.session",
        "clarity_chat.planted_html imports html.parser",
        "clarity_chat.planted_outsider imports clarity_chat.documents, and isn't on the list",
        "clarity_chat.planted_package imports clarity_chat.documents, and isn't on the list",
        "clarity_chat.planted_parser imports pypdf",
        "clarity_chat.planted_submodule imports the submodule clarity_chat.documents.insert",
        "clarity_chat.session imports the submodule clarity_chat.documents.reading",
    ]


def test_only_the_embedder_talks_to_the_embeddings_endpoint():
    talking = [
        path.relative_to(SRC).as_posix()
        for path in sorted((SRC / "clarity_chat").rglob("*.py"))
        if "/v1/embeddings" in path.read_text(encoding="utf-8")
    ]
    assert talking in ([], ["clarity_chat/documents/embedder.py"]), talking
