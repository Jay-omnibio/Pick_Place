from __future__ import annotations

import argparse
import re
from pathlib import Path


MERMAID_BLOCK_RE = re.compile(r"```mermaid\s*\r?\n(.*?)\r?\n```", re.DOTALL | re.IGNORECASE)


def extract_mermaid_blocks(md_text: str) -> list[str]:
    return [m.strip() + "\n" for m in MERMAID_BLOCK_RE.findall(md_text)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Mermaid blocks from Markdown into .mmd files.")
    parser.add_argument("--root", default=".", help="Repo root to scan for .md files.")
    parser.add_argument("--out", default="docs/mermaid", help="Output directory for .mmd files.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(p for p in root.rglob("*.md") if ".git" not in p.parts)
    generated = []

    for md_path in md_files:
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        blocks = extract_mermaid_blocks(text)
        if not blocks:
            continue

        rel = md_path.relative_to(root)
        rel_parent = rel.parent
        stem = rel.stem

        for idx, block in enumerate(blocks, start=1):
            mmd_name = f"{stem}_{idx:02d}.mmd"
            target = out_dir / rel_parent / mmd_name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(block, encoding="utf-8")
            generated.append(target.relative_to(root if out_dir.is_relative_to(root) else out_dir))

    index_path = out_dir / "README.md"
    with index_path.open("w", encoding="utf-8") as f:
        f.write("# Mermaid Extracts\n\n")
        f.write("Generated from Markdown Mermaid blocks.\n\n")
        f.write(f"- Source root: `{root}`\n")
        f.write(f"- Total `.mmd` files: `{len(generated)}`\n\n")
        if generated:
            f.write("## Files\n\n")
            for p in generated:
                f.write(f"- `{p.as_posix()}`\n")

    print(f"Extracted {len(generated)} Mermaid block(s) into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

