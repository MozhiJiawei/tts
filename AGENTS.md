# Agents

## Encoding Rules

- This repository uses the setting in `.editorconfig`: Python files must be saved as `utf-8-bom` with CRLF line endings.
- When editing Python files that contain Chinese text, always read and write them with an explicit encoding. In scripts, use `encoding="utf-8-sig"` so the BOM is preserved.
- Do not use shell redirection or ad-hoc scripts that write source files with the platform default encoding.
- Before finishing, verify changed Python files can still be decoded with `utf-8-sig` and that Chinese comments or log strings are readable.
- If a file already contains Chinese text, preserve the existing wording unless the task explicitly asks to rewrite it.
