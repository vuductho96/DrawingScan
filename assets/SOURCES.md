# Local UI dependencies

- Icon SVGs: Lucide Static 0.468.0, from `https://cdn.jsdelivr.net/npm/lucide-static@0.468.0/icons/`. ISC license in `vendor/lucide-LICENSE.txt`. Only icons referenced by the UI are retained.
- Excel export: SheetJS `xlsx` 0.18.5 browser bundle, from `https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js`. License in `vendor/xlsx-LICENSE.txt`.
- Fonts: installed Segoe UI / Arial and Consolas; no remote font requests.

These files are served locally by `server.py`. No network access is required for icons or Excel export at runtime.
