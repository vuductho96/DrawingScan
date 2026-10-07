# Drawing OCR — UI verification, 06/10/2026

final result: passed

Scope: implement the supplied layout in the existing application, with the drawing and OCR result areas as the primary surfaces and a collapsible left panel. This is a layout recreation with the explicit differences below, not a claim that generated sample values or desktop window chrome have been reproduced pixel for pixel.

## Evidence and environment

- Selected reference: `D:/Temp/codex-clipboard-9ad9cc0f-982e-4193-bf82-3f2e02b2d4bc.png`, 2146×1206 pixels.
- Running preview: `http://127.0.0.1:8091/`.
- Browser plugin/IAB is unavailable in this session. The user authorized Playwright fallback; validation used Python Playwright with headless Microsoft Edge and a temporary browser profile.
- Browser screenshots: `C:/Users/tho/.codex/visualizations/2026/10/03/01a10292-04b3-7fb3-abf3-8327fe29ad95/drawing-ocr-desktop.png`, `drawing-ocr-collapsed.png`, `drawing-ocr-laptop.png`, `drawing-ocr-mobile.png`.
- Viewports: 2146×1206 (same native pixel dimensions as reference, deviceScaleFactor 1), 1440×900, and 390×844. Mobile uses vertically arranged drawing/results and a collapsible overlay panel.
- Reference and implementation screenshots were opened together with `view_image` for visual comparison. The laptop and mobile screenshots were inspected as well.

## Required fidelity surfaces

| Surface | Comparison and outcome |
| --- | --- |
| Typography | Segoe UI/Arial for application chrome, Consolas for dimension inputs. Header, sidebar labels and numeric text have explicit sizes, with larger type at desktop widths. Labels and table values remain readable at laptop size. |
| Layout / spacing | Dark application header, left information panel, central drawing, right result tabs and table, bottom timing/export bar follow the source hierarchy. Collapsing the left panel increases drawing width. Rounded panels, restrained separators and consistent controls preserve the source rhythm. |
| Colors | Navy background/surfaces, blue view/result controls, cyan/mint scan and completion state, white engineering drawing. Flat tokens replace the mock's decorative gradient treatment. |
| Image quality | The drawing and region previews come from the actual PDF pipeline, with aspect ratio preserved. Source region crops have cyan/purple/amber borders. UI icons are vendored library SVGs. No screen-sized raster is used as the application UI. |
| Copy / content | Drawing OCR, filename, settings/history, nominal/Tol −/Tol +, image/result tabs, scan/F5 and Excel actions are retained. Visible output values are real recognition results. Manual review status replaces the mock's automatic OK labels. |

## Interaction checks

- Real sample PDF loads with 43 detected boxes and 43 result rows; selected region synchronizes with drawing, table and crop view.
- Left panel collapses/expands and the drawing refits. Narrow resize collapses the panel rather than leaving it over the main content.
- Zoom in/out, fit to frame and image-only/region-overlay controls respond to state.
- Result/crop tabs switch content. Crop selection highlights the corresponding region.
- Manual review and important flags toggle. Changing a numeric field resets manual review status and updates export data.
- An actual `.xlsx` download was opened as a ZIP and checked for the edited value `41.43`. The temporary test workbook was removed afterward.
- Settings expose exactly the two retained YOLO models and one OCR model. History lists completed scans.
- At 1440 pixels, document width is 1440 and result scroll area width equals its scrollWidth (411); no hidden horizontal result columns. At 390 pixels, document width equals viewport width.
- Final browser check: no page errors or console warnings/errors. Correct page title, populated drawing/results and no error overlay.
- Existing backend unit tests: 7/7 passed. JavaScript syntax check passed.

## Issues fixed during QA

- Busy state originally disabled the split checkbox before reading whether splitting was enabled. The request now derives support from selected model metadata.
- Desktop typography was enlarged to match the reference hierarchy more closely.
- The expanded sidebar covered the primary surfaces after a desktop-to-mobile resize. Narrow resize now collapses it automatically.
- Mobile sticky status bar overlapped result rows. It now stays in normal document flow.
- Export is disabled if the displayed result belongs to a different document than the selected filename.

## Intentional differences and limits

- Web application controls replace native desktop minimize/maximize/close buttons; the mock's custom logo/version are represented by a library scan icon and the application name.
- Metadata comes from user input and filename, not invented user/mold data. Default date is 06/10/2026 for this task; native date formatting follows the browser locale.
- “OK” requires manual verification and is not inferred from OCR confidence. Checkbox selection determines exported rows.
- A compact selected-region source preview is included below the table to help verify the OCR; the dedicated image tab shows all crops.
- The mock's “2 lỗ so sánh” control has no existing supported workflow and was omitted.
- Fullscreen, multi-page traversal and arbitrary uploaded files are wired, but browser testing used the existing single-page sample and did not cover every PDF/image format. OCR accuracy itself is unchanged by this UI work.
- History/review/metadata storage is local to the browser. No shared database or multi-user history was added.

No blocking layout or interaction mismatches remain for the requested drawing/results/left-panel workflow. Backend weights and dictionaries were not modified by the UI redesign.
