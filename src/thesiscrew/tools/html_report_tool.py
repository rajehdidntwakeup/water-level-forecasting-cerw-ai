"""Deterministic HTML final report builder for the frontend specialist agent.

Reads Phase 5 Markdown reports and numeric artifacts, converts Markdown to
HTML, embeds interactive Plotly charts, and writes a standalone, styled,
browser-ready document to output/final_report.html.
"""

import json
import os
import re
import html as _html
from typing import Any, Optional

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

OUTPUT_DIR = os.environ.get("PEGELHUB_OUTPUT_DIR", "output")
DATA_DIR = os.environ.get("PEGELHUB_DATA_DIR", os.path.join(OUTPUT_DIR, "data"))


def _safe_print(message: str) -> None:
    """Print a message, replacing characters the console cannot encode."""
    import sys
    try:
        print(message)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding if sys.stdout else "utf-8"
        print(message.encode(encoding or "utf-8", errors="replace").decode(encoding or "utf-8"))


class BuildHtmlReportInput(BaseModel):
    style: str = Field(
        default="academic",
        description="CSS theme: 'academic' (thesis style) or 'dashboard' (dark UI).",
    )
    include_charts: bool = Field(
        default=True,
        description="Embed interactive Plotly charts from numeric artifacts.",
    )
    focus: str = Field(
        default="full",
        description="Report scope: 'full' (all thesis sections) or 'forecasting' (only forecast results, charts, and key setup).",
    )


class BuildHtmlReportTool(BaseTool):
    name: str = "build_html_report"
    description: str = (
        "Generate a standalone HTML final report at output/final_report.html. "
        "Reads the Phase 5 Markdown reports and numeric artifacts, converts Markdown "
        "to styled HTML, embeds interactive Plotly charts, adds a navigation sidebar, "
        "and writes the final document. Call this tool once to produce the report."
    )
    args_schema: type[BaseModel] = BuildHtmlReportInput

    def _run(
        self,
        style: str = "academic",
        include_charts: bool = True,
        focus: str = "full",
    ) -> str:
        output_path = os.path.join(OUTPUT_DIR, "final_report.html")
        forecasting_focus = focus.lower() == "forecasting"
        sources = self._collect_forecast_sources() if forecasting_focus else self._collect_sources()

        sections: list[tuple[str, str]] = []
        if forecasting_focus:
            setup_section = self._build_forecast_setup_section()
            if setup_section:
                sections.append(("Forecast Setup", setup_section))

        for title, path in sources:
            if not os.path.exists(path):
                sections.append(
                    (title, f'<p class="missing">Source not found: {path}</p>')
                )
                continue
            with open(path, "r", encoding="utf-8") as f:
                md = f.read()
            md = self._unwrap_fenced_block(md)
            if not md.strip():
                sections.append(
                    (title, f'<p class="missing">Source file is empty: {path}</p>')
                )
                continue
            if self._is_agent_completion_message(md):
                # Skip agent chatter; upstream tool output is not a real report section.
                continue
            sections.append((title, self._markdown_to_html(md)))

        verification = self._load_verification_inputs()
        charts = self._build_charts(verification) if include_charts else []
        metrics = self._build_metrics_summary(verification)
        if forecasting_focus:
            metrics.update(self._build_forecast_setup_metrics())

        html = self._render_document(sections, charts, metrics, style=style, focus=focus)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        validation = self.validate_html(html, path=output_path)
        if not validation.get("valid", False):
            _safe_print(f"[build_html_report] validation warning: {validation.get('reason', 'unknown')}")

        status = "valid" if validation["valid"] else f"warning: {validation['reason']}"
        return f"HTML final report written to {output_path} ({len(html)} chars, {validation['sections']} sections, {validation['charts']} charts, focus={focus}, {status})"


    @staticmethod
    def validate_html(html: str, path: Optional[str] = None) -> dict[str, Any]:
        """Check that generated HTML looks like a real final report.

        Returns a dict with keys 'valid' (bool), 'size' (int), 'sections' (int),
        'charts' (int), and 'reason' (str | None).
        """
        from html.parser import HTMLParser

        size = len(html)
        if size < 20_000:
            return {"valid": False, "size": size, "sections": 0, "charts": 0, "reason": "too small to be a styled report"}

        required_tags = ["html", "head", "body", "style", "nav", "section", "script"]
        lower = html.lower()
        missing = [tag for tag in required_tags if f"<{tag}" not in lower]
        if missing:
            return {"valid": False, "size": size, "sections": 0, "charts": 0, "reason": f"missing required tags: {', '.join(missing)}"}

        sections = html.count('class="report-section"')
        charts = html.count('class="chart"')

        # Structural parse check.
        class StackChecker(HTMLParser):
            def __init__(self):
                super().__init__()
                self.stack: list[str] = []
                self.errors: list[str] = []
                self.void = {"meta", "link", "br", "hr", "img", "input", "!doctype"}
            def handle_starttag(self, tag, attrs):
                if tag not in self.void:
                    self.stack.append(tag)
            def handle_endtag(self, tag):
                if tag in self.void:
                    return
                if self.stack and self.stack[-1] == tag:
                    self.stack.pop()
                else:
                    self.errors.append(f"unexpected </{tag}>")

        checker = StackChecker()
        try:
            checker.feed(html)
        except Exception as e:
            return {"valid": False, "size": size, "sections": sections, "charts": charts, "reason": f"HTML parse error: {e}"}

        if checker.stack or checker.errors:
            return {"valid": False, "size": size, "sections": sections, "charts": charts, "reason": f"unclosed tags: {checker.stack[-5:]}, errors: {checker.errors[:3]}"}

        return {"valid": True, "size": size, "sections": sections, "charts": charts, "reason": None}

    def _collect_sources(self) -> list[tuple[str, str]]:
        """Return report sources in display order.

        Split data/model reports are optional; if they are missing or only contain
        agent completion chatter, the main Phase 5 documentation already covers
        those sections so we skip them rather than printing an ugly placeholder.
        """
        candidates = [
            ("Project Overview", os.path.join(OUTPUT_DIR, "phase5_report.md")),
            ("Documentation", os.path.join(OUTPUT_DIR, "phase5_final_documentation.md")),
            ("Data Discovery & Ingestion", os.path.join(OUTPUT_DIR, "data", "data_output.md")),
            ("Models & Verification", os.path.join(OUTPUT_DIR, "models", "models_output.md")),
            ("Verification Details", os.path.join(OUTPUT_DIR, "phase5_verification.md")),
        ]
        result: list[tuple[str, str]] = []
        for title, path in candidates:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                md = f.read()
            md = self._unwrap_fenced_block(md)
            if self._is_agent_completion_message(md) or not md.strip():
                continue
            result.append((title, path))
        return result

    def _collect_forecast_sources(self) -> list[tuple[str, str]]:
        """Return only forecast-relevant sources for the forecasting-focused report.

        The focused report keeps the narrative short: verification details only.
        Forecast setup and charts are generated separately and inserted in the
        desired menu/main order.
        """
        candidates = [
            ("Verification Details", os.path.join(OUTPUT_DIR, "phase5_verification.md")),
        ]
        result: list[tuple[str, str]] = []
        for title, path in candidates:
            if not os.path.exists(path):
                continue
            with open(path, "r", encoding="utf-8") as f:
                md = f.read()
            md = self._unwrap_fenced_block(md)
            if self._is_agent_completion_message(md) or not md.strip():
                continue
            result.append((title, path))
        return result

    @staticmethod
    def _format_target_label(target: str) -> str:
        """Convert a snake_case target variable into a human-readable label."""
        if not target:
            return target
        lower = target.lower()
        # Recognised units so the label reads nicely.
        unit_map = {
            "_cm": " (cm)",
            "_m3s": " (m³/s)",
            "_m3_s": " (m³/s)",
            "_cms": " (m³/s)",
            "_c": " (°C)",
            "_celsius": " (°C)",
            "_mm": " (mm)",
            "_kmh": " (km/h)",
            "_m": " (m)",
        }
        unit = ""
        for suffix, label in unit_map.items():
            if lower.endswith(suffix):
                unit = label
                target = target[: -len(suffix)]
                break
        words = target.replace("_", " ").split()
        return " ".join(w.capitalize() for w in words) + unit

    def _build_forecast_setup_section(self) -> str:
        """Build a compact HTML section describing the forecast setup from inputs."""
        input_path = os.path.join(os.path.dirname(OUTPUT_DIR), "input", "research_area.json")
        if not os.path.exists(input_path):
            input_path = "input/research_area.json"
        setup = {
            "station": "Korneuburg",
            "river": "Donau (Danube)",
            "country": "Austria",
            "latitude": 48.345,
            "longitude": 16.337,
            "target": "water_level_cm",
            "horizons": [1, 6, 12, 24, 48, 72, 168],
            "primary_model": "Gradient Boosting Machine",
        }
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            setup["station"] = data.get("primary_station", setup["station"])
            setup["river"] = data.get("river", setup["river"])
            setup["country"] = data.get("country", setup["country"])
            setup["latitude"] = data.get("latitude", setup["latitude"])
            setup["longitude"] = data.get("longitude", setup["longitude"])
            setup["target"] = data.get("target_variable", setup["target"])
            setup["horizons"] = data.get("forecast_horizons_hours", setup["horizons"])
        except Exception:
            pass

        target_label = self._format_target_label(str(setup["target"]))
        horizons_html = ", ".join(f"<span class=\"horizon-tag\">+{h}h</span>" for h in setup["horizons"])
        return f"""<div class=\"forecast-setup\">
<div class=\"setup-card\">
<h3>Station</h3>
<p><strong>{_html.escape(str(setup['station']))}</strong></p>
<p>{_html.escape(str(setup['river']))}, {_html.escape(str(setup['country']))}</p>
<p class=\"coords\">{setup['latitude']:.3f}°N, {setup['longitude']:.3f}°E</p>
</div>
<div class=\"setup-card\">
<h3>Target</h3>
<p><strong>{_html.escape(target_label)}</strong></p>
<p>Hourly {target_label.lower()} forecast</p>
</div>
<div class=\"setup-card\">
<h3>Horizons</h3>
<p class=\"horizons\">{horizons_html}</p>
</div>
<div class=\"setup-card\">
<h3>Primary Model</h3>
<p><strong>{_html.escape(str(setup['primary_model']))}</strong></p>
<p>One model per horizon</p>
</div>
</div>"""

    def _build_forecast_setup_metrics(self) -> dict[str, Any]:
        """Return static setup metadata for the forecasting hero cards."""
        input_path = os.path.join(os.path.dirname(OUTPUT_DIR), "input", "research_area.json")
        if not os.path.exists(input_path):
            input_path = "input/research_area.json"
        target = "water_level_cm"
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                target = json.load(f).get("target_variable", target)
        except Exception:
            pass
        return {
            "station": "Korneuburg",
            "river": "Donau (Danube)",
            "target": self._format_target_label(target),
            "model": "Gradient Boosting Machine",
        }

    @staticmethod
    def _is_agent_completion_message(text: str) -> bool:
        """Heuristic to detect CrewAI agent completion chatter vs. real report content."""
        stripped = text.strip()
        if not stripped:
            return True
        lower = stripped.lower()
        strong_completion_phrases = [
            "i have successfully",
            "i have completed",
            "report compilation complete",
            "file confirmation",
            "files created",
            "standalone html final report has been",
            "the html report has been",
            "the final report has been",
        ]
        if any(phrase in lower for phrase in strong_completion_phrases):
            return True
        # Short or emoji-heavy agent summaries without real tables/code are not reports.
        if len(stripped) < 400:
            return True
        return False

    @staticmethod
    def _unwrap_fenced_block(text: str) -> str:
        """If a source is wrapped in a single code fence, strip it."""
        stripped = text.strip()
        if stripped.startswith("```"):
            first_newline = stripped.find("\n")
            if first_newline != -1:
                opener = stripped[:first_newline].strip()
                # Accept any language tag (markdown, json, etc.) or bare fence.
                if opener == "```" or re.fullmatch(r"```\w+", opener):
                    last_fence = stripped.rfind("\n```")
                    if last_fence != -1:
                        inner = stripped[first_newline + 1:last_fence].strip()
                        return inner
        return text

    def _load_json_artifact(self, path: str) -> dict[str, Any]:
        """Load a JSON artifact, handling optional code fences and errors."""
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            raw = self._unwrap_fenced_block(raw)
            return json.loads(raw)
        except Exception:
            return {}

    @staticmethod
    def _inline_md_to_html(text: str) -> str:
        """Escape raw HTML, then render lightweight inline Markdown."""
        line = _html.escape(text)
        # Images before links so link regex doesn't capture image markdown.
        line = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r'<img alt="\1" src="\2">', line)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', line)
        line = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", line)
        line = re.sub(r"\*(.+?)\*", r"<em>\1</em>", line)
        line = re.sub(r"`(.+?)`", r"<code>\1</code>", line)
        return line

    def _markdown_to_html(self, md: str) -> str:
        """Enhanced Markdown-to-HTML converter with nested lists and blockquotes."""
        lines = md.splitlines()
        html_lines: list[str] = []
        para_lines: list[str] = []
        list_stack: list[tuple[str, int]] = []  # (tag, indent)
        in_code = False
        code_buffer: list[str] = []
        code_lang = ""
        in_table = False
        table_rows: list[list[str]] = []
        in_blockquote = False
        bq_lines: list[str] = []

        def flush_code():
            nonlocal in_code, code_lang
            if in_code and code_buffer:
                lang_attr = f' class="language-{code_lang}"' if code_lang else ""
                html_lines.append(
                    f"<pre><code{lang_attr}>{_html.escape(chr(10).join(code_buffer))}</code></pre>"
                )
                code_buffer.clear()
                code_lang = ""
                in_code = False

        def flush_table():
            nonlocal in_table
            if in_table and table_rows:
                # Drop the Markdown separator row if present.
                data_rows = [r for i, r in enumerate(table_rows) if i != 1 or not all(re.match(r"^[:-]+$", c.strip()) for c in r)]
                headers = data_rows[0]
                body_rows = data_rows[1:]
                html_lines.append('<div class="table-wrap">')
                html_lines.append("<table><thead><tr>")
                for h in headers:
                    html_lines.append(f"<th>{self._inline_md_to_html(h.strip())}</th>")
                html_lines.append("</tr></thead><tbody>")
                for row in body_rows:
                    html_lines.append("<tr>")
                    for cell in row:
                        html_lines.append(f"<td>{self._inline_md_to_html(cell.strip())}</td>")
                    html_lines.append("</tr>")
                html_lines.append("</tbody></table>")
                html_lines.append("</div>")
                table_rows.clear()
                in_table = False

        def flush_para():
            if para_lines:
                joined = " ".join(para_lines)
                html_lines.append(f"<p>{joined}</p>")
                para_lines.clear()

        def flush_blockquote():
            nonlocal in_blockquote
            if in_blockquote and bq_lines:
                inner = self._markdown_to_html("\n".join(bq_lines))
                html_lines.append(f"<blockquote>{inner}</blockquote>")
                bq_lines.clear()
                in_blockquote = False

        def flush_lists_up_to(indent: int):
            nonlocal list_stack
            while list_stack and list_stack[-1][1] >= indent:
                tag, _ = list_stack.pop()
                html_lines.append(f"</{tag}>")

        def ensure_list(tag: str, indent: int):
            nonlocal list_stack
            # Close deeper lists first.
            flush_lists_up_to(indent)
            if list_stack and list_stack[-1][0] != tag:
                old_tag, _ = list_stack.pop()
                html_lines.append(f"</{old_tag}>")
            if not list_stack or list_stack[-1][0] != tag:
                html_lines.append(f"<{tag}>")
                list_stack.append((tag, indent))

        for raw_line in lines:
            stripped = raw_line.strip()

            if stripped.startswith("```"):
                flush_para()
                flush_table()
                flush_blockquote()
                if in_code:
                    flush_code()
                else:
                    in_code = True
                    code_lang = stripped[3:].strip() or ""
                continue
            if in_code:
                code_buffer.append(raw_line)
                continue

            if re.fullmatch(r"-{3,}", stripped):
                flush_para()
                flush_table()
                flush_blockquote()
                flush_lists_up_to(0)
                html_lines.append("<hr>")
                continue

            # Blockquotes
            if stripped.startswith("> "):
                flush_para()
                flush_table()
                flush_lists_up_to(0)
                in_blockquote = True
                bq_lines.append(stripped[2:])
                continue
            else:
                flush_blockquote()

            # Tables
            if stripped.startswith("|") and stripped.endswith("|"):
                flush_para()
                flush_lists_up_to(0)
                cells = [c.strip() for c in stripped[1:-1].split("|")]
                table_rows.append(cells)
                in_table = True
                continue
            else:
                flush_table()

            # Headings
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                flush_para()
                flush_lists_up_to(0)
                level = len(heading_match.group(1))
                content = heading_match.group(2).strip()
                html_lines.append(f"<h{level}>{self._inline_md_to_html(content)}</h{level}>")
                continue

            # Lists with indentation tracking
            ul_match = re.match(r"^(\s*)[-*]\s+(\[[ xX]\]\s+)?(.+)$", raw_line)
            ol_match = re.match(r"^(\s*)\d+\.\s+(.+)$", raw_line)
            if ul_match or ol_match:
                flush_para()
                flush_table()
                flush_blockquote()
                if ul_match:
                    indent = len(ul_match.group(1))
                    task = ul_match.group(2)
                    item_text = ul_match.group(3)
                    tag = "ul"
                    prefix = '<input type="checkbox" disabled' + (' checked' if "x" in (task or "").lower() else "") + '> ' if task else ""
                else:
                    indent = len(ol_match.group(1))
                    item_text = ol_match.group(2)
                    tag = "ol"
                    prefix = ""
                ensure_list(tag, indent)
                html_lines.append(f"<li>{prefix}{self._inline_md_to_html(item_text)}</li>")
                continue

            # Empty lines close paragraphs and lists.
            if not stripped:
                flush_para()
                flush_lists_up_to(0)
                continue

            # Continuation of a paragraph.
            para_lines.append(self._inline_md_to_html(raw_line))

        flush_para()
        flush_lists_up_to(0)
        flush_code()
        flush_table()
        flush_blockquote()

        return "\n".join(html_lines)

    @staticmethod
    def _normalize_horizon_key(key: str) -> str:
        """Normalize 'h1', 'h=1', '1' etc. to 'h=1'."""
        s = str(key).strip().lower()
        s = s.replace("h=", "h")
        s = s.replace("h", "")
        s = s.replace("=", "")
        digits = re.sub(r"\D", "", s)
        return f"h={digits}" if digits else key

    def _normalize_verification(self, verification: dict[str, Any]) -> dict[str, Any]:
        """Normalize mixed verification artifact formats to a consistent schema.

        Handles flat dict-of-dicts, list-of-dicts under 'horizons', and the
        nested {"horizons": {"1": {...}, ...}} format produced by the baseline
        verification tool."""
        normalized: dict[str, Any] = {}

        def _extract_metrics(raw: Any, nested_key: str | None = None) -> dict[str, Any]:
            """Pull horizon-keyed metric dicts out of a variety of wrappers."""
            out: dict[str, Any] = {}
            if not isinstance(raw, dict):
                return out

            # List-of-dicts under 'horizons'.
            horizons_val = raw.get("horizons")
            if isinstance(horizons_val, list):
                for entry in horizons_val:
                    if not isinstance(entry, dict):
                        continue
                    h_key = entry.get("horizon_hours", entry.get("horizon", entry.get("h")))
                    if h_key is None:
                        continue
                    h = self._normalize_horizon_key(str(h_key))
                    out[h] = dict(entry)
            # Nested dict-of-dicts under 'horizons'.
            elif isinstance(horizons_val, dict):
                for h_key, metrics in horizons_val.items():
                    if not isinstance(metrics, dict):
                        continue
                    h = self._normalize_horizon_key(str(h_key))
                    out[h] = dict(metrics)

            # Flat dict-of-dicts at the top level (skip structural/metadata keys).
            skip_keys = {"horizons", "prediction_files", "source_file", "column", "metadata", "notes"}
            if nested_key and isinstance(raw.get(nested_key), dict):
                source = raw[nested_key]
            else:
                source = raw
            for k, v in source.items():
                if k in skip_keys or not isinstance(v, dict):
                    continue
                h = self._normalize_horizon_key(str(k))
                if h not in out:
                    out[h] = dict(v)
            return out

        # GBM overall metrics.
        gbm = _extract_metrics(verification.get("gbm_overall", {}), nested_key=None)
        for h, metrics in gbm.items():
            if not isinstance(metrics, dict):
                continue
            if "RMSE" in metrics and "RMSE_cm" not in metrics:
                metrics["RMSE_cm"] = metrics.pop("RMSE")
            if "rmse" in metrics and "RMSE_cm" not in metrics:
                metrics["RMSE_cm"] = metrics["rmse"]
            if "nse" in metrics and "NSE" not in metrics:
                metrics["NSE"] = metrics["nse"]
        normalized["gbm_overall"] = gbm

        # Persistence metrics.
        persistence = _extract_metrics(verification.get("persistence", {}), nested_key="metrics")
        for h, metrics in persistence.items():
            if not isinstance(metrics, dict):
                continue
            if "rmse" not in metrics and "RMSE" in metrics:
                metrics["rmse"] = metrics.pop("RMSE")
            if "nse" not in metrics and "NSE" in metrics:
                metrics["nse"] = metrics.pop("NSE")
            if "RMSE_cm" in metrics and "rmse" not in metrics:
                metrics["rmse"] = metrics["RMSE_cm"]
        normalized["persistence"] = {"metrics": persistence}

        if "skill_vs_persistence" in verification:
            normalized["skill_vs_persistence"] = verification["skill_vs_persistence"]
        return normalized

    def _load_verification_inputs(self) -> dict[str, Any]:
        """Load verification artifacts and merge the best available metrics per horizon.

        Strategy:
          * GBM: prefer the detailed verification_inputs values where they are valid
            (no error key), otherwise fall back to verification_baseline.
          * Persistence: verification_inputs covers all seven horizons, so use it
            as the primary source and fill any gaps from verification_baseline.
          * Preserve walk-forward, stratified, and metadata from verification_inputs.
        """
        inputs = self._load_json_artifact(os.path.join(OUTPUT_DIR, "models", "verification_inputs.json"))
        baseline = self._load_json_artifact(os.path.join(OUTPUT_DIR, "models", "verification_baseline.json"))

        norm_inputs = self._normalize_verification(inputs) if inputs else {}
        norm_baseline = self._normalize_verification(baseline) if baseline else {}

        merged: dict[str, Any] = {}

        # Keep non-metric blocks from the detailed verification file.
        for key in ("walk_forward", "stratified", "metadata"):
            if inputs and key in inputs:
                merged[key] = inputs[key]

        # GBM: inputs first when valid, baseline as fallback.
        gbm: dict[str, Any] = {}
        for source in (norm_inputs, norm_baseline):
            for h, metrics in source.get("gbm_overall", {}).items():
                if not isinstance(metrics, dict) or h in gbm:
                    continue
                if "error" in metrics:
                    continue
                gbm[h] = metrics
        merged["gbm_overall"] = gbm

        # Persistence: inputs has the full set, baseline fills gaps.
        persistence: dict[str, Any] = {}
        for source in (norm_inputs, norm_baseline):
            for h, metrics in source.get("persistence", {}).get("metrics", {}).items():
                if isinstance(metrics, dict) and h not in persistence:
                    persistence[h] = metrics
        merged["persistence"] = {"metrics": persistence}

        if inputs and "skill_vs_persistence" in inputs:
            merged["skill_vs_persistence"] = inputs["skill_vs_persistence"]
        elif baseline and "skill_vs_persistence" in baseline:
            merged["skill_vs_persistence"] = baseline["skill_vs_persistence"]

        return merged

    def _build_metrics_summary(self, verification: dict[str, Any]) -> dict[str, Any]:
        """Extract key headline metrics for the hero cards."""
        gbm = verification.get("gbm_overall", {})
        persistence = verification.get("persistence", {}).get("metrics", {})
        if not gbm:
            return {}

        best_h = None
        best_rmse = float("inf")
        for h, metrics in gbm.items():
            if isinstance(metrics, dict) and "RMSE_cm" in metrics:
                rmse = metrics["RMSE_cm"]
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_h = h

        horizons = sorted(
            [h for h, m in gbm.items() if isinstance(m, dict) and "RMSE_cm" in m],
            key=lambda h: int(h.split("=")[-1]) if "=" in h else h,
        )
        last_h = horizons[-1] if horizons else None
        return {
            "best_horizon": best_h,
            "best_rmse": best_rmse,
            "best_nse": gbm.get(best_h, {}).get("NSE") if best_h else None,
            "horizon_count": len(horizons),
            "long_horizon": last_h,
            "long_rmse": gbm.get(last_h, {}).get("RMSE_cm") if last_h else None,
            "pers_best": persistence.get(horizons[0], {}).get("rmse") if horizons else None,
        }

    def _build_charts(self, verification: dict[str, Any]) -> list[dict[str, Any]]:
        """Build Plotly chart specs from numeric artifacts."""
        charts: list[dict[str, Any]] = []
        gbm = verification.get("gbm_overall", {})
        persistence = verification.get("persistence", {}).get("metrics", {})

        horizons = sorted(
            {h for h in gbm.keys() if isinstance(gbm.get(h), dict)} |
            {h for h in persistence.keys()},
            key=lambda h: int(h.split("=")[-1]) if "=" in h else h,
        )
        horizon_labels = [f"h{h.split('=')[-1]}" if "=" in h else h for h in horizons]

        if horizons:
            gbm_rmse = []
            persistence_rmse = []
            gbm_nse = []
            persistence_nse = []
            for h in horizons:
                g = gbm.get(h, {})
                p = persistence.get(h, {})
                gbm_rmse.append(g.get("RMSE_cm", g.get("rmse")))
                persistence_rmse.append(p.get("rmse"))
                gbm_nse.append(g.get("NSE"))
                persistence_nse.append(p.get("nse"))

            charts.append({
                "id": "chart-rmse-comparison",
                "title": "RMSE by Horizon",
                "data": [
                    {
                        "x": horizon_labels,
                        "y": gbm_rmse,
                        "name": "GBM RMSE",
                        "type": "bar",
                        "marker": {"color": "#2563eb"},
                    },
                    {
                        "x": horizon_labels,
                        "y": persistence_rmse,
                        "name": "Persistence RMSE",
                        "type": "bar",
                        "marker": {"color": "#94a3b8"},
                    },
                ],
                "layout": {
                    "xaxis": {"title": "Forecast horizon"},
                    "yaxis": {"title": "RMSE (cm)"},
                    "barmode": "group",
                    "margin": {"t": 40, "b": 50},
                    "legend": {"orientation": "h", "y": -0.2},
                },
            })

            charts.append({
                "id": "chart-nse-comparison",
                "title": "Nash-Sutcliffe Efficiency by Horizon",
                "data": [
                    {
                        "x": horizon_labels,
                        "y": gbm_nse,
                        "name": "GBM NSE",
                        "type": "scatter",
                        "mode": "lines+markers",
                        "marker": {"color": "#2563eb", "size": 8},
                        "line": {"width": 3},
                    },
                    {
                        "x": horizon_labels,
                        "y": persistence_nse,
                        "name": "Persistence NSE",
                        "type": "scatter",
                        "mode": "lines+markers",
                        "marker": {"color": "#94a3b8", "size": 8},
                        "line": {"width": 3, "dash": "dash"},
                    },
                ],
                "layout": {
                    "xaxis": {"title": "Forecast horizon"},
                    "yaxis": {"title": "NSE", "range": [-1, 1]},
                    "margin": {"t": 40, "b": 50},
                    "legend": {"orientation": "h", "y": -0.2},
                    "shapes": [
                        {"type": "line", "x0": 0, "x1": 1, "xref": "paper", "y0": 0, "y1": 0, "line": {"color": "#ef4444", "dash": "dot", "width": 1}}
                    ],
                },
            })

        # Time series for each available horizon.
        for h in [1, 6, 12, 24, 48, 72, 168]:
            pred_path = os.path.join(OUTPUT_DIR, "models", f"gbm_predictions_h{h}.csv")
            if not os.path.exists(pred_path):
                continue
            try:
                import pandas as pd

                df = pd.read_csv(pred_path, parse_dates=["timestamp"], nrows=500)
                if {"timestamp", "actual", "predicted"}.issubset(df.columns):
                    labels = [str(t) for t in df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")]
                    charts.append({
                        "id": f"chart-time-series-h{h}",
                        "title": f"Measured vs Predicted (h={h})",
                        "data": [
                            {
                                "x": labels,
                                "y": df["actual"].tolist(),
                                "name": "Measured",
                                "type": "scatter",
                                "mode": "lines",
                                "line": {"width": 2, "color": "#0f172a"},
                            },
                            {
                                "x": labels,
                                "y": df["predicted"].tolist(),
                                "name": "Predicted",
                                "type": "scatter",
                                "mode": "lines",
                                "line": {"width": 2, "dash": "dash", "color": "#2563eb"},
                            },
                        ],
                        "layout": {
                            "xaxis": {"title": "Time"},
                            "yaxis": {"title": "Water level (cm)"},
                            "margin": {"t": 40, "b": 50},
                            "legend": {"orientation": "h", "y": -0.2},
                        },
                    })
            except Exception:
                continue

        # Forward-looking forecast chart: recent measured + future predicted points.
        recent_path = os.path.join(OUTPUT_DIR, "models", "forward_predictions_recent.csv")
        forward_path = os.path.join(OUTPUT_DIR, "models", "forward_predictions.csv")
        if os.path.exists(recent_path) and os.path.exists(forward_path):
            try:
                import pandas as pd

                recent_df = pd.read_csv(recent_path, parse_dates=["timestamp"])
                forward_df = pd.read_csv(forward_path, parse_dates=["timestamp"])
                forward_df = forward_df.dropna(subset=["predicted"])

                recent_labels = [str(t) for t in recent_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")]
                future_labels = [str(t) for t in forward_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")]

                charts.append({
                    "id": "chart-forward-forecast",
                    "title": "Forward Forecast (Measured + Predicted)",
                    "data": [
                        {
                            "x": recent_labels,
                            "y": recent_df["actual"].tolist(),
                            "name": "Measured (recent)",
                            "type": "scatter",
                            "mode": "lines",
                            "line": {"width": 2, "color": "#0f172a"},
                        },
                        {
                            "x": future_labels,
                            "y": forward_df["predicted"].tolist(),
                            "name": "Predicted",
                            "type": "scatter",
                            "mode": "lines+markers",
                            "marker": {"size": 8, "color": "#2563eb"},
                            "line": {"width": 2, "dash": "dash", "color": "#2563eb"},
                        },
                    ],
                    "layout": {
                        "xaxis": {"title": "Time"},
                        "yaxis": {"title": "Water level (cm)"},
                        "margin": {"t": 40, "b": 50},
                        "legend": {"orientation": "h", "y": -0.2},
                        "shapes": [
                            {
                                "type": "line",
                                "x0": recent_labels[-1] if recent_labels else None,
                                "x1": recent_labels[-1] if recent_labels else None,
                                "y0": 0,
                                "y1": 1,
                                "xref": "x",
                                "yref": "paper",
                                "line": {"color": "#94a3b8", "dash": "dot", "width": 1},
                            }
                        ],
                        "annotations": [
                            {
                                "x": recent_labels[-1] if recent_labels else None,
                                "y": 1,
                                "xref": "x",
                                "yref": "paper",
                                "text": "now",
                                "showarrow": False,
                                "font": {"color": "#94a3b8", "size": 11},
                                "xanchor": "left",
                            }
                        ] if recent_labels else [],
                    },
                })
            except Exception:
                pass

        # Scatter actual vs predicted for the first available short horizon.
        for h in [1, 6, 12, 24, 48, 72, 168]:
            pred_path = os.path.join(OUTPUT_DIR, "models", f"gbm_predictions_h{h}.csv")
            if not os.path.exists(pred_path):
                continue
            try:
                import pandas as pd

                df = pd.read_csv(pred_path)
                if {"actual", "predicted"}.issubset(df.columns):
                    charts.append({
                        "id": "chart-scatter",
                        "title": f"Predicted vs Actual (h={h})",
                        "data": [
                            {
                                "x": df["actual"].tolist(),
                                "y": df["predicted"].tolist(),
                                "name": f"h={h}",
                                "type": "scatter",
                                "mode": "markers",
                                "marker": {"size": 6, "opacity": 0.7, "color": "#2563eb"},
                            }
                        ],
                        "layout": {
                            "xaxis": {"title": "Actual (cm)"},
                            "yaxis": {"title": "Predicted (cm)"},
                            "margin": {"t": 40, "b": 50},
                            "shapes": [
                                {"type": "line", "x0": df["actual"].min(), "x1": df["actual"].max(), "y0": df["actual"].min(), "y1": df["actual"].max(), "line": {"color": "#94a3b8", "dash": "dash", "width": 2}}
                            ],
                        },
                    })
                    break
            except Exception:
                continue

        return charts

    def _render_document(
        self,
        sections: list[tuple[str, str]],
        charts: list[dict[str, Any]],
        metrics: dict[str, Any],
        style: str,
        focus: str = "full",
    ) -> str:
        generated = self._format_datetime()
        forecasting_focus = focus.lower() == "forecasting"

        metrics_html = self._render_metrics_cards(metrics, focus=focus)

        chart_cards = "\n".join(
            f'<div class="chart-card"><h3>{_html.escape(c["title"])}</h3>'
            f'<div id="{c["id"]}" class="chart"></div></div>'
            for c in charts
        )

        chart_scripts = "\n".join(
            f"Plotly.newPlot('{c['id']}', {json.dumps(c['data'])}, {json.dumps(c.get('layout', {}))}, {{responsive: true, displayModeBar: true, displaylogo: false}});"
            for c in charts
        )

        charts_section = ("Interactive Charts", f"""
    <p class="section-intro">
      Hover, zoom, and pan each chart. Use the mode bar at the top-right to export
      figures or toggle traces.
    </p>
    {chart_cards}
""")

        # Forecasting focus: menu and page order = Forecast Setup, Interactive Charts, Verification Details.
        if forecasting_focus:
            ordered_sections: list[tuple[str, str]] = []
            for section in sections:
                ordered_sections.append(section)
                if section[0] == "Forecast Setup":
                    ordered_sections.append(charts_section)
            # Fallback if Forecast Setup is missing: still include charts.
            if charts_section not in ordered_sections:
                ordered_sections.append(charts_section)
        else:
            ordered_sections = list(sections) + [charts_section]

        def _section_anchor(title: str, index: int) -> str:
            return "charts" if title == "Interactive Charts" else f"section-{index}"

        nav_items = "\n".join(
            f'<li><a href="#{_section_anchor(title, i)}" data-nav="{_section_anchor(title, i)}">{_html.escape(title)}</a></li>'
            for i, (title, _) in enumerate(ordered_sections)
        )
        section_html = "\n".join(
            f'<section id="{_section_anchor(title, i)}" class="report-section">'
            f'<h2 class="section-title">{_html.escape(title)}</h2>'
            f'{content}</section>'
            for i, (title, content) in enumerate(ordered_sections)
        )

        css = self._academic_css() if style == "academic" else self._dashboard_css()
        report_label = "Forecast Report" if forecasting_focus else "Thesis Report"
        hero_title = "Water Level Forecast" if forecasting_focus else "Water Level Forecasting for Korneuburg"
        hero_lead = (
            "Hourly water-level forecasts for Korneuburg on the Austrian Danube, "
            "from +1h to +168h ahead. The report highlights model performance, "
            "forecast horizons, and measured-vs-predicted comparisons."
        ) if forecasting_focus else (
            "A machine-learning approach to hourly water-level forecasting on the Austrian Danube, "
            "comparing gradient-boosted models against persistence and official forecasts across "
            "horizons from +1h to +168h."
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Water Level Forecasting — Final Report</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
{css}
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">
    <span class="brand-icon">🌊</span>
    <div>
      <div class="brand-title">Water Level Forecasting</div>
      <div class="brand-subtitle">Korneuburg / Danube, Austria · Final Report</div>
    </div>
  </div>
  <div class="topbar-meta">
    <span class="badge">{report_label}</span>
    <button id="theme-toggle" class="theme-toggle" aria-label="Toggle dark mode">🌙</button>
  </div>
</header>
<nav class="sidebar">
  <h2>Sections</h2>
  <ul>
    {nav_items}
  </ul>
  <div class="nav-meta">Generated {generated}</div>
</nav>
<div class="main">
  <section class="hero">
    <h1>{hero_title}</h1>
    <p class="hero-lead">
      {hero_lead}
    </p>
    {metrics_html}
  </section>
  {section_html}
  <footer class="report-footer">
    <p>Generated by PegelHub Thesis Crew on {generated}.</p>
    <p>Report sources: Phase 5 Markdown documentation and model verification artifacts.</p>
  </footer>
</div>
<script>
  (function() {{
    const toggle = document.getElementById('theme-toggle');
    const html = document.documentElement;
    const saved = localStorage.getItem('pegelhub-theme');
    if (saved) html.setAttribute('data-theme', saved);
    function updateIcon() {{
      const isDark = html.getAttribute('data-theme') === 'dark';
      toggle.textContent = isDark ? '☀️' : '🌙';
      toggle.setAttribute('aria-label', isDark ? 'Switch to light mode' : 'Switch to dark mode');
    }}
    updateIcon();
    toggle.addEventListener('click', function() {{
      const next = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
      html.setAttribute('data-theme', next);
      localStorage.setItem('pegelhub-theme', next);
      updateIcon();
    }});

    // Highlight active nav item on scroll.
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.sidebar a[data-nav]');
    const observer = new IntersectionObserver(function(entries) {{
      entries.forEach(function(entry) {{
        if (entry.isIntersecting) {{
          navLinks.forEach(function(link) {{
            link.classList.toggle('active', link.getAttribute('data-nav') === entry.target.id);
          }});
        }}
      }});
    }}, {{ rootMargin: '-20% 0px -60% 0px', threshold: 0 }});
    sections.forEach(function(section) {{ observer.observe(section); }});

    {chart_scripts}
  }})();
</script>
</body>
</html>"""

    @staticmethod
    def _format_datetime() -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _render_metrics_cards(self, metrics: dict[str, Any], focus: str = "full") -> str:
        if not metrics:
            return ""
        forecasting_focus = focus.lower() == "forecasting"
        cards = []

        # Forecasting focus: lead with the operational setup.
        if forecasting_focus:
            if metrics.get("station"):
                cards.append(
                    f'<div class="metric-card"><div class="metric-value">{_html.escape(str(metrics["station"]))}</div>'
                    f'<div class="metric-label">Station</div></div>'
                )
            if metrics.get("target"):
                cards.append(
                    f'<div class="metric-card"><div class="metric-value">{_html.escape(str(metrics["target"]))}</div>'
                    f'<div class="metric-label">Target</div></div>'
                )
            if metrics.get("model"):
                cards.append(
                    f'<div class="metric-card"><div class="metric-value">{_html.escape(str(metrics["model"]))}</div>'
                    f'<div class="metric-label">Model</div></div>'
                )

        if metrics.get("best_horizon") is not None:
            label = metrics["best_horizon"].replace("=", "=") if isinstance(metrics["best_horizon"], str) else str(metrics["best_horizon"])
            cards.append(
                f'<div class="metric-card highlight"><div class="metric-value">{label}</div>'
                f'<div class="metric-label">Best horizon</div></div>'
            )
        if metrics.get("best_rmse") is not None:
            cards.append(
                f'<div class="metric-card"><div class="metric-value">{metrics["best_rmse"]:.2f} cm</div>'
                f'<div class="metric-label">Best RMSE</div></div>'
            )
        if metrics.get("best_nse") is not None:
            cards.append(
                f'<div class="metric-card"><div class="metric-value">{metrics["best_nse"]:.3f}</div>'
                f'<div class="metric-label">Best NSE</div></div>'
            )
        if metrics.get("horizon_count"):
            cards.append(
                f'<div class="metric-card"><div class="metric-value">{metrics["horizon_count"]}</div>'
                f'<div class="metric-label">Horizons evaluated</div></div>'
            )
        if not cards:
            return ""
        return f'<div class="metrics-grid">{"".join(cards)}</div>'

    def _academic_css(self) -> str:
        return """
:root {
  --bg: #f8fafc;
  --fg: #0f172a;
  --muted: #475569;
  --accent: #2563eb;
  --accent-light: #dbeafe;
  --accent-dark: #1d4ed8;
  --card-bg: #ffffff;
  --border: #e2e8f0;
  --shadow: 0 4px 6px -1px rgba(0,0,0,0.05), 0 2px 4px -1px rgba(0,0,0,0.03);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.08), 0 4px 6px -2px rgba(0,0,0,0.04);
  --radius: 12px;
  --sidebar-width: 280px;
  --topbar-height: 64px;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-heading: Georgia, "Times New Roman", serif;
}
html[data-theme="dark"] {
  --bg: #0f172a;
  --fg: #f1f5f9;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --accent-light: #0c4a6e;
  --accent-dark: #7dd3fc;
  --card-bg: #1e293b;
  --border: #334155;
  --shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  font-family: var(--font-body);
  background: var(--bg);
  color: var(--fg);
  line-height: 1.65;
}
.topbar {
  position: fixed;
  top: 0; left: 0; right: 0;
  height: var(--topbar-height);
  background: var(--card-bg);
  border-bottom: 1px solid var(--border);
  box-shadow: var(--shadow);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 1.5rem;
  z-index: 100;
}
.brand { display: flex; align-items: center; gap: 0.75rem; }
.brand-icon { font-size: 1.6rem; }
.brand-title { font-weight: 700; font-size: 1.1rem; line-height: 1.2; }
.brand-subtitle { font-size: 0.8rem; color: var(--muted); }
.topbar-meta { display: flex; align-items: center; gap: 0.75rem; }
.badge {
  background: var(--accent-light);
  color: var(--accent-dark);
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}
.theme-toggle {
  background: var(--accent-light);
  border: none;
  border-radius: 50%;
  width: 36px; height: 36px;
  cursor: pointer;
  font-size: 1.1rem;
  display: flex; align-items: center; justify-content: center;
  transition: transform 0.2s, background 0.2s;
}
.theme-toggle:hover { transform: scale(1.05); background: var(--border); }
.sidebar {
  position: fixed;
  top: var(--topbar-height); left: 0; bottom: 0;
  width: var(--sidebar-width);
  background: var(--card-bg);
  border-right: 1px solid var(--border);
  padding: 1.5rem;
  overflow-y: auto;
  z-index: 90;
}
.sidebar h2 {
  font-family: var(--font-heading);
  font-size: 1.1rem;
  margin: 0 0 1rem 0;
  color: var(--accent-dark);
}
.sidebar ul { list-style: none; padding: 0; margin: 0; }
.sidebar li { margin: 0.25rem 0; }
.sidebar a {
  display: block;
  color: var(--muted);
  text-decoration: none;
  padding: 0.45rem 0.6rem;
  border-radius: 6px;
  font-size: 0.92rem;
  transition: background 0.15s, color 0.15s;
  border-left: 3px solid transparent;
}
.sidebar a:hover { color: var(--accent-dark); background: var(--bg); }
.sidebar a.active {
  color: var(--accent-dark);
  background: var(--accent-light);
  border-left-color: var(--accent);
  font-weight: 600;
}
.nav-meta {
  margin-top: 2rem;
  font-size: 0.75rem;
  color: var(--muted);
  line-height: 1.4;
}
.main {
  margin-left: var(--sidebar-width);
  margin-top: var(--topbar-height);
  padding: 2rem 2.5rem;
  max-width: 1000px;
}
.hero {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-dark) 100%);
  color: #fff;
  border-radius: var(--radius);
  padding: 2.25rem;
  margin-bottom: 2rem;
  box-shadow: var(--shadow-lg);
}
html[data-theme="dark"] .hero { background: linear-gradient(135deg, #0c4a6e 0%, #1e293b 100%); }
.hero h1 {
  font-family: var(--font-heading);
  font-size: 2.25rem;
  margin: 0 0 0.75rem 0;
  color: #fff;
}
.hero-lead {
  font-size: 1.05rem;
  margin: 0 0 1.5rem 0;
  opacity: 0.95;
  max-width: 720px;
}
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 1rem;
  margin-top: 1.5rem;
}
.metric-card {
  background: rgba(255,255,255,0.15);
  backdrop-filter: blur(4px);
  border-radius: 10px;
  padding: 1rem;
  text-align: center;
  border: 1px solid rgba(255,255,255,0.2);
}
.metric-card.highlight { background: rgba(255,255,255,0.25); }
.metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  line-height: 1.2;
}
.metric-label {
  font-size: 0.75rem;
  opacity: 0.9;
  margin-top: 0.25rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.report-section {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 2rem 2.25rem;
  margin-bottom: 2rem;
  box-shadow: var(--shadow);
}
.section-title {
  font-family: var(--font-heading);
  font-size: 1.7rem;
  color: var(--accent-dark);
  margin-top: 0;
  margin-bottom: 1.25rem;
  padding-bottom: 0.6rem;
  border-bottom: 2px solid var(--accent-light);
}
.section-intro { color: var(--muted); margin-bottom: 1.5rem; }
h1 { font-family: var(--font-heading); font-size: 2rem; margin-top: 0; color: var(--accent-dark); }
h2 { font-family: var(--font-heading); font-size: 1.45rem; margin-top: 1.75rem; color: var(--accent-dark); }
h3 { font-size: 1.15rem; color: var(--fg); margin-top: 1.5rem; opacity: 0.9; }
h4 { font-size: 1rem; color: var(--muted); margin-top: 1.25rem; }
p { margin: 0.85rem 0; }
a { color: var(--accent); }
img { max-width: 100%; height: auto; border-radius: 6px; }
blockquote {
  border-left: 4px solid var(--accent);
  background: var(--accent-light);
  padding: 1rem 1.25rem;
  margin: 1rem 0;
  border-radius: 0 8px 8px 0;
  color: var(--fg);
}
html[data-theme="dark"] blockquote { background: rgba(56,189,248,0.1); }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
th, td { border: 1px solid var(--border); padding: 0.55rem 0.75rem; text-align: left; }
th { background: var(--accent-light); font-weight: 600; color: var(--accent-dark); }
tr:nth-child(even) td { background: rgba(0,0,0,0.02); }
html[data-theme="dark"] tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
pre {
  background: #0f172a;
  color: #e2e8f0;
  padding: 1rem;
  border-radius: 8px;
  overflow-x: auto;
  font-size: 0.88rem;
  line-height: 1.5;
}
code {
  background: var(--accent-light);
  color: var(--accent-dark);
  padding: 0.15rem 0.35rem;
  border-radius: 4px;
  font-size: 0.9em;
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}
pre code { background: transparent; color: inherit; padding: 0; }
.table-wrap { overflow-x: auto; border-radius: 8px; border: 1px solid var(--border); }
ul, ol { padding-left: 1.5rem; margin: 0.75rem 0; }
li { margin: 0.35rem 0; }
li input[type="checkbox"] { margin-right: 0.4rem; }
.missing { color: #dc2626; font-style: italic; }
.chart-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow);
}
.chart-card h3 { margin-top: 0; margin-bottom: 0.75rem; color: var(--accent-dark); }
.chart { width: 100%; min-height: 420px; }
.forecast-setup {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 1rem;
  margin: 1rem 0 1.5rem 0;
}
.setup-card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1rem;
  box-shadow: var(--shadow);
}
.setup-card h3 {
  margin: 0 0 0.5rem 0;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: var(--accent-dark);
}
.setup-card p {
  margin: 0.25rem 0;
  font-size: 0.95rem;
}
.setup-card .coords {
  font-size: 0.85rem;
  color: var(--muted);
}
.setup-card .horizons {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.horizon-tag {
  background: var(--accent-light);
  color: var(--accent-dark);
  padding: 0.2rem 0.5rem;
  border-radius: 999px;
  font-size: 0.8rem;
  font-weight: 600;
}
.report-footer {
  margin-top: 1rem;
  padding: 1.5rem;
  text-align: center;
  color: var(--muted);
  font-size: 0.85rem;
  border-top: 1px solid var(--border);
}
.report-footer p { margin: 0.25rem 0; }
@media print {
  .topbar, .sidebar, .theme-toggle { display: none; }
  .main { margin-left: 0; margin-top: 0; max-width: none; padding: 1rem; }
  .report-section, .chart-card { box-shadow: none; border: 1px solid #ccc; page-break-inside: avoid; }
  .hero { background: #1a4a6e !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  .chart { page-break-inside: avoid; }
}
@media (max-width: 900px) {
  .sidebar { position: relative; width: 100%; top: auto; border-right: none; border-bottom: 1px solid var(--border); }
  .main { margin-left: 0; }
  .metrics-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 600px) {
  .main { padding: 1rem; }
  .hero { padding: 1.5rem; }
  .hero h1 { font-size: 1.6rem; }
  .report-section { padding: 1.25rem; }
  .metrics-grid { grid-template-columns: 1fr; }
  .chart { min-height: 320px; }
}
""".strip()

    def _dashboard_css(self) -> str:
        # Dashboard theme is just the dark palette applied by default.
        base = self._academic_css()
        # Replace the :root block with a dark-first root and add an auto dark hint.
        dark_root = """:root {
  color-scheme: dark;
  --bg: #0f172a;
  --fg: #f1f5f9;
  --muted: #94a3b8;
  --accent: #38bdf8;
  --accent-light: #0c4a6e;
  --accent-dark: #7dd3fc;
  --card-bg: #1e293b;
  --border: #334155;
  --shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
  --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4);
  --radius: 12px;
  --sidebar-width: 280px;
  --topbar-height: 64px;
  --font-body: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-heading: Georgia, "Times New Roman", serif;
}"""
        # Swap :root blocks roughly.
        new_css = re.sub(r":root \{[^}]+\}", dark_root, base, count=1)
        return new_css
