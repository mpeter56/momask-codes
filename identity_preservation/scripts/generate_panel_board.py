"""
generate_panel_board.py — Interactive HTML dashboard, "Unnoticed Dance"-style.

Adapted to the identity-preservation-spectrum concept. Reads
rules_metadata.json produced by run_rules_blend.py and builds a page where:

  - Rows       = the 10 spectrum intensities (0.09 → 1.00)
  - Columns    = SOURCE | TRANSFORMED | RULE (retention bar)
  - A dropdown at the top lets you switch between the 10 rules live
    (α-FLOOR, ROOT TRACK, FOOT LOCK, UPPER PRESERVE, LOWER PRESERVE,
    SPINE AXIS, SKELETON SCALE, TIMING, LOW-FREQUENCY, POSE SIGNATURE).
  - Master play / pause / scrub / speed apply to every video at once.

Usage:
    conda activate momask
    python generate_panel_board.py
"""
from __future__ import annotations

import argparse
import html as html_module
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional


DEFAULT_DANCE_VALUES = [0.09, 0.11, 0.28, 0.39, 0.40, 0.51, 0.63, 0.74, 0.80, 1.00]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend-dir", default=str(Path.home() / "Downloads" / "momask-codes-main" / "identity_preservation" / "outputs" /
                                                "pipeline_output"))
    ap.add_argument("--webcam", default="", help="path to the source webcam mp4 (auto-detects if empty)")
    ap.add_argument("--out-dir", default=str(Path.home() / "Downloads" / "momask-codes-main" / "identity_preservation" / "outputs" / "panel_board"))
    ap.add_argument("--title", default="Spectrum Panel Board")
    return ap.parse_args()


def find_webcam_source(explicit_path: str) -> Optional[Path]:
    if explicit_path:
        p = Path(explicit_path).expanduser()
        return p if p.exists() else None
    input_dir = Path.home() / "Downloads" / "momask-codes-main" / "input_videos"
    if not input_dir.exists():
        return None
    candidates = sorted(input_dir.glob("webcam*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def descriptor_for(d: float) -> str:
    if d < 0.15:  return "identity preserved"
    if d < 0.30:  return "gentle deviation"
    if d < 0.45:  return "mild transformation"
    if d < 0.60:  return "mid transformation"
    if d < 0.75:  return "moderate dance"
    if d < 0.90:  return "strong dance"
    return "intensively dancing"


def color_for(d: float) -> str:
    r = min(255, int(80 + d * 200))
    g = min(255, int(120 - d * 50))
    b = min(255, int(210 - d * 200))
    return f"rgb({r},{g},{b})"


def link_video(src: Path, dst_dir: Path, dst_name: str) -> Optional[str]:
    if not src.exists():
        return None
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / dst_name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy(src, dst)
    return f"videos/{dst_name}"


def main() -> int:
    args = parse_args()
    blend_dir = Path(args.blend_dir)
    out_dir   = Path(args.out_dir)
    videos_dir = out_dir / "videos"

    # ---------- load rules metadata -----------------------------------------
    meta_path = blend_dir / "rules_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"[fatal] {meta_path} not found — run run_rules_blend.py first")
    meta = json.loads(meta_path.read_text())
    dance_values = meta["dance_values"]
    rules        = meta["rules"]     # list of dicts

    # ---------- symlink webcam ----------------------------------------------
    src_webcam = find_webcam_source(args.webcam)
    webcam_rel = link_video(src_webcam, videos_dir, "source_webcam.mp4") if src_webcam else None
    print(f"[panel] source webcam: {src_webcam if src_webcam else '(not found)'}")

    # ---------- symlink every (rule, intensity) mp4 -------------------------
    # metric_map[(rule_key, d)] = perturbation amount (0 = source, 1 = max)
    # Backwards compatible: if entries have "retention" from the older
    # generator pipeline we use that; otherwise use "perturbation".
    metric_map: dict[tuple[str, float], float] = {}
    for entry in meta["entries"]:
        rule_key = entry["rule"]; d = entry["d"]
        src_mp4 = blend_dir / entry["mp4"]
        if src_mp4.exists():
            link_video(src_mp4, videos_dir, entry["mp4"])
        metric_map[(rule_key, round(d, 2))] = entry.get(
            "perturbation", entry.get("retention", 0.0)
        )

    # ---------- HTML rendering ----------------------------------------------
    session_name = src_webcam.stem if src_webcam else "(no session)"
    default_rule = rules[0]["key"] if rules else ""

    # Build the rule-selector <option> tags
    rule_options = "\n".join(
        f'<option value="{r["key"]}" data-tagline="{html_module.escape(r["tagline"])}" '
        f'data-preserves="{html_module.escape(r["preserves"])}" '
        f'data-borrows="{html_module.escape(r["borrows"])}">{html_module.escape(r["title"])}</option>'
        for r in rules
    )

    # Build the 10 panel rows.  Each row's <video> starts with the default rule's video;
    # the rule <select> just swaps the src attribute when changed.
    rows_html = []
    for d in dance_values:
        d_key = round(d, 2)
        retention = metric_map.get((default_rule, d_key), 0.0)
        c = color_for(d)
        src_cell = (f'<video preload="metadata" muted playsinline src="{html_module.escape(webcam_rel)}"></video>'
                    if webcam_rel else '<div class="missing">source missing</div>')
        # attach data attribute so JS can rebuild the URL per rule
        rows_html.append(f"""
            <div class="panel-row" data-d="{d:.2f}">
              <div class="cell label">
                <div class="d-value" style="color: {c}">d = {d:.2f}</div>
                <div class="descriptor">{descriptor_for(d)}</div>
              </div>
              <div class="cell video-cell source-cell">{src_cell}</div>
              <div class="cell video-cell trans-cell">
                <video preload="metadata" muted playsinline data-rule-video
                       src="videos/blended_{default_rule}_d{d:.2f}.mp4"></video>
              </div>
              <div class="cell rule">
                <div class="alpha-label">perturbation strength</div>
                <div class="alpha-value" data-retention>{retention:.3f}</div>
                <div class="alpha-bar"><div data-retention-bar style="width: {retention*100:.1f}%"></div></div>
              </div>
            </div>""")

    retention_json = json.dumps(
        {r["key"]: {f"{d:.2f}": metric_map.get((r["key"], round(d, 2)), 0.0) for d in dance_values}
         for r in rules}
    )

    css = """
    :root {
        --bg-0:#0a0a12; --bg-1:#14141f; --bg-2:#1c1c2b; --line:#2a2a3e;
        --text-hi:#e8e6ff; --text-lo:#8b8aa8; --accent:#e8b880;
    }
    * { box-sizing: border-box; }
    body { background: var(--bg-0); color: var(--text-hi); margin: 0;
           font-family: "Playfair Display","Iowan Old Style",Georgia,serif; }
    header { padding: 20px 32px; border-bottom: 1px solid var(--line);
             display: flex; justify-content: space-between; align-items: baseline;
             background: var(--bg-1); }
    header h1 { margin: 0; font-weight: 300; letter-spacing: 4px; font-size: 26px; }
    header .session { font-size: 13px; color: var(--text-lo); font-style: italic; }
    main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 74px); }
    aside { background: var(--bg-1); padding: 18px 22px; border-right: 1px solid var(--line);
            display: flex; flex-direction: column; gap: 22px; }
    aside h3 { font-weight: 400; letter-spacing: 3px; color: var(--text-lo);
               text-transform: uppercase; font-size: 11px; margin: 0 0 8px 0; }
    button { background: var(--bg-2); color: var(--text-hi); border: 1px solid var(--line);
             padding: 8px 14px; font-family: inherit; font-size: 13px; cursor: pointer;
             border-radius: 3px; margin-right: 6px; margin-bottom: 6px; }
    button:hover { background: #262637; }
    button.primary { background: var(--accent); color: #2a1a08; border-color: var(--accent); }
    input[type=range] { width: 100%; accent-color: var(--accent); }
    select { background: var(--bg-2); color: var(--text-hi); border: 1px solid var(--line);
             padding: 10px; font-family: inherit; font-size: 14px; width: 100%; border-radius: 3px; }
    .rule-info { background: var(--bg-2); padding: 12px; border-left: 3px solid var(--accent);
                 margin-top: 10px; font-size: 12px; color: var(--text-lo); line-height: 1.5; }
    .rule-info strong { color: var(--text-hi); font-weight: 400; }
    .webcam-preview { border: 1px solid var(--line); padding: 8px; background: #000; }
    .webcam-preview video { width: 100%; display: block; }
    .webcam-preview p { margin: 6px 0 0 0; color: var(--text-lo); font-size: 11px; letter-spacing: 2px; }
    .panels { padding: 22px 28px; display: flex; flex-direction: column; gap: 10px; }
    .column-headers, .panel-row {
        display: grid; grid-template-columns: 100px 1fr 1fr 200px; gap: 12px;
    }
    .column-headers { color: var(--text-lo); font-size: 11px; letter-spacing: 3px;
                      text-transform: uppercase; padding: 0 6px; }
    .panel-row { align-items: center; background: var(--bg-1); padding: 10px;
                 border: 1px solid var(--line); border-radius: 4px; }
    .cell.label { display: flex; flex-direction: column; justify-content: center; }
    .cell.label .d-value { font-size: 22px; font-weight: 300; }
    .cell.label .descriptor { font-size: 11px; color: var(--text-lo); font-style: italic; margin-top: 3px; }
    .cell.video-cell { background: #000; border: 1px solid var(--line);
                        aspect-ratio: 16/9; overflow: hidden;
                        display: flex; align-items: center; justify-content: center; }
    .cell.video-cell video { width: 100%; height: 100%; object-fit: contain; display: block; }
    .cell.rule { padding: 6px 12px; }
    .cell.rule .alpha-label { color: var(--text-lo); font-size: 10px; letter-spacing: 2px; text-transform: uppercase; }
    .cell.rule .alpha-value { font-size: 20px; font-weight: 300; }
    .cell.rule .alpha-bar { height: 6px; background: var(--bg-2); margin-top: 6px; border-radius: 3px; overflow: hidden; }
    .cell.rule .alpha-bar > div { height: 100%; background: linear-gradient(to right, #4d5ef0, #e8b880); transition: width 0.3s; }
    .missing { color: #644d4d; font-style: italic; font-size: 12px; text-align: center; }
    footer { padding: 16px 32px; color: var(--text-lo); font-size: 11px; letter-spacing: 2px;
             border-top: 1px solid var(--line); background: var(--bg-1); }
    """

    js = f"""
    const PERTURBATION = {retention_json};

    const videos = () => Array.from(document.querySelectorAll('.panel-row video'));

    function playAll()    {{ videos().forEach(v => v.play().catch(()=>{{}})); }}
    function pauseAll()   {{ videos().forEach(v => v.pause()); }}
    function restartAll() {{ videos().forEach(v => {{ v.currentTime = 0; }}); playAll(); }}
    function setRate(r)   {{
        videos().forEach(v => v.playbackRate = r);
        document.getElementById('speed-val').textContent = r + 'x';
    }}
    function scrubAll(t)  {{
        videos().forEach(v => {{ if (!isNaN(v.duration)) v.currentTime = t * v.duration; }});
    }}

    function selectRule(ruleKey) {{
        // Swap every trans-cell <video>'s src to the new rule
        document.querySelectorAll('.panel-row').forEach(row => {{
            const d = row.dataset.d;
            const video = row.querySelector('[data-rule-video]');
            video.src = `videos/blended_${{ruleKey}}_d${{d}}.mp4`;
            video.load();
            // Update retention display
            const retention = (PERTURBATION[ruleKey] || {{}})[d] || 0.0;
            row.querySelector('[data-retention]').textContent = retention.toFixed(3);
            row.querySelector('[data-retention-bar]').style.width = (retention * 100).toFixed(1) + '%';
        }});
        // Update the rule-info card
        const sel = document.getElementById('rule-select');
        const opt = sel.options[sel.selectedIndex];
        document.getElementById('rule-tagline').textContent = opt.dataset.tagline;
        document.getElementById('rule-preserves').textContent = opt.dataset.preserves;
        document.getElementById('rule-borrows').textContent = opt.dataset.borrows;
    }}

    setInterval(() => {{
        const first = videos()[0];
        if (first && first.duration) {{
            document.getElementById('master-scrub').value = first.currentTime / first.duration;
            document.getElementById('time-display').textContent =
                first.currentTime.toFixed(2) + 's / ' + first.duration.toFixed(2) + 's';
        }}
    }}, 100);
    """

    # Rule-info block (starts filled with the default rule's data)
    first_rule = rules[0] if rules else {"tagline": "", "preserves": "", "borrows": ""}

    ctrls = f"""
    <aside>
      <div>
        <h3>Session</h3>
        <div style="font-size:13px; line-height: 1.5;">{html_module.escape(session_name)}</div>
      </div>

      <div>
        <h3>Rule</h3>
        <select id="rule-select" onchange="selectRule(this.value)">
          {rule_options}
        </select>
        <div class="rule-info">
          <div><strong id="rule-tagline">{html_module.escape(first_rule.get('tagline', ''))}</strong></div>
          <div style="margin-top:6px;">preserves: <span id="rule-preserves">{html_module.escape(first_rule.get('preserves', ''))}</span></div>
          <div>borrows: <span id="rule-borrows">{html_module.escape(first_rule.get('borrows', ''))}</span></div>
        </div>
      </div>

      <div>
        <h3>Playback</h3>
        <button class="primary" onclick="playAll()">▶ Play all</button>
        <button onclick="pauseAll()">⏸ Pause</button>
        <button onclick="restartAll()">↺ Restart</button>
        <div style="margin-top: 10px;">
          <label style="font-size:11px; color:var(--text-lo); letter-spacing: 2px;">SPEED <span id="speed-val">1x</span></label>
          <input type="range" min="0.25" max="2" step="0.25" value="1" oninput="setRate(parseFloat(this.value))">
        </div>
        <div style="margin-top: 10px;">
          <label style="font-size:11px; color:var(--text-lo); letter-spacing: 2px;">MASTER TIMELINE</label>
          <input id="master-scrub" type="range" min="0" max="1" step="0.001" value="0" oninput="scrubAll(parseFloat(this.value))">
          <div id="time-display" style="font-size:11px; color: var(--text-lo);">0.00s / 0.00s</div>
        </div>
      </div>

      { ('<div class="webcam-preview"><video src="' + html_module.escape(webcam_rel) + '" muted playsinline controls></video><p>SOURCE WEBCAM</p></div>') if webcam_rel else '<div class="missing">no source webcam attached</div>' }
    </aside>
    """

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html_module.escape(args.title)}</title>
  <meta name="viewport" content="width=1400">
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>{html_module.escape(args.title.upper())}</h1>
    <div class="session">10 identity-protection rules × 10 spectrum intensities · {html_module.escape(session_name)}</div>
  </header>
  <main>
    {ctrls}
    <div class="panels">
      <div class="column-headers">
        <div>intensity</div>
        <div>source</div>
        <div>transformed (current rule)</div>
        <div>perturbation</div>
      </div>
      {''.join(rows_html)}
    </div>
  </main>
  <footer>identity_rules.py — 10 deterministic rules · pipeline_output outputs</footer>
  <script>{js}</script>
</body>
</html>
"""

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(body)
    print(f"\n[panel] wrote {out_dir}/index.html")
    print(f"[panel] {len(rules)} rules × {len(dance_values)} intensities available")
    print(f"\nTo view:")
    print(f"    cd \"{out_dir}\" && python -m http.server 8000")
    print(f"    open http://localhost:8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
