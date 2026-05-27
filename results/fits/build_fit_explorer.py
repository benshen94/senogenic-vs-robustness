#!/usr/bin/env python3
"""Build a double-clickable HTML explorer for archived SR fits."""

from __future__ import annotations

import json
from pathlib import Path


ARCHIVE_DIR = Path(__file__).resolve().parent
INDEX_PATH = ARCHIVE_DIR / "index.json"
HTML_PATH = ARCHIVE_DIR / "fit_explorer.html"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_records(index: dict) -> list[dict]:
    records = []

    for item in index["records"]:
        record_path = ARCHIVE_DIR / item["record_json"]
        record = load_json(record_path)
        record["explorer_png"] = item.get("png")
        record["explorer_record_json"] = item.get("record_json")
        records.append(record)

    return records


def make_payload(index: dict, records: list[dict]) -> str:
    payload = {
        "archive": {
            "archive_name": index["archive_name"],
            "archive_created_utc": index["archive_created_utc"],
            "project_cwd": index["project_cwd"],
            "record_count": index["record_count"],
            "completed_record_count": index["completed_record_count"],
        },
        "records": records,
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_html(payload_json: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SR Fit Explorer</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --panel: #ffffff;
      --ink: #181818;
      --muted: #6b665e;
      --line: #d7d0c3;
      --accent: #146c78;
      --accent-2: #953f2f;
      --soft: #e8f1f2;
      --warn: #f6e3b5;
      --shadow: 0 10px 28px rgba(35, 30, 20, 0.10);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.45 Arial, Helvetica, sans-serif;
    }}

    header {{
      padding: 22px 28px 18px;
      border-bottom: 1px solid var(--line);
      background: #fffaf0;
    }}

    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }}

    header p {{
      margin: 0;
      color: var(--muted);
      max-width: 920px;
    }}

    main {{
      display: grid;
      grid-template-columns: minmax(300px, 360px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      min-height: calc(100vh - 92px);
    }}

    .sidebar,
    .details {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}

    .sidebar {{
      padding: 14px;
      align-self: start;
      position: sticky;
      top: 18px;
      max-height: calc(100vh - 36px);
      overflow: auto;
    }}

    .controls {{
      display: grid;
      gap: 10px;
      margin-bottom: 14px;
    }}

    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}

    input,
    select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      font: inherit;
      padding: 9px 10px;
    }}

    .fit-list {{
      display: grid;
      gap: 8px;
    }}

    .fit-button {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      color: var(--ink);
      padding: 10px;
      text-align: left;
      cursor: pointer;
    }}

    .fit-button:hover {{
      border-color: var(--accent);
    }}

    .fit-button.active {{
      border-color: var(--accent);
      background: var(--soft);
    }}

    .fit-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 4px;
      font-weight: 700;
    }}

    .fit-meta {{
      color: var(--muted);
      font-size: 13px;
    }}

    .serial {{
      display: inline-block;
      border-radius: 999px;
      background: #f0ebe2;
      color: #39352f;
      font-size: 12px;
      font-weight: 700;
      padding: 3px 7px;
      white-space: nowrap;
    }}

    .details {{
      min-width: 0;
      overflow: hidden;
    }}

    .details-head {{
      display: grid;
      gap: 10px;
      padding: 18px 20px 14px;
      border-bottom: 1px solid var(--line);
    }}

    h2 {{
      margin: 0;
      font-size: 24px;
      line-height: 1.2;
      letter-spacing: 0;
    }}

    .description {{
      color: var(--muted);
      margin: 0;
    }}

    .tag-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}

    .tag {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 8px;
      background: #fbfaf8;
      color: #38332d;
      font-size: 13px;
    }}

    .tag.warn {{
      background: var(--warn);
      border-color: #dfbf68;
    }}

    .content {{
      display: grid;
      gap: 18px;
      padding: 18px 20px 22px;
    }}

    .image-wrap {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfaf8;
      padding: 10px;
      min-height: 260px;
    }}

    .image-wrap img {{
      display: block;
      width: 100%;
      height: auto;
      border-radius: 4px;
      background: #fff;
    }}

    .missing-image {{
      display: grid;
      place-items: center;
      min-height: 260px;
      color: var(--muted);
      text-align: center;
    }}

    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}

    section {{
      min-width: 0;
    }}

    h3 {{
      margin: 0 0 8px;
      font-size: 16px;
      letter-spacing: 0;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background: #fff;
    }}

    th,
    td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}

    th {{
      width: 42%;
      color: var(--muted);
      font-weight: 700;
      background: #fbfaf8;
    }}

    tr:last-child th,
    tr:last-child td {{
      border-bottom: 0;
    }}

    code {{
      overflow-wrap: anywhere;
      color: var(--accent-2);
      font-family: Menlo, Consolas, monospace;
      font-size: 13px;
    }}

    .empty {{
      color: var(--muted);
      padding: 14px;
      border: 1px dashed var(--line);
      border-radius: 8px;
      background: #fbfaf8;
    }}

    @media (max-width: 900px) {{
      main {{
        grid-template-columns: 1fr;
      }}

      .sidebar {{
        position: static;
        max-height: none;
      }}

      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SR Fit Explorer</h1>
    <p>Double-clickable browser for the archived fits. Pick a fitted curve, check the fitted region and constraints, then use the serial name in code.</p>
  </header>

  <main>
    <aside class="sidebar">
      <div class="controls">
        <label>
          Search
          <input id="search" type="search" placeholder="USA, cohort, 1920, tail90...">
        </label>
        <label>
          Data type
          <select id="dataType"></select>
        </label>
        <label>
          Country
          <select id="country"></select>
        </label>
        <label>
          Status
          <select id="status"></select>
        </label>
      </div>
      <div id="fitList" class="fit-list"></div>
    </aside>

    <article class="details">
      <div class="details-head">
        <h2 id="title"></h2>
        <p id="description" class="description"></p>
        <div id="tags" class="tag-row"></div>
      </div>
      <div class="content">
        <div id="imageWrap" class="image-wrap"></div>
        <div class="grid">
          <section>
            <h3>Fitted For</h3>
            <table id="fittedFor"></table>
          </section>
          <section>
            <h3>Parameters</h3>
            <table id="parameters"></table>
          </section>
          <section>
            <h3>Metrics</h3>
            <table id="metrics"></table>
          </section>
          <section>
            <h3>Files and References</h3>
            <table id="files"></table>
          </section>
          <section>
            <h3>Confidence Intervals</h3>
            <table id="ci"></table>
          </section>
          <section>
            <h3>Procedure</h3>
            <table id="procedure"></table>
          </section>
        </div>
      </div>
    </article>
  </main>

  <script>
    const DATA = {payload_json};

    const state = {{
      selectedName: DATA.records.find((record) => record.status === "completed")?.name || DATA.records[0]?.name,
      search: "",
      dataType: "all",
      country: "all",
      status: "all",
    }};

    const nodes = {{
      search: document.getElementById("search"),
      dataType: document.getElementById("dataType"),
      country: document.getElementById("country"),
      status: document.getElementById("status"),
      fitList: document.getElementById("fitList"),
      title: document.getElementById("title"),
      description: document.getElementById("description"),
      tags: document.getElementById("tags"),
      imageWrap: document.getElementById("imageWrap"),
      fittedFor: document.getElementById("fittedFor"),
      parameters: document.getElementById("parameters"),
      metrics: document.getElementById("metrics"),
      files: document.getElementById("files"),
      ci: document.getElementById("ci"),
      procedure: document.getElementById("procedure"),
    }};

    function unique(values) {{
      return Array.from(new Set(values.filter(Boolean))).sort();
    }}

    function formatValue(value) {{
      if (value === null || value === undefined || value === "") {{
        return "";
      }}

      if (typeof value === "number") {{
        if (Math.abs(value) >= 1000) {{
          return value.toLocaleString(undefined, {{ maximumFractionDigits: 0 }});
        }}
        return Number.parseFloat(value.toPrecision(7)).toString();
      }}

      if (Array.isArray(value)) {{
        return value.map(formatValue).join(", ");
      }}

      if (typeof value === "object") {{
        return Object.entries(value)
          .map(([key, nestedValue]) => `${{key}}: ${{formatValue(nestedValue)}}`)
          .join("<br>");
      }}

      return String(value);
    }}

    function makeOption(value, label) {{
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      return option;
    }}

    function setOptions(select, values, allLabel) {{
      select.innerHTML = "";
      select.appendChild(makeOption("all", allLabel));
      for (const value of values) {{
        select.appendChild(makeOption(value, value));
      }}
    }}

    function setupFilters() {{
      const dataTypes = unique(DATA.records.map((record) => record.fitted_for?.data_type));
      const countries = unique(DATA.records.flatMap((record) => record.fitted_for?.countries || []));
      const statuses = unique(DATA.records.map((record) => record.status));

      setOptions(nodes.dataType, dataTypes, "All data types");
      setOptions(nodes.country, countries, "All countries");
      setOptions(nodes.status, statuses, "All statuses");

      nodes.search.addEventListener("input", () => {{
        state.search = nodes.search.value.trim().toLowerCase();
        render();
      }});

      nodes.dataType.addEventListener("change", () => {{
        state.dataType = nodes.dataType.value;
        render();
      }});

      nodes.country.addEventListener("change", () => {{
        state.country = nodes.country.value;
        render();
      }});

      nodes.status.addEventListener("change", () => {{
        state.status = nodes.status.value;
        render();
      }});
    }}

    function recordMatches(record) {{
      const fittedFor = record.fitted_for || {{}};
      const searchText = [
        record.name,
        record.label,
        record.description,
        record.constraints,
        fittedFor.data_type,
        fittedFor.year,
        fittedFor.sex,
        ...(fittedFor.countries || []),
      ].join(" ").toLowerCase();

      if (state.search && !searchText.includes(state.search)) {{
        return false;
      }}

      if (state.dataType !== "all" && fittedFor.data_type !== state.dataType) {{
        return false;
      }}

      if (state.country !== "all" && !(fittedFor.countries || []).includes(state.country)) {{
        return false;
      }}

      if (state.status !== "all" && record.status !== state.status) {{
        return false;
      }}

      return true;
    }}

    function getVisibleRecords() {{
      return DATA.records.filter(recordMatches);
    }}

    function makeRows(table, rows) {{
      table.innerHTML = "";

      if (!rows.length) {{
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 2;
        cell.className = "empty";
        cell.textContent = "No saved values for this fit.";
        row.appendChild(cell);
        table.appendChild(row);
        return;
      }}

      for (const [key, value] of rows) {{
        const row = document.createElement("tr");
        const head = document.createElement("th");
        const cell = document.createElement("td");
        head.textContent = key;
        cell.innerHTML = formatValue(value);
        row.appendChild(head);
        row.appendChild(cell);
        table.appendChild(row);
      }}
    }}

    function parameterRows(record) {{
      const params = record.summary?.fitted_parameters || {{}};
      return Object.entries(params);
    }}

    function metricRows(record) {{
      const metrics = record.summary?.metrics || {{}};

      if (Object.values(metrics).some((value) => typeof value === "object" && value !== null)) {{
        return Object.entries(metrics).flatMap(([group, values]) => (
          Object.entries(values || {{}}).map(([key, value]) => [`${{group}} ${{key}}`, value])
        ));
      }}

      return Object.entries(metrics);
    }}

    function ciRows(record) {{
      const ci = record.ci || record.summary?.ci || [];
      return ci.map((item) => [
        item.parameter,
        `${{formatValue(item.estimate)}} [${{formatValue(item.ci95_lower)}}, ${{formatValue(item.ci95_upper)}}]`,
      ]);
    }}

    function fileRows(record) {{
      const rows = [
        ["serial name", `<code>${{record.name}}</code>`],
        ["record JSON", `<code>${{record.explorer_record_json || record.archive_paths?.record_json || ""}}</code>`],
        ["PNG", record.explorer_png ? `<code>${{record.explorer_png}}</code>` : ""],
        ["CI CSV", record.archive_paths?.ci_csv ? `<code>${{record.archive_paths.ci_csv}}</code>` : ""],
        ["source script", record.source_script ? `<code>${{record.source_script}}</code>` : ""],
        ["source summary", record.source_summary ? `<code>${{record.source_summary}}</code>` : ""],
      ];

      return rows.filter(([, value]) => value);
    }}

    function procedureRows(record) {{
      const summary = record.summary || {{}};
      return [
        ["constraints", record.constraints || summary.constraint],
        ["focus age window", summary.focus_age_window || record.fitted_for?.age_window],
        ["optimization n", summary.opt_n],
        ["final n", summary.final_n],
        ["CI n", summary.ci_n],
        ["archive created UTC", record.archive_created_utc],
      ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    }}

    function fittedForRows(record) {{
      const fittedFor = record.fitted_for || {{}};
      return [
        ["countries", fittedFor.countries],
        ["data type", fittedFor.data_type],
        ["year", fittedFor.year],
        ["sex", fittedFor.sex],
        ["age window", fittedFor.age_window],
      ].filter(([, value]) => value !== undefined && value !== null && value !== "");
    }}

    function tag(label, isWarning = false) {{
      const span = document.createElement("span");
      span.className = isWarning ? "tag warn" : "tag";
      span.textContent = label;
      return span;
    }}

    function renderList(visibleRecords) {{
      nodes.fitList.innerHTML = "";

      if (!visibleRecords.length) {{
        const empty = document.createElement("div");
        empty.className = "empty";
        empty.textContent = "No fits match these filters.";
        nodes.fitList.appendChild(empty);
        return;
      }}

      if (!visibleRecords.some((record) => record.name === state.selectedName)) {{
        state.selectedName = visibleRecords[0].name;
      }}

      visibleRecords.forEach((record, index) => {{
        const fittedFor = record.fitted_for || {{}};
        const button = document.createElement("button");
        button.className = record.name === state.selectedName ? "fit-button active" : "fit-button";
        button.type = "button";
        button.addEventListener("click", () => {{
          state.selectedName = record.name;
          render();
        }});

        button.innerHTML = `
          <div class="fit-title">
            <span>${{record.label || record.name}}</span>
            <span class="serial">#${{index + 1}}</span>
          </div>
          <div class="fit-meta">
            ${{record.name}}<br>
            ${{formatValue(fittedFor.countries)}} · ${{fittedFor.data_type || ""}} · ${{fittedFor.year || ""}} · ages ${{formatValue(fittedFor.age_window)}}
          </div>
        `;

        nodes.fitList.appendChild(button);
      }});
    }}

    function renderDetails(record) {{
      const fittedFor = record.fitted_for || {{}};
      nodes.title.textContent = record.label || record.name;
      nodes.description.textContent = record.description || "";

      nodes.tags.innerHTML = "";
      nodes.tags.appendChild(tag(record.status || "unknown", record.status !== "completed"));
      nodes.tags.appendChild(tag(record.name));
      if (fittedFor.data_type) nodes.tags.appendChild(tag(fittedFor.data_type));
      if (fittedFor.year) nodes.tags.appendChild(tag(String(fittedFor.year)));
      if (fittedFor.age_window) nodes.tags.appendChild(tag(`ages ${{formatValue(fittedFor.age_window)}}`));
      for (const country of fittedFor.countries || []) nodes.tags.appendChild(tag(country));

      nodes.imageWrap.innerHTML = "";
      if (record.explorer_png) {{
        const img = document.createElement("img");
        img.alt = `${{record.label || record.name}} fit plot`;
        img.src = record.explorer_png;
        nodes.imageWrap.appendChild(img);
      }} else {{
        const missing = document.createElement("div");
        missing.className = "missing-image";
        missing.textContent = "No completed PNG was saved for this record.";
        nodes.imageWrap.appendChild(missing);
      }}

      makeRows(nodes.fittedFor, fittedForRows(record));
      makeRows(nodes.parameters, parameterRows(record));
      makeRows(nodes.metrics, metricRows(record));
      makeRows(nodes.files, fileRows(record));
      makeRows(nodes.ci, ciRows(record));
      makeRows(nodes.procedure, procedureRows(record));
    }}

    function render() {{
      const visibleRecords = getVisibleRecords();
      renderList(visibleRecords);

      const record = DATA.records.find((item) => item.name === state.selectedName) || visibleRecords[0] || DATA.records[0];
      if (record) {{
        renderDetails(record);
      }}
    }}

    setupFilters();
    render();
  </script>
</body>
</html>
"""


def main() -> None:
    index = load_json(INDEX_PATH)
    records = load_records(index)
    payload_json = make_payload(index, records)
    HTML_PATH.write_text(build_html(payload_json), encoding="utf-8")
    print(HTML_PATH)


if __name__ == "__main__":
    main()
