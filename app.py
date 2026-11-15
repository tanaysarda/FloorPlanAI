from pathlib import Path
import sqlite3
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from assistant.qa_engine import QAEngine

# ---------------------------------------------------------------------------
# Page Config & Architectural Styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ask Your Floor Plan",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: rgba(79, 139, 249, 0.08);
        border: 1px solid rgba(79, 139, 249, 0.25);
        padding: 0.85rem 1rem;
        border-radius: 0.6rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers (PyArrow-free & Schema-Adaptive)
# ---------------------------------------------------------------------------
def find_db_path(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    for candidate in [base_dir / "floorplans.db", base_dir / "floorplan.db"]:
        if candidate.exists():
            return candidate
    dbs = sorted(base_dir.glob("*.db"))
    return dbs[0] if dbs else None


def get_table_columns(cur: sqlite3.Cursor, table_name: str) -> list[str]:
    rows = cur.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [r[1] for r in rows]


def pick_column(columns: list[str], candidates: list[str], fallback: str | None = None) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return fallback


def render_table(rows: list[dict] | pd.DataFrame) -> None:
    """Render tabular data via Markdown/HTML so PyArrow DLLs are never loaded."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        st.info("No rows returned.")
        return
    try:
        st.markdown(df.to_markdown(index=False))
    except ImportError:
        st.write(df.to_html(index=False), unsafe_allow_html=True)


def render_room_chart(room_counts: dict[str, int]) -> None:
    """Render a clean Matplotlib bar chart without Altair/PyArrow."""
    if not room_counts:
        st.info("No room data available.")
        return

    names = [str(k) for k in room_counts.keys()]
    counts = list(room_counts.values())

    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    bars = ax.bar(names, counts, color="#4F8BF9", width=0.52, edgecolor="#7AA7FF")
    ax.bar_label(bars, padding=3, color="#E6EAF2", fontsize=9, fontweight="bold")
    ax.tick_params(colors="#CBD5E1", labelsize=9)
    ax.set_ylabel("Count", color="#94A3B8", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#475569")
    ax.spines["bottom"].set_color("#475569")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sidebar: Dataset Selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📐 Ask Your Floor Plan")
    st.caption("Deterministic SQL-backed floor plan assistant")

    dataset_choice = st.radio(
        "Dataset",
        options=["Synthetic test data", "Real dataset (Kaggle)"],
        index=0,
    )

    output_dir = (
        Path("outputs/synthetic")
        if dataset_choice == "Synthetic test data"
        else Path("outputs/real")
    )
    db_path = find_db_path(output_dir)

    st.divider()
    st.subheader("Quick Sample Questions")
    sample_questions = [
        "How many plans are in the database?",
        "How many rooms does plan 3 have?",
        "Which plans have more than 3 doors?",
        "What is the largest room in plan 1?",
        "Which rooms in plan 2 have no windows?",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_question"] = q

# ---------------------------------------------------------------------------
# Guard: Missing Database
# ---------------------------------------------------------------------------
if db_path is None or not db_path.exists():
    st.error("No database found for this dataset yet.")
    st.code(
        "python cli.py build-synthetic --reset   # synthetic test data\n"
        "python cli.py ingest --images <dir> --masks <dir> --class-map configs/<map>.json   # real dataset",
        language="bash",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load Summary Stats Dynamically from SQLite
# ---------------------------------------------------------------------------
conn = sqlite3.connect(str(db_path))
conn.row_factory = sqlite3.Row
cur = conn.cursor()

summary_cols = get_table_columns(cur, "plan_summary")
id_col = pick_column(summary_cols, ["plan_id", "id"], fallback=summary_cols[0])
name_col = pick_column(summary_cols, ["name", "plan_name", "filename"])
room_cnt_col = pick_column(summary_cols, ["room_count", "num_rooms", "rooms"])
door_cnt_col = pick_column(summary_cols, ["door_count", "num_doors", "doors"])
win_cnt_col = pick_column(summary_cols, ["window_count", "num_windows", "windows"])

plans = [dict(r) for r in cur.execute(f"SELECT * FROM plan_summary ORDER BY {id_col}").fetchall()]

room_cols = get_table_columns(cur, "rooms")
label_col = pick_column(room_cols, ["label", "room_type", "category", "class_name", "name", "type"])

if label_col:
    room_rows = cur.execute(
        f"SELECT {label_col} AS grp, COUNT(*) AS cnt FROM rooms GROUP BY {label_col} ORDER BY cnt DESC"
    ).fetchall()
    room_distribution = {r["grp"]: r["cnt"] for r in room_rows}
else:
    room_distribution = {}

total_plans = len(plans)
total_rooms = (
    sum(p.get(room_cnt_col, 0) or 0 for p in plans)
    if room_cnt_col
    else cur.execute("SELECT COUNT(*) FROM rooms").fetchone()[0]
)
total_doors = (
    sum(p.get(door_cnt_col, 0) or 0 for p in plans)
    if door_cnt_col
    else cur.execute("SELECT COUNT(*) FROM doors").fetchone()[0]
)
total_windows = (
    sum(p.get(win_cnt_col, 0) or 0 for p in plans)
    if win_cnt_col
    else cur.execute("SELECT COUNT(*) FROM windows").fetchone()[0]
)

# ---------------------------------------------------------------------------
# Top KPI Cards
# ---------------------------------------------------------------------------
st.subheader(f"Dataset Overview — {dataset_choice}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Floor Plans", total_plans)
c2.metric("Total Rooms", total_rooms)
c3.metric("Total Doors", total_doors)
c4.metric("Total Windows", total_windows)

st.divider()

# ---------------------------------------------------------------------------
# Main Layout: Visual Inspector (Left) + Q&A Assistant (Right)
# ---------------------------------------------------------------------------
left_col, right_col = st.columns([1.05, 1.15], gap="large")

with left_col:
    st.subheader("Plan Visuals & Distribution")

    if plans:
        plan_options = {}
        for p in plans:
            pid = p[id_col]
            pname = p.get(name_col, "") if name_col else ""
            label = f"Plan {pid} — {pname}" if pname else f"Plan {pid}"
            plan_options[label] = p

        selected_label = st.selectbox("Inspect Floor Plan", list(plan_options.keys()))
        selected_plan = plan_options[selected_label]
        pid = selected_plan[id_col]
        plan_name = selected_plan.get(name_col, f"synthetic_plan_{int(pid):02d}") if name_col else f"synthetic_plan_{int(pid):02d}"

        overlays_dir = output_dir / "overlays"
        comp_img = overlays_dir / f"{plan_name}_comparison.png"
        ov_img = overlays_dir / f"{plan_name}_overlay.png"

        if comp_img.exists():
            st.image(str(comp_img), caption=f"{plan_name} (Original vs. Segmentation Overlay)", use_container_width=True)
        elif ov_img.exists():
            st.image(str(ov_img), caption=f"{plan_name} Overlay", use_container_width=True)
        else:
            st.info(f"No overlay image found in `{overlays_dir}` for `{plan_name}`.")

        r_cnt = selected_plan.get(room_cnt_col, "-") if room_cnt_col else "-"
        d_cnt = selected_plan.get(door_cnt_col, "-") if door_cnt_col else "-"
        w_cnt = selected_plan.get(win_cnt_col, "-") if win_cnt_col else "-"
        st.caption(f"**Plan {pid} Breakdown:** {r_cnt} rooms · {d_cnt} doors · {w_cnt} windows")

    st.markdown("**Room Type Distribution Across Dataset**")
    render_room_chart(room_distribution)

with right_col:
    st.subheader("Ask a Plain-Language Question")

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    engine = QAEngine(db_path=str(db_path))

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("View Executed SQL & Raw Rows"):
                    st.code(msg["sql"], language="sql")
                    if msg.get("rows"):
                        render_table(msg["rows"])

    prefill = st.session_state.pop("prefill_question", None)
    user_prompt = st.chat_input("e.g., How many rooms does plan 3 have?") or prefill

    if user_prompt:
        st.session_state["messages"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        result = engine.ask(user_prompt)
        answer_text = (
            result.get("answer")
            if isinstance(result, dict)
            else getattr(result, "answer", str(result))
        )
        sql_query = (
            result.get("sql")
            if isinstance(result, dict)
            else getattr(result, "sql", None)
        )
        rows = (
            result.get("rows")
            if isinstance(result, dict)
            else getattr(result, "rows", None)
        )

        with st.chat_message("assistant"):
            st.markdown(answer_text)
            if sql_query:
                with st.expander("View Executed SQL & Raw Rows"):
                    st.code(sql_query, language="sql")
                    if rows:
                        render_table(rows)

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer_text,
                "sql": sql_query,
                "rows": rows,
            }
        )

conn.close()