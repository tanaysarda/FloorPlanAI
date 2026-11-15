"""Command-line interface (works with only the standard library + OpenCV/NumPy).

  python cli.py build-synthetic            # generate SYNTHETIC test data, ingest, overlays, JSON export
  python cli.py ingest --images D --masks D --class-map F --dataset NAME   # real dataset (see README)
  python cli.py ask "How many rooms does plan 2 have?" [--kind synthetic|real] [--json]
  python cli.py chat                       # interactive questions
  python cli.py stats                      # dataset summary
  python cli.py eval                       # run the evaluation question set
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import config
from assistant.llm import get_provider
from assistant.qa_engine import QAEngine, QAResult
from database import db


def _print_result(res: QAResult, out, show_sql: bool = True) -> None:
    print(f"\nAnswer [{res.status}, source={res.source}]: {res.answer}", file=out)
    for note in res.notes:
        print(f"Note: {note}", file=out)
    if show_sql and res.sql:
        print(f"\nSQL:      {res.sql}\nParams:   {list(res.params)}", file=out)
    if res.rows:
        print("Evidence:", file=out)
        print("  " + " | ".join(res.columns), file=out)
        for r in res.rows[:20]:
            print("  " + " | ".join(str(v) for v in r), file=out)
        if len(res.rows) > 20 or res.truncated:
            print(f"  ... ({len(res.rows)} rows shown/returned)", file=out)


def _banner(paths: config.Paths, out) -> None:
    if paths.kind == "synthetic":
        print("*** SYNTHETIC TEST DATA - not the Kaggle dataset. Results are not real-dataset results. ***", file=out)


def cmd_build_synthetic(args, out) -> int:
    from processing.loader import FolderPairAdapter
    from processing.pipeline import ingest_dataset, write_report
    from processing.segmentation import ClassMap
    from processing.synthetic import generate_fixtures
    paths = config.paths_for("synthetic", Path(args.output_root) if args.output_root else None)
    generate_fixtures(config.SYNTHETIC_DIR)
    if args.reset and paths.db_path.exists():
        paths.db_path.unlink()
    if args.reset and paths.overlays_dir.exists():
        shutil.rmtree(paths.overlays_dir)
    conn = db.connect(paths.db_path)
    try:
        db.init_db(conn, "synthetic")
        adapter = FolderPairAdapter(config.SYNTHETIC_DIR / "images", config.SYNTHETIC_DIR / "masks",
                                    "synthetic", is_synthetic=True)
        report = ingest_dataset(adapter, ClassMap.from_json(config.CONFIG_DIR / "class_map_synthetic.json"), conn,
                                overlays_dir=paths.overlays_dir, comparison_figures=True)
        db.export_json(conn, paths.json_path)
        write_report(report, paths.report_path)
        summary = db.dataset_summary(conn)
    finally:
        conn.close()
    _banner(paths, out)
    print(f"Processed {len(report.processed)} plans, skipped {len(report.skipped)}. DB: {paths.db_path}", file=out)
    print(f"Summary: {json.dumps({k: v for k, v in summary.items() if k != 'room_labels'})}", file=out)
    print(f"Overlays: {paths.overlays_dir}  JSON: {paths.json_path}", file=out)
    return 0


def cmd_ingest(args, out) -> int:
    from processing.loader import FolderPairAdapter
    from processing.pipeline import ingest_dataset, write_report
    from processing.segmentation import ClassMap
    paths = config.paths_for("real", Path(args.output_root) if args.output_root else None)
    try:
        class_map = ClassMap.from_json(args.class_map)
    except (OSError, ValueError) as exc:
        print(f"Could not load the class map: {exc}", file=out)
        return 2
    adapter = FolderPairAdapter(args.images, args.masks, args.dataset, mask_suffix=args.mask_suffix)
    if not adapter.samples():
        print(f"No images found in {args.images}. Nothing to ingest.", file=out)
        return 2
    conn = db.connect(paths.db_path)
    try:
        db.init_db(conn, "real")
        report = ingest_dataset(adapter, class_map, conn, overlays_dir=paths.overlays_dir, max_overlays=args.max_overlays,
                                comparison_figures=True)
        db.export_json(conn, paths.json_path)
        write_report(report, paths.report_path)
    finally:
        conn.close()
    print(f"Processed {len(report.processed)}, skipped {len(report.skipped)} (see {paths.report_path}).", file=out)
    return 0


def _engine(kind: str, root: Optional[str], no_llm: bool) -> tuple[QAEngine, config.Paths]:
    paths = config.paths_for(kind, Path(root) if root else None)
    config.load_env_file()
    return QAEngine(paths.db_path, None if no_llm else get_provider()), paths


def cmd_ask(args, out) -> int:
    engine, paths = _engine(args.kind, args.output_root, args.no_llm)
    res = engine.ask(args.question)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str), file=out)
    else:
        _banner(paths, out)
        _print_result(res, out)
    return 0 if res.status in ("answered", "no_results", "unsupported", "invalid_plan", "clarification") else 1


def cmd_chat(args, out, inp=sys.stdin) -> int:
    engine, paths = _engine(args.kind, args.output_root, args.no_llm)
    _banner(paths, out)
    print("Ask a question about the floor plans (empty line or 'quit' to exit).", file=out)
    for line in inp:
        q = line.strip()
        if q.lower() in ("", "quit", "exit"):
            break
        _print_result(engine.ask(q), out)
    return 0


def cmd_stats(args, out) -> int:
    paths = config.paths_for(args.kind, Path(args.output_root) if args.output_root else None)
    try:
        conn = db.connect(paths.db_path, read_only=True)
    except db.DatabaseError:
        print("Database not found. Run `python cli.py build-synthetic` first.", file=out)
        return 1
    try:
        _banner(paths, out)
        print(json.dumps(db.dataset_summary(conn), indent=2), file=out)
    finally:
        conn.close()
    return 0


def cmd_eval(args, out) -> int:
    from evaluation.evaluator import run_evaluation, write_report
    paths = config.paths_for(args.kind, Path(args.output_root) if args.output_root else None)
    qfile = Path(args.questions) if args.questions else config.ROOT / "evaluation" / f"test_questions_{args.kind}.json"
    engine = QAEngine(paths.db_path, None)   # evaluation never depends on an LLM
    report = run_evaluation(engine, qfile)
    write_report(report, paths.output_dir)
    _banner(paths, out)
    s = report["summary"]
    print(f"Questions: {s['total']}  Correct: {s['correct']}  Incorrect: {s['incorrect']}  "
          f"Accuracy: {s['accuracy_pct']}%", file=out)
    for f in report["failures"]:
        print(f"  FAIL #{f['id']}: {f['question']} -> {f['reason']}", file=out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    def common(sp, kind=True):
        sp.add_argument("--output-root", default=None, help="override the outputs/ directory")
        if kind:
            sp.add_argument("--kind", choices=config.DATASET_KINDS, default="synthetic")
    s = sub.add_parser("build-synthetic"); common(s, False); s.add_argument("--reset", action="store_true")
    s.set_defaults(fn=cmd_build_synthetic)
    s = sub.add_parser("ingest"); common(s, False)
    s.add_argument("--images", required=True); s.add_argument("--masks", required=True)
    s.add_argument("--class-map", required=True); s.add_argument("--dataset", default="kaggle_floorplans")
    s.add_argument("--mask-suffix", default=""); s.add_argument("--max-overlays", type=int, default=None)
    s.set_defaults(fn=cmd_ingest)
    s = sub.add_parser("ask"); common(s); s.add_argument("question"); s.add_argument("--json", action="store_true")
    s.add_argument("--no-llm", action="store_true"); s.set_defaults(fn=cmd_ask)
    s = sub.add_parser("chat"); common(s); s.add_argument("--no-llm", action="store_true"); s.set_defaults(fn=cmd_chat)
    s = sub.add_parser("stats"); common(s); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("eval"); common(s); s.add_argument("--questions", default=None); s.set_defaults(fn=cmd_eval)
    return p


def main(argv: Optional[List[str]] = None, out=None) -> int:
    out = out or sys.stdout
    args = build_parser().parse_args(argv)
    return args.fn(args, out)


if __name__ == "__main__":
    sys.exit(main())
