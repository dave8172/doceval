"""
doceval CLI — entry point for `doceval run`.

Usage:
    doceval run \\
        --docs    ./dataset/docs \\
        --labels  ./dataset/labels \\
        --extractor my_module:extract \\
        [--output report.md] \\
        [--name "my extractor v2"] \\
        [--quiet]
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import click

from .harness import run_eval
from .report import generate_report


def _load_extractor(spec: str):
    """Load a function from a 'module:function' specifier."""
    if ":" not in spec:
        raise click.BadParameter(
            f"Extractor must be specified as 'module:function', got: {spec!r}"
        )
    module_path, fn_name = spec.rsplit(":", 1)

    # Allow relative imports by ensuring cwd is on sys.path
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise click.BadParameter(f"Cannot import module '{module_path}': {exc}") from exc

    if not hasattr(module, fn_name):
        raise click.BadParameter(
            f"Function '{fn_name}' not found in module '{module_path}'"
        )
    return getattr(module, fn_name)


@click.group()
def cli():
    """doceval — eval harness for document extraction pipelines."""


@cli.command()
@click.option("--docs",      required=True,  type=click.Path(exists=True, file_okay=False), help="Directory of documents to evaluate.")
@click.option("--labels",    required=True,  type=click.Path(exists=True, file_okay=False), help="Directory of label JSON files (one per document).")
@click.option("--extractor", required=True,  help="Extractor function as 'module:function'.")
@click.option("--output",    default=None,   help="Path for the Markdown report (default: auto-named in cwd).")
@click.option("--json-out",  default=None,   help="Path for the raw JSON results.")
@click.option("--name",      default=None,   help="Display name for the extractor in reports.")
@click.option("--quiet",     is_flag=True,   help="Suppress per-document progress output.")
def run(docs, labels, extractor, output, json_out, name, quiet):
    """Run evaluation and generate an accuracy report."""
    extract_fn = _load_extractor(extractor)
    extractor_name = name or extractor

    total_docs = sum(
        1 for p in Path(docs).iterdir()
        if p.suffix.lower() in {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".webp"}
    )

    if not quiet:
        click.echo(f"doceval run")
        click.echo(f"  docs:      {docs}")
        click.echo(f"  labels:    {labels}")
        click.echo(f"  extractor: {extractor_name}")
        click.echo(f"  documents: {total_docs} found\n")

    def _progress(idx, total, filename, result):
        if quiet:
            return
        if result is None:
            click.echo(f"[{idx:3}/{total}] {filename} ... ", nl=False)
        else:
            if result.error:
                click.echo(f"ERROR: {result.error}")
            else:
                pct = result.accuracy * 100
                cost = f"  ${result.cost_usd:.4f}" if result.cost_usd is not None else ""
                click.echo(
                    f"{result.fields_correct}/{result.fields_total} "
                    f"({pct:.0f}%)  {result.duration_s:.1f}s{cost}"
                )

    result = run_eval(
        docs_dir=docs,
        labels_dir=labels,
        extract_fn=extract_fn,
        extractor_name=extractor_name,
        progress_callback=_progress,
    )

    # Console summary
    click.echo(f"\n{'='*50}")
    click.echo(
        f"Overall: {result.fields_correct}/{result.fields_total} fields correct "
        f"({result.overall_accuracy * 100:.1f}%)"
    )
    click.echo(f"Successful: {result.successful}  Failed: {result.failed}")
    if result.failure_modes:
        click.echo("\nFailure modes:")
        for mode, count in sorted(result.failure_modes.items(), key=lambda x: -x[1]):
            click.echo(f"  {mode:16s}: {count}")
    if result.avg_cost_per_doc_usd is not None:
        click.echo(
            f"\nCost: ${result.avg_cost_per_doc_usd:.4f}/doc avg  "
            f"(${result.total_cost_usd:.4f} total)"
        )

    # Write markdown report
    report_path = Path(output) if output else Path(f"eval-report-{result.run_id}.md")
    report_path.write_text(generate_report(result))
    click.echo(f"\nReport → {report_path}")

    # Write JSON results
    if json_out:
        json_path = Path(json_out)
        _write_json(result, json_path)
        click.echo(f"JSON   → {json_path}")

    # Exit non-zero if any extractions failed
    if result.failed > 0:
        sys.exit(1)


def _write_json(result, path: Path):
    from dataclasses import asdict
    path.write_text(json.dumps(asdict(result), indent=2))


def main():
    cli()
