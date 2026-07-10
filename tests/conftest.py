"""Shared fixtures for doceval's test suite."""

import json

import pytest


@pytest.fixture
def paired_dataset(tmp_path):
    """A tiny docs/ + labels/ dataset: 2 documents, one JSON label file each."""
    docs_dir = tmp_path / "docs"
    labels_dir = tmp_path / "labels"
    docs_dir.mkdir()
    labels_dir.mkdir()

    docs_dir.joinpath("invoice_001.pdf").write_bytes(b"fake-pdf-1")
    docs_dir.joinpath("invoice_002.pdf").write_bytes(b"fake-pdf-2")

    labels_dir.joinpath("invoice_001.json").write_text(json.dumps({
        "vendor": "Acme Corp", "total": "100.00",
    }))
    labels_dir.joinpath("invoice_002.json").write_text(json.dumps({
        "vendor": "Beta LLC", "total": "200.00",
    }))
    return docs_dir, labels_dir
