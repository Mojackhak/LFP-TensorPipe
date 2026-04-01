from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree

from paper.pd.submission.minimize_tracked_revision_docx import minimize_tracked_revision_docx
from paper.pd.submission.tracked_revision import (
    NSMAP,
    RevisionMetadata,
    build_plain_run,
    build_revision_run,
    extract_revision_text,
    revision_segments,
    rewrite_paragraph_xml,
    w_tag,
)


def test_revision_segments_keep_large_lexical_changes_as_whole_words() -> None:
    segments = revision_segments(
        "Representative validation cases",
        "Representative application cases",
    )
    assert segments == [
        ("equal", "Representative "),
        ("delete", "validation"),
        ("insert", "application"),
        ("equal", " cases"),
    ]


def test_revision_segments_descend_to_characters_for_typo_scale_change() -> None:
    segments = revision_segments("High-resolution", "Higher-resolution")
    assert segments == [
        ("equal", "High"),
        ("insert", "er"),
        ("equal", "-resolution"),
    ]


def test_rewrite_paragraph_xml_preserves_paragraph_properties() -> None:
    paragraph = etree.Element(w_tag("p"))
    paragraph_properties = etree.SubElement(paragraph, w_tag("pPr"))
    etree.SubElement(paragraph_properties, w_tag("pStyle"), {w_tag("val"): "Heading2"})
    metadata = RevisionMetadata(author="Codex", revision_date="2026-04-01T00:00:00Z")

    next_revision_id = rewrite_paragraph_xml(
        paragraph=paragraph,
        old_text="Graphical",
        new_text="graphical",
        start_revision_id=10,
        insert_metadata=metadata,
    )

    assert next_revision_id == 12
    children = list(paragraph)
    assert children[0].tag == w_tag("pPr")
    assert children[1].tag == w_tag("del")
    assert children[1].get(w_tag("id")) == "10"
    assert extract_revision_text(children[1]) == "G"
    assert children[2].tag == w_tag("ins")
    assert children[2].get(w_tag("id")) == "11"
    assert extract_revision_text(children[2]) == "g"
    assert "".join(children[3].xpath(".//w:t/text()", namespaces=NSMAP)) == "raphical"


def test_minimize_tracked_revision_docx_rewrites_only_coarse_pairs(tmp_path: Path) -> None:
    metadata = RevisionMetadata(author="OpenAI", revision_date="2026-03-31T05:48:58Z")
    input_docx = tmp_path / "tracked.docx"
    _write_minimal_tracked_docx(input_docx, metadata)

    summary = minimize_tracked_revision_docx(input_docx=input_docx, in_place=True)

    assert summary["candidate_paragraphs"] == 1
    assert summary["rewritten_paragraphs"] == 1
    assert summary["starting_revision_id"] == 9
    assert summary["ending_revision_id"] == 10
    backup_docx = Path(summary["backup_docx"])
    assert backup_docx.exists()

    with ZipFile(input_docx) as zip_file:
        document_root = etree.fromstring(zip_file.read("word/document.xml"))
        settings_root = etree.fromstring(zip_file.read("word/settings.xml"))

    body = document_root.find("w:body", namespaces=NSMAP)
    assert body is not None
    paragraphs = body.findall("w:p", namespaces=NSMAP)
    first_paragraph = paragraphs[0]
    second_paragraph = paragraphs[1]

    first_children = list(first_paragraph)
    assert first_children[0].tag == w_tag("pPr")
    assert first_children[1].tag == w_tag("del")
    assert first_children[1].get(w_tag("id")) == "9"
    assert extract_revision_text(first_children[1]) == "G"
    assert first_children[2].tag == w_tag("ins")
    assert first_children[2].get(w_tag("id")) == "10"
    assert extract_revision_text(first_children[2]) == "g"
    assert "".join(first_children[3].xpath(".//w:t/text()", namespaces=NSMAP)) == "raphical"

    second_children = list(second_paragraph)
    assert second_children[0].tag == w_tag("r")
    assert second_children[1].tag == w_tag("del")
    assert second_children[1].get(w_tag("id")) == "7"
    assert second_children[2].tag == w_tag("ins")
    assert second_children[2].get(w_tag("id")) == "8"
    assert settings_root.find("w:trackRevisions", namespaces=NSMAP) is not None


def _write_minimal_tracked_docx(path: Path, metadata: RevisionMetadata) -> None:
    document_root = etree.Element(w_tag("document"), nsmap={"w": NSMAP["w"]})
    body = etree.SubElement(document_root, w_tag("body"))

    coarse_paragraph = etree.SubElement(body, w_tag("p"))
    paragraph_properties = etree.SubElement(coarse_paragraph, w_tag("pPr"))
    etree.SubElement(paragraph_properties, w_tag("pStyle"), {w_tag("val"): "Heading2"})
    coarse_paragraph.append(build_revision_run("del", "Graphical", 5, metadata))
    coarse_paragraph.append(build_revision_run("ins", "graphical", 6, metadata))

    mixed_paragraph = etree.SubElement(body, w_tag("p"))
    mixed_paragraph.append(build_plain_run("Keep "))
    mixed_paragraph.append(build_revision_run("del", "validation", 7, metadata))
    mixed_paragraph.append(build_revision_run("ins", "application", 8, metadata))

    settings_root = etree.Element(w_tag("settings"), nsmap={"w": NSMAP["w"]})
    etree.SubElement(settings_root, w_tag("trackRevisions"))

    with ZipFile(path, "w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr(
            "word/document.xml",
            etree.tostring(document_root, xml_declaration=True, encoding="UTF-8", standalone="yes"),
        )
        zip_file.writestr(
            "word/settings.xml",
            etree.tostring(settings_root, xml_declaration=True, encoding="UTF-8", standalone="yes"),
        )
