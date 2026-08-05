import csv
import hashlib
import tempfile
import unittest
from pathlib import Path

from phase3_data_collection.collectors import AnnotationRecord, AnnotationWriter, scrape_links
from phase3_data_collection.storage import create_media_manifest, write_csv


class DataCollectionTest(unittest.TestCase):
    def test_scrape_links_filters_and_resolves_urls(self):
        parser_fixture = '<a href="images/cat.jpg">cat</a><a href="/notes.txt">notes</a>'
        self.assertEqual(parser_fixture.count("href"), 2)
        # Exercise the parser through a local file URL without network access.
        with tempfile.TemporaryDirectory() as temporary:
            page = Path(temporary) / "index.html"
            page.write_text(parser_fixture, encoding="utf-8")
            links = scrape_links(page.as_uri(), extensions=(".jpg",))
            self.assertEqual(links, [f"{Path(temporary).as_uri()}/images/cat.jpg"])

    def test_annotations_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            annotation_path = Path(temporary) / "labels.csv"
            writer = AnnotationWriter(annotation_path)
            writer.append(AnnotationRecord("cat.jpg", "cat", "ana", "clear image"))
            with annotation_path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(rows[0]["label"], "cat")
            output = write_csv([{"id": 1, "label": "cat"}], Path(temporary) / "data.csv")
            self.assertTrue(output.exists())

    def test_media_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "media"
            root.mkdir()
            (root / "cat.jpg").write_bytes(b"image")
            (root / "clip.mp4").write_bytes(b"video")
            (root / "noise.wav").write_bytes(b"audio")
            manifest = create_media_manifest(root, Path(temporary) / "manifest.csv")
            text = manifest.read_text(encoding="utf-8")
            self.assertIn("image", text)
            self.assertIn("video", text)
            self.assertIn("audio", text)


if __name__ == "__main__":
    unittest.main()
