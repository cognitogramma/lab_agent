"""Tests for the figure ledger, concentrated on the things that cost days.

    python tools/figure_ledger/test_figure_ledger.py

Standard library only, and runnable without pytest, so this stays testable on a
machine where nothing has been installed.
"""

from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parent))

import images  # noqa: E402
import ledger  # noqa: E402
import serve  # noqa: E402


class Netpbm(unittest.TestCase):
    """The extensions lie. Identification is by content, always."""

    def test_a_ppm_named_png_is_identified_as_netpbm(self):
        data = b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0])
        found = images.identify(data)

        self.assertIn("PPM", found.format)
        self.assertTrue(found.needs_conversion)
        self.assertTrue(found.displayable)

    def test_p6_binary_pixmap_decodes(self):
        data = b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0])
        width, height, samples, channels = images.decode_netpbm(data)

        self.assertEqual((width, height, channels), (2, 1, 3))
        self.assertEqual(samples, bytes([255, 0, 0, 0, 255, 0]))

    def test_p5_binary_graymap_decodes(self):
        data = b"P5\n2 2\n255\n" + bytes([0, 85, 170, 255])
        width, height, samples, channels = images.decode_netpbm(data)

        self.assertEqual((width, height, channels), (2, 2, 1))
        self.assertEqual(samples, bytes([0, 85, 170, 255]))

    def test_p4_binary_bitmap_decodes_and_inverts(self):
        """In PBM 1 means black, which is inverted from every other format."""
        data = b"P4\n8 1\n" + bytes([0b10000000])
        _w, _h, samples, _c = images.decode_netpbm(data)

        self.assertEqual(samples[0], 0)      # bit set -> black
        self.assertEqual(samples[1], 255)    # bit clear -> white

    def test_p3_plain_pixmap_decodes(self):
        data = b"P3\n2 1\n255\n255 0 0  0 255 0\n"
        _w, _h, samples, channels = images.decode_netpbm(data)

        self.assertEqual(channels, 3)
        self.assertEqual(samples, bytes([255, 0, 0, 0, 255, 0]))

    def test_p1_plain_bitmap_decodes(self):
        _w, _h, samples, _c = images.decode_netpbm(b"P1\n2 1\n1 0\n")
        self.assertEqual(samples, bytes([0, 255]))

    def test_comments_anywhere_in_the_header_are_skipped(self):
        """Netpbm allows comments between the width and the height."""
        data = b"P6\n# from pdfimages\n2 # width\n1\n255\n" + bytes([1, 2, 3, 4, 5, 6])
        width, height, _s, _c = images.decode_netpbm(data)

        self.assertEqual((width, height), (2, 1))

    def test_sixteen_bit_samples_are_rescaled(self):
        data = b"P5\n1 1\n65535\n" + bytes([255, 255])
        _w, _h, samples, _c = images.decode_netpbm(data)

        self.assertEqual(samples, bytes([255]))

    def test_maxval_below_255_is_rescaled(self):
        data = b"P5\n2 1\n15\n" + bytes([0, 15])
        _w, _h, samples, _c = images.decode_netpbm(data)

        self.assertEqual(samples, bytes([0, 255]))

    def test_truncated_data_is_refused_with_the_counts(self):
        with self.assertRaises(ValueError) as error:
            images.decode_netpbm(b"P6\n4 4\n255\n" + b"\x00" * 5)
        self.assertIn("short", str(error.exception))

    def test_a_converted_netpbm_comes_out_as_a_real_png(self):
        data = b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0])
        body, content_type, found = images.to_displayable(data)

        self.assertEqual(content_type, "image/png")
        self.assertEqual(body[:8], b"\x89PNG\r\n\x1a\n")
        self.assertIn("PPM", found.format)


class Identification(unittest.TestCase):
    def test_real_formats_are_named(self):
        cases = {
            b"\x89PNG\r\n\x1a\n" + b"\x00" * 8: "PNG",
            b"\xff\xd8\xff\xe0rest": "JPEG",
            b"GIF89a-rest": "GIF",
            b"BM....rest": "BMP",
            b"RIFF\x00\x00\x00\x00WEBPrest": "WebP",
        }
        for data, expected in cases.items():
            self.assertEqual(images.identify(data).format, expected)

    def test_an_undisplayable_format_is_named_rather_than_refused_blankly(self):
        """'This is TIFF' says what to do; 'cannot display' says nothing."""
        with self.assertRaises(ValueError) as error:
            images.to_displayable(b"II*\x00" + b"\x00" * 20)

        message = str(error.exception)
        self.assertIn("TIFF", message)
        self.assertIn("convert it first", message)

    def test_a_pdf_is_named_too(self):
        self.assertEqual(images.identify(b"%PDF-1.7\n").format, "PDF")

    def test_something_starting_P5_that_is_not_netpbm_is_not_claimed(self):
        self.assertNotIn("Netpbm", images.identify(b"P5XYZ").format)


class Dates(unittest.TestCase):
    """Archive material often carries only a year. Precision is recorded."""

    def test_the_three_accepted_precisions(self):
        for value, precision in (
            ("2021", "year"),
            ("2021-06", "month"),
            ("2021-06-14", "day"),
        ):
            checked = ledger.check_date(value)
            self.assertTrue(checked.ok, checked.error)
            self.assertEqual(checked.precision, precision)

    def test_an_impossible_day_is_refused(self):
        checked = ledger.check_date("2021-02-30")
        self.assertFalse(checked.ok)
        self.assertIn("28 days", checked.error)

    def test_a_leap_day_is_accepted_in_a_leap_year_only(self):
        self.assertTrue(ledger.check_date("2020-02-29").ok)
        self.assertFalse(ledger.check_date("2021-02-29").ok)

    def test_a_bad_month_is_refused(self):
        self.assertFalse(ledger.check_date("2021-13").ok)

    def test_blank_is_allowed(self):
        checked = ledger.check_date("")
        self.assertTrue(checked.ok)
        self.assertEqual(checked.precision, "none")

    def test_a_shape_that_is_not_a_date_is_refused(self):
        for value in ("June 2021", "21-06", "2021/06", "2021-6-1"):
            self.assertFalse(ledger.check_date(value).ok, value)


class Records(unittest.TestCase):
    def test_normalise_derives_the_precision(self):
        record = ledger.normalise({"date": "2021-06"}, "a.png")
        self.assertEqual(record["datePrecision"], "month")

    def test_an_impossible_date_refuses_the_whole_record(self):
        with self.assertRaises(ledger.ValidationError):
            ledger.normalise({"date": "2021-02-30"}, "a.png")

    def test_protein_class_is_dropped_for_a_non_protein(self):
        """Otherwise a search for globular proteins returns RNA."""
        record = ledger.normalise(
            {"sampleType": "RNA", "proteinClass": "Globular"}, "a.png"
        )
        self.assertEqual(record["proteinClass"], "")

    def test_protein_class_survives_for_a_protein(self):
        record = ledger.normalise(
            {"sampleType": "Protein", "proteinClass": "Globular"}, "a.png"
        )
        self.assertEqual(record["proteinClass"], "Globular")

    def test_unknown_fields_are_dropped(self):
        record = ledger.normalise({"whatever": "x"}, "a.png")
        self.assertNotIn("whatever", record)

    def test_logged_needs_a_method_a_sample_type_and_a_goal(self):
        self.assertFalse(ledger.is_logged({"methods": ["SLS"]}))
        self.assertFalse(ledger.is_logged({"methods": ["SLS"], "sampleType": "RNA"}))
        self.assertTrue(
            ledger.is_logged(
                {"methods": ["SLS"], "sampleType": "RNA", "goals": ["Kinetics"]}
            )
        )


class Storage(unittest.TestCase):
    def test_an_existing_ledger_is_resumed_not_replaced(self):
        """The 13 figures already labelled must survive a restart."""
        existing = [
            {
                "image": "acsomega_fig01_p04.png",
                "filename": "acsomega_fig01_p04.png",
                "date": "2021",
                "dateKind": "Published",
                "methods": ["SLS"],
                "measured": ["Ir"],
                "sampleType": "RNA",
                "goals": ["Dialysis", "Kinetics"],
                "goalDetails": ["Dialysis: Guanidine-HCl (Gdn)"],
                "notes": "Aggregation window.",
                "datePrecision": "year",
            }
        ]
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "ledger.json").write_text(json.dumps(existing), encoding="utf-8")

            loaded = ledger.load(folder)
            self.assertIn("acsomega_fig01_p04.png", loaded)
            record = loaded["acsomega_fig01_p04.png"]
            self.assertEqual(record["organization"], "")   # absent field filled in
            self.assertEqual(record["methods"], ["SLS"])
            self.assertEqual(record["notes"], "Aggregation window.")

    def test_a_corrupt_ledger_raises_rather_than_being_overwritten(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "ledger.json").write_text("{not json", encoding="utf-8")

            with self.assertRaises(ledger.ValidationError) as error:
                ledger.load(folder)
            self.assertIn("not being overwritten", str(error.exception))

    def test_save_leaves_no_temporary_files_behind(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            ledger.save(folder, {"a.png": ledger.blank_record("a.png")})

            leftovers = [p.name for p in folder.iterdir() if p.name != "ledger.json"]
            self.assertEqual(leftovers, [])

    def test_a_saved_ledger_round_trips(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            record = ledger.normalise(
                {"date": "2021", "methods": ["SLS"], "sampleType": "RNA"}, "a.png"
            )
            ledger.save(folder, {"a.png": record})

            loaded = ledger.load(folder)
            self.assertEqual(loaded["a.png"]["methods"], ["SLS"])
            self.assertEqual(loaded["a.png"]["datePrecision"], "year")

    def test_csv_export_joins_lists_without_breaking_the_delimiter(self):
        record = ledger.normalise(
            {"methods": ["DLS", "SLS"], "goalDetails": ["Kinetics: Aggregation"]},
            "a.png",
        )
        text = ledger.to_csv({"a.png": record})

        self.assertIn("DLS; SLS", text)
        self.assertIn("Kinetics: Aggregation", text)
        self.assertEqual(text.splitlines()[0].split(",")[0], "image")


class Serving(unittest.TestCase):
    """The HTTP surface: traversal blocked, state injected, formats named."""

    def setUp(self):
        self._directory = TemporaryDirectory()
        self.folder = Path(self._directory.name)
        (self.folder / "sub").mkdir()
        # A Netpbm file wearing a .png extension -- the exact trap.
        (self.folder / "liar.png").write_bytes(
            b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0])
        )
        (self.folder / "sub" / "real.tif").write_bytes(b"II*\x00" + b"\x00" * 20)
        self.store = serve.Store(self.folder)

    def tearDown(self):
        self._directory.cleanup()

    def test_the_scan_recurses_and_identifies_by_content(self):
        found = {entry["image"]: entry for entry in self.store.images}

        self.assertIn("liar.png", found)
        self.assertIn("sub/real.tif", found)
        self.assertIn("PPM", found["liar.png"]["format"])
        self.assertTrue(found["liar.png"]["displayable"])
        self.assertIn("TIFF", found["sub/real.tif"]["format"])
        self.assertFalse(found["sub/real.tif"]["displayable"])

    def test_path_traversal_is_refused(self):
        for attempt in ("../secret", "../../etc/passwd", "sub/../../outside"):
            with self.assertRaises((PermissionError, FileNotFoundError), msg=attempt):
                self.store.resolve(attempt)

    def test_a_real_file_inside_the_folder_resolves(self):
        self.assertTrue(self.store.resolve("liar.png").is_file())

    def test_the_page_arrives_with_its_data_already_in_it(self):
        """A page that renders empty and fills in later looks broken."""
        page = serve.build_page(self.store).decode("utf-8")

        self.assertNotIn("__INITIAL_STATE__", page)
        self.assertIn("liar.png", page)
        self.assertIn('"images"', page)
        self.assertIn('"records"', page)

    def test_a_script_closing_tag_in_a_filename_cannot_break_out(self):
        page = serve.build_page(self.store).decode("utf-8")
        self.assertNotIn("</script>", page.split("<script>")[1].split("renderList")[0])

    def test_saving_writes_the_ledger_to_disk(self):
        self.store.put("liar.png", {"date": "2021", "methods": ["SLS"]})

        written = json.loads((self.folder / "ledger.json").read_text(encoding="utf-8"))
        self.assertEqual(written[0]["image"], "liar.png")
        self.assertEqual(written[0]["datePrecision"], "year")

    def test_saving_an_impossible_date_is_refused(self):
        with self.assertRaises(ledger.ValidationError):
            self.store.put("liar.png", {"date": "2021-02-30"})


class LiveServer(unittest.TestCase):
    """One end-to-end pass over real HTTP, including the Netpbm conversion."""

    def setUp(self):
        self._directory = TemporaryDirectory()
        self.folder = Path(self._directory.name)
        (self.folder / "liar.png").write_bytes(
            b"P6\n2 1\n255\n" + bytes([255, 0, 0, 0, 255, 0])
        )
        (self.folder / "nope.tif").write_bytes(b"MM\x00*" + b"\x00" * 20)

        serve.Handler.store = serve.Store(self.folder)
        self.server = serve.Server(("127.0.0.1", 0), serve.Handler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self._directory.cleanup()

    def get(self, path):
        return urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5)

    def test_the_index_serves_a_populated_page(self):
        body = self.get("/").read().decode("utf-8")
        self.assertIn("liar.png", body)
        self.assertIn("Copy previous", body)

    def test_a_netpbm_file_is_served_as_a_png(self):
        response = self.get("/image/liar.png")
        self.assertEqual(response.headers["Content-Type"], "image/png")
        self.assertEqual(response.read()[:8], b"\x89PNG\r\n\x1a\n")

    def test_an_undisplayable_file_reports_its_format(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get("/image/nope.tif")
        self.assertIn("TIFF", json.loads(error.exception.read())["error"])

    def test_traversal_over_http_is_refused(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            self.get("/image/../../../etc/passwd")
        self.assertIn(error.exception.code, (403, 404))

    def test_save_then_csv_export(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/save",
            data=json.dumps(
                {
                    "image": "liar.png",
                    "record": {
                        "date": "2021-06",
                        "methods": ["SLS"],
                        "sampleType": "RNA",
                        "goals": ["Kinetics"],
                    },
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        saved = json.loads(urllib.request.urlopen(request, timeout=5).read())
        self.assertTrue(saved["saved"])
        self.assertEqual(saved["record"]["datePrecision"], "month")

        csv_text = self.get("/export.csv").read().decode("utf-8")
        self.assertIn("liar.png", csv_text)
        self.assertIn("SLS", csv_text)
        self.assertTrue((self.folder / "ledger.json").is_file())

    def test_a_bad_date_over_http_returns_the_reason(self):
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/save",
            data=json.dumps(
                {"image": "liar.png", "record": {"date": "2021-02-30"}}
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request, timeout=5)
        self.assertIn("28 days", json.loads(error.exception.read())["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
