"""Shared test helpers: isolated application home, sample-data access and a tiny PDF writer."""
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

import config  # noqa: E402
import evaluation as ev  # noqa: E402
from database import connect  # noqa: E402


class IsolatedHome(unittest.TestCase):
    """Every test gets a fresh RRT_HOME, so the real data/ folder is never touched."""

    def setUp(self):
        self._old = os.environ.get("RRT_HOME")
        self.home = Path(tempfile.mkdtemp(prefix="rrt_test_"))
        os.environ["RRT_HOME"] = str(self.home)
        config.ensure_dirs()
        self.conn = connect()

    def tearDown(self):
        self.conn.close()
        ev._close_file_logger()
        if self._old is None:
            os.environ.pop("RRT_HOME", None)
        else:
            os.environ["RRT_HOME"] = self._old
        shutil.rmtree(self.home, ignore_errors=True)

    def count(self, table, pid=None):
        if pid:
            return self.conn.execute("SELECT COUNT(*) FROM %s WHERE paper_id=?" % table, (pid,)).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]


def sample(name):
    return str(PROJECT / "sample_data" / name)


def make_pdf(path, pages):
    """Write a minimal text PDF. pages is a list of pages, each a list of text lines."""
    objs = []

    def add(body):
        objs.append(body)
        return len(objs)

    add("<< /Type /Catalog /Pages 2 0 R >>")
    add(None)  # pages placeholder
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    kids = []
    for lines in pages:
        stream = "BT /F1 11 Tf 50 750 Td 14 TL\n" + "".join("(%s) Tj T*\n" % ln.replace("(", "[").replace(")", "]") for ln in lines) + "ET"
        c = add("<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
        p = add("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents %d 0 R /Resources << /Font << /F1 %d 0 R >> >> >>" % (c, font))
        kids.append(p)
    objs[1] = "<< /Type /Pages /Count %d /Kids [%s] >>" % (len(kids), " ".join("%d 0 R" % k for k in kids))
    out = "%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out.encode("latin-1")))
        out += "%d 0 obj\n%s\nendobj\n" % (i, body)
    xref = len(out.encode("latin-1"))
    out += "xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += "".join("%010d 00000 n \n" % o for o in offsets)
    out += "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    Path(path).write_bytes(out.encode("latin-1"))
    return path
