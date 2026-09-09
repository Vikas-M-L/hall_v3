"""Exercise the actual Checker/Autopsy render paths with supplied model evidence."""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]


def test_checker_banner_export_and_autopsy_agree(monkeypatch):
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import models.clip_wrapper as clip_module
    import models.gemini_wrapper as gemini_module
    import studio_lib

    class FakeClip:
        def __init__(self, cfg):
            pass

        def encode_regions(self, image, **kwargs):
            return object()

        def evidence(self, text, regions):
            return 1 - 0.2640286921100182, None, True

        def similarity(self, text, regions):
            return 1 - 0.43983436138792464

        def suggested_window(self):
            return None

        def observed_cosine_range(self):
            return None

    class FakeGemini:
        def verify(self, image, text):
            return {"supported": False, "raw": "NO. There are only two cats, sleeping, not eating."}

    upload = io.BytesIO((ROOT / "assets/red_square.jpg").read_bytes())
    upload.name = "fixture.jpg"
    monkeypatch.setattr(st, "file_uploader", lambda *a, **kw: upload)
    monkeypatch.setattr(clip_module, "CLIPWrapper", FakeClip)
    monkeypatch.setattr(gemini_module, "GeminiWrapper", FakeGemini)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(str(ROOT / "app.py"))
        app.session_state["claim_text"] = "5 cats are eating"
        app.run(timeout=30)
        next(b for b in app.button if b.label == "Run verification").click().run(timeout=30)
        assert not app.exception
        saved = app.session_state["last_run"]
        assert saved["out"]["final_verdict"] == "contradicted"
        assert len(saved["out"]["claims"]) == 3
        counted = next(r for r in saved["out"]["claims"] if r["claim_type"] == "count")
        assert counted["diagnosis"]["mechanism"] == "M2"
        assert saved["out"]["claims"][0]["risk"] == pytest.approx(.35193152674897143)
        assert any(e.value == "CONTRADICTED" for e in app.error)
        assert any("verdict" in d.value.columns and d.value.iloc[0]["verdict"] == "contradicted" for d in app.dataframe)
        before = saved["out"]["events"]
        next(s for s in app.selectbox if s.label == "Inspect fact").select(1).run(timeout=30)
        assert not app.exception
        assert app.session_state["last_run"]["out"]["events"] == before

        monkeypatch.setattr(studio_lib, "get_clip", lambda *a: FakeClip({}))
        autopsy = AppTest.from_file(str(ROOT / "pages/1_Autopsy.py"))
        autopsy.session_state["last_run"] = saved
        autopsy.run(timeout=30)
        assert not autopsy.exception
        assert any("CONTRADICTED" in m.value for m in autopsy.markdown)
    finally:
        st.cache_resource.clear()
