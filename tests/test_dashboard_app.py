"""Headless smoke test of the dashboard (Streamlit AppTest): every shape x each of its VSD recordings renders without
error (including the across-VSD comparison grid), and shapes without a recording show a message.
Run: scripts/py -m pytest tests/test_dashboard_app.py  (~3 min; skipped without schira_dashboard/cache/gandalf_maps.npz)"""
from pathlib import Path
import pytest
DASH = Path(__file__).resolve().parents[1] / "schira_dashboard"
APP = str(DASH / "schira_dash_app.py")
pytestmark = pytest.mark.skipif(not (DASH / "cache/gandalf_maps.npz").is_file(),
                                reason="dashboard cache not built (scripts/py schira_dashboard/prepare_data.py)")


def test_dashboard_app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=180).run()
    assert not at.exception, at.exception
    n = n_none = n_fig12 = 0
    for u in range(len(at.selectbox(key="shape").options)):
        at.selectbox(key="shape").select_index(u).run()
        assert not at.exception, (u, [e.value for e in at.exception])
        vs = [s for s in at.selectbox if s.key == "vsd"]
        if not vs:
            assert at.info, u; n_none += 1; continue
        opts = list(vs[0].options)
        idx = range(len(opts))
        if len(opts) > 20:   # thesis grids: test a thesis plate of each day, a recording of each day, a blank, the last
            pick = lambda f: next(i for i, o in enumerate(opts) if f(o))
            idx = sorted({pick(lambda o: o.startswith("THESIS") and "20/11/18" in o), pick(lambda o: o.startswith("THESIS") and "10/07/18" in o),
                          pick(lambda o: o.startswith("201118")), pick(lambda o: o.startswith("100718")),
                          pick(lambda o: "blank trials" in o), len(opts) - 1})
        for v in idx:
            at.selectbox(key="vsd").select_index(v).run(); n += 1
            assert not at.exception, (u, v, [e.value for e in at.exception])
            joanna = [b for b in at.button if str(b.label).startswith("Use Joanna")]
            if joanna:
                joanna[0].click().run(); n_fig12 += 1
                assert not at.exception, (u, v, "joanna button", [e.value for e in at.exception])
    at.selectbox(key="shape").select_index(0).run()
    at.radio(key="mode").set_value("rigid").run()
    next(b for b in at.button if str(b.label).startswith("Optimise")).click().run()
    assert not at.exception, [e.value for e in at.exception]

    # stretched Schira + Joanna's 2-anchor registration on a 20/11/18 and a 10/07/18 recording
    def pick_vsd(day):
        for u in range(len(at.selectbox(key="shape").options)):
            at.selectbox(key="shape").select_index(u).run()
            vs = [s for s in at.selectbox if s.key == "vsd"]
            if vs and any(str(o).startswith(day) for o in vs[0].options):
                at.selectbox(key="vsd").select_index(next(i for i, o in enumerate(vs[0].options) if str(o).startswith(day))).run()
                return
        raise AssertionError(day)
    at.radio(key="model").set_value("joint").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.radio(key="mode").value == "anchors" and abs(at.number_input(key="r").value - 0.539) < 1e-9
    for day in ("201118", "100718"):
        pick_vsd(day)
        assert not at.exception, (day, [e.value for e in at.exception])
        assert any("Mean distance (non-anchor)" in m.label for m in at.metric), day
    assert at.radio(key="model").value == "joint" and at.number_input(key="a").disabled
    next(b for b in at.button if str(b.label).startswith("Optimise")).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert any("r 0.5" in s.value for s in at.success), [s.value for s in at.success]
    assert at.radio(key="model").value == "free" and not at.number_input(key="a").disabled
    next(b for b in at.button if str(b.label) == "Run joint fit").click().run()
    assert not at.exception, [e.value for e in at.exception]
    msg = [s.value for s in at.success if s.value.startswith("Joint fit")]
    assert msg, [s.value for s in at.success]
    print(msg[0])
    print(f"dashboard OK: {n} shape/VSD views rendered ({n_fig12} with the thesis Fig. 12 comparison), {n_none} shapes without a VSD, "
          "optimise button OK, anchor mode + stretch + joint fit OK")
