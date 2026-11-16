import io, contextlib, numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import run_100718 as R
# 10 distinct landmarks from opt-points-10-07-18.xlsx: (stim col, stim row) -> (cam col, cam row)
pts = [((21,26),(44,39)), ((14,14),(21,36)), ((14,38),(60,30)), ((35,26),(47,45)), ((21,14),(25,41)),
       ((21,37),(58,36)), ((21,9),(24,46)), ((21,43),(70,36)), ((4,26),(37,24)), ((38,26),(48,51))]
P = np.array([[s[1], s[0], 0, 0, c[1], c[0]] for s, c in pts])
R.use(P); print("xlsx landmarks 10/07/18 (n=10 distinct)"); R.models()
