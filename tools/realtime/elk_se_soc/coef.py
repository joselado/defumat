import sys
import numpy as np
d = np.load(sys.argv[1]); t = d["times"]; j = d["current"][:, 2]; vol = float(d["volume"]); k = d["kappa"][:, 2]
for f in sys.argv[2:]:
    e = np.loadtxt(f); n = len(e); je = e[:, 3] / vol
    tt, jj, kk = t[:n], j[:n], k[:n]
    m = (tt > 60) & (tt < 175)
    diff = je - jj[:n]
    c = np.dot(diff[m], kk[m]) / np.dot(kk[m], kk[m])
    resid = diff[m] - c * kk[m]
    post = (tt >= 190)
    print(f, "t_last", tt[-1], "coef of kappa in (Je-Jd)", c, "x Omega =", c * vol,
          "| resid/max|Jd| in pulse", np.abs(resid).max() / np.abs(jj[m]).max(),
          "| relL2 after pulse", (np.linalg.norm(diff[post]) / np.linalg.norm(jj[post])) if post.any() else None,
          "| relL2 in pulse", np.linalg.norm(diff[m]) / np.linalg.norm(jj[m]),
          "| relL2 in pulse after removing c*kappa", np.linalg.norm(resid) / np.linalg.norm(jj[m]))
