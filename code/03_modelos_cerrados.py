"""Paso 3a: empírico, semi-físico, físico (y variantes) con validación por condición y por nivel de factor.
Salida: out/preds_cerrados.pkl  {modelo: {(tipo, split): {imagen: perfil_predicho(101)}}}
Uso:  python 03_modelos_cerrados.py [--rapido]   (--rapido: menos arranques, solo para probar)"""
import sys, time, pickle, numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
from datos_modelos import *

QUICK = '--rapido' in sys.argv
ONLY_PHYS = '--solo-fisicos' in sys.argv      # repite solo la familia física y la mezcla con lo ya guardado
NSTART = 3 if QUICK else 8
XG = GRID

# ------------------------------------------------------------------ 1. empírico (ridge por distancia)
def fit_empirical(tr, lam=0.05, xmin=0.0):
    B = np.zeros((len(XG), 4))
    for k in range(len(XG)):
        ok = [i for i in tr if M[i, k]]
        if len(ok) < 5: B[k] = B[k - 1]; continue
        X = X4[ok]; A = X.T @ X + lam * np.diag([0, 1, 1, 1]); B[k] = np.linalg.solve(A, X.T @ Pm[ok, k])
    return lambda i: post(X4[i] @ B.T)

# ------------------------------------------------------------------ 2. semi-físico  T = Tb + a e^{-x/L} + c
def semi_curve(p, z, bg):
    la, lL, lc = (p[0] + z @ p[1:4]), (p[4] + z @ p[5:8]), (p[8] + z @ p[9:12])
    a, L, c = np.exp(la), np.exp(lL), np.exp(lc)
    return bg[:, None] + a[:, None] * np.exp(-XG[None] / L[:, None]) + c[:, None]
def fit_semi(tr, lam=1.0, xmin=0.0, seed=0):
    tr = np.array(tr); Y = Pm[tr]; Mk = M[tr] & (XG[None] >= xmin)
    def res(p):
        r = (semi_curve(p, ZC[tr], BG[tr]) - Y)[Mk]
        return np.r_[r, np.sqrt(lam) * p[[1, 2, 3, 5, 6, 7, 9, 10, 11]]]
    rng = np.random.default_rng(seed); best = None
    for s in range(3):
        p0 = np.r_[np.log(40), 0, 0, 0, np.log(1.2), 0, 0, 0, np.log(3), 0, 0, 0] + (s > 0) * rng.normal(0, .3, 12)
        o = least_squares(res, p0, method='trf', max_nfev=300)
        if best is None or o.cost < best.cost: best = o
    p = best.x
    return lambda i: post(semi_curve(p, ZC[[i]], BG[[i]])[0])

# ------------------------------------------------------------------ 3. físico (balance de energía, 12 parámetros)
NSUB = 4; XF = np.linspace(0, 5, 100 * NSUB + 1); DX = (XF[1] - XF[0]) * 1e-2; XMID = 0.5 * (XF[1:] + XF[:-1])
PHYS_NAMES = ['lnA0', 'g', 'ln(d0/df-1)', 'ln lambda', 'ln Delta0', 'delta', 'kappa', 'ln ell', 'x0', 'a0', 'a1', 'p']
# g >= 0 (más ventilador, más convección) y delta <= 0 (el ventilador reduce el aire caliente): signos físicos
PLO = np.array([np.log(.3), 0., np.log(.01), np.log(.1), np.log(.1), -.95, -.9, np.log(.2), 0., 60., 0., .05])
PHI = np.array([np.log(100), .9, np.log(3), np.log(5), np.log(60), 0., .9, np.log(10), 1., 160., 4., 3.])
PINI = np.array([np.log(6), .2, np.log(.3), np.log(1.), np.log(8), -.3, .3, np.log(2.), .3, 100., 1., .3])
def phys_curve(p, idx, v_arr):
    idx = np.atleast_1d(idx); Tb = BG[idx][:, None]; f = F_C[idx][:, None]; t = T_C[idx][:, None]
    df = DM[idx][:, None]; vf = v_arr[idx][:, None]; Tn = TN[idx][:, None]
    d = df + df * np.exp(p[2]) * np.exp(-XMID[None] / np.exp(p[3]))                 # d(x): de d0 a df
    v = vf * df ** 2 / d ** 2                                                      # conservación de masa
    h = np.exp(p[0]) * (1 + p[1] * f) / np.sqrt(d)                                 # h = A d^-1/2
    k = 4 * h / (RHO * CP * d * v)                                                 # 1/m
    Tinf = Tb + np.exp(p[4]) * (1 + p[5] * f) * (1 + p[6] * t) * np.exp(-XMID[None] / np.exp(p[7]))
    e = np.where(XMID[None] < p[8], 1.0, np.exp(-k * DX))                          # parte plana hasta x0
    T = np.empty((len(idx), len(XF))); T[:, 0] = p[9] + p[10] * (Tn[:, 0] - 100)
    for j in range(len(XMID)): T[:, j + 1] = Tinf[:, j] + (T[:, j] - Tinf[:, j]) * e[:, j]
    eta = 1 - np.exp(-np.concatenate([d[:, :1], d], axis=1) * 1e3 / p[11])                          # mezcla en el píxel (fibra sub-píxel)
    Tapp = Tb + eta * (T - Tb)
    return Tapp[:, ::NSUB]
def fit_phys(tr, v_arr=V, xmin=0.0, seed=0, info=None):
    tr = np.array(tr); Mk = M[tr] & (XG[None] >= xmin); Y = Pm[tr]
    res = lambda p: (phys_curve(p, tr, v_arr) - Y)[Mk]
    rng = np.random.default_rng(seed); best = None; costs = []
    for s in range(NSTART):
        p0 = PINI.copy() if s == 0 else PLO + (PHI - PLO) * rng.uniform(.05, .95, len(PLO))
        try: o = least_squares(res, p0, bounds=(PLO, PHI), method='trf', max_nfev=120, x_scale=(PHI - PLO))
        except Exception: continue
        costs.append(o.cost)
        if best is None or o.cost < best.cost: best = o
    if info is not None:
        info['starts_at_best'] = int(np.sum(np.array(costs) < best.cost * 1.02)); info['n_starts'] = len(costs)
        span = PHI - PLO; info['at_bounds'] = int(np.sum((best.x - PLO < 1e-3 * span) | (PHI - best.x < 1e-3 * span))); info['params'] = best.x
    p = best.x
    return lambda i: post(phys_curve(p, [i], v_arr)[0])

# ------------------------------------------------------------------ 4. híbrido: empírico + predicción física como regresor
def fit_hybrid(tr, v_arr=V, seed=0):
    pf = fit_phys(tr, v_arr, seed=seed)
    PH = np.array([pf(i) for i in range(N)])             # predicción física (en entrenamiento: dentro de muestra)
    B = np.zeros((len(XG), 5))
    for k in range(len(XG)):
        ok = [i for i in tr if M[i, k]]
        X = np.c_[X4[ok], PH[ok, k]]; A = X.T @ X + 0.05 * np.diag([0, 1, 1, 1, 1]); B[k] = np.linalg.solve(A, X.T @ Pm[ok, k])
    return lambda i: post((np.c_[np.tile(X4[i], (len(XG), 1)), PH[i]] * B).sum(1))

MODELS = {
    'empirical': lambda tr: fit_empirical(tr),
    'semi': lambda tr: fit_semi(tr),
    'physics': lambda tr: fit_phys(tr),
    'hybrid': lambda tr: fit_hybrid(tr),
    'semi_x05': lambda tr: fit_semi(tr, xmin=0.5),
    'physics_x05': lambda tr: fit_phys(tr, xmin=0.5),
    'physics_effrpm': lambda tr: fit_phys(tr, v_arr=V_EFF),
}
def task(args):
    model, key, tr, te = args
    info = {}
    f = MODELS[model](tr)
    return model, key, {i: f(i) for i in te}

if __name__ == '__main__':
    sp = splits(); t0 = time.time()
    which = [m for m in MODELS if (not QUICK or m in ('empirical', 'semi', 'physics')) and (not ONLY_PHYS or m in ('physics', 'hybrid', 'physics_x05', 'physics_effrpm'))]
    keys = list(sp)[:4] + [k for k in sp if k[0] == 'extrap'][:1] if QUICK else list(sp)
    jobs = [(m, k, *sp[k]) for m in which for k in keys]
    with Pool(2) as pool: res = pool.map(task, jobs, chunksize=1)
    P = {m: {} for m in which}
    for m, k, d in res: P[m][k] = d
    if ONLY_PHYS:
        old = pickle.load(open(out('preds_cerrados.pkl'), 'rb')); old.update(P); P = old
    pickle.dump(P, open(out('preds_cerrados_quick.pkl' if QUICK else 'preds_cerrados.pkl'), 'wb'))
    print('listo en %.0f s' % (time.time() - t0))
