"""Paso 3b: PINN multicondición (solo convección) con validación por condición y por nivel de factor.
Red MLP 4->64->64->64->1 sobre (x, t, r, f); h = exp(a0 + a.z) es entrenable; pérdida = datos + lp*ODE + lb*borde.
Salida: out/preds_pinn.pkl  {'pinn': {(tipo, split): {imagen: perfil(101)}}}
Uso: python 03b_pinn.py [--rapido]   (--rapido: 200 pasos, 3 particiones, solo para probar)"""
import sys, time, pickle, numpy as np, torch
from multiprocessing import Pool
from datos_modelos import *

QUICK = '--rapido' in sys.argv
STEPS = 200 if QUICK else 2500
LP, LB, SC, X0 = 1.0, 10.0, 30.0, 0.01          # peso física, peso borde, escala de salida (°C), cm->m
torch.set_num_threads(1)

class Net(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.m = torch.nn.Sequential(torch.nn.Linear(4, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64), torch.nn.Tanh(),
                                  torch.nn.Linear(64, 64), torch.nn.Tanh(), torch.nn.Linear(64, 1))
        s.a = torch.nn.Parameter(torch.tensor([np.log(120.), 0., 0., 0.], dtype=torch.float64))   # ln h = a0 + a.z
    def T(s, x, z, b): return b + SC * s.m(torch.cat([x / 5.0 * 2 - 1, z], 1))
    def h(s, z): return torch.exp(s.a[0] + z @ s.a[1:])

def fit(tr, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); tr = np.array(tr)
    ii, xs, ys = [], [], []
    for i in tr:
        k = np.where(M[i])[0][::2]; ii += [i] * len(k); xs.append(GRID[k]); ys.append(Pm[i, k])
    ii = np.array(ii); x = torch.tensor(np.concatenate(xs))[:, None]; y = torch.tensor(np.concatenate(ys))[:, None]
    z = torch.tensor(ZC[ii]); b = torch.tensor(BG[ii])[:, None]
    i0 = np.array([i for i in tr if M[i, 0]]); z0 = torch.tensor(ZC[i0]); b0 = torch.tensor(BG[i0])[:, None]; y0 = torch.tensor(Pm[i0, 0])[:, None]
    nc = len(tr) * 40; ci = tr[rng.integers(0, len(tr), nc)]
    cx = torch.tensor(rng.uniform(0.2, 5, nc))[:, None].requires_grad_(True)
    cz = torch.tensor(ZC[ci]); cb = torch.tensor(BG[ci])[:, None]; cD = torch.tensor(DM[ci])[:, None]; cv = torch.tensor(V[ci])[:, None]
    net = Net().double(); opt = torch.optim.Adam(net.parameters(), lr=4e-3); sch = torch.optim.lr_scheduler.StepLR(opt, 800, 0.4)
    for it in range(STEPS):
        opt.zero_grad()
        Ld = torch.mean(((net.T(x, z, b) - y) / 10) ** 2)
        Lb = torch.mean(((net.T(torch.zeros_like(y0), z0, b0) - y0) / 10) ** 2)
        Tp = net.T(cx, cz, cb); dT = torch.autograd.grad(Tp.sum(), cx, create_graph=True)[0]
        Rr = RHO * CP * cv * dT / X0 + 4 * net.h(cz)[:, None] / cD * (Tp - cb)              # residuo de la ecuación (1)
        Lp = torch.mean((Rr / (RHO * CP * cv / X0 * 10)) ** 2)
        (Ld + LB * Lb + LP * Lp).backward(); opt.step(); sch.step()
    return net

def pred(net, i):
    with torch.no_grad():
        n = len(GRID)
        T = net.T(torch.tensor(GRID)[:, None], torch.tensor(ZC[i])[None].repeat(n, 1), torch.full((n, 1), float(BG[i]), dtype=torch.float64))
    return post(T.numpy()[:, 0])

def task(a):
    key, tr, te = a; net = fit(tr)
    return key, {i: pred(net, i) for i in te}, float(net.h(torch.tensor(ZC[te[0]])[None]).item())

if __name__ == '__main__':
    sp = splits(); t0 = time.time()
    keys = list(sp)[:2] + [k for k in sp if k[0] == 'extrap'][:1] if QUICK else list(sp)
    with Pool(2) as pool: res = pool.map(task, [(k, *sp[k]) for k in keys], chunksize=1)
    P = {'pinn': {k: d for k, d, h in res}}; H = {k: h for k, d, h in res}
    pickle.dump(P, open(out('preds_pinn_quick.pkl' if QUICK else 'preds_pinn.pkl'), 'wb')); pickle.dump(H, open(out('h_pinn.pkl'), 'wb'))
    print('listo en %.0f s' % (time.time() - t0))
