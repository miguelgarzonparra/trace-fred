"""Paso 4: métricas de validación, bootstrap, tablas LaTeX y figuras de la sección Results.
Lee out/preds_cerrados.pkl y out/preds_pinn.pkl (si existe). Escribe out/tab_loco.tex, tab_extrap.tex,
out/results_summary.json, fig_pred y fig_loco_scatter."""
import json, pickle, os, importlib.util, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from datos_modelos import *

P = pickle.load(open(out('preds_cerrados.pkl'), 'rb'))
if os.path.exists(out('preds_pinn.pkl')): P.update(pickle.load(open(out('preds_pinn.pkl'), 'rb')))
NAME = {'empirical': 'Empirical', 'semi': 'Semi-physical', 'physics': 'Physics-based', 'pinn': 'PINN', 'hybrid': 'Hybrid',
        'semi_x05': 'Semi-physical ($x\\geq0.5$)', 'physics_x05': 'Physics-based ($x\\geq0.5$)', 'physics_effrpm': 'Physics-based (eff. speed)'}
MAIN = [m for m in ('empirical', 'semi', 'physics', 'pinn') if m in P]
KEY_L = [k for k in splits() if k[0] == 'loco']; KEY_X = [k for k in splits() if k[0] == 'extrap']
S = {}

def errs(m, key, xmin=0.0):
    e = []
    for i, p in P[m][key].items():
        mk = M[i] & (GRID >= xmin); e.append((p - Pm[i])[mk])
    e = np.concatenate(e); return e
def rmse(e): return float(np.sqrt(np.mean(e ** 2)))
def pooled(m, keys, xmin=0.0): return rmse(np.concatenate([errs(m, k, xmin) for k in keys if k in P[m]]))

# ---- LOCO ------------------------------------------------------------------------------------------------------
SS = {m: np.array([np.sum(errs(m, k) ** 2) for k in KEY_L if k in P[m]]) for m in P}
NN = {m: np.array([len(errs(m, k)) for k in KEY_L if k in P[m]]) for m in P}
rng = np.random.default_rng(0); B = rng.integers(0, len(KEY_L), (2000, len(KEY_L)))
tab = []
for m in P:
    if len(SS[m]) != len(KEY_L): continue
    r = np.sqrt(SS[m].sum() / NN[m].sum()); ref = 'empirical'
    d = np.sqrt(SS[m][B].sum(1) / NN[m][B].sum(1)) - np.sqrt(SS[ref][B].sum(1) / NN[ref][B].sum(1))
    pc = np.sqrt(SS[m] / NN[m]); pe = np.sqrt(SS[ref] / NN[ref])
    tab.append(dict(model=m, rmse_loco=r, rmse_x05=pooled(m, KEY_L, 0.5), d_vs_emp=r - np.sqrt(SS[ref].sum() / NN[ref].sum()),
                    d_lo=np.percentile(d, 2.5), d_hi=np.percentile(d, 97.5), better_than_emp=int(np.sum(pc < pe)), n_cond=len(pc),
                    rmse_cond_median=float(np.median(pc)), p_boot=float(2 * min((d <= 0).mean(), (d >= 0).mean()))))
T = pd.DataFrame(tab).set_index('model'); T.to_csv(out('tab_loco.csv')); print(T.round(3))
S['loco'] = T.round(4).to_dict('index')

# ---- extrapolación ------------------------------------------------------------------------------------------------
X = {m: {k[1]: rmse(errs(m, k)) for k in KEY_X if k in P[m]} for m in P}
TX = pd.DataFrame(X); TX.to_csv(out('tab_extrap.csv')); print(TX.round(2)); S['extrap'] = TX.round(3).to_dict()
S['extrap_mean'] = {m: float(np.mean(list(X[m].values()))) for m in P if len(X[m]) == len(KEY_X)}

# ---- error por zona de distancia --------------------------------------------------------------------------------------
bins = [(0, .5), (.5, 1.5), (1.5, 3), (3, 5.01)]; ZB = {}
for m in P:
    row = {}
    for lo, hi in bins:
        e = np.concatenate([(p - Pm[i])[M[i] & (GRID >= lo) & (GRID < hi)] for k in KEY_L if k in P[m] for i, p in P[m][k].items()])
        row[f'{lo}-{min(hi,5)}'] = rmse(e)
    ZB[m] = row
pd.DataFrame(ZB).T.to_csv(out('tab_rmse_by_x.csv')); S['by_x'] = ZB
S['bias_loco'] = {m: float(np.mean(np.concatenate([errs(m, k) for k in KEY_L if k in P[m]]))) for m in P}

# ---- ajuste del modelo físico con todos los datos (parámetros y diagnóstico de arranques) --------------------------------------
spec = importlib.util.spec_from_file_location('cerr', os.path.join(os.path.dirname(os.path.abspath(__file__)), '03_modelos_cerrados.py'))
cer = importlib.util.module_from_spec(spec); spec.loader.exec_module(cer)
info = {}; pf = cer.fit_phys(list(range(N)), info=info)
S['physics_full'] = dict(starts_at_best=info['starts_at_best'], n_starts=info['n_starts'], at_bounds=info['at_bounds'],
                         params={n: float(v) for n, v in zip(cer.PHYS_NAMES, info['params'])})
full_rmse = rmse(np.concatenate([(pf(i) - Pm[i])[M[i]] for i in range(N)])); S['physics_full']['rmse_in_sample'] = full_rmse
json.dump(S, open(out('results_summary.json'), 'w'), indent=1)

# ---- tablas LaTeX -----------------------------------------------------------------------------------------------------------
L = ['\\begin{tabular}{lcccc}', '\\hline', 'Model & RMSE & $\\Delta$ vs.\\ emp. & 95\\,\\% CI & Better in \\\\',
     ' & ($^\\circ$C) & ($^\\circ$C) & ($^\\circ$C) & cond. \\\\', '\\hline']
for m in [x for x in ('empirical', 'semi', 'physics', 'pinn', 'hybrid') if x in T.index]:
    r = T.loc[m]
    if m == 'empirical': L.append(f"{NAME[m]} & {r.rmse_loco:.2f} & -- & -- & -- \\\\")
    else: L.append(f"{NAME[m]} & {r.rmse_loco:.2f} & {r.d_vs_emp:+.2f} & [{r.d_lo:+.2f}, {r.d_hi:+.2f}] & {int(r.better_than_emp)}/{int(r.n_cond)} \\\\")
L += ['\\hline', '\\end{tabular}']; open(out('tab_loco.tex'), 'w').write('\n'.join(L))
cols = [m for m in ('empirical', 'semi', 'physics', 'pinn') if m in TX.columns]
L = ['\\begin{tabular}{l' + 'c' * len(cols) + '}', '\\hline', 'Held-out level & ' + ' & '.join(NAME[c] for c in cols) + ' \\\\', '\\hline']
for lv in TX.index:
    L.append(lv.replace('temp=', 'Nozzle ').replace('rpm=', 'Speed ').replace('fan=', 'Fans ') + ' & ' + ' & '.join(f'{TX.loc[lv,c]:.2f}' for c in cols) + ' \\\\')
L.append('Mean & ' + ' & '.join(f'{TX[c].mean():.2f}' for c in cols) + ' \\\\'); L += ['\\hline', '\\end{tabular}']
open(out('tab_extrap.tex'), 'w').write('\n'.join(L))

# ---- figuras ----------------------------------------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 7, 'axes.linewidth': .6})
COL = {'empirical': '#1f77b4', 'semi': '#2ca02c', 'physics': '#ff7f0e', 'pinn': '#9467bd'}
fig, ax = plt.subplots(figsize=(3.4, 3.1))
pe = np.sqrt(SS['empirical'] / NN['empirical']); lim = 0
for m in [x for x in ('semi', 'physics', 'pinn') if x in P and len(SS[x]) == len(KEY_L)]:
    pm_ = np.sqrt(SS[m] / NN[m]); ax.plot(pe, pm_, 'o', ms=3, c=COL[m], alpha=.8, label=NAME[m].replace('$', '')); lim = max(lim, pm_.max())
lim = max(lim, pe.max()) * 1.05; ax.plot([0, lim], [0, lim], 'k-', lw=.6); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('Empirical model, RMSE per condition (°C)'); ax.set_ylabel('Other model, RMSE per condition (°C)'); ax.grid(alpha=.25, lw=.4); ax.legend(frameon=False, fontsize=6)
plt.tight_layout(); plt.savefig(out('fig_loco_scatter.pdf')); plt.savefig(out('fig_loco_scatter.png'), dpi=300); plt.close()

show = [('loco', '90_20_OFF'), ('loco', '100_30_MID'), ('loco', '110_40_ON')]
fig, axs = plt.subplots(3, 1, figsize=(3.4, 5.0), sharex=True)
for ax, key in zip(axs, show):
    te = [i for i in range(N) if '%d_%d_%s' % COND[i] == key[1]]
    for i in te: ax.plot(GRID, Pm[i], '.', c='0.6', ms=1.5)
    for m in MAIN:
        if key in P[m]: ax.plot(GRID, np.mean([P[m][key][i] for i in te], 0), c=COL[m], lw=1.1, label=NAME[m].replace('$', ''))
    t, r, f = key[1].split('_'); ax.set_title(f'{t} °C · {r} rpm · fans {f} (held out)', fontsize=6.5); ax.grid(alpha=.25, lw=.4); ax.set_ylabel('Apparent T (°C)')
axs[-1].set_xlabel('Distance from the peak, $x$ (cm)'); axs[0].plot([], [], '.', c='0.6', label='Measured'); axs[0].legend(frameon=False, fontsize=5.5, ncol=2)
plt.tight_layout(); plt.savefig(out('fig_pred.pdf')); plt.savefig(out('fig_pred.png'), dpi=300); plt.close()
print(json.dumps(S['physics_full'], indent=1)); print('noise floor', json.load(open(out('char_summary.json')))['noise_floor_C'])
