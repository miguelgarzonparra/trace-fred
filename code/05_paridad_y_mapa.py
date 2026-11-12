"""Paso 5: gráfica de paridad (predicho vs. medido, LOCO) y mapa de calor del RMSE de extrapolación.
Lee out/preds_cerrados.pkl, out/preds_pinn.pkl y out/tab_extrap.csv (producidos por 03*.py y 04_metricas_figuras.py).
Escribe out/fig_parity.(pdf|png) y out/fig_extrap_heatmap.(pdf|png)."""
import pickle, os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from datos_modelos import *

P = pickle.load(open(out('preds_cerrados.pkl'), 'rb'))
if os.path.exists(out('preds_pinn.pkl')): P.update(pickle.load(open(out('preds_pinn.pkl'), 'rb')))
NAME = {'empirical': 'Empirical', 'semi': 'Semi-physical', 'physics': 'Physics-based', 'hybrid': 'Hybrid', 'pinn': 'PINN'}
COL = {'empirical': '#1f77b4', 'semi': '#2ca02c', 'physics': '#ff7f0e', 'hybrid': '#8c564b', 'pinn': '#9467bd'}
plt.rcParams.update({'font.size': 7, 'axes.linewidth': .6})

# ---- 1. predicho vs. medido (todas las condiciones juntas) -----------------------------------------------------------
KEY_L = [k for k in splits() if k[0] == 'loco']
fig, ax = plt.subplots(figsize=(3.4, 3.4))
lo, hi = 1e9, -1e9
for m in ('pinn', 'physics', 'hybrid', 'empirical', 'semi'):          # los peores al fondo
    if m not in P: continue
    xs, ys = [], []
    for k in KEY_L:
        for i, p in P[m][k].items():
            mk = M[i].copy(); mk[::2] = False                           # un punto de cada dos (0.1 cm) para no saturar
            xs.append(Pm[i][mk]); ys.append(np.asarray(p)[mk])
    xs, ys = np.concatenate(xs), np.concatenate(ys); lo, hi = min(lo, xs.min(), ys.min()), max(hi, xs.max(), ys.max())
    ax.scatter(xs, ys, s=2.5, c=COL[m], alpha=.35, lw=0, label=NAME[m], rasterized=True)
pad = 1.5; ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k-', lw=.7); ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(lo - pad, hi + pad)
ax.set_aspect('equal'); ax.grid(alpha=.25, lw=.4)
ax.set_xlabel('Measured apparent temperature (°C)'); ax.set_ylabel('Predicted temperature, held-out condition (°C)')
lg = ax.legend(frameon=False, fontsize=6, loc='upper left', markerscale=3)
for h in lg.legend_handles: h.set_alpha(1)
plt.tight_layout(); plt.savefig(out('fig_parity.pdf'), dpi=300); plt.savefig(out('fig_parity.png'), dpi=300); plt.close()

# ---- 2. mapa de calor de la extrapolación -----------------------------------------------------------------------------
TX = pd.read_csv(out('tab_extrap.csv'), index_col=0)
print(TX.round(2))
cols = ['empirical', 'semi', 'physics', 'pinn']
rows = [('temp=90', 'Nozzle 90'), ('temp=100', 'Nozzle 100'), ('temp=110', 'Nozzle 110'), ('rpm=20', 'Speed 20'), ('rpm=30', 'Speed 30'),
        ('rpm=40', 'Speed 40'), ('fan=OFF', 'Fans Off'), ('fan=MID', 'Fans Mid'), ('fan=ON', 'Fans On')]
A = np.array([[TX.loc[r, c] for c in cols] for r, _ in rows]); mean = A.mean(0)
Z = np.vstack([A, mean])
fig, ax = plt.subplots(figsize=(3.4, 3.5))
norm = Normalize(vmin=2, vmax=9); im = ax.imshow(Z, cmap='YlOrRd', norm=norm, aspect='auto')
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        ax.text(j, i, f'{Z[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='white' if norm(Z[i, j]) > .62 else 'black',
                fontweight='bold' if i == len(rows) else 'normal')
ax.set_xticks(range(len(cols))); ax.set_xticklabels([NAME[c] for c in cols], rotation=25, ha='right')
ax.set_yticks(range(len(rows) + 1)); ax.set_yticklabels([n for _, n in rows] + ['Mean'])
for y in (2.5, 5.5, 8.5): ax.axhline(y, c='white', lw=2)
ax.set_xticks(np.arange(-.5, len(cols)), minor=True); ax.set_yticks(np.arange(-.5, len(rows) + 1), minor=True)
ax.grid(which='minor', c='white', lw=.8); ax.tick_params(which='both', length=0)
cb = plt.colorbar(im, ax=ax, fraction=.05, pad=.03); cb.set_label('RMSE (°C)'); cb.outline.set_linewidth(.4)
ax.set_ylabel('Held-out level')
plt.tight_layout(); plt.savefig(out('fig_extrap_heatmap.pdf'), dpi=300); plt.savefig(out('fig_extrap_heatmap.png'), dpi=300); plt.close()
print('mean row', mean.round(2))
