"""Paso 2: repetibilidad, ANOVA de L, efecto de factores, diámetro y coeficiente h implícito.
Salidas: out/tab_*.csv|tex, out/char_summary.json y figuras fig_profiles / fig_L / fig_diameter."""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import statsmodels.formula.api as smf, statsmodels.api as sm
from common import *

ph = pd.read_csv(out('photos.csv')); Z = np.load(out('profiles.npz'), allow_pickle=True)
Pm, names = Z['Pm'], list(Z['names'])
ph['t'] = (ph.temp - 100) / 10; ph['r'] = (ph.rpm - 30) / 10; ph['f'] = ph.fan.map(FAN_CODE)
ph['lnL'] = np.log(ph.L)
S = {}                                              # resumen numérico que usa el texto

# ---- repetibilidad ---------------------------------------------------------------------------
g = ph.groupby(['temp', 'rpm', 'fan'])
cv = (g.L.std(ddof=1) / g.L.mean()).dropna()
S['cv_L_mean_pct'] = float(100 * cv.mean()); S['cv_L_max_pct'] = float(100 * cv.max())
dev = []                                            # desviaciones de cada perfil respecto de la media de su condición
for key, idx in g.indices.items():
    P = Pm[idx]; n = len(idx); m = np.nanmean(P, 0)
    dev.append((P - m) * np.sqrt(n / (n - 1)))      # corrección por grados de libertad
dev = np.concatenate(dev); ok = ~np.isnan(dev)
S['noise_floor_C'] = float(np.sqrt(np.mean(dev[ok] ** 2)))

# ---- ANOVA de L (efectos principales + interacciones de 2.º orden; diseño con una celda faltante) ----
res = {}
for resp in ('L', 'lnL'):
    m = smf.ols(f'{resp} ~ C(temp) + C(rpm) + C(fan) + C(temp):C(rpm) + C(temp):C(fan) + C(rpm):C(fan)', ph).fit()
    a = sm.stats.anova_lm(m, typ=2); a['eta2'] = a.sum_sq / a.sum_sq.sum()
    res[resp] = a; a.to_csv(out(f'tab_anova_{resp}.csv'))
    if resp == 'L': S['anova_L_R2'] = float(m.rsquared)
S['anova_L'] = {k.replace('C(', '').replace(')', ''): dict(F=float(v.F), p=float(v['PR(>F)']), eta2=float(v.eta2))
                for k, v in res['L'].iterrows() if k != 'Residual'}

# medias por nivel y razón de ventilador
lev = {f: ph.groupby(f).L.mean().round(3).to_dict() for f in ('temp', 'rpm', 'fan')}
S['L_by_level'] = {k: {str(a): b for a, b in v.items()} for k, v in lev.items()}
cm = ph.groupby(['temp', 'rpm', 'fan']).L.mean().unstack('fan')
S['L_ratio_ON_OFF_mean'] = float((cm['ON'] / cm['OFF']).mean()); S['L_ratio_ON_OFF_n'] = int((cm['ON'] / cm['OFF']).notna().sum())
S['L_ratio_MID_OFF_mean'] = float((cm['MID'] / cm['OFF']).mean())

# regresión log-lineal de L sobre los factores codificados
mr = smf.ols('lnL ~ t + r + f', ph).fit(cov_type='cluster', cov_kwds={'groups': ph.groupby(['temp', 'rpm', 'fan']).ngroup()})
S['lnL_reg'] = {k: dict(coef=float(mr.params[k]), lo=float(mr.conf_int().loc[k, 0]), hi=float(mr.conf_int().loc[k, 1]), p=float(mr.pvalues[k])) for k in ('t', 'r', 'f')}
S['lnL_reg_R2'] = float(mr.rsquared)

# ---- diámetro ---------------------------------------------------------------------------------------
D = pd.read_csv(f'{DATA}/diametro_fred_por_condicion.csv').rename(columns={'d_fred': 'd_mm'})
D['t'] = (D.temp - 100) / 10; D['r'] = (D.rpm - 30) / 10; D['f'] = D.fan.map(FAN_CODE); D['lnd'] = np.log(D.d_mm)
md = smf.ols('lnd ~ t + r + f', D).fit()
S['d_range_mm'] = [float(D.d_mm.min()), float(D.d_mm.max())]
S['lnd_reg'] = {k: dict(coef=float(md.params[k]), p=float(md.pvalues[k])) for k in ('t', 'r', 'f')}; S['lnd_reg_R2'] = float(md.rsquared)
mp = smf.ols('lnd ~ np.log(rpm)', D).fit(); S['d_vs_rpm_exponent'] = dict(b=float(mp.params['np.log(rpm)']), lo=float(mp.conf_int().loc['np.log(rpm)', 0]), hi=float(mp.conf_int().loc['np.log(rpm)', 1]))
S['n_diam_runs_min'] = int(D.n.min())

# ---- h implícito a partir de L (h = rho cp d v / 4L); rho*cp incierto por un factor de ~2 ----------------
C = ph.groupby(['temp', 'rpm', 'fan']).L.mean().reset_index().merge(D[['temp', 'rpm', 'fan', 'd_mm']], on=['temp', 'rpm', 'fan'])
v = 2 * np.pi * R_SPOOL_M * C.rpm / 60
C['h'] = RHO * CP * (C.d_mm * 1e-3) * v / (4 * C.L * 1e-2)
S['h_implied_W_m2K'] = dict(min=float(C.h.min()), median=float(C.h.median()), max=float(C.h.max()))
C.to_csv(out('tab_L_h_conditions.csv'), index=False)

json.dump(S, open(out('char_summary.json'), 'w'), indent=1)
print(json.dumps({k: S[k] for k in ('cv_L_mean_pct', 'noise_floor_C', 'anova_L_R2', 'L_ratio_ON_OFF_mean', 'd_range_mm', 'd_vs_rpm_exponent', 'h_implied_W_m2K')}, indent=1))
print(res['L'].round(4)); print(json.dumps(S['lnL_reg'], indent=0)); print(S['L_by_level'])

# ---- tabla LaTeX del ANOVA --------------------------------------------------------------------------------
a = res['L']; lab = {'C(temp)': 'Nozzle temperature', 'C(rpm)': 'Spooler speed', 'C(fan)': 'Fans',
                     'C(temp):C(rpm)': 'Temp. $\\times$ speed', 'C(temp):C(fan)': 'Temp. $\\times$ fans', 'C(rpm):C(fan)': 'Speed $\\times$ fans'}
L = ['\\begin{tabular}{lrrrr}', '\\hline', 'Source & df & $F$ & $p$ & $\\eta^2$ \\\\', '\\hline']
for k, nm in lab.items():
    p = a.loc[k, 'PR(>F)']; L.append(f"{nm} & {int(a.loc[k,'df'])} & {a.loc[k,'F']:.1f} & {'$<$0.001' if p<1e-3 else f'{p:.3f}'} & {a.loc[k,'eta2']:.2f} \\\\")
L += [f"Residual & {int(a.loc['Residual','df'])} & & & {a.loc['Residual','eta2']:.2f} \\\\", '\\hline', '\\end{tabular}']
open(out('tab_anova.tex'), 'w').write('\n'.join(L))

# ---- figuras ---------------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 7, 'axes.linewidth': .6})
FC = {'OFF': '#1f77b4', 'MID': '#ff7f0e', 'ON': '#d62728'}
fig, axs = plt.subplots(3, 3, figsize=(7.0, 4.5), sharex=True, sharey=True)
for i, T_ in enumerate((90, 100, 110)):
    for j, R_ in enumerate((20, 30, 40)):
        ax = axs[i, j]
        for f in ('OFF', 'MID', 'ON'):
            idx = ph.index[(ph.temp == T_) & (ph.rpm == R_) & (ph.fan == f)]
            if len(idx):
                P = Pm[idx]; ax.plot(GRID, np.nanmean(P, 0), c=FC[f], lw=1.2, label=f)
                ax.fill_between(GRID, np.nanmin(P, 0), np.nanmax(P, 0), color=FC[f], alpha=.18, lw=0)
        ax.set_title(f'{T_} °C · {R_} rpm', fontsize=7); ax.grid(alpha=.25, lw=.4)
for ax in axs[-1]: ax.set_xlabel('Distance from the peak, $x$ (cm)')
for ax in axs[:, 0]: ax.set_ylabel('Apparent temperature (°C)')
axs[0, 0].legend(title='Fans', fontsize=6, title_fontsize=6, frameon=False)
plt.tight_layout(); plt.savefig(out('fig_profiles.pdf')); plt.savefig(out('fig_profiles.png'), dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(3.4, 2.5))
mk = {90: 'o', 100: 's', 110: '^'}
for f in ('OFF', 'MID', 'ON'):
    for T_ in (90, 100, 110):
        q = ph[(ph.fan == f) & (ph.temp == T_)]
        if len(q): ax.plot(q.rpm + {'OFF': -1, 'MID': 0, 'ON': 1}[f] * .9 + (T_ - 100) * .04, q.L, mk[T_], c=FC[f], ms=3, mfc='none' if T_ == 90 else FC[f], alpha=.8, mew=.7)
for f in ('OFF', 'MID', 'ON'): ax.plot([], [], 's', c=FC[f], ms=3, label=f'Fans {f}')
for T_ in (90, 100, 110): ax.plot([], [], mk[T_], c='0.3', ms=3, mfc='none' if T_ == 90 else '0.3', label=f'{T_} °C')
ax.set_xlabel('Spooler speed (rpm)'); ax.set_ylabel('Cooling length $L$ (cm)'); ax.set_xticks([20, 30, 40]); ax.grid(alpha=.25, lw=.4); ax.set_ylim(0.85, 1.95)
ax.legend(fontsize=5.5, frameon=False, ncol=2, loc='upper left'); plt.tight_layout(); plt.savefig(out('fig_L.pdf')); plt.savefig(out('fig_L.png'), dpi=300); plt.close()

fig, ax = plt.subplots(figsize=(3.4, 2.4))
for T_ in (90, 100, 110):
    q = D[D.temp == T_]; ax.errorbar(q.rpm + (T_ - 100) * .08, q.d_mm, yerr=q.d_sd, fmt=mk[T_], ms=3, c='0.25', mfc='none' if T_ == 90 else '0.25', lw=.5, capsize=1, label=f'{T_} °C')
xx = np.linspace(18, 42, 50); ax.plot(xx, np.exp(mp.params['Intercept'] + mp.params['np.log(rpm)'] * np.log(xx)), 'r-', lw=1, label=f"fit: $d\\propto n^{{{mp.params['np.log(rpm)']:.2f}}}$")
ax.set_xlabel('Spooler speed (rpm)'); ax.set_ylabel('Fiber diameter (mm)'); ax.set_xticks([20, 30, 40]); ax.grid(alpha=.25, lw=.4); ax.legend(fontsize=5.5, frameon=False)
plt.tight_layout(); plt.savefig(out('fig_diameter.pdf')); plt.savefig(out('fig_diameter.png'), dpi=300); plt.close()
