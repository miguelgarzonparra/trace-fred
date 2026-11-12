"""Paso 1: lee los BMP, extrae el perfil axial de cada foto y ajusta L.
Salidas: out/photos.csv (una fila por foto) y out/profiles.npz (perfiles remuestreados en 101 puntos)."""
import glob, os, sys, numpy as np, pandas as pd
from common import *

cal = load_calibration()
rows, Pm, Pe, bgs, Sraw = [], [], [], [], {}
for _, r in cal.iterrows():
    path = os.path.join(DATA, 'bmp', r['name'] + '.bmp')
    T = read_bmp(path); sc = float(r['scale_cm_per_px'])
    s, E, pk, tabs, bg, xs, ys = profile(T, r.fiber_ax, r.fiber_ay, r.fiber_bx, r.fiber_by)
    s = s * sc
    d, i0 = peak_after_dip(s, E)
    rows.append(dict(name=r['name'], temp=int(r.temp_C), rpm=int(r.rpm), fan=r.fan, rep=int(r.rep),
                     scale_cm_px=sc, dip_cm=s[d], peak_cm=s[i0], L=fit_L(s, E, i0),
                     L05=fit_L(s, E, i0, 0.5), L10=fit_L(s, E, i0, 1.0),
                     Tpk=tabs[i0], bg=float(np.median(bg))))
    Pm.append(resample(s, tabs, i0)); Pe.append(resample(s, E, i0)); bgs.append(float(np.median(bg)))
photos = pd.DataFrame(rows)
photos.to_csv(out('photos.csv'), index=False)
np.savez(out('profiles.npz'), names=photos.name.values, Pm=np.array(Pm), Pe=np.array(Pe), bg=np.array(bgs), grid=GRID)
print(len(photos), 'fotos;', photos.groupby(['temp', 'rpm', 'fan']).ngroups, 'condiciones')
print('L (cm): media %.2f, min %.2f, max %.2f' % (photos.L.mean(), photos.L.min(), photos.L.max()))
print('fondo (°C): %.1f a %.1f' % (photos.bg.min(), photos.bg.max()))

# verificación opcional contra resultados guardados de una versión anterior
ref = os.path.join(DATA, 'resultados_por_foto_nuevo_v2.csv')
if os.path.exists(ref):
    R = pd.read_csv(ref).set_index('name'); P = photos.set_index('name')
    print('vs guardado: max |dL| = %.3f cm, max |d peak| = %.3f cm' %
          ((P.L - R.L).abs().max(), (P.peak_cm - R.peak_cm).abs().max()))
