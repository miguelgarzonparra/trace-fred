"""Datos y particiones compartidos por los modelos (03*.py) y por las métricas (04)."""
import numpy as np, pandas as pd
from common import *

ph = pd.read_csv(out('photos.csv')); Z = np.load(out('profiles.npz'), allow_pickle=True)
Pm, BG, NAMES = Z['Pm'], Z['bg'], list(Z['names'])
D = pd.read_csv(f'{DATA}/diametro_fred_por_condicion.csv')
ph = ph.merge(D[['temp', 'rpm', 'fan', 'd_fred', 'rpm_real']], on=['temp', 'rpm', 'fan'], how='left')
N = len(ph); M = ~np.isnan(Pm)
T_C = (ph.temp.values - 100) / 10; R_C = (ph.rpm.values - 30) / 10; F_C = ph.fan.map(FAN_CODE).values.astype(float)
ZC = np.c_[T_C, R_C, F_C]                                  # factores codificados
X4 = np.c_[np.ones(N), ZC]
TN = ph.temp.values.astype(float)                          # temperatura de boquilla (°C)
DM = ph.d_fred.values * 1e-3                               # diámetro medido por condición (m)
V = 2 * np.pi * R_SPOOL_M * ph.rpm.values / 60             # velocidad de la fibra (m/s), nominal
V_EFF = 2 * np.pi * R_SPOOL_M * ph.rpm_real.values / 60    # velocidad efectiva (con la oscilación de 20 rpm)
COND = list(zip(ph.temp, ph.rpm, ph.fan))
CONDS = sorted(set(COND))

def post(T):
    """Posprocesado común: piso de 24 °C y perfil no creciente."""
    return np.minimum.accumulate(np.maximum(T, T_FLOOR))

def splits():
    """Particiones: LOCO (26) y extrapolación (se retira un nivel completo de un factor)."""
    sp = {}
    for c in CONDS:
        te = [i for i in range(N) if COND[i] == c]; tr = [i for i in range(N) if COND[i] != c]
        sp[('loco', f'{c[0]}_{c[1]}_{c[2]}')] = (tr, te)
    for name, col in (('temp', ph.temp.values), ('rpm', ph.rpm.values), ('fan', ph.fan.values)):
        for lv in sorted(set(col)):
            te = [i for i in range(N) if col[i] == lv]; tr = [i for i in range(N) if col[i] != lv]
            sp[('extrap', f'{name}={lv}')] = (tr, te)
    return sp
