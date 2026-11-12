"""Funciones comunes del análisis TRACE-FrED (lectura de BMP, perfiles, ajuste de L).

Rutas (variables de entorno, o valores por defecto):
  TRACE_DATA  carpeta con  bmp/*.bmp, bmp_data.xlsx (calibración) y diametro_fred_por_condicion.csv
  TRACE_OUT   carpeta de salida (tablas, figuras, resultados intermedios)
"""
import os, glob, numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

DATA = os.environ.get('TRACE_DATA', 'data')
OUT = os.environ.get('TRACE_OUT', 'out')
os.makedirs(OUT, exist_ok=True)

HALF_WIDTH_PX = 7                      # semiancho lateral de la franja (px)
FIT_START_CM, FIT_SPAN_CM = 0.2, 4.0   # ventana de ajuste de L, medida desde el pico
GRID = np.linspace(0, 5, 101)          # distancia al pico (cm), 101 puntos
R_SPOOL_M = 0.00765                    # radio efectivo del carrete (m)
RHO, CP = 1200.0, 1300.0               # supuestos (el material no tiene ficha técnica)
T_FLOOR = 24.0                         # piso de temperatura aparente (°C)
FAN_CODE = {'OFF': -1, 'MID': 0, 'ON': 1}

def out(name): return os.path.join(OUT, name)

# ---------------------------------------------------------------- lectura de BMP
def read_bmp(path):
    """Matriz de temperaturas (°C) que la cámara guarda tras los píxeles del BMP."""
    d = open(path, 'rb').read()
    off = int.from_bytes(d[10:14], 'little'); w = int.from_bytes(d[18:22], 'little')
    h = abs(int.from_bytes(d[22:26], 'little', signed=True)); bpp = int.from_bytes(d[28:30], 'little')
    end = off + ((w * bpp + 31) // 32) * 4 * h; t = d[end:]
    assert t[:4] == b'AAEF', 'este BMP no trae el bloque de temperaturas'
    a = np.frombuffer(t[8:8 + 196608], dtype=np.uint8).astype(int) - 48
    a = a.reshape(-1, 4); v = (a[:, 0] * 1000 + a[:, 1] * 100 + a[:, 2] * 10 + a[:, 3]) / 10
    return v.reshape(256, 192) if w <= h else v.reshape(192, 256)

def load_calibration(path=None):
    """Calibración manual por foto (escala, borde del calentador, extremos de la fibra)."""
    path = path or os.path.join(DATA, 'bmp_data.xlsx')
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    else:
        raw = pd.read_excel(path, header=None)
        if ',' in str(raw.iloc[0, 0]):                     # CSV pegado en una sola columna
            import io
            df = pd.read_csv(io.StringIO('\n'.join(raw.iloc[:, 0].astype(str))))
        else:
            df = pd.read_excel(path)
    df = df[df['done'] == 1].reset_index(drop=True)
    df['name'] = df['file'].str.replace('.bmp', '', regex=False)
    return df

# ---------------------------------------------------------------- perfil de una foto
def profile(T, ax, ay, bx, by):
    """Sigue la fibra a lo largo de la línea A->B. Devuelve (s_px, E, pk, Tabs, bg) por fila."""
    ys = np.arange(int(round(ay)), min(int(round(by)), T.shape[0] - 1) + 1)
    xl = ax + (bx - ax) * (ys - ay) / (by - ay)
    xs, E, pk, tabs, bgs = [], [], [], [], []
    for y, x0 in zip(ys, xl):
        c = int(round(x0)); lo = max(c - 3, 1); hi = min(c + 4, T.shape[1] - 1)
        j = lo + int(np.argmax(T[y, lo:hi]))                    # píxel más caliente cerca de la línea
        side = np.r_[T[y, max(j - 16, 0):max(j - 10, 0)], T[y, j + 10:j + 17]]
        bg = np.median(side)                                    # fondo local: 10-16 px a cada lado
        m = T[y, max(j - 1, 0):j + 2].max()
        E.append(np.sum(T[y, max(j - HALF_WIDTH_PX, 0):j + HALF_WIDTH_PX + 1] - bg))
        pk.append(m - bg); tabs.append(m); bgs.append(bg); xs.append(j)
    xs = np.array(xs, float)
    s = np.r_[0, np.cumsum(np.hypot(np.diff(xs), 1.0))]
    return s, np.array(E), np.array(pk), np.array(tabs), np.array(bgs), xs, ys

def peak_after_dip(s_cm, E):
    """Índices (valle, pico): se busca el último valle pronunciado (<50 % de los máximos a ambos lados,
    dentro de los primeros 3 cm; si no hay, se relaja a 70 % y 90 %) y el pico de E justo después."""
    Es = uniform_filter1d(E, 3); idx = np.where(s_cm < 3)[0]
    for frac in (.5, .7, .9):
        c = [i for i in idx[2:-2] if Es[i] <= Es[i - 1] and Es[i] <= Es[i + 1]
             and Es[i] < frac * Es[:i].max() and Es[i] < frac * Es[i:].max()]
        if c: break
    d = c[-1]; step = max(int(1.0 / (s_cm[1] - s_cm[0])), 1)
    return d, d + int(np.argmax(E[d:d + step]))

def fit_L(s_cm, E, i0, start=FIT_START_CM, span=FIT_SPAN_CM):
    """Ajusta E(x) = a exp(-x/L) + c en una ventana de `span` cm que empieza `start` cm después del pico."""
    m = (s_cm >= s_cm[i0] + start) & (s_cm <= s_cm[i0] + start + span)
    x = s_cm[m] - s_cm[m][0]; y = E[m]
    if len(x) < 10: return np.nan
    try:
        p, _ = curve_fit(lambda x, a, L, c: a * np.exp(-x / L) + c, x, y, p0=(y[0], 1.0, y[-1]),
                         bounds=([0, 0.05, -30], [5000, 20, 300]), maxfev=20000)
        return float(p[1])
    except Exception:
        return np.nan

def resample(s_cm, y, i0):
    """Perfil remuestreado sobre GRID (distancia al pico); NaN fuera del rango medido."""
    return np.interp(GRID, s_cm - s_cm[i0], y, left=np.nan, right=np.nan)

def cond_key(t, r, f): return f'{int(t)}_{int(r)}_{f}'
