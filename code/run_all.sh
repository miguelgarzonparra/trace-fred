#!/usr/bin/env bash
# Reproduce the whole analysis. Run from the repository root:  bash code/run_all.sh
# Expects data/bmp/*.bmp, data/bmp_data.xlsx and data/diametro_fred_por_condicion.csv
set -e
export TRACE_DATA=${TRACE_DATA:-data}; export TRACE_OUT=${TRACE_OUT:-out}
export PYTHONPATH=code
for s in 01_perfiles 02_caracterizacion 03_modelos_cerrados 03b_pinn 04_metricas_figuras 05_paridad_y_mapa; do python3 code/$s.py; done
