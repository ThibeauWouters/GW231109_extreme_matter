# Kilonova lightcurve estimation

## Workflow

**Step 1 — generate lightcurve data (run on cluster, requires NMMA SVD models):**
```bash
python lightcurve_estimation_fullpos.py
# output: lc_data_band.dat
```

**Step 2 — copy output locally:**
```bash
scp cluster:/path/to/lc_data_band.dat .
```

**Step 3 — generate figure (run locally, from `figures/`):**
```bash
python Fig3_plot.py \
    ../em_lightcurves/peter_estimate/lc_data_band.dat \
    ../em_lightcurves/peter_estimate/AT2017gfo.dat \
    ../../paper/GW231109_lc_with_band.pdf
```

## Files

- `lightcurve_estimation_fullpos.py` — loops over posterior samples, generates lightcurves in all filters (u, g, r, i, z, y, J, H, K), writes median/credible intervals to `lc_data_band.dat`
- `posterior_samples_with_ejecta_mass.dat` — input posterior samples with pre-computed ejecta masses
- `AT2017gfo.dat` — observed photometry of AT2017gfo for comparison
