Posterior samples from the `jester` paper (https://inspirehep.net/literature/2915009). See below for small description.

They contain: masses `M`, radii `R` and Lambdas `L`. Also `log_prob`, posterior probability during `jester` inference.

- `radio`: Only radio timing
- `radio_chiEFT`: Joint likelihood of radio timing and chiEFT.
- `GW170817`: GW170817, this implicitly also has radio timing constraints
- `J0740`: NICER PSR J0740
- `J0030`: NICER PSR J0030
- `all`: Combine jointly all the constraints from the Jester paper