import os
import numpy as np
import json
import importlib
import h5py
import json

def fetch_posterior_filename(source_dir: str):
    """
    Given a source_dir where a GW inference run was performed in /work/, return the path to the posterior HDF5 file if it exists,
    otherwise return None.

    Args:
        source_dir (str): Directory containing the run output. NOTE: It should have
                          outdir/final_result/ with an HDF5 posterior file.

    Returns:
        str or None: Path to the posterior file, or None if not found.
    """
    
    source_dir = os.path.abspath(source_dir)
    final_results_dir = os.path.join(source_dir, "outdir/final_result")

    if not os.path.exists(final_results_dir):
        return None

    posterior_files = [
        f for f in os.listdir(final_results_dir)
        if f.endswith(".h5") or f.endswith(".hdf5")
    ]

    if not posterior_files:
        print(f"WARNING: No HDF5 files found in {final_results_dir}. Trying to look for JSON files now")
        final_results_dir = os.path.join(source_dir, "outdir")
        posterior_files = [
            f for f in os.listdir(final_results_dir)
            if f.endswith(".json")
        ]
        if not posterior_files:
            return None

    # Just return the first one found, there should be only one anyways
    return os.path.join(final_results_dir, posterior_files[0])

def fetch_mass_Lambdas_samples(posterior_filename: str,
                               chirp_tilde: bool = False):
    """
    When given a posterior filename, fetch the mass and Lambda samples from it to make the 2D contour plots.

    Args:
        posterior_filename (str): Filename of the posterior HDF5 file from which to fetch the mass and Lambda samples.
    """
    
    if posterior_filename.endswith(".json"):
        with h5py.File(posterior_filename, 'r') as f:
            posterior = f['posterior']
        
        if chirp_tilde:
            # Fetch the chirp mass and lambda_tilde
            chirp_mass_source = posterior['chirp_mass_source'][:]
            mass_ratio = posterior['mass_ratio'][:]
            lambda_tilde = posterior['lambda_tilde'][:]
            delta_lambda_tilde = posterior['delta_lambda_tilde'][:]
            return chirp_mass_source, mass_ratio, lambda_tilde, delta_lambda_tilde
        else:
            # Fetch the component masses and Lambdas
            mass_1_source = posterior['mass_1_source'][:]
            mass_2_source = posterior['mass_2_source'][:]
            lambda_1 = posterior['lambda_1'][:]
            lambda_2 = posterior['lambda_2'][:]
        
            return mass_1_source, mass_2_source, lambda_1, lambda_2
    else:
        # Open with JSON
        with open(posterior_filename, 'r') as f:
            posterior = json.load(f)['posterior']['content']
            
        if chirp_tilde:
            # Fetch the chirp mass and lambda_tilde
            chirp_mass_source = posterior['chirp_mass_source']
            mass_ratio = posterior['mass_ratio']
            lambda_tilde = posterior['lambda_tilde']
            delta_lambda_tilde = posterior['delta_lambda_tilde']
            return chirp_mass_source, mass_ratio, lambda_tilde, delta_lambda_tilde
        
        else:
            # Fetch the component masses and Lambdas
            mass_1_source = posterior['mass_1_source']
            mass_2_source = posterior['mass_2_source']
            lambda_1 = posterior['lambda_1']
            lambda_2 = posterior['lambda_2']
        
            return mass_1_source, mass_2_source, lambda_1, lambda_2
        
def fetch_sampling_time(source_dir: str) -> float:
    """
    Fetch the sampling time used in the run located in source_dir.

    Args:
        source_dir (str): Directory containing the run output. NOTE: It should have
                          outdir/final_result/ with an HDF5 posterior file.

    Returns:
        float: Sampling time in seconds
    """
    
    posterior_filename = fetch_posterior_filename(source_dir)
    
    with h5py.File(posterior_filename, 'r') as f:
        return f["sampling_time"][()]
    
def fetch_log_bayes_factor(source_dir: str) -> float:
    """
    Fetch the log_bayes_factor used in the run located in source_dir.

    Args:
        source_dir (str): Directory containing the run output. NOTE: It should have
                          outdir/final_result/ with an HDF5 posterior file.

    Returns:
        float: log_bayes_factor
    """
    
    posterior_filename = fetch_posterior_filename(source_dir)
    with h5py.File(posterior_filename, 'r') as f:
        return f["log_bayes_factor"][()]
    
def fetch_waveform_approximant(source_dir: str) -> str:
    posterior_filename = fetch_posterior_filename(source_dir)
    with h5py.File(posterior_filename, "r") as f:
        waveform_approximant = (
            f["meta_data"]["likelihood"]["waveform_arguments"]["waveform_approximant"][()]
            .decode("utf-8")
            )
    return waveform_approximant

def fetch_raw_priors(source_dir: str) -> dict:
    """
    Fetch the priors directly from the HDF5 file in source_dir.
    This is useful for making the table, where we do not need the priors initialized as bilby prior objects, but just 
    want to query the type of prior and its ranges, for instance.

    Args:
        source_dir (str): Directory containing the run output.

    Returns:
        dict: Dictionary containing the priors as stored in the HDF5 file, in the raw format
    """
    
    # Fetch the HDF5 file
    posterior_file = fetch_posterior_filename(source_dir)
    
    # Load it to get the priors dict
    with h5py.File(posterior_file, 'r') as f:
        priors_bytes = f["priors"][()]        # bytes
        priors_str = priors_bytes.decode()    # bytes → string
        priors_dict = json.loads(priors_str)  # string → dict
        
    return priors_dict

def fetch_fixed_parameters(source_dir: str, 
                           fixed_params_keys: list[str]) -> dict:
    
    posterior_filename = fetch_posterior_filename(source_dir)
    fixed_params_dict = {}
    
    # Rename azimuth to ra and zenith to dec, as that is how they are stored in the posterior file
    if "azimuth" in fixed_params_keys:
        fixed_params_keys[fixed_params_keys.index("azimuth")] = "ra"
    if "zenith" in fixed_params_keys:
        fixed_params_keys[fixed_params_keys.index("zenith")] = "dec"
    
    with h5py.File(posterior_filename, 'r') as f:
        posterior = f["posterior"]
        for key in fixed_params_keys:
            # CHeck if this is indeed all the same value
            if not np.all(posterior[key][()] == posterior[key][0]):
                raise ValueError(f"WARNING: parameter {key} is not fixed, it has multiple values!")
            fixed_params_dict[key] = posterior[key][()][0]
            
    return fixed_params_dict
        
def initialize_priors(source_dir: str,
                      keys_to_fetch: list[str] = None) -> dict:
    # Fetch the HDF5 file
    posterior_file = fetch_posterior_filename(source_dir)
    
    # Load it to get the priors dict
    with h5py.File(posterior_file, 'r') as f:
        priors_bytes = f["priors"][()]        # bytes
        priors_str = priors_bytes.decode()    # bytes → string
        priors_dict = json.loads(priors_str)  # string → dict
        
        # We only need a few parameters to determine the priors on chirp_mass_source and lambda_tilde
        if keys_to_fetch is not None:
            priors_dict = {key: priors_dict[key] for key in keys_to_fetch if key in priors_dict}
        
        # Initialize the bilby priors corresponding to the info in the dict
        priors = {}
        for key, prior_info in priors_dict.items():
            module = importlib.import_module(prior_info["__module__"])
            cls = getattr(module, prior_info["__name__"])
            priors[key] = cls(**prior_info["kwargs"])
            
    return priors