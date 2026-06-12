import os
import numpy as np
from bilby.core.result import Result
from bilby.gw.conversion import generate_all_bns_parameters

def get_posterior_samples(filename: str) -> None:
    """Loads bilby result from HDF5 filename and save it into npz"""
    
    # Load in the result
    result = Result.from_json(filename)
    posterior = result.posterior # pd.DataFrame
    search_parameter_keys = result.search_parameter_keys
    
    # Add source frame masses for jester
    search_parameter_keys += ["mass_1_source", "mass_2_source"]
    
    search_parameter_keys += ["luminosity_distance"]
    
    # Search for and add network SNR
    posterior_keys = posterior.columns.tolist()
    snr_keys = [key for key in posterior_keys if "snr_" in key]
    search_parameter_keys += snr_keys
    
    posterior_sliced = posterior[search_parameter_keys]
    
    # Get it as a dict
    posterior_dict = posterior_sliced.to_dict(orient="list")
    
    # Save this to npz file
    npz_filename = filename.replace(".json", "_posterior.npz")
    np.savez(npz_filename, **posterior_dict)
    
def main():
    for network in ["et", "et_2l", "et_ce", "et_2l_ce"]:
        
        print(f"Processing network: {network}")
        
        # Find the result file (only one JSON file exists)
        subdir = f"./{network}/outdir/" 
        result_files = [f for f in os.listdir(subdir) if f.endswith(".json")]
        if len(result_files) != 1:
            raise ValueError(f"Expected exactly one JSON file in {subdir}, found {len(result_files)}")
        result_file = os.path.join(subdir, result_files[0])
        get_posterior_samples(result_file)
        
if __name__ == "__main__":
    main()