import os
import numpy as np

SAVE_DIR = "./posterior_samples/"

def save_posterior_samples(outdir: str,
                           keys_to_save: list[str] = ["masses_EOS", "radii_EOS", "Lambdas_EOS", "log_prob"],
                           downsample_factor: int = 2
                           ):
    """From the given outdir, load the file, get the samples and only get relevant ones to save into a new file."""
    
    # Fetch the filename and read the data
    filename = outdir + "/eos_samples.npz"
    data = np.load(filename)
    new_dict = {k: v for k, v in data.items() if k in keys_to_save}
    
    # Downsample if needed
    if downsample_factor > 1:
        for k in new_dict:
            new_dict[k] = new_dict[k][::downsample_factor]
    
    identifier = outdir.split("_")[-1]
    save_path = os.path.join(SAVE_DIR, identifier)
    
    print(np.shape(new_dict["masses_EOS"]))
    print(f"Saving to {save_path}")
    
    np.savez(save_path, **new_dict)
    
def main():
    outdirs = [
        "outdir_radio",
        "outdir_GW170817"
    ]
    
    for outdir in outdirs:
        save_posterior_samples(outdir)
        
    print("DONE")
    
if __name__ == "__main__":
    main()