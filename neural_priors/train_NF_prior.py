"""
Train a normalizing flow to approximate a distribution on masses and Lambdas to replicate an EOS dataset and be used in inference.
"""

import os
import matplotlib.pyplot as plt
import corner
import argparse
import numpy as np
import json
import torch.nn.functional as F
import tqdm
import time
import joblib

import matplotlib.pyplot as plt
params = {"axes.grid": True,
          "text.usetex" : False}
plt.rcParams.update(params)

### bilby imports
import bilby
from bilby.core.prior.analytical import Uniform
from bilby.gw.prior import UniformComovingVolume
from bilby.core.prior import PriorDict
from bilby.gw.conversion import (
    luminosity_distance_to_redshift,
    chirp_mass_and_mass_ratio_to_component_masses,
    component_masses_to_chirp_mass,
    lambda_1_lambda_2_to_lambda_tilde,
    lambda_1_lambda_2_to_delta_lambda_tilde
)

### glasflow imports
from glasflow.flows.nsf import CouplingNSF
from glasflow.flows.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveFlow, MaskedAffineAutoregressiveFlow
# TODO: Add support for other autoregressive flows if desired:
# MaskedPiecewiseLinearAutoregressiveFlow,
# MaskedPiecewiseQuadraticAutoregressiveFlow, MaskedPiecewiseCubicAutoregressiveAutoregressiveFlow
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

if torch.cuda.is_available():
    print(f"torch: CUDA is available. Number of devices: {torch.cuda.device_count()}")
    print(f"torch: Current CUDA device: {torch.cuda.current_device()}")
else:
    print("torch: CUDA is not available.")

# Get the device as well:
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = os.path.dirname(os.path.abspath(__file__))
EOS_DIR = os.path.join(script_dir, "..", "data", "eos")

# HTCondor submission-specific argument keys
HTCONDOR_ARGS = {'setup_submission', 'submit', 'queue'}


# def sample_ns_mass_gaussian(nb_mass_samples: int):
#     """
#     Sample from single Gaussian distribution, found the hyperparams in https://arxiv.org/pdf/2407.16669
#     """
#     mu = 1.33
#     sigma = 0.09
#     mass_samples = np.random.normal(mu, sigma, nb_mass_samples)
    
#     if len(mass_samples) == 1:
#         return mass_samples[0]
#     else:
#         return mass_samples
    
# def sample_ns_mass_double_gaussian(nb_mass_samples: int):
#     """
#     Sample from double Gaussian, found the hyperparams in https://arxiv.org/pdf/2407.16669
#     """
#     mu_1 = 1.34
#     sigma_1 = 0.07
    
#     mu_2 = 1.80
#     sigma_2 = 0.21
#     w = 0.65
    
#     # Sample from mixture of gaussians
#     u = np.random.rand(nb_mass_samples) # uniform [0,1], to determine the mode
#     mass_samples = np.where(
#         u < w,
#         np.random.normal(mu_1, sigma_1, size=nb_mass_samples),
#         np.random.normal(mu_2, sigma_2, size=nb_mass_samples)
#     )
    
#     if len(mass_samples) == 1:
#         return mass_samples[0]
#     else:
#         return mass_samples

parser = argparse.ArgumentParser(description="Train a normalizing flow prior on EOS samples.")
parser.add_argument("--population-type",
                    type=str,
                    default="event",
                    choices=["uniform", "event"],
                    help="Type of source to model")
parser.add_argument("--source-type", 
                    type=str, 
                    default="bns", 
                    choices=["bns", "nsbh", "bhns"],
                    help="Type of source to model")
parser.add_argument("--eos-samples-name", 
                    type=str, 
                    default="radio", 
                    choices=["radio", "GW170817"],
                    help="EOS samples name (default: radio)")
parser.add_argument("--N-samples-training", 
                    type=int, 
                    default=200_000, 
                    help="Number of training samples")
parser.add_argument("--N-samples-plot", 
                    type=int, 
                    default=10_000, 
                    help="Number of samples for plotting")
parser.add_argument("--m-min",
                    type=float, 
                    default=0.5, 
                    help="Minimum mass for uniform mass sampling")
parser.add_argument("--m-max",
                    type=float, 
                    default=2.2, 
                    help="Maximum mass for uniform mass sampling")
parser.add_argument("--scale-input", 
                    action="store_true", 
                    help="Scale input before NF training")
parser.add_argument("--no-scale-input", 
                    dest="scale_input", 
                    action="store_false")
parser.set_defaults(scale_input=True)
parser.add_argument("--num-epochs", 
                    type=int, 
                    default=1_000,
                    help="Number of training epochs")
parser.add_argument("--learning-rate", 
                    type=float, 
                    default=1e-3, 
                    help="Learning rate")
parser.add_argument("--batch-size", 
                    type=int, 
                    default=1024, 
                    help="Batch size")
parser.add_argument("--max-patience", 
                    type=int, 
                    default=100, 
                    help="Max patience for early stopping")
parser.add_argument("--n-transforms", 
                    type=int, 
                    default=4, 
                    help="Number of NF transforms")
parser.add_argument("--n-neurons", 
                    type=int, 
                    default=64, 
                    help="Number of neurons per layer")
parser.add_argument("--n-blocks-per-transform", 
                    type=int, 
                    default=1, 
                    help="Number of blocks per transform")
parser.add_argument("--num-bins", 
                    type=int, 
                    default=10, 
                    help="Number of bins for spline flows")
parser.add_argument("--glasflow-type", 
                    type=str, 
                    default="CouplingNSF", 
                    choices=["CouplingNSF", "MaskedPiecewiseRationalQuadraticAutoregressiveFlow", "MaskedAffineAutoregressiveFlow"],
                    help="Type of glasflow model to use")
parser.add_argument("--validation-split-fraction", 
                    type=float, 
                    default=0.2, 
                    help="Fraction of data to use for validation (default: 0.2)")

# CouplingNSF-specific hyperparameters
parser.add_argument("--batch-norm-within-blocks", 
                    action="store_true", 
                    help="Enable batch normalization within each residual block")
parser.add_argument("--no-batch-norm-within-blocks", 
                    dest="batch_norm_within_blocks", 
                    action="store_false")
parser.set_defaults(batch_norm_within_blocks=False)
parser.add_argument("--batch-norm-between-transforms", 
                    action="store_true", 
                    help="Enable batch norm between transforms")
parser.add_argument("--no-batch-norm-between-transforms", 
                    dest="batch_norm_between_transforms", 
                    action="store_false")
parser.set_defaults(batch_norm_between_transforms=False)
parser.add_argument("--activation", 
                    type=str, 
                    default="relu", 
                    choices=["relu", "gelu", "silu", "tanh"], 
                    help="Activation function to use (default: relu)")
parser.add_argument("--dropout-probability", 
                    type=float, 
                    default=0.0, 
                    help="Dropout probability (default: 0.0)")
parser.add_argument("--linear-transform", 
                    type=str, 
                    default="None",
                    choices=["permutation", "lu", "svd", "None"],
                    help="Linear transform to apply before each coupling transform")
parser.add_argument("--tail-bound", 
                    type=float, 
                    default=5.0, 
                    help="Tail bound for splines (default: 5.0)")

    
class NFPriorCreator:
    """
    Class to construct the NF prior and train it.
    """
    
    def __init__(self,
                 population_type: str = "uniform",
                 eos_samples_name: str = "radio",
                 source_type: str = "bns",
                 N_samples_training: int = 100_000,
                 N_samples_plot: int = 10_000,
                 m_min: float = 0.5,
                 m_max: float = 2.2,
                 num_epochs: int = 500,
                 learning_rate: float = 1e-3,
                 max_patience: int = 50,
                 batch_size: int = 1024,
                 scale_input: bool = True,
                 validation_split_fraction: float = 0.2,
                 # glasflow-specific training arguments:
                 n_transforms: int = 4,
                 n_neurons: int = 128,
                 n_blocks_per_transform: int = 4,
                 num_bins: int = 10,
                 glasflow_type: str = "CouplingNSF",
                 # CouplingNSF-specific arguments:
                 batch_norm_within_blocks: bool = False,
                 batch_norm_between_transforms: bool = False,
                 activation: str = "relu",
                 dropout_probability: float = 0.0,
                 linear_transform: str = "None",
                 tail_bound: float = 5.0
                 ):
        """
        Initialize the NFPriorCreator class with the necessary parameters.

        Args:
            eos_samples_name (str, optional): Name of the run from which we load the EOS samples from, which will be converted into the training data for the NF for binary systems. Defaults to `radio`, which only uses the radio timing constraints on MTOV.
            source_type (str, optional): Which kind of source to model: `bns` or `nsbh`. Defaults to "bns".
            N_samples_training (int, optional): Number of training samples to create.. Defaults to 100_000.
            N_samples_plot (int, optional): Number of samples to create the plots. Defaults to 10_000.
            m_max_BH (float, optional): If generating NSBH training data with an NF that is not conditioned on the masses, this is up to which the masses are taken. Defaults to 5.0.
            save_name (str, optional): Where to save the models etc to. Defaults to "".
            take_log_lambda (bool, optional): Whether to take the log of the Lambdas before training to deal with their massive scaling, to improve training the NF. Defaults to True.
            use_flowjax (bool, optional): Whether to use flowJAX instead of glasflow for training. Defaults to False.
            use_tilde (bool, optional): Whether to use tilde parameterization for lambdas (lambda_tilde, delta_lambda_tilde) instead of (lambda_1, lambda_2). Defaults to False.
            use_component_masses (bool, optional): Whether to use component masses (m1, m2) instead of (Mc, q). Defaults to True.
            num_epochs (int, optional): Number of training epochs. Defaults to 100.
            learning_rate (float, optional): Learning rate for training. Defaults to 1e-3.
            max_patience (int, optional): Max stops to wait before employing early stopping. Defaults to 50.
            n_transforms (int, optional): Number of transforms in NF. Defaults to 2.
            n_neurons (int, optional): Number of neurons in NF. Defaults to 64.
            batch_size (int, optional): Batch size for NF training. Defaults to 256.
            n_blocks_per_transform (int, optional): Number of blocks per transform for NF. Defaults to 2.
            scale_input (bool, optional): Whether to scale the input for the NF before training. Defaults to False.

        Raises:
            ValueError: If source type is not one of the supported types, i.e., "bns" or "nsbh".
        """
        
        self.eos_samples_name = eos_samples_name
        
        SUPPORTED_SOURCE_TYPES = ["bns", "nsbh", "bhns"]
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"source_type must be one of {SUPPORTED_SOURCE_TYPES}, got {source_type} instead.")
        self.source_type = source_type
        
        print(f"Training a normalizing flow for {self.source_type} source")
        self.N_samples_training = N_samples_training
        self.N_samples_plot = N_samples_plot
        self.m_min = m_min
        self.m_max = m_max

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_patience = max_patience
        self.n_transforms = n_transforms
        self.n_neurons = n_neurons
        self.batch_size = batch_size
        self.n_blocks_per_transform = n_blocks_per_transform
        self.scale_input = scale_input
        self.num_bins = num_bins
        self.glasflow_type = glasflow_type
        self.validation_split_fraction = validation_split_fraction
        self.batch_norm_within_blocks = batch_norm_within_blocks
        self.batch_norm_between_transforms = batch_norm_between_transforms
        self.activation = activation
        self.dropout_probability = dropout_probability
        # TODO: a bit hacky
        if linear_transform == "None":
            linear_transform = None
        self.linear_transform = linear_transform
        self.tail_bound = tail_bound
        
        # Store the NF kwargs here to dump later on
        self.nf_kwargs = {"n_transforms": self.n_transforms,
                          "n_neurons": self.n_neurons,
                          "n_blocks_per_transform": self.n_blocks_per_transform,
                          "glasflow_type": self.glasflow_type,
                          "batch_norm_within_blocks": self.batch_norm_within_blocks,
                          "batch_norm_between_transforms": self.batch_norm_between_transforms,
                          "activation": self.activation,
                          "dropout_probability": self.dropout_probability,
                          "linear_transform": self.linear_transform,
                          "tail_bound": self.tail_bound
                          }
        
        # Set whether the masses will be sampled according to the intrinsic priors for a GW event
        # NOTE: this is also the case for the comparison with Hauke's work, so we set it to True if the population type is Hauke
        
        SUPPORTED_POPULATION_TYPES = ["uniform", "event"]
        if population_type not in SUPPORTED_POPULATION_TYPES:
            raise ValueError(f"population_type must be one of {SUPPORTED_POPULATION_TYPES}, got {population_type} instead.")
        self.population_type = population_type
        
        self.is_gw_event = (self.population_type == "event")
        
        if self.source_type == "bns":
            all_names = ["chirp_mass_source", "mass_ratio", "lambda_1", "lambda_2"]
        elif self.source_type == "nsbh":
            all_names = ["chirp_mass_source", "mass_ratio", "lambda_1"]
        elif self.source_type == "bhns":
            all_names = ["chirp_mass_source", "mass_ratio", "lambda_2"]
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
            
        if self.is_gw_event:
            all_names += ["luminosity_distance"]
        self.nf_kwargs["names"] = all_names
        self.nf_kwargs["n_inputs"] = len(self.nf_kwargs["names"])
        
        print(f"Before training, we built the following NF kwargs: {self.nf_kwargs}")
        
        # Make an outdir based on the given name etc, so that everything is stored in one directory for later on
        self.outdir = os.path.join("./models/", self.population_type, self.source_type, self.eos_samples_name)
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
            print(f"Created output directory {self.outdir}")
        print(f"Everything for this model will be saved to {self.outdir}")
        
    
    def load_eos_samples_from_file(self) -> tuple[np.array, np.array, np.array]:
        """
        Load in the EOS samples from the file and clean them. This returns the masses, radii and Lambdas of the EOS samples.
        """

        # Get the data
        eos_samples_filename = f"../EOS_data/{self.eos_samples_name}.npz"
        if not os.path.exists(eos_samples_filename):
            raise ValueError(f"File {eos_samples_filename} does not exist. Please check the path or the `eos_samples_name` argument.")
        print(f"Reading the EOS data from {eos_samples_filename}")
        eos_samples = np.load(eos_samples_filename)
        masses_EOS, radii_EOS, Lambdas_EOS = eos_samples["M"], eos_samples["R"], eos_samples["L"]
        
        # Iterate over EOS and keep those that are fine
        nb_samples = len(masses_EOS)
        good_idx = np.ones(nb_samples, dtype=bool)

        # There are sometimes a few (not many) bad EOSs, so get rid of them first
        for i in range(nb_samples):
            # First, sometimes the radius can be very large for low mass stars, which is unphysical
            bad_radii = (masses_EOS[i] > 1.0) * (radii_EOS[i] > 20.0)
            if any(bad_radii):
                good_idx[i] = False
                continue
            # Second, sometimes a negative Lambda was computed, remove that
            bad_Lambdas = (Lambdas_EOS[i] < 0.0)
            if any(bad_Lambdas):
                good_idx[i] = False
                continue
            # Finally, we want the TOV mass to be above 2.0 M_odot
            bad_MTOV = np.max(masses_EOS[i]) < 2.0
            if bad_MTOV:
                good_idx[i] = False
                continue

        print("Number of good samples: ", np.sum(good_idx) / nb_samples)

        masses_EOS = masses_EOS[good_idx]
        radii_EOS = radii_EOS[good_idx]
        Lambdas_EOS = Lambdas_EOS[good_idx]
        
        return masses_EOS, radii_EOS, Lambdas_EOS
    
    def create_data(self):
        """
        Create the dataset from the EOS to neutron star systems: m1, m2, Lambda1, Lambda2.
        Masses are sampled uniformly between 1.0 and MTOV for each EOS sample, therefore, such a prior can be used for any GW event.
        Always generates NS-NS pairs - the BNS vs NSBH distinction is handled in load_training_data().
        """
        
        # TODO: at some point, use self.population_type to change the way we generate the training data
        
        # Load the EOS samples
        masses_EOS, _, Lambdas_EOS = self.load_eos_samples_from_file()
        
        # Make everything ready for sampling
        m1_list = np.empty(self.N_samples_training)
        m2_list = np.empty(self.N_samples_training)
        Lambda1_list = np.empty(self.N_samples_training)
        Lambda2_list = np.empty(self.N_samples_training)
        Mc_det_list = np.empty(self.N_samples_training) # only for event population
        dL_list = np.empty(self.N_samples_training) # only for event population
        
        # Construct the prior from sampling from the EOS set - always generate NS-NS pairs
        if self.population_type == "uniform":
            for i in range(self.N_samples_training):
                if i % (self.N_samples_training // 10) == 0:
                    print(f"{i}/{self.N_samples_training}")
                idx = np.random.randint(0, len(masses_EOS))
                m, l = masses_EOS[idx], Lambdas_EOS[idx]
                mtov = np.max(m)
                
                # Sample two masses uniformly between 1.0 and MTOV, and ensure m1 >= m2
                mass_samples = np.random.uniform(self.m_min, self.m_max, 2)
                    
                # Ensure that m1 >= m2
                m1 = np.max(mass_samples)
                m2 = np.min(mass_samples)
                
                Lambda_1 = np.interp(m1, m, l)
                Lambda_2 = np.interp(m2, m, l)
                
                # Save the sampled values
                m1_list[i] = m1
                m2_list[i] = m2
                Lambda1_list[i] = Lambda_1
                Lambda2_list[i] = Lambda_2
                
        else:
            # For GW events, we sample from the given bilby priors# Use our own bilby priors for the GW event
            print(f"create_data is following GW_event source type: {self.population_type}")
            
            priors_dict = {}
            priors_dict["chirp_mass"] = bilby.gw.prior.UniformInComponentsChirpMass(name='chirp_mass', minimum=0.98, maximum=0.995, unit='$M_{\odot}$')
            priors_dict["mass_ratio"] = bilby.gw.prior.UniformInComponentsMassRatio(name='mass_ratio', minimum=0.125, maximum=1)
            priors_dict["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=1, maximum=700.0, unit='Mpc')
            
            gw_priors = PriorDict(priors_dict)
            print(f"Loaded priors: {gw_priors}")
            samples = gw_priors.sample(size = self.N_samples_training)
            
            Mc, q, dL = samples["chirp_mass"], samples["mass_ratio"], samples["luminosity_distance"]
            
            # Convert to source frame component masses
            m1_det, m2_det = chirp_mass_and_mass_ratio_to_component_masses(Mc, q)
            z = luminosity_distance_to_redshift(dL)
            m1 = m1_det / (1 + z)
            m2 = m2_det / (1 + z)
            
            # Interp to get Lambdas on those
            for i in range(self.N_samples_training):
                idx = np.random.randint(0, len(masses_EOS))
                m, l = masses_EOS[idx], Lambdas_EOS[idx]
                Lambda1_list[i] = np.interp(m1[i], m, l)
                Lambda2_list[i] = np.interp(m2[i], m, l)
            
            # Save the sampled values as the new lists # FIXME: hacky
            m1_list = m1
            m2_list = m2
            Mc_det_list = Mc
            dL_list = dL
            
        # For numerical stability, we turn zero into a very small number with np.clip:
        Lambda1_list = np.clip(Lambda1_list, a_min=1e-4, a_max=None)
        Lambda2_list = np.clip(Lambda2_list, a_min=1e-4, a_max=None)
        
        # Also get lambda_tilde, delta_lambda_tilde for the training data
        lambda_tilde = lambda_1_lambda_2_to_lambda_tilde(Lambda1_list, Lambda2_list, m1_list, m2_list)
        delta_lambda_tilde = lambda_1_lambda_2_to_delta_lambda_tilde(Lambda1_list, Lambda2_list, m1_list, m2_list)
        
        # Also get chirp mass and mass ratio for the training data
        chirp_mass_source = component_masses_to_chirp_mass(m1_list, m2_list)
        # mass_ratio = component_masses_to_mass_ratio(m1_list, m2_list) # TODO: remove for now?
        
        save_dict = {
            "m1": np.array(m1_list),
            "m2": np.array(m2_list),
            "mass_ratio": np.array(m2_list)/np.array(m1_list),
            "lambda_1": np.array(Lambda1_list),
            "lambda_2": np.array(Lambda2_list),
            "lambda_tilde": np.array(lambda_tilde),
            "delta_lambda_tilde": np.array(delta_lambda_tilde),
            "chirp_mass_source": np.array(chirp_mass_source),
        }
        
        if self.is_gw_event:
            save_dict["chirp_mass"] = Mc_det_list
            save_dict["luminosity_distance"] = dL_list
            
        print(f"Create data will save the following data:")
        for key, value in save_dict.items():
            print(f"  {key}: range = [{np.min(value)}, {np.max(value)}]")
        
        full_save_path = os.path.join(self.outdir, f"training_data.npz")
        self.training_filename = full_save_path
        print(f"Saving to {full_save_path}:")
        np.savez(full_save_path, **save_dict)
        print(f"Saving to {full_save_path} DONE")
        
    def load_training_data(self, training_filename: str):
        """
        Loads in the preprocessed training data before feeding it into the NF for training.
        Creates parameterization-agnostic training arrays that the training methods can use.
        """
        
        data = np.load(training_filename)
        m1_raw = data["m1"]
        m2_raw = data["m2"]
        lambda_1_raw = data["lambda_1"]
        lambda_2_raw = data["lambda_2"]
        
        # Always store the original arrays first
        self.m1_raw = m1_raw
        self.m2_raw = m2_raw
        self.lambda_1_raw = lambda_1_raw
        self.lambda_2_raw = lambda_2_raw
        
        if self.is_gw_event:
            print("Using detector-frame chirp mass from the GW event and mass ratio (Mc_det, q) for training")
            train_mass_1 = data["chirp_mass"]
            train_mass_2 = data["mass_ratio"]
        else:
            print("Using source-frame chirp mass and mass ratio (Mc, q) for training")
            train_mass_1 = data["chirp_mass_source"]
            train_mass_2 = data["mass_ratio"]
        
        # Handle lambda parameterization
        train_lambda_1 = lambda_1_raw
        train_lambda_2 = lambda_2_raw
        
        # Create train-validation split using sklearn
        arrays_to_split = [train_mass_1, train_mass_2, train_lambda_1, train_lambda_2]
        if self.is_gw_event:
            # If we have GW event data, also include the luminosity distance
            arrays_to_split.append(data["luminosity_distance"])
        split_result = train_test_split(*arrays_to_split,
                                        test_size=self.validation_split_fraction,
                                        train_size=1.0-self.validation_split_fraction,
                                        random_state=42)
        
        # Unpack results
        self.train_mass_1, self.val_mass_1 = split_result[0], split_result[1]
        self.train_mass_2, self.val_mass_2 = split_result[2], split_result[3]
        self.train_lambda_1, self.val_lambda_1 = split_result[4], split_result[5]
        self.train_lambda_2, self.val_lambda_2 = split_result[6], split_result[7]
        if self.is_gw_event:
            self.train_dL, self.val_dL = split_result[8], split_result[9]
        
        print(f"Training samples: {len(self.train_mass_1)}, Validation samples: {len(self.val_mass_1)}")
        
    def train(self):
        """
        Function to build the training data based on some specifications, and then branch off to the desired subfunction for the actual training.
        """
        # Load the data, from which we infer what system we are training for
        training_filename = os.path.join(self.outdir, "training_data.npz")
        self.load_training_data(training_filename)
        
        print("Creating the training data arrays")
        
        if self.source_type == "bns":
            x = [self.train_mass_1, self.train_mass_2, self.train_lambda_1, self.train_lambda_2]
            x_val = [self.val_mass_1, self.val_mass_2, self.val_lambda_1, self.val_lambda_2]
        
        elif self.source_type == "nsbh":
            x = [self.train_mass_1, self.train_mass_2, self.train_lambda_1]
            x_val = [self.val_mass_1, self.val_mass_2, self.val_lambda_1]
        elif self.source_type == "bhns":
            x = [self.train_mass_1, self.train_mass_2, self.train_lambda_2]
            x_val = [self.val_mass_1, self.val_mass_2, self.val_lambda_2]
                
        if self.is_gw_event:
            # If we have GW event data, also include the detector-frame chirp mass and luminosity distance
            x.append(self.train_dL)
            x_val.append(self.val_dL)
            print("Including luminosity distance in training data")
                
        self.x = np.array(x).T
        self.x_val = np.array(x_val).T
        
        # Show some stuff
        print("np.shape(self.x)")
        print(np.shape(self.x))
        
        print("np.min(x)")
        print(np.min(x))
        
        print("np.max(x)")
        print(np.max(x))
            
        if self.scale_input:
            print(f"Using MinMaxScaler to scale the input data x")
    
            # Combine training and validation data for fitting the scaler. I should have actually fitted the scaler before train-test split, but this will work as well
            x_combined = np.vstack([self.x, self.x_val])
            
            # Fit scaler on combined data
            scaler = MinMaxScaler()
            scaler.fit(x_combined)
            
            # Transform both datasets using the same scaler
            self.x = scaler.transform(self.x)
            self.x_val = scaler.transform(self.x_val)
            
            # Save the scaler to a file we can unpickle later on
            scaler_savename = os.path.join(self.outdir, "scaler.gz")
            joblib.dump(scaler, scaler_savename)
            print(f"Saved sklearn scaler to {scaler_savename}")
        
        print(f"Going to start training . . .")
        start_time = time.time()
        flow = self._train_glasflow()
        end_time = time.time()
        
        print(f"Training done. Took around {(end_time - start_time)/60:.2f} minutes")
        
        # Save the model
        save_path = os.path.join(self.outdir, "model.pt")
        print(f"Saving the model weights to {save_path}")
        torch.save(flow.state_dict(), save_path)
        
        # Save the model kwargs
        save_path = save_path.replace(".pt", "_kwargs.json")
        
        # Just dump any last kwargs we want to save here before finally saving it
        self.nf_kwargs["source_type"] = self.source_type
        self.nf_kwargs["N_samples_training"] = self.N_samples_training
        self.nf_kwargs["num_epochs"] = self.num_epochs
        self.nf_kwargs["learning_rate"] = self.learning_rate
        self.nf_kwargs["max_patience"] = self.max_patience
        self.nf_kwargs["batch_size"] = self.batch_size
        self.nf_kwargs["scale_input"] = self.scale_input
        self.nf_kwargs["num_bins"] = self.num_bins
        self.nf_kwargs["eos_samples_name"] = self.eos_samples_name
        self.nf_kwargs["training_filename"] = self.training_filename
        # Save CouplingNSF-specific hyperparameters as strings for JSON serialization
        self.nf_kwargs["batch_norm_within_blocks"] = str(self.batch_norm_within_blocks)
        self.nf_kwargs["batch_norm_between_transforms"] = str(self.batch_norm_between_transforms)
        self.nf_kwargs["activation"] = str(self.activation)
        self.nf_kwargs["dropout_probability"] = self.dropout_probability
        self.nf_kwargs["linear_transform"] = str(self.linear_transform) if self.linear_transform else "None"
        self.nf_kwargs["tail_bound"] = self.tail_bound
        
        print(f"Saving the model kwargs to {save_path}")
        with open(save_path, "w") as f:
            json.dump(self.nf_kwargs, f, indent=4)
        print("DONE training!")
        
    def _train_glasflow(self):
        """
        Simple wrapper around a glasflow model to train an unconditional normalizing flow on the data.
        """
        
        # Initialize the flow model based on glasflow_type
        self.nf_kwargs["model_type"] = self.glasflow_type
        
        # TODO: Support for other activation functions is future work
        print("Only RELU supported for now")
        
        if self.glasflow_type == "CouplingNSF":
            flow = CouplingNSF(n_inputs=self.nf_kwargs["n_inputs"],
                               n_transforms=self.n_transforms,
                               n_neurons=self.n_neurons,
                               n_blocks_per_transform=self.n_blocks_per_transform,
                               num_bins=self.num_bins,
                               batch_norm_within_blocks=self.batch_norm_within_blocks,
                               batch_norm_between_transforms=self.batch_norm_between_transforms,
                               activation=F.relu, # TODO: hardcoded for now
                               dropout_probability=self.dropout_probability,
                               linear_transform=self.linear_transform,
                               tail_bound=self.tail_bound
            )
        elif self.glasflow_type == "MaskedPiecewiseRationalQuadraticAutoregressiveFlow":
            flow = MaskedPiecewiseRationalQuadraticAutoregressiveFlow(
                n_inputs=self.nf_kwargs["n_inputs"],
                n_transforms=self.n_transforms,
                n_neurons=self.n_neurons,
                n_blocks_per_transform=self.n_blocks_per_transform,
                num_bins=self.num_bins
            )
        elif self.glasflow_type == "MaskedAffineAutoregressiveFlow":
            flow = MaskedAffineAutoregressiveFlow(
                n_inputs=self.nf_kwargs["n_inputs"],
                n_transforms=self.n_transforms,
                n_neurons=self.n_neurons,
                n_blocks_per_transform=self.n_blocks_per_transform
            )
        else:
            raise ValueError(f"Unsupported glasflow_type: {self.glasflow_type}")
        flow.to(DEVICE)
        
        # Initialize early stopping parameters
        best_val_loss = np.inf
        early_stop_counter = 0
        
        # DataLoader for batching the data
        x_tensor = torch.tensor(self.x, dtype=torch.float32)
        x_val_tensor = torch.tensor(self.x_val, dtype=torch.float32)
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop with tqdm
        optimizer = optim.Adam(flow.parameters(), lr=self.learning_rate)
        train_losses = []
        val_losses = []
        
        for epoch in tqdm.tqdm(range(self.num_epochs), desc="Training", unit="epoch"):
            epoch_loss = 0.0
            flow.train()

            for (batch,) in dataloader:
                batch = batch.to(DEVICE)
                
                optimizer.zero_grad()
                loss = -flow.log_prob(inputs=batch).mean()
                loss.backward()
                optimizer.step()

                batch_loss = loss.item()
                epoch_loss += batch_loss

            epoch_loss /= len(dataloader)
            train_losses.append(epoch_loss)
            
            # Validation loss calculation
            flow.eval()
            with torch.no_grad():
                x_val_batch = x_val_tensor.to(DEVICE)
                val_loss = -flow.log_prob(inputs=x_val_batch).mean().item()
                val_losses.append(val_loss)

            # Early stopping check using validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                # Save best model
                best_model_state = flow.state_dict().copy()
            else:
                early_stop_counter += 1
                if early_stop_counter >= self.max_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model
        flow.load_state_dict(best_model_state)
        self.plot_loss(np.array(train_losses), np.array(val_losses))
        return flow
    
    def plot_loss(self, train_loss: np.array, val_loss: np.array = None) -> None:
        
        # Make a plot of the loss trajectory
        plt.figure(figsize=(12, 6))
        plt.plot(train_loss, label="Training Loss")
        if val_loss is not None:
            plt.plot(val_loss, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss Trajectory")
        plt.legend()
        save_path = os.path.join(self.outdir, "training_loss.pdf")
        print(f"Saving the training loss plot to {save_path}")
        if all(train_loss > 0.0) and (val_loss is None or all(val_loss > 0.0)):
            plt.yscale("log")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()


    def plot_training_data(self):
        training_data_filename = os.path.join(self.outdir, "training_data.npz")
        data = np.load(training_data_filename)
        names = self.nf_kwargs["names"]
        n_params = len(names)
        
        samples = np.empty((self.N_samples_plot, n_params))
        for i, name in enumerate(names):
            samples[:, i] = data[name][:self.N_samples_plot]
            
        corner.corner(samples, labels=names)
        save_name = os.path.join(self.outdir, "training_data_corner.pdf")
        plt.savefig(save_name, bbox_inches="tight")
        plt.close()
        print(f"Saved corner plot of the training data to {save_name}")

def main():
    args = parser.parse_args()
    args = vars(args)
    trainer = NFPriorCreator(**args)
    trainer.create_data()
    trainer.plot_training_data()
    trainer.train()
    
if __name__ == "__main__":
    main()