import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import corner
import copy
import json
import torch
import joblib
import warnings
from scipy.stats import gaussian_kde
from scipy.special import kl_div

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Try to import flow libraries
try:
    from glasflow.flows.nsf import CouplingNSF
    from glasflow.flows.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveFlow, MaskedAffineAutoregressiveFlow
    GLASFLOW_AVAILABLE = True
except ImportError:
    GLASFLOW_AVAILABLE = False
    print("Warning: glasflow not available")

try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    from flowjax.flows import coupling_flow, masked_autoregressive_flow, block_neural_autoregressive_flow
    from flowjax.distributions import Normal
    import equinox as eqx
    FLOWJAX_AVAILABLE = True
except ImportError:
    FLOWJAX_AVAILABLE = False
    print("Warning: flowjax not available")

# Matplotlib parameters for consistent styling
params = {
    "axes.grid": True,
    "text.usetex": False,
    "font.family": "serif",
    "ytick.color": "black",
    "xtick.color": "black",
    "axes.labelcolor": "black",
    "axes.edgecolor": "black",
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "axes.labelsize": 16,
    "legend.fontsize": 16,
    "legend.title_fontsize": 16,
    "figure.titlesize": 16
}
plt.rcParams.update(params)

# Default corner plot configuration
default_corner_kwargs = dict(
    bins=40, 
    smooth=1., 
    show_titles=False,
    label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), 
    color="blue",
    plot_density=True, 
    plot_datapoints=False, 
    fill_contours=True,
    max_n_ticks=4, 
    min_n_ticks=3,
    truth_color="red",
    save=False
)

def make_cornerplot(chains_1: np.array, 
                    chains_2: np.array,
                    name: str,
                    my_range: list = None,
                    truths: list = None,
                    labels: list = None):
    """
    Plot a cornerplot comparing training data samples and NF samples.
    """
    
    # Plot training data first
    corner_kwargs = copy.deepcopy(default_corner_kwargs)
    hist_1d_kwargs = {"density": True, "color": "blue"}
    corner_kwargs["color"] = "blue"
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    fig = corner.corner(chains_1, range=my_range, truths=truths, 
                       labels=labels, **corner_kwargs)

    # Overlay NF samples
    corner_kwargs["color"] = "red"
    hist_1d_kwargs = {"density": True, "color": "red"}
    corner_kwargs["hist_kwargs"] = hist_1d_kwargs
    corner.corner(chains_2, truths=truths, range=my_range, 
                 fig=fig, labels=labels, **corner_kwargs)

    # Add legend
    if my_range is None or len(my_range) <= 2:
        fs = 14
    else:
        fs = 24
    plt.text(0.75, 0.75, "Training data", fontsize=fs, color="blue", 
            transform=plt.gcf().transFigure)
    plt.text(0.75, 0.65, "Normalizing flow", fontsize=fs, color="red", 
            transform=plt.gcf().transFigure)

    plt.savefig(name, bbox_inches="tight")
    plt.close()

def get_parameter_labels(parameter_names: list) -> list:
    """Convert parameter names to LaTeX labels for plotting."""
    translation_dict = {
        "chirp_mass": r"$M_c$ [M$_\odot$]",
        "chirp_mass_source": r"$M_c^{\rm{source}}$ [M$_\odot$]",
        "q": r"$q$",
        "mass_ratio": r"$q$",
        "lambda_1": r"$\Lambda_1$",
        "lambda_2": r"$\Lambda_2$",
        "lambda_tilde": r"$\tilde{\Lambda}$",
        "delta_lambda_tilde": r"$\delta\tilde{\Lambda}$",
        "luminosity_distance": r"$d_L$ [Mpc]",
        "redshift": r"$z$",
        "m_1": r"$m_1$ [M$_\odot$]",
        "m_2": r"$m_2$ [M$_\odot$]"
    }
    return [translation_dict.get(name, name) for name in parameter_names]

class FlowEvaluator:
    """Evaluate normalizing flow models."""
    
    def __init__(self, model_path: str, n_samples: int = 10_000):
        self.model_path = model_path
        self.n_samples = n_samples
        self.figures_outdir = os.path.join(model_path, "figures")
        os.makedirs(self.figures_outdir, exist_ok=True)
        
        # Load model configuration
        nf_kwargs_path = os.path.join(model_path, "model_kwargs.json")
        if not os.path.exists(nf_kwargs_path):
            raise FileNotFoundError(f"Model config not found at {nf_kwargs_path}")
            
        with open(nf_kwargs_path, "r") as f:
            self.nf_kwargs = json.load(f)
        
        # Load model and scaler
        self.flow, self.scaler = self.load_model()
        
        # Load training data
        self.training_data = self.load_training_data()
    
    def load_model(self):
        """Load the trained flow model."""
        use_flowjax = self.nf_kwargs.get("use_flowjax", "False") == "True"
        n_inputs = self.nf_kwargs["n_inputs"]
        
        if use_flowjax:
            if not FLOWJAX_AVAILABLE:
                raise ImportError("flowjax is required but not installed")
                
            print(f"Loading flowJAX model with {n_inputs} inputs")
            nf_path = os.path.join(self.model_path, "model.eqx")
            
            if not os.path.exists(nf_path):
                raise FileNotFoundError(f"flowJAX model not found at {nf_path}")
            
            # Create base distribution
            base_dist = Normal(jnp.zeros(n_inputs))
            key = jr.key(42)
            
            # Choose flow type based on model configuration
            model_type = self.nf_kwargs.get("model_type", "coupling_flow")
            if model_type == "coupling_flow":
                flow = coupling_flow(
                    key=key,
                    base_dist=base_dist,
                    flow_layers=self.nf_kwargs["n_transforms"],
                    nn_width=self.nf_kwargs["n_neurons"],
                    nn_depth=self.nf_kwargs.get("nn_depth", 2)
                )
            elif model_type == "masked_autoregressive_flow":
                flow = masked_autoregressive_flow(
                    key=key,
                    base_dist=base_dist,
                    flow_layers=self.nf_kwargs["n_transforms"],
                    nn_width=self.nf_kwargs["n_neurons"],
                    nn_depth=self.nf_kwargs.get("nn_depth", 2)
                )
            elif model_type == "block_neural_autoregressive_flow":
                flow = block_neural_autoregressive_flow(
                    key=key,
                    base_dist=base_dist,
                    flow_layers=self.nf_kwargs["flow_layers"],
                    nn_depth=self.nf_kwargs["nn_depth"],
                    nn_block_dim=self.nf_kwargs["nn_block_dim"]
                )
            else:
                raise ValueError(f"Unsupported flowJAX model type: {model_type}")
            
            flow = eqx.tree_deserialise_leaves(nf_path, flow)
            
        else:
            if not GLASFLOW_AVAILABLE:
                raise ImportError("glasflow is required but not installed")
                
            print(f"Loading glasflow model with {n_inputs} inputs")
            nf_path = os.path.join(self.model_path, "model.pt")
            
            if not os.path.exists(nf_path):
                raise FileNotFoundError(f"glasflow model not found at {nf_path}")
            
            # Determine glasflow model type
            glasflow_type = self.nf_kwargs.get("glasflow_type", "CouplingNSF")
            
            if glasflow_type == "CouplingNSF":
                import torch.nn.functional as F
                flow = CouplingNSF(
                    n_inputs=n_inputs,
                    n_transforms=self.nf_kwargs["n_transforms"],
                    n_neurons=self.nf_kwargs["n_neurons"],
                    n_blocks_per_transform=self.nf_kwargs["n_blocks_per_transform"],
                    num_bins=self.nf_kwargs["num_bins"],
                    batch_norm_within_blocks=self.nf_kwargs.get("batch_norm_within_blocks", "False") == "True",
                    batch_norm_between_transforms=self.nf_kwargs.get("batch_norm_between_transforms", "False") == "True",
                    activation=F.relu,
                    dropout_probability=float(self.nf_kwargs.get("dropout_probability", 0.0)),
                    tail_bound=float(self.nf_kwargs.get("tail_bound", 5.0))
                )
            elif glasflow_type == "MaskedPiecewiseRationalQuadraticAutoregressiveFlow":
                flow = MaskedPiecewiseRationalQuadraticAutoregressiveFlow(
                    n_inputs=n_inputs,
                    n_transforms=self.nf_kwargs["n_transforms"],
                    n_neurons=self.nf_kwargs["n_neurons"],
                    n_blocks_per_transform=self.nf_kwargs["n_blocks_per_transform"],
                    num_bins=self.nf_kwargs["num_bins"]
                )
            elif glasflow_type == "MaskedAffineAutoregressiveFlow":
                flow = MaskedAffineAutoregressiveFlow(
                    n_inputs=n_inputs,
                    n_transforms=self.nf_kwargs["n_transforms"],
                    n_neurons=self.nf_kwargs["n_neurons"],
                    n_blocks_per_transform=self.nf_kwargs["n_blocks_per_transform"]
                )
            else:
                raise ValueError(f"Unsupported glasflow model type: {glasflow_type}")
            
            flow.load_state_dict(torch.load(nf_path, map_location=torch.device('cpu')))
            flow.eval()
            flow.compile()
        
        # Load scaler if it exists
        scaler_path = os.path.join(self.model_path, "scaler.gz")
        scaler = None
        if os.path.exists(scaler_path):
            print(f"Loading scaler from {scaler_path}")
            scaler = joblib.load(scaler_path)
        else:
            print("No scaler found")
        
        return flow, scaler
    
    def load_training_data(self):
        """Load training data for comparison."""
        training_data_path = os.path.join(self.model_path, "training_data.npz")
        if os.path.exists(training_data_path):
            return np.load(training_data_path)
        else:
            print("Warning: No training data found for comparison")
            return None
    
    def generate_samples(self):
        """Generate samples from the flow model."""
        use_flowjax = self.nf_kwargs.get("use_flowjax", "False") == "True"
        
        if use_flowjax:
            # flowJAX sampling
            key = jr.key(123)
            keys = jr.split(key, self.n_samples)
            
            @jax.jit
            def sample_fn(sample_key):
                return self.flow.sample(sample_key, (1,)).flatten()
            
            # Vectorize over keys
            nf_samples_jax = jax.vmap(sample_fn)(keys)
            nf_samples = np.array(nf_samples_jax)
        else:
            # glasflow sampling
            with torch.no_grad():
                nf_samples = self.flow.sample(self.n_samples).cpu().numpy()
        
        # Apply inverse scaling if scaler was used
        if self.scaler is not None:
            nf_samples = self.scaler.inverse_transform(nf_samples)
        
        # Apply inverse log transform if needed
        if self.nf_kwargs.get("take_log_lambda", "False") == "True":
            use_tilde = self.nf_kwargs.get("use_tilde", "False") == "True"
            source_type = self.nf_kwargs.get("source_type", "bns")
            
            if source_type == "bns":
                if use_tilde:
                    nf_samples[:, -2:] = np.exp(nf_samples[:, -2:])
                else:
                    nf_samples[:, -2:] = np.exp(nf_samples[:, -2:])
            elif source_type == "nsbh":
                nf_samples[:, -1] = np.exp(nf_samples[:, -1])
        
        return nf_samples
    
    def get_training_samples(self):
        """Get training samples in the same format as NF samples."""
        if self.training_data is None:
            return None
            
        parameter_names = self.nf_kwargs.get("names", [])
        training_columns = []
        
        for param_name in parameter_names:
            if param_name in self.training_data:
                training_columns.append(self.training_data[param_name])
            elif param_name == "mass_ratio" and "m1" in self.training_data and "m2" in self.training_data:
                mass_ratio = self.training_data["m2"] / self.training_data["m1"]
                training_columns.append(mass_ratio)
            elif param_name == "q" and "m1" in self.training_data and "m2" in self.training_data:
                q = self.training_data["m2"] / self.training_data["m1"]
                training_columns.append(q)
            else:
                raise KeyError(f"Parameter '{param_name}' not found in training data")
        
        return np.column_stack(training_columns)
    
    def evaluate(self):
        """Run full evaluation of the model."""
        print(f"Evaluating model at: {self.model_path}")
        print(f"Generating {self.n_samples} samples...")
        
        # Generate samples from the flow
        nf_samples = self.generate_samples()
        print(f"Generated {len(nf_samples)} samples with shape {nf_samples.shape}")
        
        # Get training samples if available
        if self.training_data is not None:
            training_samples = self.get_training_samples()
            print(f"Training samples shape: {training_samples.shape}")
            
            # Create comparison plots
            parameter_names = self.nf_kwargs.get("names", [])
            labels = get_parameter_labels(parameter_names)
            
            # Determine plot ranges
            all_data = np.concatenate([training_samples, nf_samples], axis=0)
            ranges = []
            for i in range(all_data.shape[1]):
                lower, upper = np.quantile(all_data[:, i], [0.01, 0.99])
                ranges.append([lower, upper])
            
            # Create corner plot
            corner_filename = os.path.join(self.figures_outdir, "comparison.pdf")
            make_cornerplot(training_samples, nf_samples, corner_filename, 
                           my_range=ranges, labels=labels)
            print(f"Corner plot saved to: {corner_filename}")
            
            # Check sample validity
            violations = self.check_sample_validity(nf_samples)
            if violations['total_bad_samples'] > 0:
                print(f"Warning: {violations['total_bad_samples']} bad samples ({violations['bad_percentage']:.2f}%)")
            else:
                print("All samples satisfy physical constraints")
        else:
            print("No training data available for comparison")
        
        # Print sample statistics
        print("\nSample statistics:")
        parameter_names = self.nf_kwargs.get("names", [f"param_{i}" for i in range(nf_samples.shape[1])])
        for i, param_name in enumerate(parameter_names):
            values = nf_samples[:, i]
            print(f"{param_name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, range=[{np.min(values):.3f}, {np.max(values):.3f}]")
    
    def test_sampling(self):
        """Quick test to see if model can generate samples."""
        print(f"Testing sample generation from: {self.model_path}")
        try:
            samples = self.generate_samples()
            print(f"Success! Generated {len(samples)} samples with shape {samples.shape}")
            
            # Check for issues
            if np.any(np.isnan(samples)) or np.any(np.isinf(samples)):
                print("Warning: Found NaN or infinite values")
                return False
            
            print("Sample ranges:")
            for i in range(samples.shape[1]):
                print(f"  Param {i}: [{samples[:, i].min():.3f}, {samples[:, i].max():.3f}]")
            
            return True
        except Exception as e:
            print(f"Error during sampling: {e}")
            return False
    
    def check_sample_validity(self, nf_samples):
        """Check how many samples violate physical constraints."""
        parameter_names = self.nf_kwargs.get("names", [])
        n_samples = len(nf_samples)
        violations = {'total_bad_samples': 0, 'total_samples': n_samples, 'bad_percentage': 0.0}
        
        bad_samples_mask = np.zeros(n_samples, dtype=bool)
        
        # Check mass ratio constraint: q in [0, 1]
        if "mass_ratio" in parameter_names or "q" in parameter_names:
            q_param = "mass_ratio" if "mass_ratio" in parameter_names else "q"
            q_idx = parameter_names.index(q_param)
            q_values = nf_samples[:, q_idx]
            
            q_violations = (q_values < 0) | (q_values > 1)
            violations[f'{q_param}_out_of_bounds'] = np.sum(q_violations)
            bad_samples_mask |= q_violations
        
        # Check lambda constraints: lambda_1, lambda_2 > 0
        for lambda_param in ["lambda_1", "lambda_2"]:
            if lambda_param in parameter_names:
                lambda_idx = parameter_names.index(lambda_param)
                lambda_values = nf_samples[:, lambda_idx]
                
                lambda_violations = lambda_values <= 0
                violations[f'{lambda_param}_negative'] = np.sum(lambda_violations)
                bad_samples_mask |= lambda_violations
        
        # Total bad samples
        total_bad = np.sum(bad_samples_mask)
        violations['total_bad_samples'] = total_bad
        violations['bad_percentage'] = (total_bad / n_samples) * 100
        
        return violations

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a normalizing flow model")
    parser.add_argument("--model_path", help="Path to the model directory", type=str, required=True)
    parser.add_argument("--test-only", action="store_true", help="Only test if model can generate samples")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of samples to generate")
    
    args = parser.parse_args()
    
    try:
        evaluator = FlowEvaluator(args.model_path, args.n_samples)
        
        if args.test_only:
            success = evaluator.test_sampling()
            return success
        else:
            evaluator.evaluate()
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)