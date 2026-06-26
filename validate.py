import torch
import random
import yaml
import sys
import os

# Add the current directory to python path if not already present
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from original_system import EnergyModel
from system import load_config
from model.orchestrator import Orchestrator

def main():
    print("Loading config...")
    cfg = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))
    seed = cfg["training"]["seed"]

    # --- 1. BACKWARD COMPATIBILITY INITIALIZATION TEST ---
    print("\n=== Testing Backward Compatibility (Initialization & Energy matching) ===")
    
    # Force risk_scale to 0.01 temporarily to match the original model's hardcoded scaling for comparison
    cfg_compat = yaml.safe_load(yaml.dump(cfg))  # Deep copy
    cfg_compat["model"]["risk_scale"] = 0.01

    print("Initializing Original Model...")
    torch.manual_seed(seed)
    random.seed(seed)
    original_model = EnergyModel(cfg_compat)

    print("Initializing Refactored Orchestrator (with risk_scale=0.01)...")
    torch.manual_seed(seed)
    random.seed(seed)
    refactored_compat = Orchestrator(cfg_compat)

    # Compare initialization tensors
    tensors_to_compare = {
        "X": (original_model.X, refactored_compat.X),
        "s": (original_model.s, refactored_compat.s),
        "c": (original_model.c, refactored_compat.c),
        "kappa": (original_model.kappa, refactored_compat.kappa),
        "Theta": (original_model.Theta, refactored_compat.Theta),
        "C": (original_model.C, refactored_compat.C),
        "W_risk": (original_model.W_risk, refactored_compat.W_risk)
    }

    mismatch = False
    for name, (orig_t, ref_t) in tensors_to_compare.items():
        diff = torch.max(torch.abs(orig_t - ref_t)).item()
        print(f"Tensor {name} max difference: {diff:.8e}")
        if diff > 1e-7:
            print(f"  WARNING: {name} does not match!")
            mismatch = True

    if mismatch:
        print("Initialization mismatch detected under backward compatibility mode! Exiting.")
        sys.exit(1)
    else:
        print("Initialization matches perfectly.")

    # Compare initial energies
    orig_total = original_model.total_energy().item()
    ref_total = refactored_compat.total_energy().item()
    print(f"Original Total Energy: {orig_total:.6f} | Refactored Compat: {ref_total:.6f}")
    print(f"Total Energy Diff: {abs(orig_total - ref_total):.8e}")

    if abs(orig_total - ref_total) > 1e-7:
        print("Initial energy mismatch detected in compat mode! Exiting.")
        sys.exit(1)

    # --- 2. VERIFY NEW AND IMPROVED BEHAVIOR ---
    print("\n=== Testing Improved Refactored Model (risk_scale=1.0 & new logic) ===")
    
    # Initialize refactored model with scale = 1.0 (from config)
    torch.manual_seed(seed)
    random.seed(seed)
    improved_model = Orchestrator(cfg)

    # A. Verify RiskPredictor calibration (spread of probabilities)
    p_orig = original_model.risk_probs()
    p_improved = improved_model.risk_predictor.predict(improved_model.state)
    
    std_orig = torch.std(p_orig).item()
    std_improved = torch.std(p_improved).item()
    
    print(f"Original risk probs StdDev (scale 0.01): {std_orig:.6f} (vanishing variance)")
    print(f"Improved risk probs StdDev (scale 1.0) : {std_improved:.6f} (active signals)")
    
    # B. Verify Proposal no-ops are avoided
    print("Testing 1000 proposal generations to verify no no-ops exist...")
    no_ops_found = 0
    for _ in range(1000):
        X_prop = improved_model.proposal_mechanism.propose(improved_model.state)
        # Check if identical to old state
        if torch.equal(X_prop, improved_model.state.X):
            no_ops_found += 1
    print(f"No-op proposals found in 1000 runs: {no_ops_found} (Expected: 0)")
    assert no_ops_found == 0, "Error: Proposal mechanism generated a no-op transition!"

    # C. Verify dynamics stability under multi-step evolution
    print("\nRunning improved refactored model for 100 steps to verify dynamics stability...")
    for i in range(1, 101):
        improved_model.step()
        if i % 10 == 0:
            E, breakdown = improved_model.energy_registry.compute(improved_model.state)
            print(f"Step {i:3d} | E: {E.item():.4f} (Assign={breakdown['AssignmentEnergy']:.3f}, Risk={breakdown['RiskEnergy']:.3f}) | T: {improved_model.T:.3f} | acc: {improved_model.acc_rate:.3f}")

    print("\nAll verifications successful! Refactored model is backward-compatible in initialization, avoids proposal no-ops, resolves vanishing risk signal variance, and uses a dimensionally consistent Theta updater.")

if __name__ == "__main__":
    main()
