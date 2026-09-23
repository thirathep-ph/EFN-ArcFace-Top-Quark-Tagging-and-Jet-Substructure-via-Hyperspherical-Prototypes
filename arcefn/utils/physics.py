"""
Physics Utilities: Optimized FastJet Feature Extraction (Native SoftDrop)
==========================================================================
Vectorized extraction of 10 advanced substructure features.
Uses Native exclusive_jets_softdrop_grooming from FastJet Python wrapper.

Order: [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), Tau21, Tau32, zg, theta_g]
"""
import numpy as np
import awkward as ak
import fastjet

def compute_features(events: ak.Array) -> np.ndarray:
    """Extract 10 substructure features via FastJet batch clustering.

    Clusters each jet using the Cambridge/Aachen algorithm (R=2.0),
    applies SoftDrop grooming (beta=0.0, z_cut=0.1), and computes:
    mass, mSD, multiplicity, nSD, sqrt(d12), sqrt(d23), tau21,
    tau32, zg, theta_g.

    Parameters
    ----------
    events : ak.Array
        Jagged array of Momentum4D Lorentz vectors (compatible with
        FastJet batch API). Must have .px, .py, .pz, .E fields.

    Returns
    -------
    np.ndarray
        Float32 array of shape (N, 10) with columns ordered as:
        [Mass, mSD, Mult, nSD, sqrt(d12), sqrt(d23), tau21, tau32,
         zg, theta_g].
    """
    # 1. Prepare constituents
    px = ak.values_astype(events.px, np.float64)
    py = ak.values_astype(events.py, np.float64)
    pz = ak.values_astype(events.pz, np.float64)
    E = ak.values_astype(events.E, np.float64)
    cons = ak.zip({"px": px, "py": py, "pz": pz, "E": E})

    # 2. Cluster Sequence (Agnostic R=2.0 for Exclusive Clustering)
    # Using a larger R ensures we don't cut off radiation when forcing n_jets=1.
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 2.0)
    cs = fastjet.ClusterSequence(cons, jet_def)
    
    # --- Native SoftDrop Grooming ---
    # Field names in native output are suffix-appended (e.g., msoftdrop)
    sd_jets = cs.exclusive_jets_softdrop_grooming(
        njets=1, 
        beta=0.0, 
        symmetry_cut=0.1, 
        R0=2.0
    )
    m_sd = ak.to_numpy(sd_jets.msoftdrop)

    # --- Standard Jets & Scales ---
    jets = cs.exclusive_jets(n_jets=1)
    mass = ak.to_numpy(jets.mass)
    multiplicity = ak.to_numpy(ak.num(events)).astype(np.float32)
    
    # Splitting scales
    d12 = ak.to_numpy(np.sqrt(np.abs(cs.exclusive_dmerge(1))))
    d23 = ak.to_numpy(np.sqrt(np.abs(cs.exclusive_dmerge(2))))

    # --- Shape Features ---
    # N-subjettiness uses the FastJet wrapper defaults: OnePass_KT axes,
    # normalized measure, beta=1.0, R0=0.8. R0 cancels in the tau21/tau32
    # ratios, so its specific value does not affect the downstream ratios.
    nj = cs.njettiness()
    tau1, tau2, tau3 = ak.to_numpy(nj[:, 0]), ak.to_numpy(nj[:, 1]), ak.to_numpy(nj[:, 2])
    tau21 = np.where(tau1 > 1e-10, tau2 / tau1, 1.0)
    tau32 = np.where(tau2 > 1e-10, tau3 / tau2, 1.0)

    # --- Groomed Variables (zg, theta_g, nSD) ---
    # Using specific field names from fastjet Python wrapper
    zg = ak.to_numpy(sd_jets.symmetrysoftdrop)
    theta_g = ak.to_numpy(sd_jets.deltaRsoftdrop)
    n_sd = ak.to_numpy(ak.num(sd_jets.constituents)).astype(np.float32)

    # 3. Final Stack (Perfect Order: ML + Basic Physics + Advanced Substructure)
    features = np.column_stack([
        mass, m_sd, multiplicity, n_sd, d12, d23, tau21, tau32, zg, theta_g
    ])

    return features.astype(np.float32)

def get_lund_coordinates(events: ak.Array) -> tuple[np.ndarray, np.ndarray]:
    """Compute Lund plane coordinates via C/A declustering.

    Parameters
    ----------
    events : ak.Array
        Jagged array of Momentum4D Lorentz vectors.

    Returns
    -------
    ln_inv_delta : np.ndarray
        ln(1/Delta) coordinates of primary Lund declusterings.
    ln_kt : np.ndarray
        ln(kt) coordinates of primary Lund declusterings.
    """
    jet_def = fastjet.JetDefinition(fastjet.cambridge_algorithm, 2.0)
    # 1. Zip constituents (Force float64 for FastJet C++ interface)
    px = ak.values_astype(events.px, np.float64)
    py = ak.values_astype(events.py, np.float64)
    pz = ak.values_astype(events.pz, np.float64)
    E = ak.values_astype(events.E, np.float64)
    cons = ak.zip({"px": px, "py": py, "pz": pz, "E": E})
    
    # 2. Cluster
    cs = fastjet.ClusterSequence(cons, jet_def)
    
    # 3. Use FastJet's native Lund Declustering (Much faster and more stable)
    # This returns all declusterings along the primary branch for the hardest jet (njets=1)
    lund_data = cs.exclusive_jets_lund_declusterings(njets=1)
    
    # Flatten the results to a single 1D array (axis=None)
    delta = ak.flatten(lund_data.Delta, axis=None)
    kt = ak.flatten(lund_data.kt, axis=None)
    
    # Filter out non-physical values and apply logs
    mask = (delta > 0) & (kt > 0)
    # Convert to numpy explicitly
    delta_np = ak.to_numpy(delta[mask])
    kt_np = ak.to_numpy(kt[mask])
    
    ln_inv_delta = np.log(1.0 / delta_np)
    ln_kt = np.log(kt_np)
    
    return ln_inv_delta, ln_kt

def print_feature_stats(features, label="Features"):
    names = ["Mass", "mSD", "Mult", "nSD", "sqrt(d12)", "sqrt(d23)", "Tau21", "Tau32", "zg", "theta_g"]
    print(f"\n  {label}:")
    for i, name in enumerate(names):
        vals = features[:, i]
        print(f"    {name:10s}: mean={np.mean(vals):.4f}, std={np.std(vals):.4f}")

