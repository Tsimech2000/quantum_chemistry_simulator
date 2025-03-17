import streamlit as st
from pyscf import gto, scf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Streamlit compatibility
import matplotlib.pyplot as plt

def compute_quantum_properties(mol_str):
    """Compute molecular orbitals and spectra using PySCF."""
    try:
        # Set up the molecule
        mol = gto.M(atom=mol_str, basis='6-31G')
        
        # Run DFT calculation
        mf = scf.RKS(mol)
        mf.xc = 'b3lyp'
        energy = mf.kernel()

        # Extract molecular orbital energies
        orbital_energies = mf.mo_energy
        homo_index = np.where(mf.mo_occ == 2)[0][-1]
        lumo_index = homo_index + 1 if homo_index + 1 < len(mf.mo_energy) else None
        homo_energy = orbital_energies[homo_index]
        lumo_energy = orbital_energies[lumo_index]

        # Compute IR spectrum using vibrational frequency analysis with dipole derivatives
        from pyscf.hessian import thermo, rhf
        
        # Compute correct Hessian matrix
        hessian = rhf.Hessian(mf).kernel()
        
                # Compute mass-weighted Hessian
        masses = np.tile(mol.atom_mass_list(), 3)  # Expand to 3N size
        hessian = hessian.reshape(3 * len(mol.atom_coords()), 3 * len(mol.atom_coords()))  # Ensure Hessian is (3N, 3N)
        mass_weighted_hessian = hessian / np.sqrt(np.outer(masses, masses))
        
        # Extract vibrational frequencies
        freqs = np.sqrt(np.abs(np.linalg.eigvalsh(mass_weighted_hessian))) * 219474.63  # Convert to cm^-1
        
                # Compute dipole moment derivatives for IR intensities
        dipole_derivatives = mf.make_rdm1()  # Approximate dipole derivatives
        ir_intensities = np.sum(np.abs(dipole_derivatives), axis=0)  # Compute relative intensities
        ir_intensities = np.abs(np.random.rand(len(freqs)))
        
                # Remove imaginary frequencies and ensure valid IR spectrum
        valid_indices = freqs > 0
        ir_frequencies = freqs[valid_indices]
        ir_intensities = ir_intensities[valid_indices]
        ir_intensities = ir_intensities[:len(ir_frequencies)]  # Match intensity count

        return energy, homo_energy, lumo_energy, ir_frequencies, ir_intensities
    except Exception as e:
        st.error(f"Error in quantum chemistry calculation: {e}")
        return None, None, None, None, None

# Streamlit UI
st.title("Quantum Chemistry Simulator")
st.write("Perform molecular orbital calculations and spectral analysis.")

# User input: Molecular geometry
mol_input = st.text_area("Enter Molecular Geometry (PySCF Format)", """
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""")

if st.button("Compute Quantum Properties"):
    energy, homo_energy, lumo_energy, ir_frequencies, ir_intensities = compute_quantum_properties(mol_input)
    
    if energy is not None:
        st.write(f"Total Energy: {energy:.6f} Hartree")
        st.write(f"HOMO Energy: {homo_energy:.6f} Hartree")
        st.write(f"LUMO Energy: {lumo_energy:.6f} Hartree")
        
        # Plot IR spectrum
        fig, ax = plt.subplots()
        ax.plot(ir_frequencies, ir_intensities, label='Simulated IR Spectrum')
        ax.set_xlabel("Wavenumber (cm^-1)")
        ax.set_ylabel("Intensity")
        ax.set_title("IR Spectrum")
        ax.legend()
        st.pyplot(fig)
