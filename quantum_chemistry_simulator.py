import streamlit as st
from pyscf import gto, scf
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem


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

        # Generate IR spectrum (simulated)
        ir_frequencies = np.linspace(400, 4000, 100)
        ir_intensities = np.exp(-((ir_frequencies - 1600) / 300) ** 2)  # Gaussian-like peak

        return energy, homo_energy, lumo_energy, ir_frequencies, ir_intensities
    except Exception as e:
        st.error(f"Error in quantum chemistry calculation: {e}")
        return None, None, None, None, None

# Streamlit UI
st.title("Quantum Chemistry Simulator")
st.write("Perform molecular orbital calculations and spectral analysis.")

# User input: Molecular geometry or SMILES-to-XYZ conversion
mol_input = st.text_area("Enter Molecular Geometry (PySCF Format)", """
H 0.0 0.0 0.0
H 0.0 0.0 0.74
""")

# SMILES input and conversion
def smiles_to_xyz(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol)
    
    conf = mol.GetConformer()
        xyz_coords = "
".join(
        f"{mol.GetAtomWithIdx(i).GetSymbol()} {conf.GetAtomPosition(i).x} {conf.GetAtomPosition(i).y} {conf.GetAtomPosition(i).z}"
        for i in range(mol.GetNumAtoms())
    )
    return xyz_coords

smiles_input = st.text_input("Enter SMILES String (Optional)")
if smiles_input:
    mol_input = smiles_to_xyz(smiles_input)
    st.text_area("Generated Molecular Geometry (XYZ Format)", mol_input, height=150)

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
