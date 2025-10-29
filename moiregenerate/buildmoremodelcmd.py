"""
Module for building moiré structures
Input files should have z-axis as vacuum direction, and interlayer distance needs to be specified
"""

import json
import os
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import ase.io as aio
from ase import Atoms
from ase.build import make_supercell, sort
from ase.constraints import FixAtoms
from ase.geometry import get_distances
from scipy import optimize

from .moregenerate import MoireData


def get_layer_thickness(structure: Atoms) -> float:
    """
    Calculate the thickness of an atomic layer

    Args:
        structure: Atomic structure object

    Returns:
        Layer thickness (Angstrom)
    """
    z_positions = structure.get_positions()[:, 2]
    return np.max(z_positions) - np.min(z_positions)


def get_rotation_matrix_3d(theta: float) -> np.ndarray:
    """
    Get 3D rotation matrix (around z-axis)

    Args:
        theta: Rotation angle (radians)

    Returns:
        3x3 rotation matrix
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_theta, -sin_theta, 0],
        [sin_theta, cos_theta, 0],
        [0, 0, 1]
    ])


def get_rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Get 2D rotation matrix

    Args:
        theta: Rotation angle (radians)

    Returns:
        2x2 rotation matrix
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    return np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])


def build_supercell(structure: Atoms, transformation_matrix: np.ndarray) -> Atoms:
    """
    Build supercell structure through given matrix

    Args:
        structure: Original atomic structure
        transformation_matrix: 3x3 transformation matrix

    Returns:
        Supercell structure
    """
    structure = structure.copy()

    # Save fixed atom constraints
    fixed_indices = np.zeros(len(structure), dtype=np.int32)
    for constraint in structure.constraints:
        if isinstance(constraint, FixAtoms):
            fixed_indices[constraint.get_indices()] = 1
    structure.arrays["fix_atoms"] = fixed_indices

    # Build supercell
    supercell = make_supercell(structure, transformation_matrix)

    return supercell


def build_moire_pattern(
    layer_a: Atoms,
    layer_b: Atoms,
    rotation_angle: float,
    supercell_matrix_a: np.ndarray,
    interlayer_distance: float,
    vacuum_thickness: float = 20.0
) -> Atoms:
    """
    Build moiré structure

    Args:
        layer_a: Lower layer atomic structure
        layer_b: Upper layer atomic structure
        rotation_angle: Rotation angle (radians)
        supercell_matrix: 2x2 supercell matrix
        interlayer_distance: Interlayer distance
        vacuum_thickness: Vacuum layer thickness

    Returns:
        Moiré structure
    """
    # Ensure supercell matrix is right-handed coordinate system
    if np.linalg.det(supercell_matrix_a) < 0:
        supercell_matrix_a = np.array([supercell_matrix_a[1], supercell_matrix_a[0]])

    cell_a = layer_a.get_cell().array
    cell_b = layer_b.get_cell().array

    # Extract 2D lattice vectors
    lattice_a = cell_a[:2, :2]
    lattice_b = cell_b[:2, :2]

    # Apply rotation
    rotation_2d = get_rotation_matrix_2d(rotation_angle)
    rotation_3d = get_rotation_matrix_3d(rotation_angle)
    lattice_b = np.dot(rotation_2d, lattice_b.T).T

    # Build supercell
    supercell_matrix_b = np.round(np.dot(supercell_matrix_a, np.dot(lattice_a, np.linalg.inv(lattice_b))))

    # Build supercell
    supercell_a = build_supercell(
        layer_a,
        [[supercell_matrix_a[0, 0], supercell_matrix_a[0, 1], 0],
         [supercell_matrix_a[1, 0], supercell_matrix_a[1, 1], 0],
         [0, 0, 1]]
    )

    supercell_b = build_supercell(
        layer_b,
        [[supercell_matrix_b[0, 0], supercell_matrix_b[0, 1], 0],
         [supercell_matrix_b[1, 0], supercell_matrix_b[1, 1], 0],
         [0, 0, 1]]
    )

    # Rotate upper layer structure
    supercell_b.positions = np.dot(rotation_3d, supercell_b.positions.T).T
    supercell_b.set_cell(np.dot(rotation_3d, supercell_b.get_cell().array.T).T)

    # Create moiré model
    moire_structure = Atoms(
        cell=(supercell_a.get_cell() + supercell_b.get_cell()) / 2,
        pbc=[True, True, False]
    )

    average_lattice = moire_structure.get_cell()

    average_lattice[2] = supercell_a.cell[2]
    supercell_a.set_cell(average_lattice, scale_atoms=True)
    average_lattice[2] = supercell_b.cell[2]
    supercell_b.set_cell(average_lattice, scale_atoms=True)

    # Adjust lower layer structure position
    min_z_a = np.min(supercell_a.get_positions()[:, 2])
    supercell_a.positions[:, 2] += (1 - min_z_a)
    max_z_a = np.max(supercell_a.get_positions()[:, 2])

    # Add lower layer to moiré structure
    moire_structure.extend(supercell_a)

    # Process upper layer structure
    min_z_b = np.min(supercell_b.get_positions()[:, 2])
    # No shift, directly add upper layer
    z_offset = -min_z_b + max_z_a + interlayer_distance
    supercell_b.positions[:, 2] += z_offset
    moire_structure.extend(supercell_b)

    # Sort atoms and apply constraints
    moire_structure = sort(moire_structure)

    fixed_atoms = np.where(moire_structure.arrays["fix_atoms"] == 1)[0]
    if len(fixed_atoms) > 0:
        moire_structure.set_constraint(FixAtoms(fixed_atoms))

    moire_structure.center(vacuum=vacuum_thickness / 2, axis=2)
    moire_structure.pbc = True

    return moire_structure


def get_moire_pattern_info(
    layer_a: Atoms,
    layer_b: Atoms,
    rotation_angle: float,
    supercell_matrix: np.ndarray,
    interlayer_distance: float,
    vacuum_thickness: float = 20.0
) -> Dict[str, Any]:
    """
    Get moiré structure information

    Args:
        layer_a: Lower layer atomic structure
        layer_b: Upper layer atomic structure
        rotation_angle: Rotation angle (radians)
        supercell_matrix: 2x2 supercell matrix
        interlayer_distance: Interlayer distance
        vacuum_thickness: Vacuum layer thickness

    Returns:
        Dictionary containing structure information
    """
    if np.linalg.det(supercell_matrix) < 0:
        supercell_matrix = np.array([supercell_matrix[1], supercell_matrix[0]])

    cell_a = layer_a.get_cell().array
    cell_b = layer_b.get_cell().array
    lattice_a = cell_a[:2, :2]
    lattice_b = cell_b[:2, :2]

    rotation_2d = get_rotation_matrix_2d(rotation_angle)
    lattice_b = np.dot(rotation_2d, lattice_b.T).T

    supercell_lattice_a = np.dot(supercell_matrix, lattice_a)
    supercell_matrix_b = np.dot(supercell_matrix, np.dot(lattice_a, np.linalg.inv(lattice_b)))
    supercell_lattice_b = np.dot(np.round(supercell_matrix_b), lattice_b)
    average_lattice = (supercell_lattice_a + supercell_lattice_b) / 2

    thickness_a = get_layer_thickness(layer_a)
    thickness_b = get_layer_thickness(layer_b)

    return {
        "description": """
        Moiré structure information:
        Amn: Transformation matrix from lower supercell to primitive cell
        Bmn: Transformation matrix from upper supercell to primitive cell
        sA: Lower layer supercell lattice vectors
        sB: Upper layer supercell lattice vectors
        layer_thickness: Layer thickness
        newlattice: Average supercell lattice
        split_height: Split height
        """,
        'Amn': supercell_lattice_a.tolist(),
        'Bmn': supercell_matrix_b.tolist(),
        'sA': supercell_lattice_a.tolist(),
        'sB': supercell_lattice_b.tolist(),
        'A': lattice_a.tolist(),
        'B': lattice_b.tolist(),
        'R': rotation_2d.tolist(),
        'split_height': 1 + thickness_a + thickness_a / 2,
        'layer_thickness': interlayer_distance,
        'layer_thickness1': thickness_a,
        'layer_thickness2': thickness_b,
        'newlattice': average_lattice.tolist(),
    }


def create_matched_structure(
    atoms1: Atoms,
    atoms2: Atoms,
    maxm: int = 10,
    layer_thickness: float = 2.5,
    angle: float = 0.0,
    mismatch: float = 0.04,
    vacuum: float = 20.0
) -> Atoms:
    """
    Create matched interface structure for two atomic structures

    Args:
        atoms1: First atomic structure (lower layer)
        atoms2: Second atomic structure (upper layer)
        maxm: Maximum supercell size
        layer_thickness: Interlayer distance
        angle: Rotation angle (degrees)
        mismatch: Allowed lattice mismatch
        vacuum: Vacuum layer thickness (Angstrom)

    Returns:
        Matched interface structure
    """
    theta = angle * np.pi / 180

    # Create MoireData object and set parameters
    moire_data = MoireData(
        lattice_A=atoms1.get_cell().array[:2, :2],
        lattice_B0=atoms2.get_cell().array[:2, :2],
        maxm=maxm,
        epsilon=mismatch,
        lepsilon=mismatch,
        maxLrate=10
    )
    moire_data.get_all_candidates()

    # Find matching structure
    moire_data.set_rotation_angle(theta)
    mn_A, area, mn_B = moire_data.find_optimal_supercell()

    if mn_A.shape != (2, 2):
        raise ValueError("Cannot find matching structure")

    # Build matched structure
    matched_structure = build_moire_pattern(
        atoms1, atoms2, angle, mn_A, layer_thickness,
        vacuum_thickness=vacuum
    )

    return matched_structure


def main():
    """Main function for command line invocation"""
    import argparse
    import tqdm

    parser = argparse.ArgumentParser(description='Generate moiré structures')
    parser.add_argument(
        "files",
        type=str,
        nargs=2,
        help='Input filenames [lower layer A] [upper layer B(real space rotation layer)]'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='tmpoutput',
        help="Output directory (default: %(default)s)"
    )
    parser.add_argument(
        "-r", "--range",
        type=float,
        default=[0, 180, 1000],
        nargs=3,
        help="Angle range: start end number (default: %(default)s)"
    )
    parser.add_argument(
        "-e", "--epsilon",
        type=float,
        default=0.04,
        help="Tolerance for lattice vector matching (epsilon). Maximum allowed relative deviation between supercell vectors when finding initial 1D matches. Controls precision of vector pairing between layers A and B. (default: %(default)s)"
    )
    parser.add_argument(
        "-l", "--lepsilon",
        type=float,
        default=0.04,
        help="Tolerance for overall lattice matching. Spectral norm of MN matrix where U = s_avg_inv * s_B and MN = U^T - I. Maximum elastic strain allowed in the most sensitive direction (default: %(default)s)"
    )
    parser.add_argument(
        "--maxl",
        type=float,
        default=50,
        help="Maximum lattice length (Angstrom) (default: %(default)s)"
    )
    parser.add_argument(
        "-m", "--maxm",
        type=int,
        default=50,
        help="Maximum supercell size for lower (default: %(default)s)"
    )
    parser.add_argument(
        "-d","--distance",
        type=float,
        default=3.04432,
        help="Interlayer distance (default: %(default)s)"
    )
    parser.add_argument(
        "--max_atoms",
        type=int,
        default=100000,
        help="Maximum allowed number of atoms (default: %(default)s)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read input files
    upper_layer = aio.read(args.files[1])  # Upper layer
    lower_layer = aio.read(args.files[0])  # Lower layer
    
    
    print(f"Lower layer file: {args.files[0]}, lattice: {lower_layer.get_cell().lengths()}, angles: {lower_layer.get_cell().angles()}")
    print(f"Upper layer file: {args.files[1]}, lattice: {upper_layer.get_cell().lengths()}, angles: {upper_layer.get_cell().angles()}")
    # Assert that a,b lattice vectors are in xy-plane and c is aligned with z-axis
    # Check that z-components of a and b vectors are near zero
    assert np.allclose(lower_layer.get_cell()[:2, 2], 0, atol=1e-6), "Lower layer: a,b vectors must be in xy-plane"
    assert np.allclose(upper_layer.get_cell()[:2, 2], 0, atol=1e-6), "Upper layer: a,b vectors must be in xy-plane"
    # Check that x and y components of c vector are near zero
    assert np.allclose(lower_layer.get_cell()[2, :2], 0, atol=1e-6), "Lower layer: c vector must be aligned with z-axis"
    assert np.allclose(upper_layer.get_cell()[2, :2], 0, atol=1e-6), "Upper layer: c vector must be aligned with z-axis" 
    # Calculate average atomic density
    density_lower = len(lower_layer) / np.abs(np.linalg.det(lower_layer.get_cell().array[:2, :2]))
    density_upper = len(upper_layer) / np.abs(np.linalg.det(upper_layer.get_cell().array[:2, :2]))
    average_density = (density_lower + density_upper) / 2

    # Initialize MoireData object
    moire_data = MoireData(
        lattice_A=lower_layer.get_cell().array[:2, :2],
        lattice_B0=upper_layer.get_cell().array[:2, :2],
        maxm=args.maxm,
        max_lattice_length=args.maxl,
        epsilon=args.epsilon,
        lepsilon=args.lepsilon,
        dtheta=0  # Will be set later
    )

    moire_data.get_all_candidates()

    # Generate angle list
    theta_list = np.linspace(
        args.range[0] * np.pi / 180,
        args.range[1] * np.pi / 180,
        int(args.range[2])
    )
    moire_data.dtheta = theta_list[1] - theta_list[0]

    print(f"Angle search range: {args.range}")

    # Store valid configurations
    valid_configurations = []

    # Search for optimal configurations
    for theta in tqdm.tqdm(theta_list, desc="Searching angles"):
        try:
            moire_data.set_rotation_angle(theta)
            mn_A, area, mn_B = moire_data.find_optimal_supercell()

            if mn_A.shape == (2, 2):
                result = moire_data.relax_with_mn(mn_A)
                if result.success:
                    valid_configurations.append((result.x, result.fun, mn_A, area))
        except Exception:
            continue

    print(f"Found {len(valid_configurations)} valid configurations")

    # Build moiré structures
    structure_info = {}

    for config in valid_configurations:
        try:
            theta_opt, mismatch_value, mn_matrix, area = config

            # Calculate supercell lattice vectors
            supercell_lattice_a = np.dot(mn_matrix, lower_layer.get_cell().array[:2, :2])

            # Calculate rotated upper layer lattice
            upper_rotated = np.dot(
                get_rotation_matrix_2d(theta_opt),
                upper_layer.get_cell().array[:2, :2].T
            ).T

            # Calculate upper layer supercell
            transform_matrix = np.dot(
                mn_matrix,
                np.dot(lower_layer.get_cell().array[:2, :2], np.linalg.inv(upper_rotated))
            )
            supercell_lattice_b = np.dot(np.round(transform_matrix), upper_rotated)

            # Estimate number of atoms
            moire_area = (np.abs(np.linalg.det(supercell_lattice_a)) +
                         np.abs(np.linalg.det(supercell_lattice_b)))
            estimated_atoms = int(moire_area * average_density)

            # Check atom count limit
            if estimated_atoms > args.max_atoms:
                print(f"Skipping angle {theta_opt * 180 / np.pi:.2f}°, "
                      f"estimated atoms {estimated_atoms} exceeds limit {args.max_atoms}")
                continue

            # Build moiré structure
            moire_structure = build_moire_pattern(
                lower_layer, upper_layer, theta_opt, mn_matrix,
                args.distance
            )

            # Generate filename
            filename = f"POSCAR-{theta_opt * 180 / np.pi:.10f}-{mismatch_value * 100:.3f}.vasp"

            # Get structure information
            info = get_moire_pattern_info(
                lower_layer, upper_layer, theta_opt, mn_matrix,
                args.distance
            )
            info['theta'] = theta_opt
            info['epsilon'] = mismatch_value

            # print(f"Actual atoms: {len(moire_structure)}, estimated atoms: {estimated_atoms}")

            structure_info[filename] = info

            # Save structure
            moire_structure.wrap()
            moire_structure.write(
                os.path.join(args.output, filename),
                format="vasp",
                direct=True
            )

        except Exception as e:
            print(f"Error processing configuration: {e}")
            raise e
            continue

    # Save structure information
    info_file = os.path.join(args.output, "informations.json")
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            existing_info = json.load(f)
        existing_info.update(structure_info)
        structure_info = existing_info

    with open(info_file, 'w') as f:
        json.dump(structure_info, f, indent=2)


if __name__ == '__main__':
    main()