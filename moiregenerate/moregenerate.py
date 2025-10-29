import numpy as np
import scipy.optimize as opt
from typing import Optional, Tuple


def rotate_2d_vector(vector: np.ndarray, theta: float) -> np.ndarray:
    """Rotate a 2D vector counterclockwise by angle theta.

    Args:
        vector: 2D vector to rotate
        theta: Rotation angle in radians

    Returns:
        Rotated 2D vector
    """
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta],
                               [sin_theta, cos_theta]])
    return np.dot(rotation_matrix, vector)


class MoireData:
    """Class for generating moiré patterns in 2D material bilayer structures."""

    def __init__(self,
                 lattice_A: Optional[np.ndarray] = None,
                 lattice_B0: Optional[np.ndarray] = None,
                 epsilon: float = 0.01,
                 lepsilon: float = 0.01,
                 maxm: int = 10,
                 max_lattice_length: float = 100.0,
                 minangle: float = np.pi / 7,
                 maxLrate: float = 3.2,
                 dtheta: float = np.pi / 180 * 0.1) -> None:
        """Initialize MoireData with lattice vectors and parameters.

        Args:
            lattice_A: Lower layer lattice vectors (2x2 array)
            lattice_B0: Upper layer lattice vectors (2x2 array, initial unrotated)
            epsilon: Tolerance for lattice vector matching
            lepsilon: Tolerance for overall lattice matching
            maxm: Maximum supercell size
            max_lattice_length: Maximum lattice length in Angstroms
            minangle: Minimum angle between vectors (radians)
            maxLrate: Maximum length ratio between vectors
            dtheta: Angle step for local optimization (radians)
        """
        # Lattice vectors (use default if not provided)
        self.lattice_A = lattice_A if lattice_A is not None else np.array([[1, 2], [2, 1]])
        self.lattice_B0 = lattice_B0 if lattice_B0 is not None else np.array([[1, 2], [2, 1]])

        # Rotation and tolerance parameters
        self.theta = 0.0  # Current rotation angle
        self.epsilon = epsilon  # Tolerance for lattice vector matching
        self.lepsilon = lepsilon  # Tolerance for overall lattice matching
        self.maxm = maxm  # Maximum supercell size
        self.max_lattice_length = max_lattice_length  # Maximum lattice length
        self.minangle = minangle  # Minimum angle between vectors
        self.maxLrate = maxLrate  # Maximum length ratio between vectors
        self.dtheta = dtheta  # Angle step for local optimization

        # Computed properties
        self.lattice_B = self.lattice_B0.copy()  # Rotated upper layer lattice vectors
        self.candidate_vectors = None  # All possible supercell vectors
        self.candidate_lengths = None  # Lengths of all candidate vectors

    def set_rotation_angle(self, theta: float) -> None:
        """Set the rotation angle for the upper layer.

        Args:
            theta: Rotation angle in radians
        """
        self.theta = theta
        self.lattice_B = rotate_2d_vector(self.lattice_B0.T, theta).T

    def _generate_supercell_candidates(self) -> None:
        """Generate all possible supercell candidate vectors."""
        candidates = []
        for i in range(-self.maxm, self.maxm + 1):
            for j in range(-self.maxm, self.maxm + 1):
                if i != 0 or j != 0:
                    candidates.append([i, j])

        candidates = np.array(candidates, dtype=np.int32)
        real_space_vectors = np.dot(candidates, self.lattice_A)
        lengths = np.linalg.norm(real_space_vectors, axis=1)

        # Filter by maximum length
        mask = lengths < self.max_lattice_length
        candidates = candidates[mask]
        lengths = lengths[mask]

        # Sort by length
        sort_order = np.argsort(lengths)
        self.candidate_vectors = candidates[sort_order]
        self.candidate_lengths = lengths[sort_order]

    def get_all_candidates(self) -> None:
        """Generate all possible supercell candidates."""
        self._generate_supercell_candidates()



    def _batch_mismatch_multi(self, mn_pairs: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Calculate mismatch for multiple supercell pairs and multiple angles.

        Args:
            mn_pairs: Array of shape (P, 2, 2) with supercell pairs
            thetas: Array of angles to evaluate

        Returns:
            Array of shape (P, T) with mismatch values (spectral norm)
        """
        thetas = np.asarray(thetas, dtype=float)

        # Precompute B(theta), C(theta) = A @ inv(B(theta))
        B_stack = np.stack([rotate_2d_vector(self.lattice_B0.T, th).T for th in thetas], axis=0)
        invB_stack = np.linalg.inv(B_stack)
        C_stack = np.matmul(self.lattice_A, invB_stack)

        # Expand mn_pairs for broadcasting
        mn_exp = mn_pairs[:, None, :, :]

        # Calculate result = mn @ C(theta)
        result = np.matmul(mn_exp, C_stack)
        mn2 = np.round(result)

        # Calculate sA = mn @ A (theta-independent)
        sA = np.matmul(mn_exp, self.lattice_A)

        # Calculate sB = mn2 @ B(theta)
        sB = np.matmul(mn2, B_stack)

        # Average lattice for stability
        sAavg = 0.5 * (sA + sB)

        # Add small regularization to avoid singular matrices
        epsI = 1e-12 * np.eye(2)[None, None, :, :]
        inv_sAavg = np.linalg.inv(sAavg + epsI)

        # Calculate transformation matrix U
        U = np.matmul(inv_sAavg, sB)

        # Calculate MN = U^T - I
        MN = np.swapaxes(U, -1, -2) - np.eye(2)[None, None, :, :]

        # Return spectral norm (maximum singular value)
        svals = np.linalg.svd(MN, compute_uv=False)
        return svals[..., 0]

    def find_optimal_supercell(self) -> Tuple[Optional[np.ndarray], float, Optional[np.ndarray]]:
        """Find optimal supercell matching using comprehensive search algorithm.

        This is the recommended method for finding minimal supercell configurations.
        It uses comprehensive search with batch processing and multi-objective optimization
        to identify the best matching supercell matrices for both layers.

        Returns:
            Tuple of (mn_A, match_area, mn_B) or (None, 0, None) if no feasible solution
            where:
            - mn_A: Optimal supercell matrix for the LOWER layer (lattice_A)
            - match_area: Area of the matched supercell
            - mn_B: Corresponding supercell matrix for the UPPER layer (lattice_B)

        Note:
            Both mn_A and mn_B are 2x2 integer matrices that define the supercell
            transformations relative to their respective lattices.

            For mn_A (relative to lattice_A):
            Each row represents a supercell vector in terms of the lower layer basis vectors.
            If mn_A = [[m11, m12], [m21, m22]], then:
            - SuperCell vector 1 = m11 * a1 + m12 * a2 (where a1, a2 are lattice_A basis)
            - SuperCell vector 2 = m21 * a1 + m22 * a2

            For mn_B (relative to lattice_B):
            Defines the matching supercell for the upper layer after rotation.
        """
        if self.candidate_vectors is None:
            self.get_all_candidates()

        # Step 1: Get candidate 1D supercell vectors
        C = np.dot(self.lattice_A, np.linalg.inv(self.lattice_B))
        result = self.candidate_vectors.dot(C)
        allchoose2 = np.round(result)
        diff = np.dot(allchoose2, self.lattice_B) - np.dot(self.candidate_vectors, self.lattice_A)
        denom = np.linalg.norm(np.dot(allchoose2, self.lattice_B), axis=1)
        denom = np.where(denom == 0, 1e-15, denom)
        dis = np.linalg.norm(diff, axis=1) / denom
        mns = self.candidate_vectors[dis < self.epsilon]

        if mns.shape[0] < 2:
            return None, 0, None

        # Step 2: Precompute geometric properties
        cart = np.dot(mns, self.lattice_A)
        lengths = np.linalg.norm(cart, axis=1)

        # Step 3: Generate all unique pairs
        i_idx, j_idx = np.triu_indices(mns.shape[0], k=1)
        cart_i, cart_j = cart[i_idx], cart[j_idx]
        len_i, len_j = lengths[i_idx], lengths[j_idx]

        # Step 4: Calculate geometric constraints
        cross_ij = np.abs(cart_i[:, 0] * cart_j[:, 1] - cart_i[:, 1] * cart_j[:, 0])
        denom_ang = np.clip(len_i * len_j, 1e-15, None)
        sinang = np.clip(cross_ij / denom_ang, 0.0, 1.0)
        angle = np.arcsin(sinang)
        # ratio = np.maximum(len_i, len_j) / np.clip(np.minimum(len_i, len_j), 1e-15, None)

        # Apply geometric filters
        # mask_geom = (angle > self.minangle) & (cross_ij > 0.1) & (ratio < self.maxLrate)
        mask_geom = (cross_ij > 0.1) 
        if not np.any(mask_geom):
            return None, 0, None

        # Step 5: Build candidate 2x2 matrices
        mn_pairs = np.stack([mns[i_idx[mask_geom]], mns[j_idx[mask_geom]]], axis=1)

        # Step 6: Batch calculate mismatches for local minimum check
        thetas = [self.theta, self.theta - self.dtheta, self.theta + self.dtheta]
        mm = self._batch_mismatch_multi(mn_pairs, thetas)
        m0, m_minus, m_plus = mm[:, 0], mm[:, 1], mm[:, 2]

        # Check for local minima
        mask_local_min = (m0 < m_minus) & (m0 < m_plus) & (m0 < self.lepsilon)
        if not np.any(mask_local_min):
            return None, 0, None

        # Step 7: Multi-objective sorting for optimal solution
        area_sel = cross_ij[mask_geom][mask_local_min]
        # ldiff_sel = np.abs(len_i[mask_geom][mask_local_min] - len_j[mask_geom][mask_local_min])
        # ang90_dev = np.abs(np.pi / 2.0 - angle[mask_geom][mask_local_min])
        mn_sel = mn_pairs[mask_local_min]

        # Sort by: area (primary), length difference (secondary), angle to 90° (tertiary)
        # order = np.lexsort((ang90_dev, ldiff_sel, area_sel))
        order = np.argsort((area_sel))
        
        best = order[0]

        # Reduce the supercell matrix to its most compact form
        from .utils import reduce_surface_mn
        mn_reduced = reduce_surface_mn(self.lattice_A, mn_sel[best])

        # Calculate the corresponding supercell matrix for the rotated B lattice
        C = np.dot(self.lattice_A, np.linalg.inv(self.lattice_B))
        result = mn_reduced.dot(C)
        mn_B = np.round(result)

        # Ensure mn_B has positive determinant (right-handed)
        if np.linalg.det(mn_B) < 0:
            mn_B = np.array([[-mn_B[1, 0], -mn_B[1, 1]], [-mn_B[0, 0], -mn_B[0, 1]]])

        return mn_reduced, area_sel[best].item(), mn_B

    def get_min_mn_one(self) -> Tuple[Optional[np.ndarray], float]:
        """Find a single valid supercell configuration.

        Returns:
            Tuple of (supercell_matrix, match_area) or (None, 0) if no match
        """
        if self.candidate_vectors is None:
            self.get_all_candidates()

        C = np.dot(self.lattice_A, np.linalg.inv(self.lattice_B))
        result = self.candidate_vectors.dot(C)
        allchoose2 = np.round(result)

        diff = np.dot(allchoose2, self.lattice_B) - np.dot(self.candidate_vectors, self.lattice_A)
        denom = np.linalg.norm(np.dot(allchoose2, self.lattice_B), axis=1)
        denom = np.where(denom == 0, 1e-15, denom)
        mismatch = np.linalg.norm(diff, axis=1) / denom

        mns = self.candidate_vectors[mismatch < self.epsilon]

        if mns.shape[0] < 2:
            return None, 0, None

        mns = mns[np.argsort(np.linalg.norm(np.dot(mns, self.lattice_A), axis=1))]
        cart = np.dot(mns, self.lattice_A)

        areas = np.abs(np.cross(cart[0], cart))
        lll = np.linalg.norm(cart, axis=1)
        angle = np.abs(np.arcsin(areas / lll / lll[0]))

        sortdep = areas + np.abs(np.linalg.det(self.lattice_A)) / 2 * lll / self.candidate_lengths[-1]
        index = np.arange(len(areas))
        index = index[np.logical_and(
            angle > self.minangle,
            lll / lll[0] < self.maxLrate,
            areas > 0.1
        )]
        index = index[np.argsort(sortdep[index])]

        if len(index) > 0:
            return np.array([mns[0], mns[index[0]]]), areas[index[0]]

        return None, 0

    def plot_vectors(self, mn: np.ndarray, theta: float) -> None:
        """Plot vector configuration for visualization.

        Args:
            mn: 2x2 supercell matrix
            theta: Rotation angle
        """
        import matplotlib.pyplot as plt

        B = rotate_2d_vector(self.lattice_B0.T, theta).T
        C = np.dot(self.lattice_A, np.linalg.inv(B))

        sA = np.dot(mn, self.lattice_A)
        result = mn.dot(C)
        mn2 = np.round(result)
        sB = np.dot(mn2, B)
        sA = (sA + sB) / 2

        # Plot vectors
        for i in range(2):
            plt.arrow(0, 0, sA[i, 0], sA[i, 1],
                     head_width=0.05, head_length=0.1, color='r')
            plt.arrow(0, 0, sB[i, 0], sB[i, 1],
                     head_width=0.05, head_length=0.1, color='b')

        # Plot lattice points
        point0 = self.candidate_vectors.dot(sA)
        point1 = self.candidate_vectors.dot(sB)
        plt.scatter(point0[:, 0], point0[:, 1], color='r', s=0.1)
        plt.scatter(point1[:, 0], point1[:, 1], color='b', s=0.1)

        plt.axis('equal')

        # Calculate and display transformation matrix
        U = np.linalg.inv(sA).dot(sB)
        point3 = np.dot(point0, U)
        MN = (U.T - np.eye(2))
        print("MN norm:", MN, np.linalg.norm(MN, ord=2))
        plt.scatter(point3[:, 0], point3[:, 1], color='g', s=0.1)
        plt.show()

    def get_mismatch(self, mn: np.ndarray, theta: float) -> float:
        """Calculate lattice mismatch for a given supercell configuration.

        Args:
            mn: 2x2 supercell matrix
            theta: Rotation angle

        Returns:
            Mismatch value (spectral norm)
        """
        B = rotate_2d_vector(self.lattice_B0.T, theta).T
        C = np.dot(self.lattice_A, np.linalg.inv(B))
        result = mn.dot(C)
        mn2 = np.round(result)

        sA = np.dot(mn, self.lattice_A)
        sB = np.dot(mn2, B)
        sA_avg = (sA + sB) / 2

        U = np.linalg.inv(sA_avg).dot(sB)
        MN = (U.T - np.eye(2))

        return np.linalg.norm(MN, ord=2)

    def relax_with_mn(self, mn: np.ndarray) -> opt.OptimizeResult:
        """Optimize rotation angle for a given supercell configuration.

        Args:
            mn: 2x2 supercell matrix

        Returns:
            Optimization result
        """
        theta = self.theta
        fun = lambda x: self.get_mismatch(mn, x)

        return opt.minimize_scalar(
            fun,
            method="bounded",
            bounds=(theta - self.dtheta, theta + self.dtheta)
        )

    def relax_with_mn2(self, mn: np.ndarray) -> opt.OptimizeResult:
        """Optimize rotation angle over full range.

        Args:
            mn: 2x2 supercell matrix

        Returns:
            Optimization result
        """
        fun = lambda x: self.get_mismatch(mn, x)

        return opt.minimize_scalar(
            fun,
            method="Golden",
            bounds=(0, np.pi)
        )


def main():
    """Main function for command-line interface."""
    import argparse
    import os
    import ase.io as aio
    import tqdm

    parser = argparse.ArgumentParser(description='Generate moiré structure')
    parser.add_argument("files", type=str, default=None, nargs=2,
                       help='Input file names (two files)')
    parser.add_argument('-o', '--output', type=str, default='tmpoutput',
                       help='Output directory')
    parser.add_argument("-r", "--range", type=float, default=[0, 180, 1000],
                       nargs=3, help="Angle range: start end num")
    parser.add_argument("-e", "--epsilon", type=float, default=0.04,
                       help="Tolerance for lattice vector matching")
    parser.add_argument("-l", "--lepsilon", type=float, default=0.04,
                       help="Tolerance for overall lattice matching")
    parser.add_argument("-m", "--maxm", type=int, default=10,
                       help="Maximum supercell size")
    parser.add_argument("--distance", type=float, default=3.04432,
                       help="Distance between supercells")
    parser.add_argument("--needshift", action='store_true',
                       help="Apply shift transformation")

    args = parser.parse_args()

    if args.files is None or len(args.files) != 2:
        print("Please provide exactly two input files")
        exit()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Read input structures
    structure_b0 = aio.read(args.files[1])
    structure_a = aio.read(args.files[0])

    print(f"{args.files[0]}: {structure_a.get_cell().lengths()}, {structure_a.get_cell().angles()}")
    print(f"{args.files[1]}: {structure_b0.get_cell().lengths()}, {structure_b0.get_cell().angles()}")

    # Generate angle list
    theta_list = np.linspace(
        args.range[0] * np.pi / 180,
        args.range[1] * np.pi / 180,
        int(args.range[2])
    )

    # Initialize moiré data with parameters
    moire = MoireData(
        lattice_A=structure_a.get_cell().array[:2, :2],
        lattice_B0=structure_b0.get_cell().array[:2, :2],
        maxm=args.maxm,
        epsilon=args.epsilon,
        lepsilon=args.lepsilon,
        dtheta=theta_list[1] - theta_list[0]
    )
    moire.get_all_candidates()

    print(f"Angle range: {args.range}")

    # Search for optimal configurations
    valid_configs = []
    all_results = []
    plot_data = []
    unique_keys = set()

    def get_unique_key(angle, mn):
        return f"{angle:.3f}_{mn[0, 0]}_{mn[0, 1]}_{mn[1, 0]}_{mn[1, 1]}"

    for theta in tqdm.tqdm(theta_list):
        try:
            moire.set_rotation_angle(theta)
            mn_A, area, mn_B = moire.find_optimal_supercell()

            if mn_A is None or mn_A.shape != (2, 2):
                continue

            result = moire.relax_with_mn(mn_A)

            # Check for uniqueness
            key = get_unique_key(result.x, mn_A)
            if key in unique_keys:
                continue
            unique_keys.add(key)

            all_results.append((result.x, area, mn_A))
            plot_data.append([result.x, area])

            if result.success:
                print(f"Success: {theta/np.pi*180:.3f}° -> {result.x/np.pi*180:.3f}°, mn_A={mn_A}, mn_B={mn_B}")
                valid_configs.append((result.x, result.fun, mn_A, area))

        except Exception as e:
            print(f"Error at angle {theta/np.pi*180:.3f}°: {e}")
            continue

    # Print valid configurations
    print("Valid configurations:")
    for config in valid_configs:
        print(f"  Angle: {config[0]/np.pi*180:.3f}°, Matrix: {config[2]}, Area: {config[3]}")


if __name__ == "__main__":
    main()