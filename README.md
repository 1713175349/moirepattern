# Moiré Pattern Generator

A Python package for generating moiré superlattice structures in 2D material bilayers with twistronics applications.

## Features

- Generate moiré patterns for 2D material bilayer structures
- Support for rotation angle scanning and supercell matching
- Advanced lattice mismatch tolerance control
- Structure optimization with minimal strain configurations
- Batch processing for multiple angles
- ASE integration for structure manipulation

## Requirements

The input structures must satisfy:
- Vacuum layer direction aligned with z-axis
- 2D lattice vectors (a, b) in the xy-plane
- c-vector aligned with z-axis (no x,y components)

## Installation

```bash
pip install .
```

Or install from source:

```bash
git clone https://github.com/1713175349/moirepattern
cd moirepattern
pip install -e .
```
or
```bash
pip install git+https://github.com/1713175349/moirepattern
```

## Usage

### Command Line Interface

```bash
moiregenerate-cmd [-h] [-o OUTPUT] [-r RANGE RANGE RANGE] [-e EPSILON] [-l LEPSILON]
                  [--maxl MAXL] [-m MAXM] [-d DISTANCE] [--max_atoms MAX_ATOMS]
                  files files
```

### Parameters

**Positional Arguments:**
- `files`: Input filenames [lower layer A] [upper layer B]

**Optional Arguments:**
- `-o OUTPUT, --output OUTPUT`: Output directory (default: tmpoutput)
- `-r RANGE RANGE RANGE, --range RANGE RANGE RANGE`: Angle range: start end number (default: [0, 180, 1000])
- `-e EPSILON, --epsilon EPSILON`: Tolerance for lattice vector matching (default: 0.04)
- `-l LEPSILON, --lepsilon LEPSILON`: Tolerance for overall lattice matching (default: 0.04)
- `--maxl MAXL`: Maximum lattice length in Angstroms (default: 50)
- `-m MAXM, --maxm MAXM`: Maximum supercell size (default: 50)
- `-d DISTANCE, --distance DISTANCE`: Interlayer distance in Angstroms (default: 3.04432)
- `--max_atoms MAX_ATOMS`: Maximum allowed number of atoms (default: 100000)

### Understanding Tolerance Parameters

1. **Epsilon (ε)**: Controls precision of initial 1D vector matching between layers A and B. Smaller values lead to more precise but fewer matches.

2. **Lepsilon (λε)**: Controls overall lattice matching quality. It's the spectral norm of the MN matrix where U = s_avg⁻¹ × s_B and MN = Uᵀ - I. Represents the maximum elastic strain allowed in the most sensitive direction.

## Examples

### Basic Usage
```bash
moiregenerate-cmd -o CrI3 -r 0 60 1000 --maxl 50 --distance 3.4 CR1.vasp CR1.vasp
```

### High Precision Search
```bash
moiregenerate-cmd -o output -e 0.01 -l 0.01 --maxm 20 graphene1.vasp graphene2.vasp
```

### Wide Angle Range
```bash
moiregenerate-cmd -o results -r 0 180 2000 --max_atoms 50000 WS2_1.vasp WS2_2.vasp
```

## Output

The program generates:
- VASP POSCAR files for each valid moiré configuration
- `informations.json` containing detailed structure information for each generated file
- Files named as: `POSCAR-{angle}-{mismatch}.vasp`

## Python API

```python
from moiregenerate import MoireData, build_moire_pattern
import numpy as np
from ase.io import read

# Read input structures
layer_a = read('graphene.vasp')
layer_b = read('hbn.vasp')

# Initialize MoireData
moire_data = MoireData(
    lattice_A=layer_a.get_cell().array[:2, :2],
    lattice_B0=layer_b.get_cell().array[:2, :2],
    maxm=10,
    epsilon=0.04,
    lepsilon=0.04
)

# Find optimal configuration
moire_data.set_rotation_angle(np.pi/6)  # 30 degrees
mn_A, area, mn_B = moire_data.find_optimal_supercell()

# Build moiré structure
moire_structure = build_moire_pattern(
    layer_a, layer_b, np.pi/6, mn_A, 3.4
)

# Save structure
moire_structure.write('moire_pattern.vasp', format='vasp')
```

## Algorithm Overview

The package uses a comprehensive search algorithm to find optimal supercell configurations:

1. Generate all possible supercell candidates within size limits
2. Filter candidates based on geometric constraints (lattice mismatch)
3. Perform batch mismatch calculations for efficiency
4. Identify local minima in this angle for every candidates
5. Select configurations with minimal area

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

