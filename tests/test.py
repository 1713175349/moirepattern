from ase.build import surface
from ase.lattice.cubic import BodyCenteredCubic

W_bulk=BodyCenteredCubic("W",latticeconstant=3.1)
ats1=surface(W_bulk,[1,1,0],5,20,periodic=True)

ats2=surface(W_bulk,[1,0,0],5,20,periodic=True)

from moiregenerate.buildmoremodelcmd import create_matched_structure
from ase.visualize import view
ats=create_matched_structure(ats1,ats2,layer_thickness=10,maxm=400,mismatch=0.001)
from ase.build import make_supercell
ats=make_supercell(ats,[[3,0,0],[0,3,0],[0,0,1]])
view(ats)