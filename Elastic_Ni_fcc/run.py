import os
from lammps import lammps
import torch
import torchani
import numpy as np

os.system("rm geo*.xyz ss*.dat sd.dat")
lmp=lammps()
#lmp.file("bend.in")
#lmp.file("relax.in")
lmp.file("in.elastic")

#lmp.file("in.check")

