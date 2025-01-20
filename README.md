# Low2High
The repository includes lammps inputs with the trained model (best0.pt), which is the final model of relabeled DFT data. 
The methods are explained in the following paper:

## Enhancing High-Fidelity Neural Network Potentials through Low-Fidelity Sampling

**1. Elastic_Ni_fcc**
   Example and log file for the elastic constant calculations
   To run this code, you need to install following codes
   
    -. Install https://github.com/gsjung0419/LMPTorch
   
    -. Install https://github.com/aiqm/torchani
   
    -. MPI (MPICH or OPENMPI), with mpi4py

**2. Training Tutorial** with stress term is available chapter 2: https://github.com/gsjung0419/TorchANITutorials  

**3.  Selected ~2,500 configurations** through active learning ("Data Distillation for Neural Network Potentials toward Foundational Dataset", https://arxiv.org/abs/2311.05407), is available in the tutorial. 

**4. DFT data** will be available with other ongoing sampling (Tensile loading and gas phases). 
