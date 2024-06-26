# Low2High
The repository includes lammps inputs with the trained model (best0.pt), which is the final model of relabeled DFT data. 
The methods are explained in the following paper:

## Enhancing High-Fidelity Neural Network Potentials through Low-Fidelity Sampling

1. Elastic_Ni_fcc
   Example and log file for the elastic constant calculations
   To run this code, you need to install following codes
   
    -. Install https://github.com/gsjung0419/LMPTorch
   
    -. Install https://github.com/aiqm/torchani
   
    -. MPI (MPICH or OPENMPI), with mpi4py

3. Training Tutorial with stress term (will be updated once the paper is accepted)

4. DFT data (will be updated once the paper is accepted)

5. EAM data (32 atoms: 20,000 configuration poitns, 108 atoms: 20,000 configurations) will be available in the other repository based on the paper "Data Distillation for Neural Network Potentials toward Foundational Dataset", https://arxiv.org/abs/2311.05407
