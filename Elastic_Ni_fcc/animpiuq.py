from __future__ import print_function
from mpi4py import MPI
from lammps import lammps,LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM, LMP_STYLE_ATOM, LMP_TYPE_VECTOR
import torch,os
import torchani
import sys,time,copy
import numpy as np

from torchani.utils import ChemicalSymbolsToInts
from model import Model

hartree2ev = np.float64(27.211386024367243)
device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
#device_str = 'cpu'
device = torch.device(device_str)
device_cpu =torch.device('cpu')

nptype=np.double
tortype=torch.float64

################# Now let's read self energies and construct energy shifter ########################
############################ Shared model in different MPI   #######################################

#torch.set_num_threads(1)
#torch.set_num_interop_threads(1)

def end_of_step_callback(lmp):
    L = lammps(ptr=lmp)
    t = L.extract_global("ntimestep", 0)
    print("### END OF STEP ###", t)

def post_force_callback(lmp, v, m):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    emodel = m
    
    L = lammps(ptr=lmp)
    pid = os.getpid()
    pid_prefix = "[{}] ".format(pid)
    
    t = L.extract_global("ntimestep", 0)
    nlocal = L.extract_global("nlocal", 0)
    nghost = L.extract_global("nghost", 0)

    idx = L.find_pair_neighlist("zero", request=0)
    mylist = L.numpy.get_neighlist(idx)

    nlocal = L.extract_global("nlocal")
    nghost = L.extract_global("nghost")

    atype = L.numpy.extract_atom("type", nelem=nlocal+nghost)
    x = L.numpy.extract_atom("x", nelem=nlocal+nghost, dim=3)
    f = L.numpy.extract_atom("f", nelem=nlocal, dim=3)
    #q = L.numpy.extract_atom("q", nelem=nlocal, dim=1) should be commented for atomic type//
    #q is for the charge or qu for visualization. atom_style should be charge or full
    
    #spx = L.numpy.extract_atom("spx", nelem=nlocal, dim=1)        


    #Extract atom style variable, need to define group
    vname = "ape"        
    ape=L.numpy.extract_variable(vname,"all",LMP_VAR_ATOM)
    vname = "auq"
    auq=L.extract_variable(vname,"all",LMP_VAR_ATOM)    

    #tmparray = L.extract_fix("10",LMP_STYLE_ATOM,LMP_TYPE_VECTOR,nlocal)
    #tmparray = L.extract_fix("10",1,1)
    #print(spx[:])
    
    #vname = "cape"            
    #cape = L.extract_compute(vname,1,1)
    #print("capte",cape)
    #print("auq",ape[:])
    
    carray = L.extract_fix("2",2,1,10)    

    #######################   NUMPY  FROM LAMMPS  ######################################
    #species
    slist=[]
    for i in range(0,nlocal+nghost):
        if(atype[i]==1):
            slist.append('C')
        else:
            slist.append('C')

    #coordinates
    lcoord=np.ndarray.tolist(x)    

    #atypes
    ltype = np.ndarray.tolist(np.transpose(atype)[0])

    #cell - box
    box = L.extract_box()
    boxlo = box[0]
    boxhi = box[1]
    xy = box[2]
    yz = box[3]
    xz = box[4]

    #Set carray for retun
    tarray=np.zeros(10,dtype=np.double)
    
    #######################       TORCH SETUP & OPERATIONS    #######################################
    if(device_str=='cpu'):
        #Loading model from python 
        
        #nn = m.nn
        species_to_tensor=m.species_to_tensor
        #model = m.model.to(device)
        aev_computer = emodel.aev_computer
        energy_shifter = emodel.energy_shifter
        
        #species
        species = species_to_tensor(slist).unsqueeze(0).to(device_str) 
        #coordinates
        coordinates = torch.tensor([lcoord],requires_grad=True,dtype=torch.float64,device=device_str) 
        
        #cell
        cell=torch.tensor([[boxhi[0]-boxlo[0],0,0],[xy,boxhi[1]-boxlo[1],0],[xz,yz,boxhi[2]-boxlo[2]]],dtype=torch.float64,device=device_str) 
        pbc = torch.tensor([0,0,0],dtype=torch.bool,device=device_str)
        
        emodel.lvalues(species,coordinates,cell,pbc,mylist)
        
        energy = torch.mean(emodel.nenergy,dim=0)
        aenergy = torch.mean(emodel.nlenergy,dim=0)
        euq = torch.std(emodel.nlenergy,dim=0)        
        force = torch.mean(emodel.nforces,dim=0).numpy()
        stress =  torch.mean(emodel.nstress,dim=0)

        #for atomic energy
        lenergy = torch.mean(emodel.nlenergy,dim=0)

        #print(aenergy)

        #Need to further make the code work for MPI communications
        nape = aenergy[:nlocal].detach().numpy()*hartree2ev
        nauq = euq[:nlocal].detach().numpy()*hartree2ev        
        euq=nauq[:,None]

        #print(np.shape(q),np.shape(nauq[:,None]))

        #Original UQ
        #q[:]=euq[:] #comment for atom stype: atomic

        #Shifted based on atom type
        zeros=np.zeros(np.shape(euq))        
        hmask = np.ma.masked_where(atype[:nlocal,:]==6,euq) #Remove type =6
        cmask = np.ma.masked_where(atype[:nlocal,:]==1,euq) #Remove type =1
        
        cmean = cmask.min()
        hmean = hmask.min()        
        hshift = np.ma.filled(np.ma.masked_where(atype[:nlocal,:]==1,zeros),fill_value=hmean) #Remove type =1
        cshift = np.ma.filled(np.ma.masked_where(atype[:nlocal,:]==6,zeros),fill_value=cmean) #Remove type =6    
        nuq = euq -hshift-cshift
        #q[:]=nuq[:]#comment for atom stype: atomic
        #print(q)
        
        f[:][:] = force[:nlocal][:]*hartree2ev
        tarray[9]=energy.sum()*hartree2ev
        for i in range(0,3):
            for j in range(0,3):
                tarray[i*3+j]=-stress[i][j]*hartree2ev

    elif(device_str=='cuda'):
        #Loading model from python 
        
        #nn = m.nn
        species_to_tensor=m.species_to_tensor
        aev_computer = emodel.aev_computer
        energy_shifter = emodel.energy_shifter
        
        #species
        species = species_to_tensor(slist).unsqueeze(0).to(device_str) 
        #coordinates
        coordinates = torch.tensor([lcoord],requires_grad=True,dtype=torch.float64,device=device_str) 
        
        #cell
        cell=torch.tensor([[boxhi[0]-boxlo[0],0,0],[xy,boxhi[1]-boxlo[1],0],[xz,yz,boxhi[2]-boxlo[2]]],dtype=torch.float64,device=device_str) 
        pbc = torch.tensor([0,0,0],dtype=torch.bool,device=device_str)
        
        emodel.lvalues(species,coordinates,cell,pbc,mylist)
        
        energy = torch.mean(emodel.nenergy,dim=0)
        aenergy = torch.mean(emodel.nlenergy,dim=0)
        euq = torch.std(emodel.nlenergy,dim=0)        
        force = torch.mean(emodel.nforces,dim=0).cpu().numpy()
        stress =  torch.mean(emodel.nstress,dim=0)

        #for atomic energy
        lenergy = torch.mean(emodel.nlenergy,dim=0)

        #For the atomic UQ, not requires for MD running.
        #print(aenergy)
        #Need to further make the code work for MPI communications
        #nape = aenergy[:nlocal].detach().numpy()*hartree2ev
        #nauq = euq[:nlocal].detach().numpy()*hartree2ev        
        #euq=nauq[:,None]

        #print(np.shape(q),np.shape(nauq[:,None]))

        #Original UQ
        #q[:]=euq[:] #comment for atom stype: atomic

        #Shifted based on atom type
        #zeros=np.zeros(np.shape(euq))        
        #hmask = np.ma.masked_where(atype[:nlocal,:]==6,euq) #Remove type =6
        #cmask = np.ma.masked_where(atype[:nlocal,:]==1,euq) #Remove type =1
        
        #cmean = cmask.min()
        #hmean = hmask.min()        
        #hshift = np.ma.filled(np.ma.masked_where(atype[:nlocal,:]==1,zeros),fill_value=hmean) #Remove type =1
        #cshift = np.ma.filled(np.ma.masked_where(atype[:nlocal,:]==6,zeros),fill_value=cmean) #Remove type =6    
        #nuq = euq -hshift-cshift
        #q[:]=nuq[:]#comment for atom stype: atomic
        #print(q)
        
        f[:][:] = force[:nlocal][:]*hartree2ev
        tarray[9]=energy.sum()*hartree2ev
        for i in range(0,3):
            for j in range(0,3):
                tarray[i*3+j]=-stress[i][j]*hartree2ev

    #Without UQ version for GPU
    """
    elif(device_str=='cuda'):
        for myrank in range(0,size):
            if(rank==myrank):
                model=copy.deepcopy(model_cpu.to(device))
                #Loading model from python 
                nn = m.nn
                species_to_tensor=m.species_to_tensor
                model = m.model.to(device)
                aev_computer = m.aev_computer
                energy_shifter = m.energy_shifter

                
                #print('############# HEY I AM USING GPU #############', rank,myrank,time.perf_counter()) #for debug
                #species
                species = species_to_tensor(slist).unsqueeze(0).to(device) 
                #coordinates
                coordinates = torch.tensor([lcoord],requires_grad=True,dtype=torch.double,device=device) 
        
                #cell
                cell=torch.tensor([[boxhi[0]-boxlo[0],0,0],[xy,boxhi[1]-boxlo[1],0],[xz,yz,boxhi[2]-boxlo[2]]],dtype=torch.float64,device=device) 
                pbc = torch.tensor([0,0,0],dtype=torch.bool,device=device)    
                displacement = torch.zeros(3,3,requires_grad=True,dtype=torch.float64,device=device)
                scaling_factor = torch.eye(3,dtype=torch.float64,device=device)+displacement
    
                scoordinates=scaled_coordinates(cell,coordinates)
                new_cell = scale_cell(cell,scaling_factor)
                new_coordinates=real_coordinates(new_cell,scoordinates)
                sspecies,saevs = aev_computer((species,new_coordinates),cell,pbc)
                atomic_energies = nn._atomic_energies((sspecies,saevs))
    
                shift_energies = energy_shifter.self_energies.clone().to(species.device)
                shift_energies = shift_energies[species]
                shift_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
                assert shift_energies.shape == atomic_energies.shape
                atomic_energies+=shift_energies

                #List for local atoms
                tlist=[]
                for iatom, neighs in mylist:
                    tlist.append(iatom)

                loclist=torch.tensor(tlist,dtype=torch.int64,device=device)
                local_energies=torch.index_select(atomic_energies,1,loclist)
                derivative = torch.autograd.grad(atomic_energies.sum(),coordinates,retain_graph=True)[0]
                force = -derivative.cpu().numpy()[0]
                f[:][:] = force[:nlocal][:]*hartree2ev
    
                stress = np.array(torch.autograd.grad(local_energies.sum(),displacement)[0].cpu())

                #Array sum
                tarray[9]=local_energies.sum()*hartree2ev
                for i in range(0,3):
                    for j in range(0,3):
                        tarray[i*3+j]=-stress[i][j]*hartree2ev
                #print('############# HEY I FINISHED MY JOB #############', rank,myrank,time.perf_counter()) #for debug
                torch.cuda.empty_cache()
            comm.Barrier()
    """
    ############################## Gather values for returning energy and stress to LAMMPS#########################
    #Gather values
    sendbuf = tarray
    recvbuf = np.empty([size, 10], dtype=np.double)

    comm.Gather(sendbuf, recvbuf, root=0)
    if rank == 0:
        tv = recvbuf.sum(axis=0)
        for i in range(10):
            carray[i]=tv[i]


