from __future__ import print_function
import torch,os
import torchani
from torchani.utils import ChemicalSymbolsToInts
import sys,copy

def scale_cell(cell,scaling_factor):
  return torch.matmul(scaling_factor,cell)

def real_coordinates(cell,scoordinates):
  coordinates = torch.matmul(scoordinates,cell)
  return coordinates

def scaled_coordinates(cell,coordinates,pbc):
  inv_cell = torch.inverse(cell)
  scoordinates = torch.matmul(coordinates,inv_cell)
  #wrappig
  #scoordinates -= scoordinates.floor()*pbc
  return scoordinates

class Model:
    def __init__(self):
        #define the number of models
        count = 0
        path = './best'+str(count)+'.pt'
        while(os.path.isfile(path)):
            count+=1
            path='./best'+str(count)
        self.nmodels = count
        #Customized models
        models=[]
        try:
            path=os.path.dirname(os.path.realpath(__file__))
        except NameError:
            path=os.getcwd()
        sae_file = os.path.join(path, 'sae_linfit_dftb.dat')
        const_file  = os.path.join(path, 'rC.params')
        consts = torchani.neurochem.Constants(const_file)
        self.aev_computer = torchani.AEVComputer(**consts)
        self.energy_shifter = torchani.neurochem.load_sae(sae_file)

        species_order =['C']#change
        num_species = len(species_order)

        self.species_to_tensor = ChemicalSymbolsToInts(species_order)
            
        aev_dim = self.aev_computer.aev_length

        """
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 1)
        )
        """
        
        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 224),
            torch.nn.GELU(),
            torch.nn.Linear(224, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 1)
        )#Change

        nns=[]
        
        self.nn = torchani.ANIModel([C_network]) # Change

        #Customized models
        models=[]
        for i in range(0,self.nmodels):
            fname = 'best'+str(i)+'.pt'
            tmp = copy.deepcopy(self.nn)
            #CPUs
            #tmp.load_state_dict(torch.load(fname,map_location='cpu'))            
            #tmodel = torchani.nn.Sequential(self.aev_computer,tmp,self.energy_shifter).to('cpu').to(torch.float64)
            #GPUs
            tmp.load_state_dict(torch.load(fname,map_location='cpu'))            
            tmodel = torchani.nn.Sequential(self.aev_computer,tmp,self.energy_shifter).to('cuda').to(torch.float64)            
            nns.append(tmp)
            models.append(tmodel)

        self.models = models
        self.nns = nns
        
        #ANI models
        """
        tmodels = torchani.models.ANI2x(periodic_table_index=True).to('cpu').to(torch.float64)
        models=[]
        for i in range(0,self.nmodels):
            models.append(tmodels[i])

        self.models=models
        """

        self.count =0

    #obsolete; function without MPI (neigh nlocal); This gets energy in the MPI domain with ghost atoms
    def gvalues(self,species,coordinates,cell,pbc):
        tenergy=[]
        tforces=[]
        tstress=[]

        #Building graph for autograd

        ndisplacement = []

        for nnp in self.models:
            displacement = torch.zeros(3,3,requires_grad=True,dtype=torch.float64)
            scaling_factor = torch.eye(3,dtype=torch.float64)+displacement
            
            scoordinates=scaled_coordinates(cell,coordinates,pbc)
            new_cell = scale_cell(cell,scaling_factor)
            new_coordinates=real_coordinates(new_cell,scoordinates)
            
            energy=nnp((species,new_coordinates),new_cell,pbc).energies
            tenergy.append(energy)
            
            derivative = torch.autograd.grad(energy.sum(), new_coordinates,retain_graph=True)[0]
            force = -derivative
            tforces.append(force)
            
            stress = torch.autograd.grad(energy.sum(), displacement)[0]
            tstress.append(stress[None,:])
            
        self.nenergy = torch.cat(tenergy,dim=0)
        self.nforces = torch.cat(tforces,dim=0)
        self.nstress = torch.cat(tstress,dim=0)


    def lvalues(self,species,coordinates,cell,pbc,mylist):
        tlenergy=[]
        tenergy=[]
        tforces=[]
        tstress=[]

        aev_computer = self.aev_computer
        energy_shifter = self.energy_shifter
        
        #List for local atoms
        tlist=[]
        for iatom, neighs in mylist:
            tlist.append(iatom)


        for nn in self.nns:
            displacement = torch.zeros(3,3,requires_grad=True,dtype=torch.float64,device=species.device)
            scaling_factor = torch.eye(3,dtype=torch.float64,device=species.device)+displacement
            
            scoordinates=scaled_coordinates(cell,coordinates,pbc)
            new_cell = scale_cell(cell,scaling_factor)
            new_coordinates=real_coordinates(new_cell,scoordinates)

            #For atomic energy
            sspecies,saevs = aev_computer((species,new_coordinates),cell,pbc)
            atomic_energies = nn._atomic_energies((sspecies,saevs))
            shift_energies = energy_shifter.self_energies.clone().to(species.device)
            shift_energies = shift_energies[species]
            shift_energies[species == torch.tensor(-1, device=species.device)] = torch.tensor(0, device=species.device, dtype=torch.double)
            assert shift_energies.shape == atomic_energies.shape
            atomic_energies+=shift_energies
            
            loclist=torch.tensor(tlist,dtype=torch.int64,device=species.device)
            local_energies=torch.index_select(atomic_energies,1,loclist)
            
            #energy=nnp((species,new_coordinates),new_cell,pbc).energies
            #derivative = torch.autograd.grad(energy.sum(), new_coordinates,retain_graph=True)[0]
            #force = -derivative
            #stress = torch.autograd.grad(energy.sum(), displacement)[0]

            derivative = torch.autograd.grad(atomic_energies.sum(),coordinates,retain_graph=True)[0]
            #force = torch.index_select(-derivative,1,loclist)
            force=-derivative
            #stress = np.array(torch.autograd.grad(local_energies.sum(),displacement)[0].cpu())
            stress = torch.autograd.grad(local_energies.sum(),displacement)[0]

            energy =torch.sum(local_energies,dim=1)

            #print("lenergy",local_energies)
            #print("energy",energy)
            
            tlenergy.append(local_energies)
            tenergy.append(energy)
            tforces.append(force)
            tstress.append(stress[None,:])

        self.nlenergy = torch.cat(tlenergy,dim=0)
        self.nenergy = torch.cat(tenergy,dim=0)
        self.nforces = torch.cat(tforces,dim=0)
        self.nstress = torch.cat(tstress,dim=0)                
        

"""
class Model:
    def __init__(self):
        try:
            path=os.path.dirname(os.path.realpath(__file__))

        except NameError:
            path=os.getcwd()
        sae_file = os.path.join(path, 'sae_linfit_ani.data')
        const_file  = os.path.join(path, 'rC.params')
        consts = torchani.neurochem.Constants(const_file)
        self.aev_computer = torchani.AEVComputer(**consts)
        self.energy_shifter = torchani.neurochem.load_sae(sae_file)

        species_order =['H','C']
        num_species = len(species_order)

        self.species_to_tensor = ChemicalSymbolsToInts(species_order)
            
        aev_dim = self.aev_computer.aev_length
        H_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 256),
            torch.nn.GELU(),
            torch.nn.Linear(256, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 1)
        )

        C_network = torch.nn.Sequential(
            torch.nn.Linear(aev_dim, 224),
            torch.nn.GELU(),
            torch.nn.Linear(224, 192),
            torch.nn.GELU(),
            torch.nn.Linear(192, 160),
            torch.nn.GELU(),
            torch.nn.Linear(160, 1)
        )

        self.nn = torchani.ANIModel([H_network,C_network])
        self.nn.load_state_dict(torch.load('force-training-best.pt',map_location='cpu'))
        self.model = torchani.nn.Sequential(self.aev_computer,self.nn,self.energy_shifter).to('cpu').to(torch.float64)
        self.count =0
    def check(self):
        self.count = self.count+1
        return self.count



device = torch.device('cpu')
coordinates = torch.tensor([[[0.03192167, 0.00638559, 0.01301679],
                             [-0.83140486, 0.39370209, -0.26395324],
                             [-0.66518241, -0.84461308, 0.20759389],
                             [0.45554739, 0.54289633, 0.81170881],
                             [0.66091919, -0.16799635, -0.91037834]]],
                           requires_grad=True, device=device,dtype=torch.float64)

# In periodic table, C = 6 and H = 1
species = torch.tensor([[6, 1, 1, 1, 1]], device=device)
cell=torch.tensor([[50,0,0],[0,50,0],[0,0,50]],requires_grad=True,dtype=torch.float64)
pbc = torch.tensor([1,1,1],dtype=torch.bool)



emodel = EnsembleModel(2)
emodel.gvalues(species,coordinates,cell,pbc)


#Check mean values and std
#Std values as UQ
print(torch.mean(emodel.nenergy,dim=0),torch.std(emodel.nenergy,dim=0))
print(torch.mean(emodel.nforces,dim=0),torch.std(emodel.nforces,dim=0))
print(torch.mean(emodel.nstress,dim=0),torch.std(emodel.nstress,dim=0))
"""
