import numpy as np
import os

#CLASS for periodic sites on the model
class Sites:
    def __init__(self,lattvec,nsite,sitepos,coord=None):

        # Initializes Sites class, Sites takes: lattice parameters
        # as a 3x3 array 'lattvec'; number of sites as an integer ; 
        # 'nsite'; site positions as a nsite x 3 array 'sitepos'  ;
        # and a string that marks wether sitepos is given in Frac-;
        # tional or Cartesian coordinates, default is 'Fractional'.
        # If positions are given in fractional coordinates they are
        # transformed to cartesian coordinates.

        self.lattvec = np.array(lattvec,dtype=np.double)
        self.nsite = nsite

        if coord is None or coord[0] == 'F' or coord[0] == 'f':
            self.posfrac = np.array(sitepos,dtype=np.double)
            self.poscart = self.frac2cart(self.lattvec,self.posfrac)
        else:
            self.poscart = np.array(sitepos,dtype=np.double)
            self.posfrac = self.cart2frac(self.lattvec,self.poscart)

    # Initialize Sites from a file    
    @classmethod
    def from_in(cls,file_name,coord=None):

        # Reads structure from an external file. The format of the
        # file is the following:
        #
        # {Descriptive line}
        # \vec{a} = a_x a_y a_z
        # \vec{b} = b_x b_y b_z
        # \vec{c} = c_x c_y c_z
        # nsite
        # n*a m*b l*c |
        # ...         | For frac coordinates
        # or
        # x y z       |
        # ...         | For cartesian coordinates        

        with open(file_name,'r') as f:
                string = f.readline()
                loadfile = [np.array(list(map(np.double, line.split()))) for line in f]
                lattvec = np.array(loadfile[0:3],dtype=np.double)
                nsite = int(loadfile[3])
                sitepos = np.array(loadfile[4:nsite+5],dtype=np.double)
                variables = cls(lattvec,nsite,sitepos,coord)
    
        return variables

    # Methods for coordinate transformations
    @staticmethod
    def frac2cart(latt,frac):
        cart = np.matmul(latt.T,frac.T)
        return cart.T

    @staticmethod
    def cart2frac(latt,cart):
        frac = np.matmul(np.linalg.inv(latt).T,cart.T)
        return frac.T

#CLASS for brillouin zone parameters
class FirstBZ():
    def __init__(self,objSites,kptmesh=None,kpoints=None,nkpt=None):

        # Initializes FirstBZ class, it feeds the object instance from
        # Class 'Sites' and the number of k-points in each direction in
        # the recirpocal space. It can given formated as a string: 'n l m'
        # or as an array ['n','l','m'] or [n,l,m]

        if isinstance(objSites,Sites):
            pass
        else:
            print('You did not input an object from Sites class, I quit this...')
            quit()

        if isinstance(kptmesh,str) and kpoints is None:
            kptarr = kptmesh.split()
            nkr1 = int(kptarr[0])
            nkr2 = int(kptarr[1])
            nkr3 = int(kptarr[2])
            self.nkpt = nkr1*nkr2*nkr3
            kp1 = np.divide(np.arange(nkr1),nkr1)
            kp2 = np.divide(np.arange(nkr2),nkr2)
            kp3 = np.divide(np.arange(nkr3),nkr3)
            self.kptfrac = np.array(np.meshgrid(kp1,kp2,kp3)).T.reshape(-1,3)
        elif isinstance(kptmesh,list) and kpoints is None:
            nkr1 = int(kptmesh[0])
            nkr2 = int(kptmesh[1])
            nkr3 = int(kptmesh[2])
            self.nkpt = nkr1*nkr2*nkr3
            kp1 = np.divide(np.arange(nkr1),nkr1)
            kp2 = np.divide(np.arange(nkr2),nkr2)
            kp3 = np.divide(np.arange(nkr3),nkr3)
            self.kptfrac = np.array(np.meshgrid(kp1,kp2,kp3)).T.reshape(-1,3)
        elif kptmesh is None and kpoints is not None:
            self.nkpt = nkpt
            self.kptfrac = kpoints
        elif kptmesh is None and kpoints is None:
            print('You did not input anything in FirstBZ class, I quit this...')
            quit()

        # Transform kpoints from fraction to cartesian coordinates
        # Define reciprocal lattice first
        self.recvec = self.get_reciprocal_lattice(objSites.lattvec)

        # Transform k-points in cartesians
        self.kptcart = objSites.frac2cart(self.recvec,self.kptfrac)

    # Initialize FirstBZ from a file    
    @classmethod
    def from_in(cls,objSites,file_name):

        # Reads structure from an external file. The format of the
        # file is the following:
        #
        # {Descriptive line}
        # nkpt
        # n*a m*b l*c                     |
        # ...                             | For frac coordinates      

        with open(file_name,'r') as f:
                kpoints = np.genfromtxt(f,dtype=np.double,skip_header=2,usecols=(0,1,2))
                nkpt = np.shape(kpoints)[0]
                variables = cls(objSites,None,kpoints,nkpt)
        return variables

    @staticmethod
    def get_reciprocal_lattice(latt):
        a1xa2 = np.cross(latt[0,:],latt[1,:])
        a2xa3 = np.cross(latt[1,:],latt[2,:])
        a3xa1 = np.cross(latt[2,:],latt[0,:])
 
        b1 = np.divide(2.0*np.pi*a2xa3,np.dot(latt[0,:],a2xa3))
        b2 = np.divide(2.0*np.pi*a3xa1,np.dot(latt[1,:],a3xa1))
        b3 = np.divide(2.0*np.pi*a1xa2,np.dot(latt[2,:],a1xa2))

        return np.array([b1,b2,b3])

#CLASS for Model Hamiltonian parameters 
class ModelParam():
    def __init__(self,*args):
        pass

    @classmethod
    def from_w90(cls,objSites,file_name,soc=None):

        # Read file from wannier90 Hamiltonian outputs, with or without SOC
        # The format of such file is in the secion 8.19 in wannier90's user
        # guide

        if isinstance(objSites,Sites):
            pass
        else:
            print('You did not input an object from Sites class, I quit this...')
            quit()

        nspin = 2

        if soc is None:
            with open(file_name,'r') as f:
                # Read data
                string = f.readline()
                nwann = int(f.readline())
                ncell = int(f.readline())
                skipline = np.floor_divide(ncell,15)+1
                transferdata = np.genfromtxt(f,dtype="3f8,2i8,1f8,1f8",names=['cells','orbitals','real','imag'],skip_header=skipline)
                # End reading data
            
            # Shape the transfer integrals to shape (nspin,nrpt,norb,norb)
            # For this we first transform transfer integrals to complex numbers
            # and then shape them as (ncells,nwann,nwann)
            dat = (transferdata['real']+1j*transferdata['imag']).reshape(ncell,nwann,nwann).transpose(0,2,1)
            # Parameters
            norb = nwann//objSites.nsite
            # Slice data matrix blockwise to an array of shape (ncell*nsite*nsite,norb,norb)
            ham_r = dat.reshape(-1,norb,nwann//norb,norb).swapaxes(1,2).reshape(-1,norb,norb)
            ham_r = np.expand_dims(ham_r,axis=0)
            # Define spin up and spin down channels 1: up, 2:down
            ham_r = np.append(ham_r,ham_r,axis=0)
        
        else:
            # Case when SOC is considered
            with open(file_name,'r') as f:
                # Read data
                string = f.readline()
                nwann = int(f.readline()) # Double nwann when compared to non-soc, spinors {up,down,up,down...}
                ncell = int(f.readline())
                skipline = np.floor_divide(ncell,15)+1
                transferdata = np.genfromtxt(f,dtype="3f8,2i8,1f8,1f8",names=['cells','orbitals','real','imag'],skip_header=skipline)
                # End reading data

            # Shape the transfer integrals to shape (nnrpt,norb*nspin,norb*nspin)
            # For this we first transform transfer integrals to complex numbers
            # and then shape them as (ncells,nwann,nwann)    
            dat = (transferdata['real']+1j*transferdata['imag']).reshape((ncell,nwann,nwann)).transpose(0,2,1)
            # Rearrange matrix elements, from a basis of {up,down,up,down...} to {up,up,down,down...}
            # This way the nwann*nwann matrix will have the following form
            # |         |          |
            # |  up up  |  up down |
            # |_________|__________|          
            # |         |          |
            # | down up | down down|
            # |         |          |
            upup = dat[:,0:nwann:2,0:nwann:2]
            updown = dat[:,0:nwann:2,1:nwann:2]
            downup = dat[:,1:nwann:2,0:nwann:2]
            downdown = dat[:,1:nwann:2,1:nwann:2]
            print(np.shape(downdown))
            #d = np.append(upup,updown,axis=2)

if __name__ == "__main__":
    site = Sites.from_in(r'C:\Users\oarcelus\Desktop\mhampy\mfhf\hf_struct.in')
    kpt = FirstBZ.from_in(site,r'C:\Users\oarcelus\Desktop\mhampy\mfhf\hf_kpt.in')
    ham = ModelParam.from_w90(site,r'C:\Users\oarcelus\Desktop\mhampy\mfhf\vose2o5_hr_soc.dat',soc=True)
