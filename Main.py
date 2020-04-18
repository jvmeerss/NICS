import numpy as np
import psi4
from HFSCF import Diag, OrthoS, OrthoF, Fock, ElEnergy, Unitary, Hermitian, EHF, SSquared
import time
import matplotlib.pyplot as plt

psi4.core.set_output_file('output.dat', True)
psi4.set_memory(int(5e8))
np_memory = 2

psi4.set_options({'basis': 'cc-pvdz'})

mol = psi4.geometry("""
O  0.000000000000 -0.143225816552 0.000000000000
H  1.638036840407 1.136548822547 -0.000000000000
H -1.638036840407 1.136548822547 -0.000000000000
units bohr
""")
mol.update_geometry()

nre = mol.nuclear_repulsion_energy()
wave = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option('basis'))
mints = psi4.core.MintsHelper(wave.basisset())

occ_a = wave.nalpha()
occ_b = wave.nbeta()

S = np.asarray(mints.ao_overlap())
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
ERI = np.asarray(mints.ao_eri())
Hcore = T + V

Sval, Svec = Diag(S)
Smin = OrthoS(Sval, Svec)

Fa = Fb = OrthoF(Hcore, Smin)
Fa_val, Fa_vec = Fb_val, Fb_vec = Diag(Fa)
Ca = Cb = Smin @ Fa_vec

Da = np.einsum('ik,jk->ij',Ca[:, :occ_a], Ca[:, :occ_a])
Db = np.einsum('ik,jk->ij',Cb[:, :occ_b], Cb[:, :occ_b])

Eza_el = np.einsum('ij,ij->', Da, 2*Hcore)
Ezb_el = np.einsum('ij,ij->', Db, 2*Hcore)

ehf = 0
DE = -Eza_el
val = 1e-12
iteration = 0
energieshf = []
iterations  =[]
total_time = time.time()

while DE > val:
    DE = 0
    iteration += 1
    iterations.append(iteration)
    # nieuwe fockmatrix maken
    Fa = Fock(Da, Db, Hcore, ERI)
    Fb = Fock(Db, Da, Hcore, ERI)
    # gemiddelde energie berekenen met zelfde density matrix als fock
    E_el = EHF(Hcore, Da, Fa, Db, Fb)
    # fock matrix orthogonaliseren en diagonaliseren
    Fa_acc = OrthoF(Fa, Smin)
    Fb_acc = OrthoF(Fb, Smin)
    Fa_acc_val, Fa_acc_vec = Diag(Fa_acc)
    Fb_acc_val, Fb_acc_vec = Diag(Fb_acc)
    Ca = Smin @ Fa_acc_vec
    Cb = Smin @ Fb_acc_vec
    # nieuwe density matrix opstellen
    Da = np.einsum('ik,jk->ij', Ca[:, :occ_a], Ca[:, :occ_a])
    Db = np.einsum('ik,jk->ij', Cb[:, :occ_b], Cb[:, :occ_b])
    energieshf.append(E_el)
    DE = abs(ehf - E_el)
    ehf = E_el
S2, Sz = SSquared(Da, Db, Ca, occ_b, occ_a)
print('SCF energy (Hartree): {}\t<S^2> = {}\t<Sz> = {}\nLoop time: {}s'.format(energieshf[-1] + nre, S2, Sz, time.time() - total_time))

