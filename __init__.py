import numpy as np
print('Imported HFSCF')

def OrthoS(eigenval, eigenvec):
    return eigenvec @ (np.linalg.inv(eigenval) ** 0.5) @ np.transpose(eigenvec)

def Diag(mat):
    eigenval = np.diag(np.linalg.eigh(mat)[0])
    eigenvec = np.linalg.eigh(mat)[1]
    return eigenval, eigenvec

def OrthoF(eigenval, eigenvec):
    return np.transpose(eigenvec) @ eigenval @ eigenvec

def Fock(Da, Db, Hcore, ERI):
    Ja = np.einsum('ijkl, kl->ij', ERI, Da)
    Jb = np.einsum('ijkl, kl->ij', ERI, Db)
    K = np.einsum('ikjl, kl->ij', ERI, Da)
    return Hcore + Ja - K + Jb

def ElEnergy(D, Hcore, Fock):
    sum = Hcore + Fock
    return np.einsum('ij,ij->', D, sum)

def Unitary(mat):
    return np.allclose(np.linalg.inv(mat),np.transpose(mat))

def Hermitian(mat):
    return np.allclose(np.transpose(mat), mat)

def EHF(Hcore, Da, Fa, Db, Fb):
    alfa = Hcore + Fa
    beta = Hcore + Fb
    suma = np.einsum('ij,ij->', Da, alfa)
    sumb = np.einsum('ij,ij->', Db, beta)
    return (suma + sumb)/2

def SSquared(Da, Db, Ca, ocb, oca):
    Pa = np.linalg.inv(Ca)
    Da_acc = Pa @ np.transpose(Da) @ np.transpose(Pa)
    Db_acc = Pa @ np.transpose(Db) @ np.transpose(Pa)
    cont = ocb - np.trace(Da_acc @ Db_acc)
    Sz = (oca - ocb)/2
    return Sz * (Sz + 1) + cont, (oca - ocb)/2
