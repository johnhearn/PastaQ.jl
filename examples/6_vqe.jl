using ITensors
using PastaQ
using Random
using HDF5

Random.seed!(1234)

input_path = "../data/vqe_LiH.h5"
fin = h5open(input_path,"r")
H  = read(fin,"H",MPO)
E0 = read(fin,"E0") 
cl_energy = read(fin,"E_cl")

N = length(H)
D = 1
epochs = 100

vqe = VQE(H,D;η=0.5)

for i in 1:epochs
  E = itervqe!(vqe,i)+cl_energy
  abs_err = abs(E - E0)
  rel_err = abs_err / abs(E0)
  println("Step: ",i,"  Energy = ",E," / ",E0," rel_err = ",rel_err)
end

