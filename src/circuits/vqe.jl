struct VQE
  H::MPO
  D::Int
  rotations::Array
  entanglers::Array
  γ::Float64 #gamma in IBM paper
  c::Float64 #c in IBM paper
  α::Float64 #alpha in BM paper
  η::Float64 #a in IBM paper
  seed::Int
end

"""
    VQE

Variational Quantum Eigensolver
"""
function VQE(H::MPO,D::Int;
             γ::Float64=0.1,
             c::Float64=0.01,
             α::Float64=0.6,
             η::Float64=0.01)
  N = length(H)
  
  # Build circuit using the RyRz structure
  rotations  = [] 
  entanglers = []
  r_layer = []
  
  # Depth 0
  # Layer of Ry
  for j in 1:N
    θ = π * rand()
    push!(r_layer,("Ry",j,(θ=θ,)))
  end
  # Layer of Rz
  for j in 1:N
    ϕ = 2*π * rand()
    push!(r_layer,("Rz",j,(ϕ=ϕ,)))
  end
  push!(rotations,r_layer)
  for d in 1:D
    r_layer = []
    e_layer = []
    # Layer of Ry
    for j in 1:N
      θ = π * rand()
      push!(r_layer,("Ry",j,(θ=θ,)))
    end
    # Layer of Rz
    for j in 1:N
      ϕ = 2*π * rand()
      push!(r_layer,("Rz",j,(ϕ=ϕ,)))
    end
    # Layers of CX
    for j in 1:2:(N-N%2)
      push!(e_layer,("CX",(j,j+1)))
    end
    for j in 2:2:(N+N%2-1)
      push!(e_layer,("CX",(j,j+1)))
    end
    push!(rotations,r_layer)
    push!(entanglers,e_layer)
  end
  
  return VQE(H,D,rotations,entanglers,γ,c,α,η)
end

"""
    updateangle!(gate::Tuple,eps::Float64)

Update single-qubit rotation gate angle.
"""
function updateangle!(gate::Tuple,eps::Float64)
  old_angle = gate[3][keys(gate[3])[1]]
  new_angle = old_angle + eps
  gate = Base.setindex(gate,Base.setindex(gate[3],new_angle,keys(gate[3])[1]),3)
  return gate
end

"""
    itervqe!(vqe::VQE,step::Int)

Performs one step of VQE.
"""
function itervqe!(vqe::VQE,step::Int)
  N = length(vqe.H)
  ψ0 = qubits(N)
  for j in 1:N
    replaceind!(vqe.H[j],inds(vqe.H[j],"Site")[1],firstind(ψ0[j],"Site"))
    replaceind!(vqe.H[j],inds(vqe.H[j],"Site")[2],prime(firstind(ψ0[j],"Site")))
  end
  
  Δ_step = [] 
  for d in 1:size(vqe.rotations)[1]
    Δ_layer = []
    for j in 1:size(vqe.rotations[d])[1]
      push!(Δ_layer,(vqe.c/step^vqe.γ)*(2*rand(0:1)-1))
    end
    push!(Δ_step,Δ_layer)
  end
  
  # θ+ evaluation
  for d in 1:size(vqe.rotations)[1]
    for r in 1:size(vqe.rotations[d])[1]
      vqe.rotations[d][r] = updateangle!(vqe.rotations[d][r],Δ_step[d][r])
    end
  end
  gates = Tuple[]

  addgates!(gates,vqe.rotations[1])
  for d in 1:vqe.D
    addgates!(gates,vqe.entanglers[d])
    addgates!(gates,vqe.rotations[d+1])
  end
  ψθ = runcircuit(ψ0,gates)
  Hψ = *(vqe.H,ψθ,method="naive")
  E_plus = inner(ψθ,Hψ)

  # θ- evaluation
  for d in 1:size(vqe.rotations)[1]
    for r in 1:size(vqe.rotations[d])[1]
      vqe.rotations[d][r] = updateangle!(vqe.rotations[d][r],-2*Δ_step[d][r])
    end
  end
  
  gates = Tuple[]
  addgates!(gates,vqe.rotations[1])
  for d in 1:vqe.D
    addgates!(gates,vqe.entanglers[d])
    addgates!(gates,vqe.rotations[d+1])
  end
  ψθ = runcircuit(ψ0,gates)
  Hψ = *(vqe.H,ψθ,method="naive")
  E_minus = inner(ψθ,Hψ)

  # Restore the parameters 
  for d in 1:size(vqe.rotations)[1]
    for r in 1:size(vqe.rotations[d])[1]
      vqe.rotations[d][r] = updateangle!(vqe.rotations[d][r],Δ_step[d][r])
    end
  end
 
  ## Update
  for d in 1:size(vqe.rotations)[1]
    for r in 1:size(vqe.rotations[d])[1]
      grad = step^(2*vqe.γ)*0.5*(E_plus-E_minus)*Δ_step[d][r]/vqe.c^2
      update = -real(grad) * vqe.η/step^(vqe.α)
      vqe.rotations[d][r] = updateangle!(vqe.rotations[d][r],update)
    end
  end
  
  # Energy evaluation
  gates = Tuple[]
  addgates!(gates,vqe.rotations[1])
  for d in 1:vqe.D
    addgates!(gates,vqe.entanglers[d])
    addgates!(gates,vqe.rotations[d+1])
  end
  ψθ = runcircuit(ψ0,gates) 
  Hpsi = *(vqe.H,ψθ,method="naive")
  E = inner(ψθ,Hψ)
  return real(E)
end

function runvqe(vqe::VQE,epochs::Int)
  for i in 1:epochs
    E = itervqe!(vqe,i)
    println("Energy = ",E)
  end
end

function addgates!(gates::Array,newgates::Array)
  for newgate in newgates
    push!(gates,newgate)
  end
end

