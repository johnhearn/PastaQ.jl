struct VQE
  N::Int
  D::Int
  H::MPO
  psi::MPS
  rotations::Array
  entanglers::Array
  γ::Float64 #gamma in IBM paper
  c::Float64 #c in IBM paper
  α::Float64 #alpha in BM paper
  η::Float64 #a in IBM paper
  seed::Int
end

function VQE(N::Int,D::Int,H::MPO;
             γ::Float64=0.1,
             c::Float64=0.01,
             α::Float64=0.6,
             η::Float64=0.01,
             seed::Int=1234)
  Random.seed!(seed) 
  psi = qubits(N)
  for j in 1:N
    replaceind!(psi[j],firstind(psi[j],"Site"),firstind(H[j],"Site"))
  end
  noprime!(psi)
  # Build circuit using the RyRz structure
  rotations  = [] 
  entanglers = []
  r_layer = []
  
  # Depth 0
  # Layer of Ry
  for j in 1:N
    θ = π * rand()
    push!(r_layer,(gate="Ry",site=j,params=(θ=θ,)))
  end
  # Layer of Rz
  for j in 1:N
    ϕ = 2*π * rand()
    push!(r_layer,(gate="Rz",site=j,params=(ϕ=ϕ,)))
  end
  push!(rotations,r_layer)
  for d in 1:D
    r_layer = []
    e_layer = []
    # Layer of Ry
    for j in 1:N
      θ = π * rand()
      push!(r_layer,(gate="Ry",site=j,params=(θ=θ,)))
    end
    # Layer of Rz
    for j in 1:N
      ϕ = 2*π * rand()
      push!(r_layer,(gate="Rz",site=j,params=(ϕ=ϕ,)))
    end
    # Layers of CX
    for j in 1:2:(N-N%2)
      push!(e_layer,(gate = "Cx", site = [j,j+1]))
    end
    for j in 2:2:(N+N%2-1)
      push!(e_layer,(gate = "Cx", site = [j,j+1]))
    end
    push!(rotations,r_layer)
    push!(entanglers,e_layer)
  end
  
  return VQE(N,D,H,psi,rotations,entanglers,γ,c,α,η,seed)
end

function updateangle!(gate::NamedTuple,eps::Float64)
  old_angle = gate[:params][keys(gate[:params])[1]]
  new_angle = old_angle + eps
  gate = Base.setindex(gate,Base.setindex(gate[:params],new_angle,keys(gate[:params])[1]),:params)
  return gate
end

function itervqe!(vqe::VQE,step::Int)
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
  resetqubits!(vqe.psi)
  gates = []
  addgates!(gates,vqe.rotations[1])
  for d in 1:vqe.D
    addgates!(gates,vqe.entanglers[d])
    addgates!(gates,vqe.rotations[d+1])
  end
  tensors = compilecircuit(vqe.psi,gates) 
  psitheta = runcircuit(vqe.psi,tensors)
  Hpsi = vqe.H * psitheta
  E_plus = inner(psitheta,Hpsi)

  # θ- evaluation
  for d in 1:size(vqe.rotations)[1]
    for r in 1:size(vqe.rotations[d])[1]
      vqe.rotations[d][r] = updateangle!(vqe.rotations[d][r],-2*Δ_step[d][r])
    end
  end
  resetqubits!(vqe.psi)
  gates = []
  addgates!(gates,vqe.rotations[1])
  for d in 1:vqe.D
    addgates!(gates,vqe.entanglers[d])
    addgates!(gates,vqe.rotations[d+1])
  end
  tensors = compilecircuit(vqe.psi,gates) 
  psitheta = runcircuit(vqe.psi,tensors)
  Hpsi = vqe.H * psitheta
  E_minus = inner(psitheta,Hpsi)

  # Restore the parameters 
  for d in 1:size(vqe.rotations)[1]
    for r in 1:size(vqe.rotations[d])[1]
      vqe.rotations[d][r] = updateangle!(vqe.rotations[d][r],Δ_step[d][r])
    end
  end
 
  # Update
  for d in 1:size(vqe.rotations)[1]
    for r in 1:size(vqe.rotations[d])[1]
      grad = step^(2*vqe.γ)*0.5*(E_plus-E_minus)*Δ_step[d][r]/vqe.c^2
      update = -real(grad) * vqe.η/step^(vqe.α)
      vqe.rotations[d][r] = updateangle!(vqe.rotations[d][r],update)
    end
  end
  
  # Energy evaluation
  resetqubits!(vqe.psi)
  gates = []
  addgates!(gates,vqe.rotations[1])
  for d in 1:vqe.D
    addgates!(gates,vqe.entanglers[d])
    addgates!(gates,vqe.rotations[d+1])
  end
  tensors = compilecircuit(vqe.psi,gates) 
  psitheta = runcircuit(vqe.psi,tensors)
  Hpsi = vqe.H * psitheta
  E = inner(psitheta,Hpsi)
  return real(E)
end

function runvqe(vqe::VQE,epochs::Int)
  for i in 1:epochs
    E = itervqe!(vqe,i)
    println("Energy = ",E)
  end
end

