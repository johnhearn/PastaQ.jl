struct Optimizer
  η::Float64
end

function Optimizer(η::Float64)
  return Optimizer(η)
end

function updateSGD!(M::Union{MPS,MPO},G::Union{MPS,MPO},opt::Optimizer)
  for j in 1:length(M)
    M[j] = M[j] - η * G[j]
  end
end
