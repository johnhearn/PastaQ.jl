var documenterSearchIndex = {"docs":
[{"location":"QuantumTomography.html#Quantum-tomography-1","page":"Quantum Tomography","title":"Quantum tomography","text":"","category":"section"},{"location":"QuantumTomography.html#Initialization-1","page":"Quantum Tomography","title":"Initialization","text":"","category":"section"},{"location":"QuantumTomography.html#Tomography-1","page":"Quantum Tomography","title":"Tomography","text":"","category":"section"},{"location":"QuantumTomography.html#","page":"Quantum Tomography","title":"Quantum Tomography","text":"tomography\nPastaQ._tomography","category":"page"},{"location":"QuantumTomography.html#PastaQ.tomography","page":"Quantum Tomography","title":"PastaQ.tomography","text":"tomography(data::Array, L::LPDO; optimizer::Optimizer, kwargs...)\ntomography(data::Array, ψ::MPS; optimizer::Optimizer, kwargs...)\n\nRun quantum state tomography using a the starting state model on data.\n\nArguments:\n\nmodel: starting LPDO state.\ndata: training data set of projective measurements.\nbatchsize: number of data-points used to compute one gradient iteration.\nepochs: total number of full sweeps over the dataset.\ntarget: target quantum state underlying the data\nchoi: if true, compute probability using Choi matrix\nobserver!: pass an observer object (like TomographyObserver()) to keep track of measurements and fidelities.\noutputpath: write training metrics on file \n\n\n\n\n\n","category":"function"},{"location":"QuantumTomography.html#Certification-1","page":"Quantum Tomography","title":"Certification","text":"","category":"section"},{"location":"QuantumTomography.html#","page":"Quantum Tomography","title":"Quantum Tomography","text":"fidelity\nfrobenius_distance\nfidelity_bound","category":"page"},{"location":"QuantumTomography.html#PastaQ.fidelity","page":"Quantum Tomography","title":"PastaQ.fidelity","text":"fidelity(ψ::MPS, ϕ::MPS)\n\nCompute the fidelity between two MPS:\n\nF = |⟨ψ̃|ϕ̃⟩|²\n\nwhere ψ̃ and ϕ̃ are the normalized MPS.\n\n\n\n\n\nfidelity(ψ::MPS, ρ::Union{MPO, LPDO})\nfidelity(ρ::Union{MPO, LPDO}, ψ::MPS)\n\nCompute the fidelity between an MPS and MPO/LPDO:\n\nF = ⟨ψ̃|ρ̃|ψ̃⟩\n\nwhere ψ̃ and ρ̃ are the normalized MPS and MPO/LDPO.\n\n\n\n\n\n","category":"function"},{"location":"QuantumTomography.html#PastaQ.frobenius_distance","page":"Quantum Tomography","title":"PastaQ.frobenius_distance","text":"frobenius_distance(ρ::Union{MPO, LPDO}, σ::Union{MPO, LPDO})\n\nCompute the trace norm of the difference between two LPDOs and MPOs:\n\nT(ρ,σ) = sqrt(trace[(ρ̃-σ̃)†(ρ̃-σ̃)])\n\nwhere ρ̃ and σ̃ are the normalized density matrices.\n\n\n\n\n\n","category":"function"},{"location":"QuantumTomography.html#PastaQ.fidelity_bound","page":"Quantum Tomography","title":"PastaQ.fidelity_bound","text":"fidelity_bound(ρ::Union{MPO, LPDO}, σ::Union{MPO, LPDO})\n\nCompute the the following lower bound of the fidelity:\n\nF̃(ρ,σ) = trace[ρ̃† σ̃)]\n\nwhere ρ̃ and σ̃ are the normalized density matrices.\n\nThe bound becomes tight when the target state is nearly pure.\n\n\n\n\n\n","category":"function"},{"location":"Circuits.html#Circuit-Simulator-1","page":"Circuit Simulator","title":"Circuit Simulator","text":"","category":"section"},{"location":"Circuits.html#Initialization-1","page":"Circuit Simulator","title":"Initialization","text":"","category":"section"},{"location":"Circuits.html#","page":"Circuit Simulator","title":"Circuit Simulator","text":"qubits\nresetqubits!","category":"page"},{"location":"Circuits.html#PastaQ.qubits","page":"Circuit Simulator","title":"PastaQ.qubits","text":"qubits(N::Int; mixed::Bool=false)\n\nqubits(sites::Vector{<:Index}; mixed::Bool=false)\n\nInitialize qubits to:\n\nAn MPS wavefunction |ψ⟩ if mixed=false\nAn MPO density matrix ρ if mixed=true\n\n\n\n\n\n","category":"function"},{"location":"Circuits.html#PastaQ.resetqubits!","page":"Circuit Simulator","title":"PastaQ.resetqubits!","text":"resetqubits!(M::Union{MPS,MPO})\n\nReset qubits to the initial state:\n\n|ψ⟩=|0,0,…,0⟩ if M = MPS\nρ = |0,0,…,0⟩⟨0,0,…,0| if M = MPO\n\n\n\n\n\n","category":"function"},{"location":"Circuits.html#Running-a-quantum-circuit-1","page":"Circuit Simulator","title":"Running a quantum circuit","text":"","category":"section"},{"location":"Circuits.html#","page":"Circuit Simulator","title":"Circuit Simulator","text":"runcircuit","category":"page"},{"location":"Circuits.html#PastaQ.runcircuit","page":"Circuit Simulator","title":"PastaQ.runcircuit","text":"runcircuit(M::Union{MPS,MPO}, gate_tensors::Vector{<:ITensor}; kwargs...)\n\nApply the circuit to a state (wavefunction/densitymatrix) from a list of tensors.\n\n\n\n\n\nruncircuit(M::Union{MPS,MPO}, gates::Vector{<:Tuple}; noise=nothing, apply_dag=nothing, \n           cutoff=1e-15, maxdim=10000)\n\nApply the circuit to a state (wavefunction or density matrix) from a list of gates.\n\nIf an MPS |ψ⟩ is input, there are three possible modes:\n\nBy default (noise = nothing and apply_dag = nothing), the evolution U|ψ⟩ is performed.\nIf noise is set to something nontrivial, the mixed evolution ε(|ψ⟩⟨ψ|) is performed. Example: noise = (\"amplitude_damping\", (γ = 0.1,)) (amplitude damping channel with decay rate γ = 0.1)\nIf noise = nothing and apply_dag = true, the evolution U|ψ⟩⟨ψ|U† is performed.\n\nIf an MPO ρ is input, there are three possible modes:\n\nBy default (noise = nothing and apply_dag = nothing), the evolution U ρ U† is performed.\nIf noise is set to something nontrivial, the evolution ε(ρ) is performed.\nIf noise = nothing and apply_dag = false, the evolution Uρ is performed.\n\n\n\n\n\nruncircuit(N::Int, gates::Vector{<:Tuple}; process=false, noise=nothing,\n           cutoff=1e-15, maxdim=10000, kwargs...)\n\nApply the circuit to a state (wavefunction or density matrix) from a list of gates, where the state has N physical qubits.  The starting state is generated automatically based on the flags process, noise, and apply_dag.\n\nBy default (noise = nothing, apply_dag = nothing, and process = false),  the evolution U|ψ⟩ is performed where the starting state is set to |ψ⟩ = |000...⟩.  The MPS U|000...⟩ is returned.\nIf noise is set to something nontrivial, the mixed evolution ε(|ψ⟩⟨ψ|) is performed,  where the starting state is set to |ψ⟩ = |000...⟩.  The MPO ε(|000...⟩⟨000...|) is returned.\nIf process = true, the evolution U 1̂ is performed, where the starting state 1̂ = (1⊗1⊗1⊗…⊗1).  The MPO approximation for the unitary represented by the set of gates is returned.  In this case, noise must be nothing.\n\n\n\n\n\nruncircuit(M::ITensor,gate_tensors::Vector{ <: ITensor}; kwargs...)\n\nApply the circuit to a ITensor from a list of tensors.\n\n\n\n\n\nruncircuit(M::ITensor, gates::Vector{<:Tuple})\n\nApply the circuit to an ITensor from a list of gates.\n\n\n\n\n\n","category":"function"},{"location":"Optimizers.html#Optimizers-1","page":"Optimizers","title":"Optimizers","text":"","category":"section"},{"location":"Optimizers.html#Stochastic-gradient-descent-1","page":"Optimizers","title":"Stochastic gradient descent","text":"","category":"section"},{"location":"Optimizers.html#","page":"Optimizers","title":"Optimizers","text":"SGD\nPastaQ.update!(::LPDO,::Array,::SGD; kwargs...)","category":"page"},{"location":"Optimizers.html#PastaQ.SGD","page":"Optimizers","title":"PastaQ.SGD","text":"SGD(L::LPDO;η::Float64=0.01,γ::Float64=0.0)\n\nStochastic gradient descent with momentum.\n\nParameters\n\nη: learning rate\nγ: friction coefficient\nv: \"velocity\"\n\n\n\n\n\n","category":"type"},{"location":"Optimizers.html#PastaQ.update!-Tuple{LPDO,Array,SGD}","page":"Optimizers","title":"PastaQ.update!","text":"update!(L::LPDO,∇::Array,opt::SGD; kwargs...)\n\nUpdate parameters with SGD.\n\nvⱼ = γ * vⱼ - η * ∇ⱼ: integrated velocity\nθⱼ = θⱼ + vⱼ: parameter update\n\n\n\n\n\n","category":"method"},{"location":"Optimizers.html#Adagrad-1","page":"Optimizers","title":"Adagrad","text":"","category":"section"},{"location":"Optimizers.html#","page":"Optimizers","title":"Optimizers","text":"AdaGrad\nPastaQ.update!(::LPDO,::Array,::AdaGrad; kwargs...)","category":"page"},{"location":"Optimizers.html#PastaQ.AdaGrad","page":"Optimizers","title":"PastaQ.AdaGrad","text":"AdaGrad(L::LPDO;η::Float64=0.01,ϵ::Float64=1E-8)\n\nParameters\n\nη: learning rate\nϵ: shift \n∇²: square gradients (running sums)\n\n\n\n\n\n","category":"type"},{"location":"Optimizers.html#PastaQ.update!-Tuple{LPDO,Array,AdaGrad}","page":"Optimizers","title":"PastaQ.update!","text":"update!(L::LPDO,∇::Array,opt::AdaGrad; kwargs...)\n\nupdate!(ψ::MPS,∇::Array,opt::AdaGrad; kwargs...)\n\nUpdate parameters with AdaGrad.\n\ngⱼ += ∇ⱼ²: running some of square gradients\nΔθⱼ = η * ∇ⱼ / (sqrt(gⱼ+ϵ) \nθⱼ = θⱼ - Δθⱼ: parameter update\n\n\n\n\n\n","category":"method"},{"location":"Optimizers.html#Adadelta-1","page":"Optimizers","title":"Adadelta","text":"","category":"section"},{"location":"Optimizers.html#","page":"Optimizers","title":"Optimizers","text":"AdaDelta\nPastaQ.update!(::LPDO,::Array,::AdaDelta; kwargs...)","category":"page"},{"location":"Optimizers.html#PastaQ.AdaDelta","page":"Optimizers","title":"PastaQ.AdaDelta","text":"AdaDelta(L::LPDO;γ::Float64=0.9,ϵ::Float64=1E-8)\n\nParameters\n\nγ: friction coefficient\nϵ: shift \n∇²: square gradients (decaying average)\nΔθ²: square updates (decaying average)\n\n\n\n\n\n","category":"type"},{"location":"Optimizers.html#PastaQ.update!-Tuple{LPDO,Array,AdaDelta}","page":"Optimizers","title":"PastaQ.update!","text":"update!(L::LPDO,∇::Array,opt::AdaDelta; kwargs...)\n\nupdate!(ψ::MPS,∇::Array,opt::AdaDelta; kwargs...)\n\nUpdate parameters with AdaDelta\n\ngⱼ = γ * gⱼ + (1-γ) * ∇ⱼ²: decaying average\nΔθⱼ = ∇ⱼ * sqrt(pⱼ) / sqrt(gⱼ+ϵ) \nθⱼ = θⱼ - Δθⱼ: parameter update\npⱼ = γ * pⱼ + (1-γ) * Δθⱼ²: decaying average\n\n\n\n\n\n","category":"method"},{"location":"Optimizers.html#Adam-1","page":"Optimizers","title":"Adam","text":"","category":"section"},{"location":"Optimizers.html#","page":"Optimizers","title":"Optimizers","text":"Adam\nPastaQ.update!(::LPDO,::Array,::Adam; kwargs...)","category":"page"},{"location":"Optimizers.html#PastaQ.Adam","page":"Optimizers","title":"PastaQ.Adam","text":"Adam(L::LPDO;η::Float64=0.001,\n     β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7)\n\nAdam(ψ::MPS;η::Float64=0.001,\n     β₁::Float64=0.9,β₂::Float64=0.999,ϵ::Float64=1E-7)\n\nParameters\n\nη: learning rate\nβ₁: decay rate 1 \nβ₂: decay rate 2\nϵ: shift \n∇: gradients (decaying average)\n∇²: square gradients (decaying average)\n\n\n\n\n\n","category":"type"},{"location":"Optimizers.html#PastaQ.update!-Tuple{LPDO,Array,Adam}","page":"Optimizers","title":"PastaQ.update!","text":"update!(L::LPDO,∇::Array,opt::Adam; kwargs...)\n\nupdate!(ψ::MPS,∇::A0rray,opt::Adam; kwargs...)\n\nUpdate parameters with Adam\n\n\n\n\n\n","category":"method"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"(Image: alt text) (Image: Tests) <!–- (Image: codecov) (Image: ) –> (Image: ) (Image: License) (Image: arXiv)","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"PLEASE NOTE THIS IS PRE-RELEASE SOFTWARE      ","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"EXPECT ROUGH EDGES AND BACKWARD INCOMPATIBLE UPDATES","category":"page"},{"location":"index.html#A-Package-for-Simulation,-Tomography-and-Analysis-of-Quantum-Computers-1","page":"Introduction","title":"A Package for Simulation, Tomography and Analysis of Quantum Computers","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"PastaQ is a julia package for simulation and benchmarking of quantum computers using a combination of machine learning and tensor-network algorithms.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"The main features of PastaQ are:","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Simulation of quantum circuits. The package provides a simulator based on Matrix Product States (MPS) to simulate quantum circuits compiled into a set of quantum gates. Noisy circuits are simulated by specifying a noise model of interest, which is applied to each quantum gate.\nQuantum state tomography. Data-driven reconstruction of an unknown quantum wavefunction or density operators, learned respectively with an MPS and a Locally-Purified Density Operator (LPDO). The reconstruction can be certified by fidelity measurements with the target quantum state (if known, and if it admits an efficient tensor-network representation).\nQuantum process tomography. Data-driven reconstruction of an unknown quantum channel, characterized in terms of its Choi matrix (using a similar approach to quantum state tomography). The channel can be unitary (i.e. rank-1 Choi matrix) or noisy.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"PastaQ is developed at the Center for Computational Quantum Physics of the Flatiron Institute, and it is supported by the Simons Foundation.","category":"page"},{"location":"index.html#Installation-1","page":"Introduction","title":"Installation","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"The PastaQ package can be installed with the Julia package manager. From the Julia REPL, type ] to enter the Pkg REPL mode and run:","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"~ julia","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"julia> ]\n\npkg> add https://github.com/GTorlai/PastaQ.jl","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Please note that right now, PastaQ.jl requires that you use Julia v1.4 or later.","category":"page"},{"location":"index.html#Documentation-1","page":"Introduction","title":"Documentation","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"<!–-","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"STABLE –  documentation of the most recently tagged version.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"–>","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"DEVEL – documentation of the in-development version.","category":"page"},{"location":"index.html#Code-Overview-1","page":"Introduction","title":"Code Overview","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"The algorithms implemented in PastaQ rely on a tensor-network representation of quantum states, quantum circuits and quantum channels, which is provided by the ITensor package.","category":"page"},{"location":"index.html#Simulation-of-quantum-circuits-1","page":"Introduction","title":"Simulation of quantum circuits","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"A quantum circuit is built out of a collection of elementary quantum gates. In PastaQ, a quantum gate is described by a data structure g = (\"gatename\",sites,params) consisting of a gatename string identifying a particular gate, a set of sites identifying which qubits the gate acts on, and a set of gate parameters params (e.g. angles of qubit rotations). A comprehensive set of gates is provided, including Pauli matrices, phase and T gates, single-qubit rotations, controlled gates, Toffoli gate and others. Additional user-specific gates can be added. Once a set of gates is specified, the output quantum state (represented as an MPS) is obtained with the runcircuit function.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using PastaQ\n\nN = 4   # Number of qubits\n\n# Building a circuit data-structure\ngates = [(\"X\" , 1),                        # Pauli X on qubit 1\n         (\"CX\", (1, 3)),                   # Controlled-X on qubits [1,3]\n         (\"Rx\", 2, (θ=0.5,)),              # Rotation of θ around X\n         (\"Rn\", 3, (θ=0.5, ϕ=0.2, λ=1.2)), # Arbitrary rotation with angles (θ,ϕ,λ)\n         (\"√SWAP\", (3, 4)),                # Sqrt Swap on qubits [2,3]\n         (\"T\" , 4)]                        # T gate on qubit 4\n\n# Returns the MPS at the output of the quantum circuit: `|ψ⟩ = Û|0,0,…,0⟩`\n# First the gate (\"X\" , 1) is applied, then (\"CX\", (1, 3)), etc.\nψ = runcircuit(N, gates)\n# This is equivalent to:\n# julia> ψ0 = qubits(N) # Initialize |ψ⟩ to |0,0,…⟩\n# julia> ψ = runcircuit(ψ0,gates) # Run the circuit","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"The unitary circuit can be approximated by a MPO, running the runcircuit function with the flag process=true. Below is an example for a random quantum circuit.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"(Image: alt text)","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using PastaQ\n\n# Example 1a: random quantum circuit\n\nN = 4     # Number of qubits\ndepth = 4 # Depth of the circuit\n\n# Generate a random quantum circuit built out of layers of single-qubit random\n# rotations + CX gates, alternating between even and of odd layers.\ngates = randomcircuit(N, depth)\n\n@show gates\n\n# Returns the MPS at the output of the quantum circuit: `|ψ⟩ = Û|0,0,…,0⟩`\nψ = runcircuit(N, gates)\n\n# Generate the MPO for the unitary circuit:\nU = runcircuit(N, gates; process=true)","category":"page"},{"location":"index.html#Noisy-gates-1","page":"Introduction","title":"Noisy gates","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"If a noise model is provided, a local noise channel is applied after each quantum gate. A noise model is described by a string identifying a set of Kraus operators, which can depend on a set of additional parameters. The runcircuit function in this setting returns the MPO for the output mixed density operator. The full quantum channel has several (and equivalent) mathematical representations. Here we focus on the Choi matrix, which is obtained by applying a given channel ε to half of N pairs of maximally entangled states.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"(Image: alt text)","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using PastaQ\n\n# Example 1b: noisy quantum circuit\n\nN = 4     # Number of qubits\ndepth = 4 # Depth of the quantum circuit\ngates = randomcircuit(N, depth) # random circuit\n\n# Run the circuit using an amplitude damping channel with decay rate `γ=0.01`.\n# Returns the MPO for the mixed density operator `ρ = ε(|0,0,…⟩⟨0,0,̇…|), where\n# `ε` is the quantum channel.\nρ = runcircuit(N, gates; noise = (\"amplitude_damping\", (γ = 0.01,)))\n\n# Compute the Choi matrix of the channel\nΛ = runcircuit(N, gates; process = true, noise = (\"amplitude_damping\", (γ = 0.01,)))","category":"page"},{"location":"index.html#Generation-of-projective-measurements-1","page":"Introduction","title":"Generation of projective measurements","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"For a given quantum circuit, with or without noise, different flavors of measurement data can be obtained with the function getsamples(...) If one is interested in the quantum state at the output of the circuit, the function carries out a set of projective measurements in arbitrary local bases. By default, each qubit is measured randomly in the bases corresponding to the Pauli matrices. The output quantum state, given as an MPS wavefunction or MPO density operators for unitary and noisy circuits respectively, is also returned with the data.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using PastaQ\n\n# Example 2: generation of measurement data\n\n# Set parameters\nN = 4                           # Number of qubits\ndepth = 4                       # Depth of random circuit\nnshots = 1000                   # Number of measurements\ngates = randomcircuit(N, depth) # Build gates\n\n\n# 2a) Output state of a noiseless circuit. By default, each projective measurement\n# is taken in basis randomly drawn from the the Pauli group. Also returns the output MPS.\ndata, ψ = getsamples(N, gates, nshots)\n\n#  Note: the above is equivalent to:\n# > bases = randombases(N, nshots; localbasis = [\"X\",\"Y\",\"Z\"])\n# > ψ = runcircuit(N, gates)\n# > data = getsamples(ψ, bases)\n\n# 2b) Output state of a noisy circuit. Also returns the output MPO\ndata, ρ = getsamples(N, gates, nshots; noise = (\"amplitude_damping\", (γ = 0.01,)))","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"For quantum process tomography of a unitary or noisy circuit, the measurement data consists of pairs of input and output states to the channel. Each input state is a product state of random single-qubit states. Be default, these are set to the six eigenstates of the Pauli matrices (an overcomplete basis). The output states are projective measurements for a set of different local bases. It returns the MPO unitary circuit (noiseless) or the Choi matrix (noisy).","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"# 2c) Generate data for quantum process tomography, consisting of input states\n# to a quantum channel, and the corresponding projective measurements\n# at the output. By defaul, the states prepared at the inputs are selected from\n# product states of eigenstates of Pauli operators, while measurements bases are\n# sampled from the Pauli group.\n\n# Unitary channel, returns the MPO unitary circuit\ndata, U = getsamples(N, gates, nshots; process=true)\n\n# Noisy channel, returns the Choi matrix\ndata, Λ = getsamples(N, gates, nshots; process = true, noise = (\"amplitude_damping\", (γ = 0.01,)))","category":"page"},{"location":"index.html#Quantum-tomography-1","page":"Introduction","title":"Quantum tomography","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"PastaQ provides a set of algorithms to reconstruction quantum states and channels from measurement data. Both problems have a similar setup: given a data set of measurements underlying an unknown target quantum state (or channel), a variational tensor network is optimized to minimize the distance between the data and the probability distribution that the variational model associates to the measurement outcomes.","category":"page"},{"location":"index.html#State-tomography-1","page":"Introduction","title":"State tomography","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Quantum state tomography consists of reconstructing an unknown quantum state underlying a set of measurement data. The ingredients for state tomography are a variational quantum state, a data-driven reconstruction algorithm and an optimization routine. In PastaQ, the variational quantum states provided are an MPS and an LPDO, for the reconstruction of a pure quantum wavefunction and a mixed density operator respectively. The reconstruction algorithm is based on unsupervised machine learning of probability distributions. A widely used approach consists of optimizing a model distribution by minimizing the Kullbach-Leibler (KL) divergence between the model and the unknown target distribution, which is approximated by the training data. For quantum states, the measurement data is made of projective measurements in arbitrary local bases, and the model probability distribution is obtained by contracting the variational tensor network with a set of projectors corresponding to the eigenstates of the observed measurement outcome.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"Given a set of training data, the first step is the definition of the quantum state to be optimized. A random wavefunction or density operator is built using the function randomstate(N) and the appropriate flag mixed. Next, one defines a specific optimizer to be used in reconstruction, such as Stochastic Gradient Descent (SGD). Quantum state tomography is carried out by calling the function tomography, with inputs the initial starting state ψ0, the training data set data, and the optimizer opt. Additional inputs include the number of training iterations (epochs), the number of samples used for a single gradient update (batch_size), as well as the target quantum state (target) if available. During the training, the cost function is printed, as well as the fidelity against the target quantum state, if target is provided.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"(Image: alt text)","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using PastaQ\n\n# Load the training data, as well as the target quantum state from file.\ndata, target = loadsamples(\"PATH_TO_DATAFILE.h5\")\nN = size(data)[2] # Number of qubits\n\n# 1. Reconstruction with a variational wavefunction:\n#\n# Initialize a variational MPS with bond dimension χ = 10.\nψ0 = randomstate(N; χ = 10)\n\n# Initialize stochastic gradient descent with learning rate η = 0.01\nopt = SGD(η = 0.01)\n\n# Run quantum state tomography\nψ = tomography(data, ψ0; optimizer = opt, target = target)\n\n# 2. Reconstruction with a variational density matrix:\n#\n# Initialize a variational LPDO with bond dimension χ = 10 and Kraus dimension ξ = 2.\nρ0 = randomstate(N; mixed = true, χ = 10, ξ = 2)\n\n# Run quantum state tomography\nρ = tomography(data, ρ0; optimizer = opt, target = target)","category":"page"},{"location":"index.html#Process-tomography-1","page":"Introduction","title":"Process tomography","text":"","category":"section"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"In quantum process tomography, the object being reconstructed is a quantum channel ε, fully specified by its Choi matrix Λ (defined over 2N qubits). In practice, process tomography reduces to quantum state tomography of the Choi matrix, where the training data consists of input states to the channel, and output projective measurements. For the special case of a unitary (noiseless) channel U, the Choi matrix has rank-1 and is equivalent to a pure state obtained by being the legs of the unitary operator U.","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"(Image: alt text)","category":"page"},{"location":"index.html#","page":"Introduction","title":"Introduction","text":"using PastaQ\n\n# Load the training data, as well as the target quantum state from file.\ndata, target = loadsamples(\"PATH_TO_DATAFILE.h5\")\nN = size(data)[2] # Number of qubits\n\n# 1. Reconstruction with a variational MPO:\n#\n# Initialize a variational MPO with bond dimension χ = 10.\nU0 = randomprocess(N; χ = 10)\n\n# Initialize stochastic gradient descent with learning rate η = 0.01\nopt = SGD(η = 0.01)\n\n# Run quantum state tomography\nU = tomography(data, U0; optimizer = opt, target = target)\n\n# 2. Reconstruction with a variational density matrix:\n#\n# Initialize a variational LPDO with bond dimension χ = 10 and Kraus dimension ξ = 2.\nΛ0 = randomprocess(N; mixed = true, χ = 10, ξ = 2)\n\n# Run quantum state tomography\nΛ = tomography(data, Λ0; optimizer = opt, target = target)","category":"page"},{"location":"RandomStates.html#Random-states-1","page":"Random states","title":"Random states","text":"","category":"section"},{"location":"RandomStates.html#Quantum-states-1","page":"Random states","title":"Quantum states","text":"","category":"section"},{"location":"RandomStates.html#","page":"Random states","title":"Random states","text":"randomstate","category":"page"},{"location":"RandomStates.html#PastaQ.randomstate","page":"Random states","title":"PastaQ.randomstate","text":"randomstate(N::Int64; kwargs...)\n\nGenerates a random quantum state of N qubits\n\nArguments\n\nN: number of qubits\nmixed: if false (default), generate a random MPS; if true, generates a random LPDO\ninit: initialization criteria: \"randompars\" initializes random tensor components;  \"circuit initializes with a random quantum circuit (MPS only).\nσ: size of the 0-centered uniform distribution in init=\"randpars\". \nχ: bond dimension of the MPS/LPDO\n'ξ`: kraus dimension (LPDO)\nnormalize: if true, return normalize state\ncplx: if true (default), returns complex-valued state\n\n\n\n\n\nrandomstate(M::Union{MPS,MPO,LPDO}; kwargs...)\n\nGenerate a random state with same Hilbert space (i.e. site indices) of a reference state M.\n\n\n\n\n\n","category":"function"},{"location":"RandomStates.html#Quantum-channels-1","page":"Random states","title":"Quantum channels","text":"","category":"section"},{"location":"RandomStates.html#","page":"Random states","title":"Random states","text":"randomprocess","category":"page"},{"location":"RandomStates.html#PastaQ.randomprocess","page":"Random states","title":"PastaQ.randomprocess","text":"randomprocess(N::Int64; kwargs...)\n\nGenerates a random quantum procecss of N qubits.\n\nArguments\n\nN: number of qubits\nmixed: if false (default), generates a random MPO; if true, generates a random LPDO.\ninit: initialization criteria, set to \"randompars\" (see randomstate).\nσ: size of the 0-centered uniform distribution in init=\"randpars\". \nχ: bond dimension of the MPO/LPDO.\n'ξ`: kraus dimension (LPDO).\ncplx: if true (default), returns complex-valued state.\n\n\n\n\n\nrandomprocess(M::Union{MPS,MPO}; kwargs...)\n\nGenerate a random process with same Hilbert space (i.e. input and output indices)of a reference process M.\n\n\n\n\n\n","category":"function"}]
}
