# AlgorithmEngineering.jl - Main module
"""
AlgorithmEngineering

A Julia package for algorithm engineering research, featuring:
- ADMM: Alternating Direction Method of Multipliers framework
- FEM: Finite Element Method implementation

Author: Natasha Hirt <nhirt@mit.edu>
"""
module AlgorithmEngineering

# Include submodules
include("FEM/FEM.jl")
include("ADMM/ADMM.jl")

# Re-export submodules
using .FEM
using .ADMM

export FEM, ADMM

end # module AlgorithmEngineering

