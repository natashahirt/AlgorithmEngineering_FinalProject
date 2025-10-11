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
include("ADMM/ADMM.jl")
include("FEM/FEM.jl")

# Re-export submodules
using .ADMM
using .FEM

export ADMM, FEM

end # module AlgorithmEngineering

