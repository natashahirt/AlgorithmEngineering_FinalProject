# runtests.jl is the standard julia interface for running tests
# from the Pkg environment, run "test" and it will automatically
# install all the required dependencies and then run the testsets

# include("test_admm_core.jl")
# include("test_fem.jl")
include("test_topopt.jl")
