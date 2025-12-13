"""
    create_hmm()

Utility to create a 3 state HMM for testing purposes. 
"""
function create_hmm()
    # Define HMM params
    α = 10.0

    trans = zeros(undef, 3, 3)
    init = zeros(undf, 3)

    # fill transition matrix
    for i in 1:3
        dir_vector = zeros(3)
        dir_vector[i] = α # make sticky!
        trans[i, :] = rand(Dirichlet(dir_vector))
    end

    # fill initial distribution
    init = rand(Dirichlet(ones(3)))

    # Define Emission Models
end