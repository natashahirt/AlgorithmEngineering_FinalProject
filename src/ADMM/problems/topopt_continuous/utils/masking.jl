function apply_element_mask!(ϕ::Vector{Float64}, mask::Union{BitMatrix, Nothing})
    if !isnothing(mask)
        nelx, nely = size(mask)
        for j in 1:nely
            for i in 1:nelx
                if !mask[i,j] # if this value is false
                    element_id = (j-1) * nelx + i
                    ϕ[element_id] = 0.0
                end
            end
        end
    end
end

function unmasked_elements(ϕ::Vector{Float64}, mask::Union{BitMatrix, Nothing})
    if !isnothing(mask)
        active = Vector{Int}(undef, sum(mask))
        x = 1
        nelx, nely = size(mask)
        for j in 1:nely
            for i in 1:nelx
                if mask[i,j] # if this value is false
                    element_id = (j-1) * nelx + i
                    active[x] = element_id
                    x += 1
                end
            end
        end
    else
        active = eachindex(ϕ)
    end
    return active
end