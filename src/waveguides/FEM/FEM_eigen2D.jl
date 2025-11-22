using CompScienceMeshes
using BEAST

include("FEM/FEM_curlcurl.jl")
##

#constants
μ0 = 4π*1e-7
ε0 = 8.854e-12
c0 = 1/sqrt(ε0*μ0)

#2D TE modes
"""
    TE2D(msh; basis = nothing)

finds the TEₘₙ mode of a 2D waveguide.
input arguments: msh -> 2D mesh of type- Mesh{3, 3, Float64}(SVector{3, Float64}
          kwarg: basis -> to input the basis function of the mesh
return arguments: vector of numerical wavenumber, vector of wavelength,
vector of frequency, arrays of eigenvectors, vector of eigenvalues,
basis function.
"""
function TE2D(msh; basis = nothing)
    bodymesh = msh   

    bnd = boundary(bodymesh)

    edges = submesh(!in(bnd), skeleton(bodymesh,1))

    #tangential electric field on PEC === 0
    X = BEAST.nedelec(bodymesh, edges)
    
    if !isnothing(basis)
        X = basis
    end

   #calling different material parameters for local operator
    Op_s = curlcurl(1.0, 1.0)
    
    S = assemble(Op_s, X, X)
    M = assemble(BEAST.Identity(), X, X) 

    #eigenvalues and eigenvectors
    k2, V_2d = eigen(Matrix(S), Matrix(M));

    #to remove null space solutions
    t = deepcopy(k2)
    
    indices = []
    for i in length(t):-1:1
        if isnan(t[i])
            deleteat!(t, i)
            append!(indices, i)
        elseif t[i] <= 0
            deleteat!(t, i)
            append!(indices, i)
        elseif isapprox(t[i], 0, atol = 0.1)
            deleteat!(t, i)
            append!(indices, i)
        elseif t[i] == Inf
            deleteat!(t, i)
            append!(indices, i)
        end
    end 
    t

    valid_k2 = [k2[i] for i in length(indices) + 1 : length(k2)]
    valid_V2 = [V_2d[:, i] for i in length(indices) + 1 : size(V_2d, 2)]
            
    #numerical wavenumber, wavelength, frequency
    k = sqrt.(t)
    λ = 2π ./ k
    f = c0 ./ λ

    return k, λ, f, valid_V2, valid_k2, X
end


#2D TM modes
"""
    TM2D(msh; basis = nothing)

finds the TMₘₙ mode of a 2D waveguide.
input arguments: msh -> 2D mesh of type- Mesh{3, 3, Float64}(SVector{3, Float64}
          kwarg: basis -> to input the basis function of the mesh
return arguments: vector of numerical wavenumber, vector of wavelength,
vector of frequency, arrays of eigenvectors, vector of eigenvalues,
basis function.
"""
function TM2D(msh; basis = nothing)
    bodymesh = msh   
    
    X = BEAST.nedelec(bodymesh)

    if !isnothing(basis)
        X = basis
    end

    #calling different material parameters for local operator
    Op_s = curlcurl(1.0, 1.0)
    
    S = assemble(Op_s, X, X)
    M = assemble(BEAST.Identity(), X, X) 

    #eigenvalues and eigenvectors
    k2, V_2d = eigen(Matrix(S), Matrix(M));

    #to remove null space solutions
    t = deepcopy(k2)
        
    indices = []
    for i in length(t):-1:1
        if isnan(t[i])
            deleteat!(t, i)
            append!(indices, i)
        end
        if t[i] <= 0
            deleteat!(t, i)
            append!(indices, i)
        end
        if isapprox(t[i], 0, atol = 0.1)
            deleteat!(t, i)
            append!(indices, i)
        end
        if t[i] == Inf
            deleteat!(t, i)
            append!(indices, i)
        end
    end 
    t

    valid_k2 = [k2[i] for i in length(indices) + 1 : length(k2)]
    valid_V2 = [V_2d[:, i] for i in length(indices) + 1 : size(V_2d, 2)]

    #numerical wavenumber, wavelength, frequency
    k = sqrt.(t)
    λ = 2π ./ k
    f = c0 ./ λ

    return k, λ, f, valid_V2, valid_k2, X, k2
end


"""
    analytical_freq(a, b, m, n)

gives the specific analytical frequency for the input arguments:
length, breadth, mode number along length, mode number along breadth
"""
function analytical_freq2d(a, b, m, n) sqrt((m/a)^2 + (n/b)^2)*c0/2 end