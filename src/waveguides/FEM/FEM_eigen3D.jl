using CompScienceMeshes
using BEAST

include("FEM_curlcurl.jl")

##
#constants
μ0 = 4π*1e-7
ε0 = 8.854e-12
c0 = 1/sqrt(ε0*μ0)


#TE modes in a 3D cavity
"""
    TE3D(bodymesh, basis = nothing)

finds the TEₘₙₚ mode of a 3D waveguide.
input arguments: bodymesh -> 3D mesh of type- Mesh{3, 4, Float64}(SVector{3, Float64}
          kwarg: basis -> to input the basis function of the mesh
return arguments: vector of numerical wavenumber, vector of wavelength,
vector of frequency, arrays of eigenvectors, vector of eigenvalues,
basis function.
"""
function TE3D(bodymesh, basis = nothing)

    bnd = skeleton(boundary(bodymesh), 1)
    edges = submesh(!in(bnd), skeleton(bodymesh, 1))

    X = BEAST.nedelecc3d(bodymesh, edges)
    if !isnothing(basis)
        X = basis
    end
    
    #calling different material parameters for local operator
    Op_s = curlcurl(1.0, 1.0)
    
    S = assemble(Op_s, X, X)
    M = assemble(BEAST.Identity(), X, X) 

    #eigenvalues and eigenvectors
    k2, V_3d = eigen(Matrix(S), Matrix(M));

    #to find the dimension of null space   
    t = real.(deepcopy(k2))
        
    indices = []
    for i in length(t):-1:1
        if isnan(t[i])
            deleteat!(t, i)
            append!(indices, i) 
        elseif real(t[i]) <= 0
            deleteat!(t, i)
            append!(indices, i)
        elseif isapprox(real(t[i]), 0, atol = 0.1)
            deleteat!(t, i)
            append!(indices, i)    
        elseif real(t[i]) == Inf
            deleteat!(t, i)
            append!(indices, i)
        end
    end 
    
    valid_k2 = [k2[i] for i in length(indices) + 1 : length(k2)]
    valid_V3 = [V_3d[:, i] for i in length(indices) + 1 : size(V_3d, 2)]

    #numerical wavenumber, wavelength, frequency
    k = sqrt.(t) 
    λ = real(2π ./ k)
    f = sort(c0 ./ λ)
    return k, λ, f, valid_V3, valid_k2, X, k2
end


#TM modes in a 3D cavity
"""
    TM3D(bodymesh, basis = nothing)

finds the TMₘₙₚ mode of a 3D waveguide.
input arguments: bodymesh -> 3D mesh of type- Mesh{3, 4, Float64}(SVector{3, Float64}
          kwarg: basis -> to input the basis function of the mesh
return arguments: vector of numerical wavenumber, vector of wavelength,
vector of frequency, arrays of eigenvectors, vector of eigenvalues,
basis function.
"""
function TM3D(bodymesh, basis = nothing)

    X = BEAST.nedelecc3d(bodymesh)

    if !isnothing(basis)
        X = basis
    end
    
    #calling different material parameters for local operator
    Op_s = curlcurl(1.0, 1.0)
    
    S = assemble(Op_s, X, X)
    M = assemble(BEAST.Identity(), X, X) 

    #eigenvalues and eigenvectors
    k2, V = eigen(Matrix(S), Matrix(M));

    #to find the dimension of null space   
    t = real.(deepcopy(k2))
        
    indices = []
    for i in length(t):-1:1
        if isnan(t[i])
            deleteat!(t, i)
            append!(indices, i) 
        elseif real(t[i]) <= 0
            deleteat!(t, i)
            append!(indices, i)
        elseif isapprox(real(t[i]), 0, atol = 0.1)
            deleteat!(t, i)
            append!(indices, i)    
        elseif real(t[i]) == Inf
            deleteat!(t, i)
            append!(indices, i)
        end
    end 
    
    valid_k2 = [k2[i] for i in length(indices) + 1 : length(k2)]
    valid_V2 = [V_2d[:, i] for i in length(indices) + 1 : size(V_2d, 2)]

    #numerical wavenumber, wavelength, frequency
    k = sqrt.(t) 
    λ = real(2π ./ k)
    f = sort(c0 ./ λ)
    return k, λ, f, valid_V2, valid_k2, X, k2
end


"""
    analytical_freq3d(a, b, c, m, n, p)

gives the specific analytical frequency for the input arguments:
length, breadth, depth, mode number along length, mode number along breadth,
mode number along depth.
"""
function analytical_freq3d(a, b, c, m, n, p) c0*sqrt((m/a)^2+ (n/b)^2 + (p/c)^2)/2 end


"""
    ordered_freq3d(a, b, c, num; field = :electric)

returns an ordered list of the first num numbers of field mode frequency
"""
function ordered_freq3d(a, b, c, num; field = :electric)
    v = [(a, "mz"), (b, "ny"), (c, "pz")]
    sort!(v, by = x -> x[1])
#    lengths = [Int(num/v[3][1]), Int(num/v[2][1]), Int(num/v[1][1])]
    #=
    if v[1][2] == "c"
        pz = num
    elseif v[1][2] == "a"
        mx = num
    else 
        ny = num
    end=#
    ordered_modes = []
    for p in 1:num
        for m in 0:num
            for n in 0:num
                (m == 0 && n == 0) && continue
                    append!(ordered_modes, analytical_freq3d(a, b, c, m, n, p))
            end
        end
    end
    sort!(ordered_modes)
    return ordered_modes[1:num]

end
