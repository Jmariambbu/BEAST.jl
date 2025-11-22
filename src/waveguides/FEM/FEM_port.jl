using BEAST
using CompScienceMeshes
using LinearAlgebra
using IterativeSolvers

include("FEM_curlcurl.jl")
include("FEM_eigen2D.jl")
include("FEM_eigen3D.jl")

##
#constants
μ0 = 4π*1e-7
ep = 8.854e-12
c0 = 1/sqrt(ep*μ0)


function beta(ana_freq, k_10)
        ωₐ = 2*pi*ana_freq
        k2 = ωₐ^2*μ0*ep
        @show k2
        β = sqrt(complex(k2 - k_10^2))
    return β, k2
end


function single_mode(mesh, a, b, c, p1, p2, ind; field = :electric)

    #eigenvector for mode at input port
    edges = submesh(!in(boundary(p1)), skeleton(p1,1))
    X_inc = BEAST.nedelec(p1, edges)
    kₜ, l, f, V = TE2D(p1, basis = X_inc) 
    
    #2d input field
    E_inc = V[ind] 

    #analytical frequency and propagation constant
    k_10 = kₜ[ind]
    ana_freq3d = last(ordered_freq3d(a, b, c, ind, field = field))
    β, k2 = beta(ana_freq3d, k_10)
    #electric field excludes only mesh with PEC boundaries
    #exclude all boundaries
    int_edges = submesh(!in(skeleton(boundary(mesh), 1)), skeleton(mesh, 1))
    #adding the port meshes back in
    edges = submesh(!in(boundary(p1)), skeleton(p1,1))
    int_edges_w_ports = union(int_edges, edges)
    edges = submesh(!in(boundary(p2)), skeleton(p2,1))
    int_edges_w_ports = union(int_edges_w_ports, edges)

    @assert cells(int_edges) != skeleton(mesh, 1).faces

    #basis function for the waveguide
    Y = BEAST.nedelecc3d(mesh, int_edges_w_ports)

    Op_s = curlcurl(1.0, 1.0)

    #n × basis function projected on input port
    Y₁ = ttrace(Y, p1)

    #n × basis function projected on output port
    Y₂ = ttrace(Y, p2)

    #Eₜ = eₜ * exp(-jβz)
    ez::Vector{ComplexF64} = []
    for i in eachindex(Y.pos)
        append!(ez, cos(β*Y.pos[i][3]) + im*sin(β*Y.pos[i][3]))
    end

    ℐ = BEAST.Identity()
    # ∫ [(∇ × E)⋅(∇ × Wᵢ)] dx
    S = assemble(Op_s, Y, Y)
    # ∫_Γₚ₁ [j β (n × Wᵢ)⋅(n × E)] dS
    A = assemble(ℐ, Y₁, Y₁).*im.*β
    # ∫_Γₚ₂ [j β (n × Wᵢ)⋅(n × E)] dS
    As = assemble(ℐ, Y₂, Y₂).*im.*β
    # ∫ (E ⋅ Wᵢ) dx
    M = assemble(ℐ, Y, Y)*k2

    new_S  = S + A + As - M 

    # ∫_Γₚ₁ [2j β (Wᵢ⋅E_inc)] dS  
    b = assemble(BEAST.NCross(), Y₁, X_inc)*E_inc.*im.*2β

    x = IterativeSolvers.idrs(new_S, b; log = false)

    return x, Y, Y₁, Y₂, X_inc, E_inc, β, ez
end