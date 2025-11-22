using BEAST
using LinearAlgebra
using StaticArrays
#using ImpedancePredictionFEM
using CompScienceMeshes



struct curlcurl{T,U} <: BEAST.LocalOperator
    α::T
    μ::U
end
function BEAST.integrand(localop::curlcurl, kerneldata, x, f, g)

    dfx = f.curl
    dgx = g.curl

    Tx = 1/kerneldata.μ

    α = localop.α

    return α * dot(dfx, Tx * dgx)
end
struct KernelValsFEMLocalOperator{U}
    μ::U
end

function BEAST.kernelvals(localop::curlcurl, p)

    μ = localop.μ

    return KernelValsFEMLocalOperator(μ)
end
BEAST.scalartype(localop::curlcurl) = typeof(localop.α)



#=
function realvertices(mesh::Mesh)
    active_nodes = realnodes(mesh)
    return mesh.vertices[active_nodes]
end

function realnodes(mesh::Mesh)
    active_nodes = Int64[]
    for n in skeleton(mesh, 0).faces
        push!(active_nodes, n[1])
    end
    return active_nodes
end






#=

# build continuous linear Lagrange elements on a 3D manifold 
function BEAST.lagrangec0d1(mesh, vertexlist::Vector, ::Type{Val{4}})

    T = coordtype(mesh)
    U = universedimension(mesh)

    cellids, ncells = vertextocellmap(mesh)

    Cells = cells(mesh)
    Verts = vertices(mesh)

    # create the local shapes
    fns = Vector{BEAST.Shape{T}}[]
    pos = Vector{vertextype(mesh)}()

    sizehint!(fns, length(vertexlist))
    sizehint!(pos, length(vertexlist))
    for v in vertexlist

        numshapes = ncells[v]
        numshapes == 0 && continue

        shapes = Vector{BEAST.Shape{T}}(undef,numshapes)
        for s in 1: numshapes
            c = cellids[v,s]
            # cell = mesh.faces[c]
            cell = Cells[c]

            localid = something(findfirst(isequal(v), cell),0)
            @assert localid != 0

            shapes[s] = BEAST.Shape(c, localid, T(1.0))
        end

        push!(fns, shapes)
        push!(pos, Verts[v])
    end

    NF = 4
    BEAST.LagrangeBasis{1,0,NF}(mesh, fns, pos)
end

=#


function full_coeff_vector(X, freenodes, u_F, topnodes, u_top, bottomnodes, u_bottom)

    u = zeros(length(X.fns))
    for (i, index_on_X) in enumerate(freenodes)
        u[index_on_X] = u_F[i] 
    end
    for (i, index_on_X) in enumerate(topnodes)
        u[index_on_X] = u_top[i] 
    end
    for (i, index_on_X) in enumerate(bottomnodes)
        u[index_on_X] = u_bottom[i] 
    end

    return u
end

=#