using BEAST
using CompScienceMeshes
using Plotly
using PlotlyJS

include("FEM_curlcurl.jl")

#constants
μ0 = 4π*1e-7
ε0 = 8.854e-12
c0 = 1/sqrt(ε0*μ0)
a = 2.0
b = 1.0
c = 3.0
l1 = -1.0
l2 = -1.0


#mesh geometry
wireframe = CompScienceMeshes.wireframe
msh = meshrectangle(a, b, 0.5)
CompScienceMeshes.translate!(msh, [l1, l2, 0.0])

#2d te modes
function TE(msh; X2 = nothing)
    bodymesh = msh   
    #Plotly.plot(wireframe(bodymesh)) 
    bnd = boundary(bodymesh)

    edges = submesh(!in(bnd), skeleton(bodymesh,1))

    free_edges = BEAST.rowvals(connectivity(edges, skeleton(bodymesh, 2)))
    bnd_edges = BEAST.rowvals(connectivity(bnd, skeleton(bodymesh, 2)))
    
    X = BEAST.nedelec(bodymesh, edges)
    
    if !isnothing(X2)
        X = X2
    end

   #calling different material parameters for local operator
    Op_s = curlcurl(1.0, 1.0)
    
    S = assemble(Op_s, X, X)
    M = assemble(BEAST.Identity(), X, X) 

    #eigenvalues and eigenvectors
    k2, V_2d = eigen(Matrix(S), Matrix(M));

        #to remove null space solutions
    t = deepcopy(k2)
        
    #k == real.(k)
    #@show t
    indices = []
    for i in length(t):-1:1
            #@show i
            #@show length(t)
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
    valid_indices = collect(length(indices) + 1: length(k2))

    valid_V2 = [V_2d[:, i] for i in length(indices) + 1 : size(V_2d, 2)]

            #numerical wavenumber, wavelength, frequency
    k = sqrt.(t)


    λ = 2π ./ k
    f = c0 ./ λ
    return k, λ, f, valid_V2, X; valid_indices, valid_k2
end

function eps_r(x) #eps_r is infinity in a PEC
    if (x[1] > -a - l1 && x[1] < a + l1)
        if (x[2] > -b - l2 && x[2] < b + l2)
            #if (x[3] >= -1.0 && x[3] <= 1.0)
                return 1.0
            #end
        end
    end
    return Inf64
end
#2d tm modes
function TM()
    bodymesh = msh   
    #Plotly.plot(wireframe(bodymesh)) 
    
    X = BEAST.nedelec(bodymesh)
    #calling different material parameters for local operator
    Op_s = curlcurl(1.0, 1.0)
    
    S = assemble(Op_s, X, X)
    M = assemble(BEAST.Identity(), X, X) 

    #eigenvalues and eigenvectors
    k2, V_2d = eigen(Matrix(S), Matrix(M));

        #to remove null space solutions
    t = deepcopy(k2)
        
    #k == real.(k)
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

            #numerical wavenumber, wavelength, frequency
    k = sqrt.(t)

    λ = 2π ./ k
    f = c0 ./ λ
    return k, λ, f
end

#assemble stiffness and mass matrices
k, l, f = TM()


##

#testing with the analytical solution
f1_TM = sqrt(a^2 + b^2)*c0/(2*a*b)
f1_TE = c0/(2*a)

function analytical_freq(a, b, m, n)
    return sqrt((m/a)^2 + (n/b)^2)*c0/2
end


#convergence study

using Test
using DataFrames

h = [1.0/2, 1/2^2, 1/2^3, 1/2^4]

function num_soln()
    n = length(h)
    K = Array{Float64}(undef, (n, 7))
    Λ = Array{Float64}(undef, (n, 7))
    F = Array{Float64}(undef, (n, 7))
    v2 = []
    for i in eachindex(h)
        msh = meshrectangle(a, b, h[i])
        translate!(msh, [l1, l2, 0.0])
        k, l, f, v = TE(msh)
        #@show k
        K[i, :], Λ[i, :], F[i, :] = k[1:7], l[1:7], f[1:7]
        append!(v2, [v])
    end
    return K, Λ, F, v2
end

K, Λ, F, v2 = num_soln();

#_ , _ , _ , te10_2dvector = TE(msh)
te10_2dvector = v2[1]

num_TE10(i) = DataFrame(wavenumber = K[:, i], wavelength = Λ[:, i], frequency = F[:, i])

n_f(i) = num_TE10(i).frequency
#append!(n_f[i], num_TE10(i).frequency)
#n_l(i) = num_TE10(i).wavelength
#n_k(i) = num_TE10(i).wavenumber


#convergence
ana_frq = Matrix(undef, 9, 3)
for i in [0, 1, 2]
    for j in [0, 1, 2]
        ana_frq[(j + 1) + 3*i, :] .= analytical_freq(a, b, i, j), i, j
    end
end
ana_frq
a_f = DataFrame(frequency=ana_frq[:][1], m = ana_frq[:][2], n = ana_frq[:][3])
p(i, j) = log((n_f(j)[i] - n_f(j)[i + 1])/(n_f(j)[i + 1] - n_f(j)[i + 2]))/log(h[i]/h[i + 1])
function convergence(j)
    P = Vector{Float64}(undef, length(h) - 2)
    for i in eachindex(h)
        P[i] = p(i, j)
        if i == length(h) - 2
            break
        end
    end
return P
end
F[end, 1:7]
F
#n_f([1:7])
using Plots
j = 3

m = 1

n_ = 1
Plots.plot(reverse(h), n_f(j)/(10^7), label = "numerical")

scatter!([0.5], [analytical_freq(a, b, m, n_)/(10^7)], label = "analytical", title = "Frequency comparison m=$m n=$n")
#scatter!([f1_TE/(10^7)], label = "analytical")

Plots.plot(reverse(h[1:length(convergence(j))]), convergence(j), title = "convergence")
scatter!([2], label = "expected")

F[:, 4]

#plot fields





##

#extension to 3D
h1 = 0.5
l3 = 0.0
msh = tetmeshcuboid(a, b, c, h1)
translate!(msh, [l1, l2, l3])
bodymesh = msh   

function is_CSM_tet(s)
   CSM_tet = false

   c = cross(s.tangents[1], s.tangents[2])
   dot(c,s.tangents[3]) > 0.0 && (CSM_tet = true)

   return CSM_tet
end

for (j,face) in enumerate(bodymesh.faces)
    tet = simplex(bodymesh.vertices[face]...)
    if is_CSM_tet(tet) == false
        bodymesh.faces[j] = face[[2,1,3,4]]
    end
end

for face in bodymesh.faces
    tet = simplex(bodymesh.vertices[face]...)
    @assert is_CSM_tet(tet)
    #println(is_CSM_tet(tet))
end


#function TE3D()
    
    Plotly.plot(wireframe(bodymesh)) 
    bnd = skeleton(boundary(bodymesh), 1)
    Plotly.plot(wireframe(bnd))
    #edge or face boundary excluded???
    edges = submesh(!in(bnd), skeleton(bodymesh, 1))
    #faces = submesh(!in(bnd), skeleton(bodymesh,2))
    #TE
    Plotly.plot(wireframe(edges))
    X = BEAST.nedelecc3d(bodymesh, edges)
    #TM
    #X = BEAST.nedelecc3d(bodymesh)

    #calling different material parameters for local operator
    Op_s = curlcurl(1.0, 1.0)
    
    S = assemble(Op_s, X, X)
    M = assemble(BEAST.Identity(), X, X) 

    #eigenvalues and eigenvectors
    k2, V = eigen(Matrix(S), Matrix(M));

    #to find the dimension of null space
    #Plotly.plot(wireframe(bnd))
    n_verts = numvertices(bodymesh);

    bnd_faces = []; 
    for i in eachindex(bnd.faces)
        append!(bnd_faces, bnd.faces[i])
    end

    
    unique!(bnd_faces);

    n_freenodes = n_verts - length(bnd_faces)

    #numerical wavenumber, wavelength, frequency
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
    t
    k2
    k = sqrt.(t) 

    λ = real(2π ./ k)

    f = sort(c0 ./ λ)

    for i in length(f):-1:1
        if f[i] < 9*10^1 
            deleteat!(f, i)
        end
    end
    f


function analytical_freq3d(a, b, c, m, n, p)
    return c0*sqrt((m/a)^2+ (n/b)^2 + (p/c)^2)/2
end


#visualization of electric and magnetic fields
analytical_freq3d(a, b, c, 1, 0, 1)
indices


valid_indices = setdiff(collect(1:length(k2)), indices)
valid_k2 = [k2[i] for i in valid_indices]
valid_k2[9]

t_vert = [0.0, -0.5, 0.0]
t_vertices = []
for i in eachindex(bodymesh.vertices)
    if bodymesh.vertices[i][1] >= t_vert[1] 
        if bodymesh.vertices[i][2] >= t_vert[2]
            if bodymesh.vertices[i][3] >= t_vert[3]
                append!(t_vertices, i)
            end
        end
    end
end

t_vertices

t_faces = []

for i in eachindex(bodymesh.faces)
    t = collect(bodymesh.faces[i])
    if sum(in.(t, Ref(t_vertices)))==4
        append!(t_faces, i)
    end
end

t_faces


new_mesh = Mesh(bodymesh.vertices, bodymesh.faces[t_faces])
##
fcr, geo = facecurrents(V[:, valid_indices[1]], X) # for fine mesh instead of strcX

#fcr

f = Array{Float64}(undef, length(fcr));

for i in eachindex(fcr)
    val = sum(fcr[i][1:3])/3
    #@show val
    f[i] = val
end


PlotlyJS.plot(patch(geo, f, opacity = 0.5, surface_count = 100))

