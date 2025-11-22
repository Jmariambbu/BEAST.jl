using BEAST
using CompScienceMeshes
using Plotly
using LinearAlgebra
using Krylov

include("FEM_curlcurl.jl")
include("FEM_eigen2.jl")


#constants
μ0 = 4π*1e-7
ϵ0 = 8.854e-12
c0 = 1/sqrt(ε0*μ0)

#mesh size
h = 0.1

#waveguide mesh
a = 1.0
b = 1.0
c = 2.0
bodymesh = tetmeshcuboid(a, b, c, h)
translate!(bodymesh, [-a/2, -b/2, 0.0])
#Plotly.plot(wireframe(bodymesh))

#orienting the faces of the tetrahedrons that the normals face out 
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
end

#port1 & port2 mesh (faces)
p1_nodes = []
for i in eachindex(bodymesh.vertices)
    if bodymesh.vertices[i][3] == 0.0 #coord z = 0 at port 1
        append!(p1_nodes, i)
    end
end
p2_nodes = []
for i in eachindex(bodymesh.vertices)
    if bodymesh.vertices[i][3] == c #coord z = c at port 2
        append!(p2_nodes, i)
    end
end

#tetrahedrons containing coords z = 0 || z = c
p1_tri = []
for i in eachindex(bodymesh.faces)
    if sum(in.(bodymesh.faces[i], Ref(p1_nodes))) == 3
        append!(p1_tri, i)
    end
end
p2_tri = []
for i in eachindex(bodymesh.faces)
    if sum(in.(bodymesh.faces[i], Ref(p2_nodes))) == 3
        append!(p2_tri, i)
    end
end
p1_nodes = bodymesh.faces[p1_tri]
p2_nodes = bodymesh.faces[p2_tri]
p1_msh = Mesh(bodymesh.vertices, p1_nodes)
p2_msh = Mesh(bodymesh.vertices, p2_nodes)

#port faces
p1 = submesh(skeleton(p1_msh, 2), boundary(bodymesh))
p2 = submesh(skeleton(p2_msh, 2), boundary(bodymesh))

#excluding the edges along the rim of port 1
edges = submesh(!in(boundary(p1)), skeleton(p1,1))
Plotly.plot(wireframe(edges))

#eigenvector for mode at port 1
E_inc = BEAST.nedelec(p1, edges) #2d nedelec excluding the rim of the port
kₜ, l, f, V, E_inc = TE(p1; X2 = E_inc) #returns te modes

#selecting te10 mode
ind = 1
te10 = V[ind] 

#compare analytical frequency for te10 with numerical
#fₐ = analytical_freq(a, 1.0, 1, 0)
#f[ind]

k_10 = kₜ[ind]

ω = f[ind]*2pi
k = ω^2*μ0*ϵ0
β = sqrt(complex(k - k_10^2))

#to compare with the numerically computed field at port 2
test::ComplexF64 = cos(β*c) + im*sin(β*c)  #e^(-jβz)|_(z=c)

#meshing only the interior edges of the waveguide and the ports
int_edges = submesh(!in(skeleton(boundary(bodymesh), 1)), skeleton(bodymesh, 1))
edges = submesh(!in(boundary(p1)), skeleton(p1,1))
int_edges_w_ports = union(int_edges, edges)
edges = submesh(!in(boundary(p2)), skeleton(p2,1))
int_edges_w_ports = union(int_edges_w_ports, edges)

Plotly.plot(wireframe(int_edges_w_ports))

@assert cells(int_edges) != skeleton(bodymesh, 1).faces

numcells(int_edges_w_ports)

#3d nedelec only on interior edges (excluding walls and the rim of the ports)
Y = BEAST.nedelecc3d(bodymesh, int_edges_w_ports)

#curl-curl operator
Op_s = curlcurl(1.0, 1.0)
#nedelec along port 1 face (including the rim of the port)
Y₁ = ttrace(Y, p1)
#likewise
Y₂ = ttrace(Y, p2)

ℐ = BEAST.Identity()

#∫_Ω (∇ × Wᵢ)⋅(∇ × E) dΩ
S = assemble(Op_s, Y, Y)
#∫_Γₚ₁ jβ (n̂ × Wᵢ)⋅(n̂ × E) dΓₚ₁
A = assemble(ℐ, Y₁, Y₁).*im.*β
#∫_Γₚ₂ jβ (n̂ × Wᵢ)⋅(n̂ × E) dΓₚ₂
As = assemble(ℐ, Y₂, Y₂).*im.*β
#∫_Ω Wᵢ⋅E dΩ
M = assemble(ℐ, Y, Y).*k 

S  = S  + A  + As  - M 

#∫_Γₚ₁ 2jβ Wᵢ⋅Eₜₑ₁₀ dΓₚ₁
b = assemble(ℐ, ttrace(Y, p1), E_inc)*te10.*im.*2β

x = S \b

#output mode - numerical
fcr₂, geo₂ = facecurrents(x, ttrace(Y, p2))
f₂ = Array{Complex}(undef, length(fcr₂));
for i in eachindex(fcr₂)
    val = sum(fcr₂[i][1:3])
    f₂[i] = val
end
plot1 = PlotlyJS.plot(patch(geo₂, real.(f₂), opacity = 0.9, surface_count = 1000))

#input mode
fcr₁, geo₁ = facecurrents(te10, E_inc)
f₁ = Array{Float64}(undef, length(fcr₁));
for i in eachindex(fcr₁)
    val = sum(fcr₁[i][1:3])
    f₁[i] = val
end
plot2 = PlotlyJS.plot(patch(geo₁, f₁, opacity = 0.9, surface_count = 100))

#output mode - analytical
f₃ = Array{Complex}(undef, length(fcr₂));
for i in eachindex(fcr₁)
    val = sum(fcr₁[i][1:3])*test
    f₃[i] = val
end
plot3 = PlotlyJS.plot(patch(geo₂, real.(f₃), opacity = 0.9, surface_count = 100))

