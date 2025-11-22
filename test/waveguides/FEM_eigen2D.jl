using Plotly
using Plots
include("../../src/waveguides/FEM/FEM_eigen2D.jl")

#variables
a = 2.0;
b = 1.0;
h = 0.1;
lx = -1.0;
ly = -1.0;

#mesh geometry
msh = meshrectangle(a, b, h); #conflicts with Plots
CompScienceMeshes.translate!(msh, [lx, ly, 0.0]);
Plotly.plot(wireframe(msh))

k, λ, f, valid_V2, valid_k2, X = TE2D(msh);

fcr, geo = facecurrents(valid_V2[1], X)
f = Array{Float64}(undef, length(fcr));
for i in eachindex(fcr)
    val = sum(fcr[i][1:3])
    #@show val
    f[i] = val
end

Plotly.plot(patch(geo, f, opacity = 2, surface_count = 100))

##

x = Array{Float64}(undef, length(fcr));
y = Array{Float64}(undef, length(fcr));
z = Array{Float64}(undef, length(fcr));
u = Array{Float64}(undef, length(fcr));
v = Array{Float64}(undef, length(fcr));
w = Array{Float64}(undef, length(fcr));
val = fcr;
for i in range(1, length(fcr))
    f = geo.faces[i]
    simp = simplex(geo.vertices[f])
    cen = CompScienceMeshes.center(simp)
    bar = cen.cart
    x[i] = bar[1]
    y[i] = bar[2]
    z[i] = bar[3]
    u[i] = val[i][1]
    v[i] = val[i][2]
    w[i] = val[i][3]
end

Plotly.plot(
    cone(
        x=x,
        y=y,
        z=z,
        u=u,
        v=v,
        w=w,
        sizemode="absolute",
        sizeref=2,
        anchor="tail"
    ),
    Layout(
        scene=attr(
        colorscale = colors.Blues_8,
        camera_eye=attr(x=-1.57, y=1.36, z=0.58))
    ))

##

Plots.quiver(x, y, quiver = (0.1.*u, 0.1.*v))

##





