struct ColumnModelData{T}
    N  :: Int
    L  :: T
    Fb :: T
    Fu :: T
    Bz :: T
    K₀ :: T 
    α  :: T 
    g  :: T 
    f  :: T 
    t  :: Array{T, 1}
    U  :: Array{Array{T, 1}, 1}
    V  :: Array{Array{T, 1}, 1}
    T  :: Array{Array{T, 1}, 1}
    S  :: Array{Array{T, 1}, 1}
    E  :: Array{Array{T, 1}, 1} # mean kinetic energy
end

"""
    ColumnModelData(datapath)

Construct ColumnModelData from a standardized dataset.
"""
function ColumnModelData(datapath)
    iters = iterations(datapath)

    t = times(datapath)

    U = [ getdata("U", datapath, i) for i in 1:length(iters) ]
    V = [ getdata("V", datapath, i) for i in 1:length(iters) ]
    T = [ getdata("T", datapath, i) for i in 1:length(iters) ]
    S = [ getdata("S", datapath, i) for i in 1:length(iters) ]
    E = [ 0.5*(U[i].^2 .+ V[i].^2)  for i in 1:length(iters) ]

    K₀ = getconstant("K₀", datapath)
     α = getconstant("α", datapath)
     g = getconstant("g", datapath)
     f = getconstant("f", datapath)
    K₀ = getconstant("K₀", datapath)
    K₀ = getconstant("K₀", datapath)
    Fb = getbc("Fb", datapath)
    Fu = getbc("Fu", datapath)
    Bz = getic("Bz", datapath)
    N, L = getgridparams(datapath)

    ColumnModelData(N, L, Fb, Fu, Bz, K₀, α, g, f, t, U, V, T, S, E)
end

struct ColumnModel{M, T}
    turbmodel :: M
    dt :: T
end
