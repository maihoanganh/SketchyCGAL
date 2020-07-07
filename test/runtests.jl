"""
Shor's relaxation for MAXCUT

"""

using MAT
using LinearAlgebra
using SparseArrays



#call weight matrix from GSET
file = matopen("../GSET/g1.mat")
Problem=read(file, "A")
close(file)


n = UInt32(size(Problem,1))
C = sparse(diagm(Problem*ones(n))) - Problem
C = (C+C').*0.5
C = -C.*0.25


println("size of weight matrix: ",n)



Primitive1(x::Vector{Float64})= C*x
Primitive2(y::Vector{Float64},x::Vector{Float64})= [@inbounds @fastmath y[i]*x[i] for i in 1:n]
Primitive3(x::Vector{Float64})= x.^2

a = Vector{Float32}([n;n])
b = ones(Float64,n)

R = UInt16(10) # rank/sketch size parameter
maxit = UInt64(1e4) # limit on number of iterations


include("./src/SketchyCGAL.jl")
using .SketchyCGAL

@time obj, U, Delt = SketchyCGAL.CGAL(n,Primitive1,Primitive2,Primitive3, a, b, R, maxit;STOPTOL=1e-3);
