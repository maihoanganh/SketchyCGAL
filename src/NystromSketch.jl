
    
"""
NYSTROMSKETCH implements a class definition for the sketching method
described [TYUC2017Nys].

[TYUC2017Nys] J.A. Tropp, A. Yurtsever, M. Udell and V. Cevher. Fixed-
Rank Approximation of a Positive-Semidefinite Matrix from Streaming
Data. In Proc. 31st Conference on Neural Information Processing Systems
(NIPS), Long Beach, CA, USA, December 2017.

Coded by: Ngoc Hoang Anh Mai - nhmai@laas.fr
     This work is based on https://github.com/alpyurtsever/SketchyCGAL/
"""


abstract type SketchingMethod end


mutable struct NystromSketch 
    Omega::Matrix{Float64}    # (n x k) dimensional test matrix for the range of A (std Gaussian + orthonormalization)
    S::Matrix{Float64}         # (n x k) dimensional range sketch
    
    ## Constructor
    function NystromSketch(n::UInt32,R::UInt16)
        obj=new()
        if R > n
            println("Sketch-size cannot be larger than the problem size.") 
        end
        
        obj.Omega = randn(Float64,n,R)
        
        obj.S = zeros(Float64,n,R)
        return obj
    end

    
end
   

   
## Reconstruct
function Reconstruct(obj::NystromSketch)
    eps=2.2204e-16*rand(1)[1]
    S = obj.S
    n = size(S,1)
    # nu = eps*norm(Y)
    sigma = sqrt(n)*eps*maximum([norm(S[:,j]) for j in 1:size(S,2)])
    S += sigma*obj.Omega
    B = obj.Omega' * S
    B = 0.5*(B+B')
    
    if !any(x->x!=0, B)
        U = zeros(Float64,n,1)
        Delta = Float64(0)
        return U,Delta
    else
        C = cholesky(B)
        F= svd(S/C)
        U=F.U
        Sigma=F.S
        Delta = maximum([Sigma.^2 .- sigma;0])
        return U,Delta
    end
    #=if nargout == 1
        U = U*Delta*U'
    end=#
    
end

## Rank One Update
function RankOneUpdate( obj::NystromSketch, v::Vector{Float64}, eta::Float64 )
    obj.S = (1-eta)*obj.S + eta*(v*(v'*obj.Omega))
    return obj
end

## Property set methods
function setS(obj::NystromSketch,value::Matrix{Float64})
    if isequal(size(value),size(obj.S)) || isempty(obj.S)
        obj.S = value
    else
        println("Size of input does not match with sketch size.")
    end
    return obj
end 
        

        
