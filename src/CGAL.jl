


  """
    CGAL This function implements CGAL - SketchyCGAL - ThinCGAL for solving
    semidefinite programs of the following form:

          minimize  <C,X>  subj.to   X is symmetric positive semidefinite matrix of size n
                                     al <= tr(X) <= au
                                     <A_i,X> = b_i, i=1,...,d

     Coded by: Ngoc Hoang Anh Mai - nhmai@laas.fr
     This work is based on https://github.com/alpyurtsever/SketchyCGAL/
     
    """

#=include("../MKL/src/MKL.jl")
using .MKL=#

function CGAL(n::UInt32,Primitive1,Primitive2,Primitive3,a::Vector{Float32},b,R::UInt16,T::UInt64;STOPTOL=1e-4)
    
    
    d=length(b)



    SAVEHIST = unique([[2^i for i in 0:floor(UInt64,log2(T))];T])
    mySketch=NystromSketch(n, R)
    U,Delt=Reconstruct(mySketch)
    t=UInt64(1)
    z = zeros(Float64,d)
    objective = Float64(0)
    TRACE = Float64(0)
    ObjCond = Float64(Inf)
    FeasCond = Float64(Inf)
    
    
    
    
    
    
    y = zeros(Float64,d)
    beta=Float64(0)
    eta=Float64(0)
    vt=Vector{Float64}(undef,d)
    u=Vector{Float64}(undef,n)
    sig=UInt64(0)
    a_t=Float64(0)
    dualUpdate=Vector{Float64}(undef,d)
   
    println("- SketchyCGAL SDP Solver - Beta.V.0.0")
    
    for t in 1:T
        
        @inbounds beta = sqrt(t+1)
        @inbounds eta = 2/(t+1)
        @inbounds vt = y + (z-b).*beta
     
        u,sig= ApproxMinEvecLanczos(u::Vector{Float64}->Primitive1(u) + Primitive2(vt,u),n, ceil(UInt64,(t^0.25)*log(n)))

        if sig > 0
            a_t = minimum(a)
        else
            a_t = maximum(a)
        end
        @inbounds u *= sqrt(a_t)

        @inbounds TRACE = (1-eta)*TRACE + eta*a_t
        
        # Check stopping criteria
        
        FeasCond = norm(z - b)
        @inbounds ObjCond = objective + ((y + (z-b).*beta)'*b)[1] + 0.5*beta*FeasCond^2 - ((Primitive1(u) + Primitive2(vt,u))'*u)[1]
        if FeasCond <= STOPTOL && ObjCond <= STOPTOL
            println("* status = stopping criteria met")
            @inbounds Delt += (TRACE-Delt)/R
            return objective,U, Delt
        end

        @inbounds z = (1-eta)*z + eta*Primitive3(u)

        @inbounds objective = (1-eta)*objective + eta*(u'*Primitive1(u))[1]

        mySketch=RankOneUpdate(mySketch,u,eta)
        @inbounds dualUpdate = z - b

        # Update the DUAL
        @inbounds y += dualUpdate.*minimum([1, 4 * sqrt(t+2) * eta^2 * maximum(a)^2 / norm(dualUpdate)^2])
        
      
        if t in SAVEHIST
            U,Delt=Reconstruct(mySketch)
            @inbounds Delt += (TRACE-Delt)/R
            @inbounds U = U.*sqrt(Delt)
            println("--------------------------")
            println(" iter=$(t) ")
            println(" stopObj=$(ObjCond) ")
            println(" stopFeas=$(FeasCond) ")
            println(" primalObj=$(objective) ")
        
        end
        if t == T 
            println("* status = maximum number of iterations achieved")
            @inbounds Delt += (TRACE-Delt)/R
            return objective,U, Delt
        end  
        
    end
end





## Lanczos method
function ApproxMinEvecLanczos(M, n::UInt32, q::UInt64)
    
   
    # Approximate minimum eigenvector
    # Vanilla Lanczos method

    q = minimum([q, n-1])                  # Iterations < dimension!

   

    Q = Matrix{Float64}(undef,n, q+1)                  # Lanczos vectors

    aleph = Vector{Float64}(undef,q)                 # Diagonal Lanczos coefs
    beth = Vector{Float64}(undef,q)                  # Off-diagonal Lanczos coefs

    Q[:,1] = randn(Float64,n)               # First Lanczos vector is random
    Q[:,1] = Q[:,1]./norm(Q[:,1])

    ii=q
    
    tol=sqrt(n)*2.2204e-16 
    
    for i=1:q
        
        Q[:, i+1] = M(Q[:, i])				# Apply M to previous Lanczos vector
        @inbounds aleph[i] = (Q[:, i]' * Q[:, i+1])[1]		# Compute diagonal coefficients

        if i == 1                    # Lanczos iteration
            @inbounds Q[:, i+1] -=  Q[:, i].*aleph[i]
        else
            @inbounds Q[:, i+1] -=  Q[:, i].*aleph[i] + Q[:, i-1].*beth[i-1] 
        end

        @inbounds beth[i] = norm(Q[:, i+1])            # Compute off-diagonal coefficients
        
        if  beth[i] < tol
            ii=i
            break
        end
        @inbounds Q[:, i+1] = Q[:, i+1] ./ beth[i]
        
    end
    
    
  
        
    E=smallest_eig(Symmetric(diagm(0 =>aleph[1:ii], 1 =>beth[1:(ii-1)], -1 =>beth[1:(ii-1)])))

    v = Q[:, 1:ii] * E.vectors[:,1]
    nv = norm(v)
    xi = E.values[1]*nv
       
    v = v./nv
        
    
    return v, xi
end


function smallest_eig(A::Symmetric{Float64,Array{Float64,2}})
    #BLAS.vendor()
    return eigen(A,1:1)
end
