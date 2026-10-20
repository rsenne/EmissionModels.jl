#= Backs the DDM emission hooks (density, sampler, CDF) with
   SequentialSamplingModels' Wiener implementation. Everything else lives in
   src/ssm/ddm.jl. =#
module EmissionModelsSequentialSamplingModelsExt

using EmissionModels: EmissionModels
using SequentialSamplingModels: SequentialSamplingModels, DDM

function EmissionModels._ddm_logpdf(
    ν::Real, α::Real, z::Real, τ::Real, choice::Real, rt::Real
)
    T = float(promote_type(typeof(ν), typeof(α), typeof(z), typeof(τ), typeof(rt)))
    # Cases SequentialSamplingModels.pdf rejects; here they have zero density.
    (choice == 1 || choice == 2) || return T(-Inf)
    rt > τ || return T(-Inf)
    #= Promote parameters and rt together before handing them to SSM: its
       series internals require one homogeneous element type, and mixed
       precisions (e.g. Float32 emission, Float64 rt) would MethodError.
       `float` only maps Integer inputs to a type where ±Inf is representable;
       every other Real, ForwardDiff duals (this is the path fit!
       differentiates), Float32, BigFloat, etc.pass through unchanged.
       The series can also overflow at extreme parameters; treat both as
       zero density. =#
    p = try
        SequentialSamplingModels.pdf(DDM(T(ν), T(α), T(z), T(τ)), Int(choice), T(rt))
    catch e
        #= Only numeric edge cases (the series can misbehave at extreme
           parameters) count as zero density. Re-raise programming errors such
           as MethodError so a real precision/dispatch bug is not masked. =#
        (e isa InterruptException || e isa MethodError) && rethrow()
        return T(-Inf)
    end
    return (isfinite(p) && p > 0) ? T(log(p)) : T(-Inf)
end

function EmissionModels._ddm_rand(rng, ν::Real, α::Real, z::Real, τ::Real)
    return rand(rng, DDM(ν, α, z, τ))
end

# A closure here is boxed inside `try` on Julia 1.10, so `_ddm_cdf` would infer `Any`.
@inline function _ssm_cdf(::Type{T}, ν, α, z, τ, choice, rt) where {T}
    return SequentialSamplingModels.cdf(DDM(T(ν), T(α), T(z), T(τ)), Int(choice), T(rt))
end

function EmissionModels._ddm_cdf(ν::Real, α::Real, z::Real, τ::Real, choice::Real, rt::Real)
    T = float(promote_type(typeof(ν), typeof(α), typeof(z), typeof(τ), typeof(rt)))
    # Cases SequentialSamplingModels.cdf rejects; here they carry zero mass.
    (choice == 1 || choice == 2) || return zero(T)
    rt > τ || return zero(T)
    # Same homogeneous-type promotion and overflow guard as `_ddm_logpdf`.
    F = try
        #= SSM's zero-drift branch (|ν| < sqrt(eps(T))) returns the cdf with the
           wrong sign; interpolate between ±thr instead, which is O(thr²) exact. =#
        thr = max(T(1e-6), 2 * sqrt(eps(T)))
        if abs(ν) < thr
            Fm = _ssm_cdf(T, -thr, α, z, τ, choice, rt)
            Fp = _ssm_cdf(T, thr, α, z, τ, choice, rt)
            Fm + (T(ν) + thr) / (2 * thr) * (Fp - Fm)
        else
            _ssm_cdf(T, ν, α, z, τ, choice, rt)
        end
    catch e
        #= Same narrowing as `_ddm_logpdf`: numeric edge cases carry zero mass,
        but MethodError/InterruptException propagate. =#
        (e isa InterruptException || e isa MethodError) && rethrow()
        return zero(T)
    end
    # Series truncation can stray slightly outside [0, 1].
    return isfinite(F) ? clamp(T(F), zero(T), one(T)) : zero(T)
end

end
