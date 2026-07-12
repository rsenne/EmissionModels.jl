#= Supplies the two numerical primitives the DDM emissions need — the Wiener
   first-passage-time density (Navarro & Fuss) and the rejection sampler —
   from SequentialSamplingModels.jl. Everything else (types, interface,
   M-step) lives in src/ssm/ddm.jl. =#
module EmissionModelsSequentialSamplingModelsExt

using EmissionModels: EmissionModels
using SequentialSamplingModels: SequentialSamplingModels, DDM

function EmissionModels._ddm_logpdf(
    ν::Real, α::Real, z::Real, τ::Real, choice::Real, rt::Real
)
    T = float(promote_type(typeof(ν), typeof(α), typeof(z), typeof(τ), typeof(rt)))
    # zero-density observations, which SequentialSamplingModels.pdf rejects
    (choice == 1 || choice == 2) || return T(-Inf)
    rt > τ || return T(-Inf)
    #= The series evaluation can overflow at extreme parameter values probed
       during optimization; treat those points as zero density rather than
       aborting an EM run. =#
    p = try
        SequentialSamplingModels.pdf(DDM(ν, α, z, τ), Int(choice), rt)
    catch
        return T(-Inf)
    end
    return (isfinite(p) && p > 0) ? T(log(p)) : T(-Inf)
end

function EmissionModels._ddm_rand(rng, ν::Real, α::Real, z::Real, τ::Real)
    return rand(rng, DDM(ν, α, z, τ))
end

function EmissionModels._ddm_cdf(ν::Real, α::Real, z::Real, τ::Real, choice::Real, rt::Real)
    return SequentialSamplingModels.cdf(DDM(ν, α, z, τ), Int(choice), rt)
end

end
