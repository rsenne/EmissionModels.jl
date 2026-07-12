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
    # The series can overflow at extreme parameters; treat those as zero density.
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
