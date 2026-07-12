#=
  Generic benchmark fucntions. Each model is described by a spec NamedTuple:

    name       display name used as the suite key
    model      a parameterized instance used for logdensityof/rand and to
               simulate the fitting data
    fresh      zero-argument constructor returning unfit parameters, called
               in `setup` (with evals=1) so every fit! measurement starts
               from the same cold state instead of refitting a converged model
    controlled whether the model takes a control_seq (GLMs) or not
    buffer     preallocated output vector for multivariate rand!, or nothing
               for scalar-observation models benchmarked via rand

  `add_model_benchmarks!` registers logdensityof / rand(!) / fit! benchmarks
  for one spec. cfg carries the shared rng, control vector x, design matrix X,
  and weights w.
=#
using BenchmarkTools
using EmissionModels
using Random
using Random: rand!

#= logdensityof benchmarks sweep the full observation sequence and sum, as the
   HMM forward/backward pass does.=#
logd_seq(d, ys, x) = sum(y -> logdensityof(d, y; control_seq=x), ys)
logd_seq(d, ys) = sum(y -> logdensityof(d, y), ys)

function simulate(rng, model, X::AbstractMatrix)
    return [rand(rng, model; control_seq=view(X, i, :)) for i in axes(X, 1)]
end
simulate(rng, model, n::Int) = [rand(rng, model) for _ in 1:n]

function add_model_benchmarks!(suite, spec, cfg)
    (; name, model, fresh, controlled, buffer) = spec
    (; rng, x, X, w) = cfg

    if controlled
        ys = simulate(rng, model, X)
        suite["logdensityof"][name] = @benchmarkable logd_seq($model, $ys, $x)
        if buffer === nothing
            suite["rand"][name] = @benchmarkable rand($rng, $model; control_seq=$x)
        else
            suite["rand!"][name] = @benchmarkable rand!(
                $rng, $model, $buffer; control_seq=$x
            )
        end
        suite["fit!"][name] = @benchmarkable fit!(d, $ys, $w; control_seq=$X) setup = (
            d = $(fresh)()
        ) evals = 1
    else
        ys = simulate(rng, model, size(X, 1))
        suite["logdensityof"][name] = @benchmarkable logd_seq($model, $ys)
        if buffer === nothing
            suite["rand"][name] = @benchmarkable rand($rng, $model)
        else
            suite["rand!"][name] = @benchmarkable rand!($rng, $model, $buffer)
        end
        suite["fit!"][name] = @benchmarkable fit!(d, $ys, $w) setup = (d = $(fresh)()) evals =
            1
    end
    return suite
end
