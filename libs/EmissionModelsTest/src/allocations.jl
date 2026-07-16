#=
  Benchmark helpers for allocation regression tests. Each one calls the
  operation `n` times inside a type-stable function and accumulates a result,
  so `@allocated` on the call measures steady-state cost per call without
  global-scope noise. Warm up with `n = 1` before measuring.
=#

#=
  On Julia 1.10 each `@allocated` measurement of these benchmark loops reports
  a constant ~16 B of measurement overhead (independent of the rep count; gone
  on 1.11+). `ALLOC_SLOP` absorbs it.
=#
const ALLOC_SLOP = VERSION < v"1.11" ? 32 : 0

function bench_logd(d, y, x, n)
    s = 0.0
    for _ in 1:n
        s += logdensityof(d, y; control_seq=x)
    end
    return s
end

function bench_logd_unctrl(d, y, n)
    s = 0.0
    for _ in 1:n
        s += logdensityof(d, y)
    end
    return s
end

function bench_logd_ddm(d, y, c, n)
    s = 0.0
    for _ in 1:n
        s += logdensityof(d, y, c)
    end
    return s
end

function bench_rand_scalar(rng, d, x, n)
    s = 0.0
    for _ in 1:n
        s += rand(rng, d; control_seq=x)
    end
    return s
end

function bench_rand_int(rng, d, x, n)
    s = 0
    for _ in 1:n
        s += rand(rng, d; control_seq=x)
    end
    return s
end

function bench_rand_ddm(rng, d, c, n)
    s = 0.0
    for _ in 1:n
        s += rand(rng, d, c).rt
    end
    return s
end

function bench_rand_unctrl_scalar(rng, d, n)
    s = 0.0
    for _ in 1:n
        s += rand(rng, d)
    end
    return s
end

function bench_rand_unctrl_vec(rng, d, n)
    s = 0.0
    for _ in 1:n
        s += rand(rng, d)[1]
    end
    return s
end

function bench_rand!_v(rng, d, out, x, n)
    s = 0.0
    for _ in 1:n
        rand!(rng, d, out; control_seq=x)
        s += out[1]
    end
    return s
end

function bench_rand!_i(rng, d, out, x, n)
    s = 0
    for _ in 1:n
        rand!(rng, d, out; control_seq=x)
        s += out[1]
    end
    return s
end

function bench_rand!_unctrl(rng, d, out, n)
    s = 0.0
    for _ in 1:n
        rand!(rng, d, out)
        s += out[1]
    end
    return s
end
