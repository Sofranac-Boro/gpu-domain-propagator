module Definitions

export gdp_inf,gdp_eps,max_num_rounds,
eps_lt,eps_le,eps_gt,eps_ge,eps_floor,
eps_eq,eps_ceil,gdp_vartype

# Sequential implementation of MIP
using MultiFloats

const gdp_inf = 1e20
const gdp_eps = Float64x8(1e-130) #e-15
const max_num_rounds = 1e3

# real_abs(x) = abs(x)
eps_le(x,y) = x - y <= gdp_eps
eps_lt(x,y) = x - y < -gdp_eps
eps_ge(x,y) = x - y >= -gdp_eps
eps_gt(x,y) = x - y >  gdp_eps
eps_eq(x,y) = abs(x - y) <= gdp_eps
eps_ceil(x) = ceil(Float64(x-gdp_eps))
eps_floor(x) = floor(Float64(x+gdp_eps))

@enum gdp_vartype begin
    gdp_binary=0
    gdp_integer=1
    gdp_continuous=3
end

# @enum gdp_retcode begin
#     gdp_okay=1
#     gdp_error=0
#     gdp_not_implemented=-18
# end
end