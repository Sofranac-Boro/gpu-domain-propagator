module Definitions

export gdp_inf,gdp_eps,max_num_rounds,
eps_lt,eps_le,eps_gt,eps_ge,eps_floor,
eps_eq,eps_ceil,gdp_vartype,is_neg_inf,is_pos_inf

# Sequential implementation of MIP
using MultiFloats

const gdp_inf = 1e20
const gdp_eps = Float64x8(1e-6) #e-15
const max_num_rounds = 100

# real_abs(x) = abs(x)
eps_le(x,y) = x - y <= gdp_eps
eps_lt(x,y) = x - y < -gdp_eps
eps_ge(x,y) = x - y >= -gdp_eps
eps_gt(x,y) = x - y >  gdp_eps
eps_eq(x,y) = abs(x - y) <= gdp_eps
eps_ceil(x) = ceil(Float64(x-gdp_eps))
eps_floor(x) = floor(Float64(x+gdp_eps))
is_pos_inf(x) = eps_ge(x,gdp_inf)
is_neg_inf(x) = eps_le(x,-gdp_inf)

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