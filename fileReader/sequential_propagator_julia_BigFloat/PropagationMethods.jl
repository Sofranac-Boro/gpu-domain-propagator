module PropagationMethods

include("Definitions.jl")

using .Definitions

using Printf
# using MultiFloats


export ActivitiesTuple,NewBoundTuple,mark_constraints,adjust_lower_bound,adjust_upper_bound,is_lb_better,
is_ub_better,is_lb_worse,is_ub_worse,tighten_var_upper_bound,tighten_var_lower_bound,
can_constraints_be_tightened,are_constraints_infeasible,compute_activities,tighten_variable,
sequential_compute_activities


mutable struct ActivitiesTuple
    min_act::BigFloat
    max_act::BigFloat
    max_act_delta::BigFloat
    min_act_inf::Int64
    max_act_inf::Int64
end

mutable struct NewBoundTuple
    lb_is_tightened::Bool
    ub_is_tightened::Bool
    newb::Array{BigFloat,1}
end

# translate mark constraints as well

function mark_constraints(var_idx,csc_col_ptrs,csc_row_ind,cons_marked)
    for i = csc_col_ptrs[var_idx]:csc_col_ptrs[var_idx+1]-1
        cons_marked[csc_row_ind[i]] = 1
    end
end

function adjust_upper_bound(is_var_count,ub)
    return is_var_count ? ub : eps_floor(ub)
end

function adjust_lower_bound(is_var_cont,lb)
    return is_var_cont ? lb : eps_ceil(lb)
end

function is_lb_better(lb,ub,newlb)
    @assert(eps_le(lb, ub))  #query
    return eps_gt(newlb, lb)
end

function is_ub_better(lb,ub,newub)
    @assert(eps_le(lb, ub))  #query
    return eps_lt(newub, ub)
end

function is_lb_worse(lb,ub,newlb)
    @assert(eps_le(lb, ub))  #query
    return eps_lt(newlb,lb)
end

function is_ub_worse(lb,ub,newub)
    @assert(eps_le(lb,ub))
    return eps_gt(newub,ub)
end

function tighten_var_upper_bound(coeff,slack,surplus,num_inf_contr,lb,ub,is_var_cont)

    newb_tuple = NewBoundTuple(false,false,[lb,ub])

    new_bound = BigFloat(0.0)
    if num_inf_contr==0
        new_bound = eps_gt(coeff,0) ? slack/coeff : surplus/coeff
        new_bound = new_bound + lb
    elseif (num_inf_contr==1) && (eps_le(lb,-gdp_inf))
        new_bound = eps_gt(coeff,0) ? slack/coeff : surplus/coeff
    else
        return newb_tuple
    end

    new_bound = adjust_upper_bound(is_var_cont, new_bound)

    if is_ub_better(lb,ub,new_bound)
        newb_tuple.ub_is_tightened = true
        newb_tuple.newb = [lb,new_bound]
        @assert(eps_le(lb,new_bound))
        return newb_tuple
    end

    return newb_tuple
end

function tighten_var_lower_bound(coeff,slack,surplus,num_inf_contr,lb,ub,is_var_cont)
    newb_tuple = NewBoundTuple(false,false,[lb,ub])
    newb=BigFloat(0.0)
    if num_inf_contr==0
        newb = eps_gt(coeff,0) ? surplus/coeff : slack/coeff
        newb=newb+ub
    elseif (num_inf_contr==1) && (ub, gdp_inf)
        newb=eps_gt(coeff,0) ? surplus/coeff : slack/coeff
    else
        return newb_tuple
    end

    newb = adjust_lower_bound(is_var_cont,newb)

    if is_lb_better(lb,ub,newb)
        newb_tuple.lb_is_tightened = true
        newb_tuple.newb = [newb,ub]
        @assert(eps_le(newb,ub))
        return newb_tuple
    end
    return newb_tuple
end

function can_constraints_be_tightened(min_act,max_act,num_min_act_inf,num_max_act_inf,
    lhs,rhs,max_act_delta)

    # print("entered here \n")

    if ((num_min_act_inf > 1) && (num_max_act_inf > 1))
        return false
    end
    if eps_lt(max_act_delta,gdp_inf)
        return !eps_le(max_act_delta,min(rhs-min_act,max_act-lhs))
    end
    return true
end

function are_constraints_infeasible(min_act,max_act,rhs,lhs)
    return (min_act>rhs) || (max_act<lhs)
end

function compute_activities(cons_idx,col_ind,row_ind,vals,ubs,lbs)
    acts_tuple=ActivitiesTuple(0.0,0.0,0.0,0,0)

    min_activity=BigFloat(0.0)
    max_activity=BigFloat(0.0)
    min_act_inf=0
    max_act_inf=0
    n_vars_in_cons = row_ind[cons_idx+1] - row_ind[cons_idx]

    # follow cons call? only for debugging

    # @printf("Printing constraint %d", cons_idx)

    for var = 1:n_vars_in_cons
        val_idx=row_ind[cons_idx] + var - 1
        var_idx=col_ind[val_idx]
        coeff=vals[val_idx]
        lb=lbs[var_idx]
        ub=ubs[var_idx]

        # @printf("%f    %f    [%f  %f]",coeff,var_idx,lb,ub)
        max_act_delta=abs(coeff)*(ub-lb)
        if (eps_gt(max_act_delta,acts_tuple.max_act_delta))
            acts_tuple.max_act_delta=max_act_delta
        end #query in C++ curly braces are only needed if there are more 1 lines following a condition

        is_min_act_inf = eps_gt(coeff,0) ? eps_le(lb,-gdp_inf) : eps_ge(ub,gdp_inf)
        is_max_act_inf = eps_gt(coeff,0) ? eps_ge(ub,gdp_inf) : eps_le(lb,-gdp_inf)

        min_act_inf = min_act_inf+is_min_act_inf
        max_act_inf = max_act_inf+is_max_act_inf

        if is_min_act_inf==0
            min_activity += eps_gt(coeff, 0) ? coeff*lb : coeff*ub
        end
        if is_max_act_inf==0
            max_activity += eps_gt(coeff, 0) ? coeff*ub : coeff*lb
        end
    end
    acts_tuple.min_act=min_activity
    acts_tuple.max_act=max_activity
    acts_tuple.min_act_inf=min_act_inf
    acts_tuple.max_act_inf=max_act_inf
    return acts_tuple
end

function tighten_variable(coeff,lhs,rhs,min_act,max_act,num_min_act_inf,num_max_act_inf,
    is_var_cont,lb,ub)
# printing variables
    # print("coeff ",)
    print("coeff ", coeff, ", ")
    print("lhs ", lhs, ", ")
    print("rhs ", rhs, ", ")
    print("min_act ", min_act, ", ")
    print("max_act ", max_act, ", ")
    print("num_max_act_inf ", num_max_act_inf, ", ")
    print("num_min_act_inf ", num_min_act_inf, ", ")
    print("is_var_cont ", is_var_cont, ", ")
    print("lb ", lb, ", ")
    print("ub ", ub, ", ")

    slack = rhs-min_act
    print("slack ", slack,", ")
    surplus = lhs-max_act

    fif_a = (eps_gt(coeff*(ub-lb),rhs-min_act))
    print("coeff*(ub-lb) ", coeff*(ub-lb),", ")
    fif_b = (eps_lt(rhs,gdp_inf))
    fif_c = (eps_gt(min_act,-gdp_inf))
    # print("minus infinity ", -gdp_inf,"\n")
    # print("fif")
    print("fif_a ",fif_a,"\n")
    # print("fif_b ",fif_b,"\n")
    # print("fif_c ",fif_c,"\n")


    first_if = ( fif_a && fif_b && fif_c )

    second_if = ((eps_gt(coeff*(ub-lb),max_act-lhs) &&
                 (eps_gt(lhs,-gdp_inf)) &&
                 (eps_lt(max_act,gdp_inf))))

    third_if = ((eps_gt(coeff*(lb-ub),rhs-min_act)) &&
                (eps_lt(rhs,gdp_inf)) &&
                (eps_lt(max_act,gdp_inf)))

    fo_if_a = (eps_gt(coeff*(lb-ub),max_act-lhs))
    fo_if_b = (eps_gt(lhs,-gdp_inf))
    fo_if_c = (eps_lt(max_act,gdp_inf))
    
    fourth_if = (fo_if_a && fo_if_b && fo_if_c)

    # initialization of return data
    newbds = NewBoundTuple(false,false,[lb,ub])
    if eps_gt(coeff,BigFloat(0.0)) # query
        if first_if
            # print("entered here")
            newbds=tighten_var_upper_bound(coeff,slack,surplus,num_min_act_inf,lb,ub,
                                            is_var_cont)
        end
        if second_if
            newbds=tighten_var_lower_bound(coeff,slack,surplus,num_max_act_inf,lb,ub,
                                            is_var_cont)
        end
    else
        if third_if
            newbds=tighten_var_lower_bound(coeff,slack,surplus,num_min_act_inf,lb,ub,
                                            is_var_cont)
        end
        if fourth_if
            # print("entered here")
            newbds=tighten_var_upper_bound(coeff,slack,surplus,num_max_act_inf,lb,ub,
                                            is_var_cont)
        end
    end
    return newbds
end

function sequential_compute_activities(n_cons,col_ind,row_ind,vals,ubs,lbs,min_act,max_act,min_act_inf,
    max_act_inf,max_act_deltas)

    for cons_idx=1:n_cons
        activities = compute_activities(cons_idx,col_ind,row_ind,vals,ubs,lbs)
        max_act[cons_idx]=activities.max_act
        min_act[cons_idx]=activities.min_act
        min_act_inf[cons_idx]=activities.min_act_inf
        max_act_inf[cons_idx]=activities.max_act_inf
        max_act_deltas[cons_idx]=activities.max_act_delta
    end
end

end