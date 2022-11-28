module PropagationMethods

include("Definitions.jl")

using .Definitions

using Printf
using MultiFloats


export ActivitiesTuple,NewBoundTuple,mark_constraints,adjust_lower_bound,adjust_upper_bound,is_lb_better,
is_ub_better,is_lb_worse,is_ub_worse,tighten_var_upper_bound,tighten_var_lower_bound,
can_constraints_be_tightened,are_constraints_infeasible,compute_activities,tighten_variable,
sequential_compute_activities

function findMin(firstTerm,secondTerm)
    termA = Float64(firstTerm)
    termB = Float64(secondTerm)
    result = min(termA,termB)
    # print(typeof(result))
    return Float64x8(result)
    
end
mutable struct ActivitiesTuple
    min_act::Float64x8
    max_act::Float64x8
    max_act_delta::Float64x8
    min_act_inf::Int64
    max_act_inf::Int64
end

mutable struct NewBoundTuple
    lb_is_tightened::Bool
    ub_is_tightened::Bool
    newb::Array{Float64x8,1}
end

# translate mark constraints as well

function mark_constraints(var_idx,csc_col_ptrs,csc_row_ind,cons_marked)
    for i = csc_col_ptrs[var_idx]:(csc_col_ptrs[var_idx+1] - 1)
        cons_marked[csc_row_ind[i]] = 1
    end
end

function adjust_upper_bound(is_var_count,ub)
    return is_var_count ? ub : eps_floor(ub)
end

function adjust_lower_bound(is_var_cont,lb)
    return is_var_cont ? lb : eps_ceil(lb)
end

# 56
function is_lb_better(lb,ub,newlb)
    @assert(eps_le(lb, ub))
    # @assert(eps_le(Float64(lb), Float64(ub)))  #query
    # print("is lb better ",eps_gt(newlb, lb),"\n")
    return eps_gt(newlb, lb)
end

function is_ub_better(lb,ub,newub)
    @assert(eps_le(lb, ub))
    # @assert(eps_le(Float64(lb), Float64(ub)))  #query
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

    new_bound = Float64x8(0.0)
    if num_inf_contr==0
        new_bound = eps_gt(coeff,0) ? slack/coeff : surplus/coeff
        new_bound = new_bound + lb
    elseif ((num_inf_contr==1) && (is_neg_inf(lb)))
        new_bound = eps_gt(coeff,0) ? slack/coeff : surplus/coeff
    else
        return newb_tuple.ub_is_tightened,newb_tuple.newb[2]
    end

    new_bound = adjust_upper_bound(is_var_cont, new_bound)

    if is_ub_better(lb,ub,new_bound)
        newb_tuple.ub_is_tightened = true
        newb_tuple.newb = [lb,new_bound]
        @assert(eps_le(lb,new_bound))
        return newb_tuple.ub_is_tightened,newb_tuple.newb[2]
    end

    return newb_tuple.ub_is_tightened,newb_tuple.newb[2]
end

function tighten_var_lower_bound(coeff,slack,surplus,num_inf_contr,lb,ub,is_var_cont)
    newb_tuple = NewBoundTuple(false,false,[lb,ub])
    newb=Float64x8(0.0)
    if num_inf_contr==0
        newb = eps_gt(coeff,0) ? surplus/coeff : slack/coeff
        newb=newb+ub
    elseif ((num_inf_contr==1) && (is_pos_inf(ub)))
        newb=eps_gt(coeff,0) ? surplus/coeff : slack/coeff
    else
        return newb_tuple.lb_is_tightened,newb_tuple.newb[1]
    end

    newb = adjust_lower_bound(is_var_cont,newb)

    if is_lb_better(lb,ub,newb)
        newb_tuple.lb_is_tightened = true
        newb_tuple.newb = [newb,ub]
        @assert(eps_le(newb,ub))
        # @assert(eps_le(newb,Float64(ub)))
        return newb_tuple.lb_is_tightened,newb_tuple.newb[1]
    end
    return newb_tuple.lb_is_tightened,newb_tuple.newb[1]
end

function can_constraints_be_tightened(min_act,max_act,num_min_act_inf,num_max_act_inf,
    lhs,rhs,max_act_delta)

    # print("entered here \n")

    if ((num_min_act_inf > 1) && (num_max_act_inf > 1))
        return false
    end
    if (eps_lt(max_act_delta,gdp_inf))
        tildeDiff = min((rhs-min_act),(max_act-lhs))
        # tildeDiff = min((Float64(rhs)-Float64(min_act)),(Float64(max_act)-Float64(lhs)))
        boolValue =  !(eps_le(max_act_delta,tildeDiff))
        return boolValue
    end
    return true
end

function are_constraints_infeasible(min_act,max_act,rhs,lhs)
    return ((min_act>rhs) || (max_act<lhs))
end

function compute_activities(cons_idx,col_ind,row_ind,vals,ubs,lbs)
    acts_tuple=ActivitiesTuple(0.0,0.0,0.0,0,0)
    # change definition to Float64 lines 148,149
    min_activity=Float64x8(0.0)
    max_activity=Float64x8(0.0)
    # min_activity=Float64(0.0)
    # max_activity=Float64(0.0)
    min_act_inf=0
    max_act_inf=0
    n_vars_in_cons = row_ind[cons_idx+1] - row_ind[cons_idx]

    # follow cons call? only for debugging

    # @printf("Printing constraint %d", cons_idx)

    for var = 1:n_vars_in_cons
        val_idx=row_ind[cons_idx] + var -1
        print("\nval_idx ",val_idx,"\n")
        var_idx=col_ind[val_idx]
        print("var_idx ",var_idx,"\n")
        # coeff=vals[val_idx]
        coeff=Float64(vals[val_idx])
        print("coeff ",coeff,"\n")
        lb= lbs[var_idx] < -Definitions.gdp_inf ? -Definitions.gdp_inf : lbs[var_idx] 
        ub=ubs[var_idx] > Definitions.gdp_inf ? Definitions.gdp_inf : ubs[var_idx]
        print("lb ",lb,"\n")
        print("ub ",ub,"\n")
        # @printf("%f    %f    [%f  %f]",coeff,var_idx,lb,ub)
        max_act_delta=abs(coeff)*(ub-lb)
        print("max_act_delta ",max_act_delta,"\n")
        print("acts_tuple.max_act_delta ",acts_tuple.max_act_delta,"\n")
        # max_act_delta=abs(Float64(coeff))*(Float64(ub)-Float64(lb))
        if (eps_gt(max_act_delta,acts_tuple.max_act_delta))
        # if (eps_gt(Float64(max_act_delta),Float64(acts_tuple.max_act_delta)))
            acts_tuple.max_act_delta=max_act_delta
        end #query in C++ curly braces are only needed if there are more 1 lines following a condition

        is_min_act_inf = eps_gt(coeff,0) ? is_neg_inf(lb) : is_pos_inf(ub)
        is_max_act_inf = eps_gt(coeff,0) ? is_pos_inf(ub) : is_neg_inf(lb)
        print("is_min_act_inf ",is_min_act_inf,"\n")
        print("is_max_act_inf ",is_max_act_inf,"\n")
        min_act_inf = min_act_inf+Int(is_min_act_inf)
        max_act_inf = max_act_inf+Int(is_max_act_inf)

        print("min_act_inf ",min_act_inf,"\n")
        print("max_act_inf ",max_act_inf,"\n")
        if is_min_act_inf==0
            min_activity += eps_gt(coeff, 0) ? coeff*lb : coeff*ub
        end
        if is_max_act_inf==0
            max_activity += eps_gt(coeff, 0) ? coeff*ub : coeff*lb
        end
        print("min_activity ",min_activity,"\n")
        print("max_activity ",max_activity,"\n")
    end
    acts_tuple.min_act=min_activity
    acts_tuple.max_act=max_activity
    acts_tuple.min_act_inf=min_act_inf
    acts_tuple.max_act_inf=max_act_inf
    print(acts_tuple,"\n")
    return acts_tuple
end

function tighten_variable(coeff,lhs,rhs,min_act,max_act,num_min_act_inf,num_max_act_inf,
    is_var_cont,lb,ub)
#= printing variables
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

    print("slack ", slack,", ")
    =#
    # print("coeff*(ub-lb) ", coeff*(ub-lb),", ")
    # fif_a = (eps_gt(coeff*(ub-lb),rhs-min_act))
    # fif_b = (eps_lt(rhs,gdp_inf))
    # fif_c = (eps_gt(min_act,-gdp_inf))
    # print("minus infinity ", -gdp_inf,"\n")
    # print("fif")
    # print("fif_a ",fif_a,"\n")
    # print("fif_b ",fif_b,"\n")
    # print("fif_c ",fif_c,"\n")
    # print("Entered method tighten v")

    ub = ub > Definitions.gdp_inf ? Definitions.gdp_inf : ub
    lb = lb < -Definitions.gdp_inf ? -Definitions.gdp_inf : lb
    #define ISPOSINF(x)    ( EPSGE(x, GDP_INF) )
    #define ISNEGINF(x)    ( EPSLE(x, -GDP_INF) )
    slack = rhs-min_act
    # slack = Float64(rhs)-Float64(min_act)
    surplus = lhs-max_act
    # surplus = Float64(lhs)-Float64(max_act)

    first_if_condition = ((eps_gt(coeff*(ub-lb),rhs-min_act)) && 
            (!is_pos_inf(rhs)) && (!is_neg_inf(min_act)))
    
    second_if_condition = (eps_gt(coeff*(ub-lb),max_act-lhs) &&
    (!is_neg_inf(lhs)) && (!is_pos_inf(max_act)))   
    
    third_if_condition = ((eps_gt(coeff*(lb-ub),rhs-min_act)) &&
                (!is_pos_inf(rhs)) && (!is_neg_inf(min_act)))
    
    fourth_if_condition = (eps_gt(coeff*(lb-ub),max_act-lhs) && 
    (!is_neg_inf(lhs)) && (!is_pos_inf(max_act)))
#=
    first_if_condition = ((eps_gt(Float64(coeff)*(Float64(ub)-Float64(lb)),Float64(rhs)-Float64(min_act))) && 
            (!is_pos_inf(Float64(rhs))) && (!is_neg_inf(Float64(min_act))))
 #( fif_a && fif_b && fif_c )


    second_if_condition = (eps_gt(Float64(coeff)*(Float64(ub)-Float64(lb)),Float64(max_act)-Float64(lhs)) &&
                (!is_neg_inf(Float64(lhs))) && (!is_pos_inf(Float64(max_act))))


    third_if_condition = ((eps_gt(Float64(coeff)*(Float64(lb)-Float64(ub)),Float64(rhs)-Float64(min_act))) &&
                (!is_pos_inf(Float64(rhs))) && (!is_neg_inf(Float64(min_act))))

    # fo_if_a = (eps_gt(coeff*(lb-ub),max_act-lhs))
    # fo_if_b = (eps_gt(lhs,-gdp_inf))
    # fo_if_c = (eps_lt(max_act,gdp_inf))
    

    fourth_if_condition = (eps_gt(Float64(coeff)*(Float64(lb)-Float64(ub)),Float64(max_act)-Float64(lhs)) && 
    (!is_neg_inf(Float64(lhs))) && (!is_pos_inf(Float64(max_act)))) #(fo_if_a && fo_if_b && fo_if_c)
=#
    # initialization of return data
    newbds = NewBoundTuple(false,false,[lb,ub])
    ctr = 0;
    if eps_gt(coeff,0.0) # query
        if first_if_condition
            # print("entered here")
            newbds.ub_is_tightened,newbds.newb[2]=tighten_var_upper_bound(coeff,slack,surplus,num_min_act_inf,lb,ub,
                                            is_var_cont)
            
                                            # newbds=tighten_var_upper_bound(coeff,slack,surplus,num_min_act_inf,lb,ub,
                                            # is_var_cont)
            # ctr = ctr +1
        end
        if second_if_condition
            newbds.lb_is_tightened,newbds.newb[1]=tighten_var_lower_bound(coeff,slack,surplus,num_max_act_inf,lb,ub,
                                            is_var_cont)
            
                                            # if ctr!=0
                # newbds=tighten_var_lower_bound(coeff,slack,surplus,num_max_act_inf,lb,newbds.newb[2],
                                            # is_var_cont)
                # newbds.ub_is_tightened = true
                
            # else
                # newbds=tighten_var_lower_bound(coeff,slack,surplus,num_max_act_inf,lb,ub,
                #                             is_var_cont)
            # end
        end
    else
        if third_if_condition
            newbds.lb_is_tightened,newbds.newb[1]=tighten_var_lower_bound(coeff,slack,surplus,num_min_act_inf,lb,ub,
                                            is_var_cont)
            # newbds=tighten_var_lower_bound(coeff,slack,surplus,num_min_act_inf,lb,ub,
            #                                 is_var_cont)
        end
        if fourth_if_condition
            # print("entered here")
            newbds.ub_is_tightened, newbds.newb[2]=tighten_var_upper_bound(coeff,slack,surplus,num_max_act_inf,lb,ub,
            is_var_cont)
            # newbds=tighten_var_upper_bound(coeff,slack,surplus,num_max_act_inf,lb,ub,
            #                                 is_var_cont)
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