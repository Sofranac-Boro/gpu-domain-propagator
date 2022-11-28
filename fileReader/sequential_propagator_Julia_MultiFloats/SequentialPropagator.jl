# bug lies here

include("PropagationMethods.jl")
include("Definitions.jl")
include("Miscellaneous.jl")
# include("TestCase.jl")

module SequentialPropagator

using ..PropagationMethods
using ..Definitions
using ..Miscellaneous

using Printf
using MultiFloats


export sequential_propagation_round,gdp_retcode_sequential_propagate

const recompute_acts_true = true
const recompute_acts_false = false
const variable_type = Definitions.gdp_continuous

function sequential_propagation_round(n_cons,n_vars,col_indices,row_indices,csc_col_ptrs,
    csc_row_indices,vals,lhss,rhss,lbs,ubs,var_types,min_acts,max_acts,min_acts_inf,
    max_acts_inf,max_acts_delta,cons_marked,cons_marked_next_round,recompute_acts)

    change_found = false

    # cons_marked = fill(0,n_cons)
    for i = 1:n_cons
    # for i = 1:1 
        # print("entered here \n")
        if cons_marked[i] == 1
            if recompute_acts
                activities = PropagationMethods.compute_activities(i,col_indices,
                row_indices,vals,ubs,lbs)
                min_acts[i]=activities.min_act
                max_acts[i]=activities.max_act
                min_acts_inf[i]=activities.min_act_inf
                max_acts_inf[i]=activities.max_act_inf
                max_acts_delta[i]=activities.max_act_delta

                # print("entered here \n")
            end
            rhs=rhss[i]
            lhs=lhss[i]

            ccbt = can_constraints_be_tightened(min_acts[i],max_acts[i],min_acts_inf[i],
                max_acts_inf[i],lhs,rhs,max_acts_delta[i])

            if ccbt
                print("\ni ",i,"\n")
                # print("entered here \n")
                # print("\nmin_acts[i] ",min_acts[i],"\n")
                # print("\nmax_acts[i] ",max_acts[i],"\n")
                # print("\nmin_acts_inf[i] ",min_acts_inf[i],"\n")
                # print("\nmax_acts_inf[i] ",max_acts_inf[i],"\n")
                print("\nlhs ",lhs,"\n")
                print("\nrhs ",rhs,"\n")
                # print("\nmax_acts_delta[i] ",max_acts_delta[i],"\n")
                num_vars_in_cons = row_indices[i+1]-row_indices[i]
                print("num_vars_in_cons ",num_vars_in_cons,"\n")
                for var=1:num_vars_in_cons
                    print("\nvar ",var,"\n")
                    val_idx=row_indices[i]+var-1
                    print("val_idx ", val_idx,"\n")
                    # print("val_idx",val_idx,"\n")
                    var_idx=col_indices[val_idx]
                    print("var_idx ", var_idx,"\n")
                    coeff=vals[val_idx]
                    print("coeff ", coeff,"\n")
                    # array of type vartypes

                    is_var_cont = (var_types[var_idx] == Int(variable_type)) #GDP_CONTINUOUS ?? to compare the types of the variables
                    print("\nis_var_cont ", is_var_cont,"\n")
                    # print(var_types[var_idx],"\n")
                    newbds = NewBoundTuple(false,false,[0.0,0.0])
                    # print(newbds,"\n")
                    newbds = tighten_variable(coeff,lhs,rhs,min_acts[i],max_acts[i],min_acts_inf[i],
                    max_acts_inf[i],is_var_cont,lbs[var_idx],ubs[var_idx])
                    if newbds.lb_is_tightened
                        lbs[var_idx]=newbds.newb[1]
                        @assert(eps_le(lbs[var_idx],ubs[var_idx]))
                        print("lb at ", var_idx," tightened to ", newbds.newb[1],"\n")
                    end
                    if newbds.ub_is_tightened
                        ubs[var_idx]=newbds.newb[2]
                        @assert(eps_le(lbs[var_idx],ubs[var_idx]))
                        # print("done")
                        print("ub at ", var_idx," tightened to ", newbds.newb[2],"\n")
                    end

                    if newbds.lb_is_tightened || newbds.ub_is_tightened
                        change_found=true
                        mark_constraints(var_idx,csc_col_ptrs,csc_row_indices,cons_marked_next_round)
                    end
                end
            end
        end
    end
    return change_found
end

function gdp_retcode_sequential_propagate(n_cons,n_vars,nnz,col_ind,row_ind,vals,
    lhss,rhss,lbs,ubs,gdp_vartype)
    if n_cons == 0 || n_vars == 0 || nnz == 0
        @printf("propagation of 0 size problem. Nothing to propagate. \n")
    end
    # print("\nn_cons ",n_cons,"\n")
    # print("\nn_vars ",n_vars,"\n")
    # print("\nnnz ",nnz,"\n")
    # print("\nvals ",vals,"\n")
    # print("\ncol_ind ",col_ind,"\n")
    # print("\nrow_ind ",row_ind,"\n")
    # print("\nlhss ",lhss,"\n")
    # print("\nrhss ",rhss,"\n")
    # print("\nlbs ",lbs,"\n")
    # print("\nubs ",ubs,"\n")
    # csr to csc? 

    # convert csr matrices to to csc in julia. Ask mathew later. write a seperate function, as an added utility

    # check test components.cu lines 16 to 18 (input) and output lines 32 to 34

    # make csc declarations as well.
    csc_vals=Array{MultiFloat{Float64,8}}(zeros(nnz))
    csc_row_ind=Array{Int64,1}(zeros(nnz))
    csc_col_ptrs=Array{Int64,1}(zeros(nnz))
# test_setup parameters to be defined here?
    cons_marked = ones(n_cons)
    cons_marked_next_round = zeros(n_cons)
    min_acts=Array{MultiFloat{Float64,8}}(zeros(n_cons))
    max_acts=Array{MultiFloat{Float64,8}}(zeros(n_cons))
    min_acts_inf=Array{MultiFloat{Float64,8}}(zeros(n_cons))
    max_acts_inf=Array{MultiFloat{Float64,8}}(zeros(n_cons))
    max_acts_delta=Array{MultiFloat{Float64,8}}(zeros(n_cons))
    # cons_marked=zeros(n_cons)
    # cons_marked_next_round=zeros(n_cons)
    
    csc_vals,csc_row_ind,csc_col_ptrs = Miscellaneous.csr_to_csc(n_cons,n_vars,vals,col_ind,row_ind)
    # print("\nprinting length of array row indices\n")
    # print("\n   ",length(csc_row_ind),"\n")
    # print("\n",csc_row_ind)

    # print("\nlength of the array column pointers\n")
    # print("\n",length(csc_col_ptrs),"\n")
    # print("\n",csc_col_ptrs)

    # print("\nlength of the array csc values\n")
    # print("\n",length(csc_vals),"\n")
    # print("\n",csc_vals)
    # for i=1:n_cons
    #     cons_marked[i]=1
    # end
    change_found=true
    # print(max_num_rounds)
    for prop_round = 1:Definitions.max_num_rounds
        # while change_found == true
        change_found = sequential_propagation_round(n_cons,n_vars,col_ind,row_ind,
        csc_col_ptrs,csc_row_ind,vals,lhss,rhss,lbs,ubs,gdp_vartype,min_acts,max_acts,
        min_acts_inf,max_acts_inf,max_acts_delta,cons_marked,cons_marked_next_round,
        recompute_acts_true)
        # end

        # print(recompute_acts_true,"\n")
    end
        # (n_cons,n_vars,col_ind,row_ind,csc_col_ptrs,csc_row_ind,vals,
        # lhss,rhss,lbs,ubs,gdp_vartype,min_acts,min_act_inf,max_act_inf,max_acts_delta,cons_marked,
        # cons_marked_next_round,recompute_acts)
    return lbs, ubs
end

end

# GDP_Retcode sequentialPropagate to be translated

# refer test_setup.cuh lines, 204 onwards

# correct solution in line 236 should be called with sequential propagate