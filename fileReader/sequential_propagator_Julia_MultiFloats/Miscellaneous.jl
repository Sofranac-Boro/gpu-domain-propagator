module Miscellaneous

export count_print_number_marked_constraints,check_input,max_cosequtive_element_diff,
print_bound_candidates,csr_to_csc

include("Definitions.jl")

using ..Definitions

using Printf
using BenchmarkTools
using SparseArrays
using MultiFloats
# use benchmarktools.jl

function count_print_number_marked_constraints(n_cons,cons_marked)
    nums_cons = 0
    for i = 1:n_cons
        if cons_marked[i] == true
            nums_cons = nums_cons + 1
        end
    end
    print("Number of marked constraints ",nums_cons)
end

function print_array(arr,size,name)

    print("%s: [",name)
    for i = 1:size
        print(arr[i])
    end
    print("]\n")
end
# print vector and arrays equivalent syntax

function print_vector(vec,name)
    print(": [ ")
    for i=1:length(vec)
        print(vec[i]," ")
    end
    print(" ]\n")
end

function print_bound_candidates(n_vars,nnz,csc_col_ptrs,newbs)
    for var_idx=1:n_vars
        nums_cons_with_var = csc_col_ptrs[var_idx+1] - csc_col_ptrs[var_idx]
        @printf("bound candidates for var_idx: %d : \n", var_idx)
        for cons=1:nums_cons_with_var
            val_idx = csc_col_ptrs[var_idx]+cons
            @printf("%f, val_idx: %d\n",csc_col_ptrs[var_idx],nums_cons_with_var)
        end
        @printf("csc indices: %d plus %d\n",csc_col_ptrs[var_idx],nums_cons_with_var)
        @printf("\n")
    end
    @printf("\n")
end

# translate all functions mentioned in debug call

function check_input(n_cons,n_vars,nnz,csr_vals,lhss,rhss,lbs,ubs,vartypes)
    for i=1:nnz
        @assert(!eps_eq(csr_vals[i],0.0))
    end

    for i=1:n_vars
        lb=lbs[i]
        ub=ubs[i]

        @assert(eps_ge(ub,lb))

        @assert(eps_lt(lb,gdp_inf) && eps_gt(ub,-gdp_inf))

        if vartypes[i] != GDP_CONTINUOUS
            @assert(eps_eq(lb,eps_ceil(lb)))
            @assert(eps_eq(ub,eps_floor(ub)))
        end
    end

    for i=1:n_cons
        @assert(eps_ge(rhss[i],lhss[i]))
    end
end

function max_cosequtive_element_diff(array,size)
    @assert(size>=1)
    ret = array[1]-array[0]
    for i=2:size-1
        if (array[i+1]-array[i]>ret)
            ret=array[i+1]-array[i]
        end
    end
    return ret
end

function csr_to_csc(m,n,non_zero_elements,column_index,row_pointer)
    # m = length(row_pointer) - 1
    # n = maximum(column_index)
    matrix = spzeros(m,n)
    for i = 1:m
        for j = row_pointer[i]:(row_pointer[i+1]-1)
            matrix[i,column_index[j]] = non_zero_elements[j]
        end
    end
    # sparse_matrix = sparse(matrix)
    nnz_csc = matrix.nzval
    row_index = matrix.rowval
    col_ptr = matrix.colptr

    return nnz_csc,row_index,col_ptr
end

end
# print("\nconverting csr to csc\n")
# print(length(row_pointer)-1,"\n")
# if length(row_pointer)-1 < maximum(column_index)
#     print("\ncondition where m is lesser than n\n")
#     dim = maximum(column_index)
#     matrix = zeros(dim,dim)
#     range = length(row_pointer)
# else
#     dim = length(row_pointer)
#     matrix = zeros(dim,dim)
#     range = maximum(column_index)
# end
    # print("\n",dim)
# println("range of i ",length(row_pointer)-1)