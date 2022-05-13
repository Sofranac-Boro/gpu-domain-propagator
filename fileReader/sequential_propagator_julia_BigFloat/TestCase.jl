include("SequentialPropagator.jl")
include("Miscellaneous.jl")
include("Definitions.jl")

using ..SequentialPropagator
using ..Miscellaneous
using ..Definitions

# using MultiFloats

# N = 8  precision
# test_case = "TwoLinesExample"

# if test_case == "Achterberg"
#     m=11
#     n=12
#     nnz=52
#     temp_vals=Array{BigFloat,1}([2, 3, 2, 9, -1, -2, -3, 5, -3, -3, 9, -2, 9, -1, 2, -4, -7, 2, 5, -2, -1, 5, 4, -5, 1, -1,
#     1, -2, 1, -1, -2, 1, -2, 4, 2, -1, 3, -2, -1, 5, 1, 2, -6, -2, -2, 1, 1, 1, 2, -2, 2, -2])
#     temp_col_ind=Array{Int64,1}([0, 7, 8, 1, 7, 8, 1, 2, 3, 1, 3, 9, 4, 8, 9, 5, 6, 9, 6, 8, 4, 6, 8, 9, 0, 1, 2, 4, 5, 7, 8,
#     9, 10, 11, 1, 3, 4, 5, 7, 8, 9, 10, 11, 0, 2, 3, 4, 7, 8, 9, 10, 11])

#     temp_col_ind = 1 .+ temp_col_ind
#     temp_row_ind=Array{Int64,1}([0, 3, 6, 9, 12, 15, 18, 20, 24, 34, 43, nnz])

#     temp_row_ind = 1 .+ temp_row_ind
#     temp_lhss=Array{BigFloat,1}([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])
#     temp_rhss=Array{BigFloat,1}([9, 0, 4, 6, 8, 3, 2, 2, 1, 2, 1])
#     temp_lbs=Array{BigFloat,1}([0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     temp_ubs=Array{BigFloat,1}([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0])

#     temp_vartypes=[Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer),
#     Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer),Int(Definitions.gdp_integer),
#     Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer)]
#     temp_cons_marked = ones(m)

#     lbs_analytical = Array{BigFloat,1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{BigFloat,1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print(ubs_analytical)
# elseif test_case == "Savelsbergh"
#     m=6
#     n=6
#     nnz=12
#     temp_vals=Array{BigFloat,1}([1, 3, 1, 2, 2, 1, -15, 1, -20, 1, -5, 1])
#     temp_col_ind=Array{Int64,1}([3, 4, 3, 4, 3, 4, 0, 3, 1, 4, 2, 5])
#     temp_col_ind = 1 .+ temp_col_ind

#     temp_row_ind=Array{Int64,1}([0, 2, 4, 6, 8, 10, nnz])
#     temp_row_ind = 1 .+ temp_row_ind

#     temp_lhss=ArrayArray{BigFloat,1}([15, 10, 20, -Definitions.gdp_inf, -Definitions.gdp_inf, 
#     -Definitions.gdp_inf])
#     temp_rhss=Array{BigFloat,1}([Definitions.gdp_inf, Definitions.gdp_inf, Definitions.gdp_inf, 
#     0, 0, 0])
#     temp_lbs=Array{BigFloat,1}([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     temp_ubs=Array{BigFloat,1}([1.0, 1.0, 1.0, Definitions.gdp_inf, Definitions.gdp_inf, 
#     Definitions.gdp_inf])

#     temp_vartypes=[Int(Definitions.gdp_binary), Int(Definitions.gdp_binary), 
#     Int(Definitions.gdp_binary), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous)]
#     temp_cons_marked = ones(m)

#     lbs_analytical = Array{BigFloat,1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{BigFloat,1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print(ubs_analytical)

# elseif test_case == "CFLPSavelsbergh"
#     m=7
#     n=16
#     nnz=28
#     temp_vals=Array{BigFloat,1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -100, 1, 1, 1,
#     -100, 1, 1, 1, -100, 1, 1, 1, -100])
#     temp_col_ind=Array{Int64,1}([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 4, 8, 12, 1, 5, 9, 
#     13, 2, 6, 10, 14, 3, 7, 11, 15])
#     temp_col_ind = 1 .+ temp_col_ind

#     temp_row_ind=Array{Int64,1}([0, 4, 8, 12, 16, 20, 24, nnz])
#     temp_row_ind = 1 .+ temp_row_ind

#     temp_lhss=Array{BigFloat,1}([80, 70, 40, -Definitions.gdp_inf, -Definitions.gdp_inf, 
#     -Definitions.gdp_inf, -Definitions.gdp_inf])
#     temp_rhss=Array{BigFloat,1}([80, 70, 40, 0, 0, 0, 0])
#     temp_lbs=Array{BigFloat,1}([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#      0.0, 0.0, 0.0])
#     temp_ubs=Array{BigFloat,1}([Definitions.gdp_inf, Definitions.gdp_inf, Definitions.gdp_inf, 
#     Definitions.gdp_inf, Definitions.gdp_inf, Definitions.gdp_inf, 
#     Definitions.gdp_inf, Definitions.gdp_inf, Definitions.gdp_inf, 
#     Definitions.gdp_inf,Definitions.gdp_inf, Definitions.gdp_inf, 1, 1, 1, 1])

#     temp_vartypes=[Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_binary), Int(Definitions.gdp_binary), Int(Definitions.gdp_binary), 
#     Int(Definitions.gdp_binary)]

#     temp_cons_marked = ones(m)

#     lbs_analytical = Array{BigFloat,1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{BigFloat,1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print(ubs_analytical)
# elseif test_case == "TwoLinesExample"
#     m=2
#     n=2
#     nnz=4
#     temp_vals=Array{BigFloat,1}([0.5, 1.0, 1.0, -1.0])
#     temp_col_ind=Array{Int64,1}([0, 1, 0, 1])
#     temp_col_ind = 1 .+ temp_col_ind

#     temp_row_ind=Array{Int64,1}([0, 2, 4])
#     temp_row_ind = 1 .+ temp_row_ind

#     temp_lhss=Array{BigFloat,1}([1.0,1.0])
#     temp_rhss=Array{BigFloat,1}([1.0,1.0])
#     temp_lbs=Array{BigFloat,1}([-2.0,-2.0])
#     temp_ubs=Array{BigFloat,1}([2.0,2.0])

#     # temp_vartypes=[Definitions.gdp_continuous, Definitions.gdp_continuous]
#     temp_vartypes = [3,3]

#     temp_cons_marked = ones(m)

#     lbs_analytical = Array{BigFloat,1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{BigFloat,1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print("analytical solution for upper bounds ",ubs_analytical,"\n")
#     print("analytical solution for lower bounds ", lbs_analytical,"\n")
# end

function convert_to_higher_floating_point_accuracy_datatype(m,n,nnz,col_ind,row_ind,vals,lhss,rhss,lbs,ubs,vartypes)
    vals            = Array{BigFloat,1}(vals)
    col_ind         = Array{Int64,1}(col_ind)
    julian_col_ind  = 1 .+ col_ind
    row_ind         = Array{Int64,1}(row_ind)
    julian_row_ind  = 1 .+ row_ind
    lhss            = Array{BigFloat,1}(lhss)
    rhss            = Array{BigFloat,1}(rhss)
    lbs             = Array{BigFloat,1}(lbs)
    ubs             = Array{BigFloat,1}(ubs)
    
    lbs_analytical  = Array{BigFloat,1}(zeros(size(lbs)))
    ubs_analytical  = Array{BigFloat,1}(zeros(size(ubs)))

    lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,julian_col_ind,julian_row_ind,vals,lhss,rhss,lbs,ubs,vartypes)

    print("analytical solution for upper bounds ",ubs_analytical,"\n")
    print("analytical solution for lower bounds ", lbs_analytical,"\n")

end
    # call sequential propagate here

# # elem = [7.5,2.9,2.8,2.7,6.8,5.7,3.8,2.4,6.2,3.2,9.7,2.3,5.0,6.6,8.1]
# elem = temp_vals #[7.5,2.9,2.8,2.7,6.8,5.7,3.8,2.4,6.2,3.2,9.7,2.3,5.8,5.0,6.6,8.1]
# col = temp_col_ind #[1,2,3,4,1,2,3,1,2,3,1,4,5,6,5,6]
# ptr = temp_row_ind #[1,5,8,11,13,15,17]

# full_matrix = Miscellaneous.csr_to_csc(elem,col,ptr)
# print(full_matrix)

# using higher precision than 64 bits for floating point numbers in Julia. (emulation)
# download and install conda on lovasz.
# passing flags in shell scripts.