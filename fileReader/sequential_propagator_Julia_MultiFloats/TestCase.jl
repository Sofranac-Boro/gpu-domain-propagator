include("SequentialPropagator.jl")
include("Miscellaneous.jl")
include("Definitions.jl")
module TestCase



using ..SequentialPropagator
using ..Miscellaneous
using ..Definitions


using MultiFloats
using Printf
using DelimitedFiles

export convert_to_higher_floating_point_accuracy_datatype


# N = 8  precision
# test_case = "TwoLinesExample"

# if test_case == "Achterberg"
#     m=11
#     n=12
#     nnz=52
#     temp_vals=Array{MultiFloat{Float64,8},1}([2, 3, 2, 9, -1, -2, -3, 5, -3, -3, 9, -2, 9, -1, 2, -4, -7, 2, 5, -2, -1, 5, 4, -5, 1, -1,
#     1, -2, 1, -1, -2, 1, -2, 4, 2, -1, 3, -2, -1, 5, 1, 2, -6, -2, -2, 1, 1, 1, 2, -2, 2, -2])
#     temp_col_ind=Array{Int64,1}([0, 7, 8, 1, 7, 8, 1, 2, 3, 1, 3, 9, 4, 8, 9, 5, 6, 9, 6, 8, 4, 6, 8, 9, 0, 1, 2, 4, 5, 7, 8,
#     9, 10, 11, 1, 3, 4, 5, 7, 8, 9, 10, 11, 0, 2, 3, 4, 7, 8, 9, 10, 11])

#     temp_col_ind = 1 .+ temp_col_ind
#     temp_row_ind=Array{Int64,1}([0, 3, 6, 9, 12, 15, 18, 20, 24, 34, 43, nnz])

#     temp_row_ind = 1 .+ temp_row_ind
#     temp_lhss=Array{MultiFloat{Float64,8},1}([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100])
#     temp_rhss=Array{MultiFloat{Float64,8},1}([9, 0, 4, 6, 8, 3, 2, 2, 1, 2, 1])
#     temp_lbs=Array{MultiFloat{Float64,8},1}([0.0 , 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     temp_ubs=Array{MultiFloat{Float64,8},1}([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0, 100.0])

#     temp_vartypes=[Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer),
#     Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer),Int(Definitions.gdp_integer),
#     Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer), Int(Definitions.gdp_integer)]
#     temp_cons_marked = ones(m)

#     lbs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print(ubs_analytical)
# elseif test_case == "Savelsbergh"
#     m=6
#     n=6
#     nnz=12
#     temp_vals=Array{MultiFloat{Float64,8},1}([1, 3, 1, 2, 2, 1, -15, 1, -20, 1, -5, 1])
#     temp_col_ind=Array{Int64,1}([3, 4, 3, 4, 3, 4, 0, 3, 1, 4, 2, 5])
#     temp_col_ind = 1 .+ temp_col_ind

#     temp_row_ind=Array{Int64,1}([0, 2, 4, 6, 8, 10, nnz])
#     temp_row_ind = 1 .+ temp_row_ind

#     temp_lhss=ArrayArray{MultiFloat{Float64,8},1}([15, 10, 20, -Definitions.gdp_inf, -Definitions.gdp_inf, 
#     -Definitions.gdp_inf])
#     temp_rhss=Array{MultiFloat{Float64,8},1}([Definitions.gdp_inf, Definitions.gdp_inf, Definitions.gdp_inf, 
#     0, 0, 0])
#     temp_lbs=Array{MultiFloat{Float64,8},1}([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#     temp_ubs=Array{MultiFloat{Float64,8},1}([1.0, 1.0, 1.0, Definitions.gdp_inf, Definitions.gdp_inf, 
#     Definitions.gdp_inf])

#     temp_vartypes=[Int(Definitions.gdp_binary), Int(Definitions.gdp_binary), 
#     Int(Definitions.gdp_binary), Int(Definitions.gdp_continuous), 
#     Int(Definitions.gdp_continuous), Int(Definitions.gdp_continuous)]
#     temp_cons_marked = ones(m)

#     lbs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print(ubs_analytical)

# elseif test_case == "CFLPSavelsbergh"
#     m=7
#     n=16
#     nnz=28
#     temp_vals=Array{MultiFloat{Float64,8},1}([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -100, 1, 1, 1,
#     -100, 1, 1, 1, -100, 1, 1, 1, -100])
#     temp_col_ind=Array{Int64,1}([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 4, 8, 12, 1, 5, 9, 
#     13, 2, 6, 10, 14, 3, 7, 11, 15])
#     temp_col_ind = 1 .+ temp_col_ind

#     temp_row_ind=Array{Int64,1}([0, 4, 8, 12, 16, 20, 24, nnz])
#     temp_row_ind = 1 .+ temp_row_ind

#     temp_lhss=Array{MultiFloat{Float64,8},1}([80, 70, 40, -Definitions.gdp_inf, -Definitions.gdp_inf, 
#     -Definitions.gdp_inf, -Definitions.gdp_inf])
#     temp_rhss=Array{MultiFloat{Float64,8},1}([80, 70, 40, 0, 0, 0, 0])
#     temp_lbs=Array{MultiFloat{Float64,8},1}([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
#      0.0, 0.0, 0.0])
#     temp_ubs=Array{MultiFloat{Float64,8},1}([Definitions.gdp_inf, Definitions.gdp_inf, Definitions.gdp_inf, 
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

#     lbs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print(ubs_analytical)
# elseif test_case == "TwoLinesExample"
#     m=2
#     n=2
#     nnz=4
#     temp_vals=Array{MultiFloat{Float64,8},1}([0.5, 1.0, 1.0, -1.0])
#     temp_col_ind=Array{Int64,1}([0, 1, 0, 1])
#     temp_col_ind = 1 .+ temp_col_ind

#     temp_row_ind=Array{Int64,1}([0, 2, 4])
#     temp_row_ind = 1 .+ temp_row_ind

#     temp_lhss=Array{MultiFloat{Float64,8},1}([1.0,1.0])
#     temp_rhss=Array{MultiFloat{Float64,8},1}([1.0,1.0])
#     temp_lbs=Array{MultiFloat{Float64,8},1}([-2.0,-2.0])
#     temp_ubs=Array{MultiFloat{Float64,8},1}([2.0,2.0])

#     # temp_vartypes=[Definitions.gdp_continuous, Definitions.gdp_continuous]
#     temp_vartypes = [3,3]

#     temp_cons_marked = ones(m)

#     lbs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_lbs)))
#     ubs_analytical = Array{MultiFloat{Float64,8},1}(zeros(size(temp_ubs)))

#     lbs_analytical, ubs_analytical = gdp_retcode_sequential_propagate(m,n,nnz,temp_col_ind,
#     temp_row_ind,temp_vals,temp_lhss,temp_rhss,temp_lbs,temp_ubs, temp_vartypes)#Definitions.gdp_vartype)

#     print("analytical solution for upper bounds ",ubs_analytical,"\n")
#     print("analytical solution for lower bounds ", lbs_analytical,"\n")
# end

function convert_to_higher_floating_point_accuracy_datatype(n,m,nnz,col_ind,row_ind,vals,lhss,rhss,lbs,ubs,vartypes)
    # number of variables is n
    # number of constraints is m
    # length of row_indices = m+1
    # length of col_ind = nnz
    # max. row_ind = nnz
    #=
    print("\n","n_vars ",n,"\n")
    print("\n","n_cons ",m,"\n")
    print("\n","nnz ",nnz,"\n")
    print("\n","col_ind ", maximum(col_ind),"\n")
    print("\n","row_ind ",length(row_ind),"\n")
    print("\n","vals ", vals,"\n")
    print("\n","lhss ", lhss,"\n")
    print("\n","rhss ",rhss,"\n")
    print("\n","lbs ",lbs,"\n")
    print("\n","ubs ",ubs,"\n")
    print("\n vartypes ",vartypes,"\n")
    # print("n_cons \n",)
    =#
    
    ind_vals_list = findall(vals[:] .>= Definitions.gdp_inf)
    vals[ind_vals_list] .= Definitions.gdp_inf

    ind_lhss_list = findall(lhss[:] .>= Definitions.gdp_inf)
    lhss[ind_lhss_list] .= Definitions.gdp_inf

    ind_rhss_list = findall(rhss[:] .>= Definitions.gdp_inf)
    rhss[ind_rhss_list] .= Definitions.gdp_inf

    ind_lhss_list = findall(lhss[:] .<= -Definitions.gdp_inf)
    lhss[ind_lhss_list] .= -Definitions.gdp_inf

    ind_rhss_list = findall(rhss[:] .<= -Definitions.gdp_inf)
    rhss[ind_rhss_list] .= -Definitions.gdp_inf

    # ind_lbs_list = findall(lbs[:] .>= Definitions.gdp_inf)
    # lbs[ind_lbs_list] .= Definitions.gdp_inf
    
    # ind_ubs_list = findall(ubs[:] .>= Definitions.gdp_inf)
    # ubs[ind_ubs_list] .= Definitions.gdp_inf

    vals            = Array{MultiFloat{Float64,8},1}(vals)
    col_ind         = Array{Int64,1}(col_ind)
    julian_col_ind  = 1 .+ col_ind
    row_ind         = Array{Int64,1}(row_ind)
    julian_row_ind  = 1 .+ row_ind
    lhss            = Array{MultiFloat{Float64,8},1}(lhss)
    rhss            = Array{MultiFloat{Float64,8},1}(rhss)
    lbs             = Array{MultiFloat{Float64,8},1}(lbs)
    ubs             = Array{MultiFloat{Float64,8},1}(ubs)
    
    lbs_analytical  = Array{MultiFloat{Float64,8},1}(zeros(size(lbs)))
    ubs_analytical  = Array{MultiFloat{Float64,8},1}(zeros(size(ubs)))

    # replace!(x -> is_pos_inf(x) ? Inf : x, ubs)
    # replace!(x -> is_neg_inf(x) ? -Inf : x, lbs)

    lbs_analytical, ubs_analytical = SequentialPropagator.gdp_retcode_sequential_propagate(m,n,nnz,julian_col_ind,julian_row_ind,vals,lhss,rhss,lbs,ubs,vartypes)

    # print("analytical solution for upper bounds ",ubs_analytical,"\n")
    # print("analytical solution for lower bounds ", lbs_analytical,"\n")

    @printf("saving result to >>> HighPrecisionResult.txt\n")
    if isfile("HighPrecisionResult.txt") == false
        io = open("HighPrecisionResult.txt","a")
    else
        rm("HighPrecisionResult.txt")
        io = open("HighPrecisionResult.txt","a")
    end
    write(io,"Lower bounds : \n")
    write(io,string(lbs_analytical))
    # writedlm("HighPrecisionResult.txt",lbs_analytical)
    # @sync
    write(io,"upper bounds : \n")
    write(io,string(ubs_analytical))
    close(io)

    return lbs_analytical,ubs_analytical
end
    # call sequential propagate here
end
# # elem = [7.5,2.9,2.8,2.7,6.8,5.7,3.8,2.4,6.2,3.2,9.7,2.3,5.0,6.6,8.1]
# elem = temp_vals #[7.5,2.9,2.8,2.7,6.8,5.7,3.8,2.4,6.2,3.2,9.7,2.3,5.8,5.0,6.6,8.1]
# col = temp_col_ind #[1,2,3,4,1,2,3,1,2,3,1,4,5,6,5,6]
# ptr = temp_row_ind #[1,5,8,11,13,15,17]

# full_matrix = Miscellaneous.csr_to_csc(elem,col,ptr)
# print(full_matrix)

# using higher precision than 64 bits for floating point numbers in Julia. (emulation)
# download and install conda on lovasz.
# passing flags in shell scripts.

# end