#ifndef __GPUPROPAGATOR_MISC_CUH__
#define __GPUPROPAGATOR_MISC_CUH__

#include <vector>
#include <string>
#include <chrono>
#include <iostream>

#define VERBOSE

#ifdef VERBOSE
#define VERBOSE_CALL(ans) {(ans);}
#else
#define VERBOSE_CALL(ans) do { } while(0)
#endif

void measureTime(const char alg_name[30], std::chrono::_V2::steady_clock::time_point start, std::chrono::_V2::steady_clock::time_point end);

template <class datatype>
void countPrintNumMarkedCons(int n_cons, int* consmarked)
{
    int num_cons = 0;
    for (int i=0; i<n_cons; i++)
    {
        if (consmarked[i] == true)
        {
            num_cons++;
        }
    }
    printf("Num marked constraints: %d\n", num_cons);
}

template <typename T>
void printArray(const T* arr, const int size, const char name[30])
{
    std::string print_str =  std::is_same<T, int>::value? "%d " : "% 06.2f ";

    printf("%s: [", name);
    for (int i=0; i< size; i++)
    {
        printf(print_str.c_str(), arr[i]);
    }
    printf("]\n");
}

template <typename datatype>
void printBoundCandidates
(
    const int n_vars,
    const int nnz,
    const int* csc_col_ptrs,
    const datatype* newbs
)
{

    for (int varidx=0; varidx<n_vars; varidx++)
    {
       // int varidx = 0;
        int num_cons_with_var = csc_col_ptrs[varidx+1] - csc_col_ptrs[varidx];

        printf("bound candidates for varidx: %d : \n", varidx);
        for (int cons=0; cons < num_cons_with_var; cons++)
        {
            int validx = csc_col_ptrs[varidx] + cons;
            printf("%.3e, validx: %d\n", newbs[validx], validx);
        }
        printf("csc indices: %d plus %d\n", csc_col_ptrs[varidx], num_cons_with_var);
        printf("\n");
    }
    printf("\n");
}

template <typename datatype>
void consistify_var_bounds(const int n_vars, datatype* lbs, datatype* ubs, const int* vartypes)
{
    // make sure no integer vars have decimal values.
    for (int j=0; j<n_vars; j++)
    {
       bool isVarCont = vartypes[j] == 3;
       ubs[j] = isVarCont? ubs[j] : floor(ubs[j]);
       lbs[j] = isVarCont? lbs[j] : ceil(lbs[j]);
    }
}

#endif