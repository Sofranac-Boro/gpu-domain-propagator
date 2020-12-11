#ifndef __GPUPROPAGATOR_MISC_CUH__
#define __GPUPROPAGATOR_MISC_CUH__

#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <assert.h>

#include "def.h"
#include "params.h"

// #define FOLLOW_CONS 581830



void measureTime(const char alg_name[30], std::chrono::_V2::steady_clock::time_point start,
                 std::chrono::_V2::steady_clock::time_point end);

template<class datatype>
void countPrintNumMarkedCons(int n_cons, int *consmarked) {
   int num_cons = 0;
   for (int i = 0; i < n_cons; i++) {
      if (consmarked[i] == true) {
         num_cons++;
      }
   }
   printf("Num marked constraints: %d\n", num_cons);
}

template<typename T>
void printArray(const T *arr, const int size, const char name[30]) {
   std::string print_str = std::is_same<T, int>::value ? "%d " : "% 06.2f ";

   printf("%s: [", name);
   for (int i = 0; i < size; i++) {
      printf(print_str.c_str(), arr[i]);
   }
   printf("]\n");
}

template<typename datatype>
void printBoundCandidates
        (
                const int n_vars,
                const int nnz,
                const int *csc_col_ptrs,
                const datatype *newbs
        ) {

   for (int varidx = 0; varidx < n_vars; varidx++) {
      // int varidx = 0;
      int num_cons_with_var = csc_col_ptrs[varidx + 1] - csc_col_ptrs[varidx];

      printf("bound candidates for varidx: %d : \n", varidx);
      for (int cons = 0; cons < num_cons_with_var; cons++) {
         int validx = csc_col_ptrs[varidx] + cons;
         printf("%.3e, validx: %d\n", newbs[validx], validx);
      }
      printf("csc indices: %d plus %d\n", csc_col_ptrs[varidx], num_cons_with_var);
      printf("\n");
   }
   printf("\n");
}

template<typename datatype>
void checkInput(
        const int n_cons,
        const int n_vars,
        const int nnz,
        const datatype *csr_vals,
        const datatype *lhss,
        const datatype *rhss,
        const datatype *lbs,
        const datatype *ubs,
        const GDP_VARTYPE *vartypes
) {
   for (int i = 0; i < nnz; i++) {
      assert(!EPSEQ(csr_vals[i], 0.0));
   }

   for (int i = 0; i < n_vars; i++) {
      assert(EPSGE(ubs[i], lbs[i]));
      // make sure all integer vars have integer values

      if (vartypes[i] != GDP_CONTINUOUS) {
         assert(EPSEQ(lbs[i], EPSCEIL(lbs[i])));
         assert(EPSEQ(ubs[i], EPSFLOOR(ubs[i])));
      }
   }

   for (int i = 0; i < n_cons; i++) {
      assert(EPSGE(rhss[i], lhss[i]));
   }
}

template<typename datatype>
datatype maxConsecutiveElemDiff(const datatype *array, const int size) {
   assert(size >= 1);

   datatype ret = array[1] - array[0];
   for (int i = 1; i < size - 1; i++) {
      if (array[i + 1] - array[i] > ret) {
         ret = array[i + 1] - array[i];
      }
   }
   return ret;
}

#endif