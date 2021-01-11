#ifndef __GPUPROPAGATOR_TESTINFRA_CUH__
#define __GPUPROPAGATOR_TESTINFRA_CUH__

#include "../lib/catch.hpp"
#include <string>
#include <stdio.h>
#include <vector>
#include <set>
#include<iostream>
#include "../src/def.h"
#include <chrono>

using namespace std;

#define TEST_EPS 1e-4

void assertDoubleEPSEQ(double val1, double val2, double eps = 1e-6);

double getRand10(double seed);

double getRand10pos(double seed);

// Template implementations

template<typename T>
void compareArrays(const int len, const T *arr1, const T *arr2, double epsilon = TEST_EPS,
                   const char name[30] = "not given") {
   for (int i = 0; i < len; i++) {
      if (!(REALABS((arr1[i]) - (arr2[i])) <= (epsilon))) {
         std::cout << "\ncomnparison failed. Tag: " << name
                   << " differing nums: " << arr1[i] << " != " << arr2[i] << " len=" << len
                   << ", i=" << i << ". num1 - num2 = " << arr1[i] - arr2[i] << "\n" << std::endl;
      }
      REQUIRE((REALABS((arr1[i]) - (arr2[i])) <= (epsilon)));
   }
}

template<typename T>
std::vector <T> arrToVector(T *arr, int size) {
   return std::vector<T>(arr, arr + size);
}

//Function to convert array to Set 
template<typename datatype>
std::set <datatype> convertToSet(const datatype *v, const int beginidx, const int endidx) {
   // Declaring the  set
   std::set <datatype> s;

   // Traverse the Vector
   for (int i = beginidx; i < endidx; i++) {
      s.insert(v[i]);
   }
   return s;
}

/*
    This funtions compares two CSC matrices with disregarding the order of rows inside each column.
*/
template<typename datatype>
void compareCSCRandomRowOrder
        (
                const int n_cols,
                const int n_rows,
                const int nnz,
                const datatype *csc_vals,
                const int *csc_row_indices,
                const int *csc_col_ptrs,
                const datatype *exp_vals,
                const int *exp_row_indices,
                const int *exp_col_ptrs
        ) {
   compareArrays<int>(n_cols + 1, csc_col_ptrs, exp_col_ptrs);
   for (int i = 0; i < n_cols; i++) {
      int beginidx = csc_col_ptrs[i];
      int endidx = csc_col_ptrs[i + 1];

      std::set <datatype> loc_vals = convertToSet<datatype>(csc_vals, beginidx, endidx);
      std::set <datatype> loc_vals_exp = convertToSet<datatype>(exp_vals, beginidx, endidx);

      std::set<int> loc_row_indices = convertToSet<int>(csc_row_indices, beginidx, endidx);
      std::set<int> loc_row_indices_exp = convertToSet<int>(exp_row_indices, beginidx, endidx);

      REQUIRE(loc_vals == loc_vals_exp);
      REQUIRE(loc_row_indices == loc_row_indices_exp);
   }
}

template<typename datatype>
void consistify_non_cont_var_bounds(
        const int n_vars,
        datatype *lbs,
        datatype *ubs,
        const GDP_VARTYPE *vartypes
) {
   // make sure no integer vars have decimal values.
   for (int j = 0; j < n_vars; j++) {
      bool isVarCont = vartypes[j] == GDP_CONTINUOUS;
      ubs[j] = isVarCont ? ubs[j] : EPSFLOOR(ubs[j]);
      lbs[j] = isVarCont ? lbs[j] : EPSCEIL(lbs[j]);
   }
}

#endif