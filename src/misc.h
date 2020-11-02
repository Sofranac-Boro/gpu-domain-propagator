#ifndef __GPUPROPAGATOR_MISC_CUH__
#define __GPUPROPAGATOR_MISC_CUH__

#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <fstream>
#include <assert.h>

// #define FOLLOW_CONS 581830

inline static void* safe_malloc(size_t n, unsigned long line)
{
   void* p = malloc(n);
   if (!p)
   {
      fprintf(stderr, "[%s:%lu]Out of memory(%lu bytes)\n",
              __FILE__, line, (unsigned long)n);
      exit(EXIT_FAILURE);
   }
   return p;
}

#ifdef VERBOSE
#define VERBOSE_CALL(ans) {(ans);}
#else
#define VERBOSE_CALL(ans) do { } while(0)
#endif

#ifdef DEBUG
#define SAFEMALLOC(n) safe_malloc(n, __LINE__)
#define DEBUG_CALL(ans) {(ans);}
#else
#define SAFEMALLOC(n) malloc(n)
#define DEBUG_CALL(ans) do { } while(0)
#endif

#ifdef FOLLOW_VAR
#define FOLLOW_VAR_CALL(varidx, ans) varidx==FOLLOW_VAR? (ans) : printf("")
#else
#define FOLLOW_VAR_CALL(varidx, ans) do { } while(0)
#endif

#ifdef FOLLOW_CONS
#define FOLLOW_CONS_CALL(considx, ans) considx==FOLLOW_CONS? (ans) : printf("")
#else
#define FOLLOW_CONS_CALL(considx, ans) do { } while(0)
#endif

#ifdef CALC_PROGRESS
#define CALC_PROGRESS_CALL(ans) {(ans);}
#else
#define CALC_PROGRESS_CALL(ans) do { } while(0)
#endif

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
void consistify_var_bounds(const int n_vars, datatype *lbs, datatype *ubs, const int *vartypes) {
   // make sure no integer vars have decimal values.
   for (int j = 0; j < n_vars; j++) {
      bool isVarCont = vartypes[j] == 3;
      ubs[j] = isVarCont ? ubs[j] : floor(ubs[j]);
      lbs[j] = isVarCont ? lbs[j] : ceil(lbs[j]);
   }
}

template<typename datatype>
void save_acts_to_file(int n_cons, datatype* minacts, datatype* maxacts, int round)
{
   std::ofstream file;
   file.open("round_" + std::to_string(round) + "_activities.txt");
   file << "constraint: min_act max_act" << std::endl;

   for (int i=0; i<n_cons; i++)
   {
      file << i <<": " << minacts[i] << " " << maxacts[i] << std::endl;
   }

   file.close();
}

//template <typename datatype>
//void print_var_changes(int varidx, datatype oldbound, datatype )
//{
//   if (varidx == FOLLOW_VAR)
//   {
//
//   }
//}
#endif