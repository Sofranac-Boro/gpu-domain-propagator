#ifndef __GPUPROPAGATOR_PROPMETHODS_CUH__
#define __GPUPROPAGATOR_PROPMETHODS_CUH__

#include <math.h>       /* fabs */
#include <omp.h>
#include "params.h"
#include "kernels/util_kernels.cuh"


struct ActivitiesTupleStruct {
    double minact;
    double maxact;
    double maxactdelta;
    int minact_inf;
    int maxact_inf;
};
typedef struct ActivitiesTupleStruct ActivitiesTuple;

template<typename datatype>
struct NewBoundTuple {
    bool is_tightened;
    datatype newb;
};

template<typename datatype>
struct NewBounds {
    NewBoundTuple<datatype> lb;
    NewBoundTuple<datatype> ub;
};


void markConstraints
        (
                const int var_idx,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                int *consmarked
        );

template<class datatype>
datatype adjustUpperBound
        (
                const bool is_var_cont,
                const datatype ub
        ) {
   return is_var_cont ? ub : EPSFLOOR(ub);
}

template<class datatype>
datatype adjustLowerBound(const bool is_var_cont, const datatype lb) {
   return is_var_cont ? lb : EPSCEIL(lb);
}

template<class datatype>
bool isLbBetter(const datatype lb, const datatype ub, const datatype newlb) {
   assert(EPSLE(lb, ub));
   return EPSGT(newlb, lb);
}

template<class datatype>
bool isUbBetter(const datatype lb, const datatype ub, const datatype newub) {
   assert(EPSLE(lb, ub));
   return EPSLT(newub, ub);
}

template<class datatype>
bool isLbWorse(const datatype lb, const datatype ub, const datatype newlb) {
   assert(EPSLE(lb, ub));
   return EPSLT(newlb, lb);
}

template<class datatype>
bool isUbWorse(const datatype lb, const datatype ub, const datatype newub) {
   assert(EPSLE(lb, ub));
   return EPSGT(newub, ub);
}

template<class datatype>
NewBoundTuple<datatype>
tightenVarUpperBound(const datatype coeff, const datatype slack, const datatype surplus, const int num_inf_contr,
                     const datatype lb, const datatype ub,
                     const bool isVarCont) {

   NewBoundTuple<datatype> newb_tuple = {false, ub}; // output
   datatype newb;

   if (num_inf_contr == 0) {
      newb = coeff > 0 ? slack / coeff : surplus / coeff;
      newb += lb;
   } else if (num_inf_contr == 1 && ISNEGINF(lb)) {
      newb = coeff > 0 ? slack / coeff : surplus / coeff;
   } else {
      return newb_tuple;
   }

   newb = adjustUpperBound(isVarCont, newb);

   if (isUbBetter(lb, ub, newb)) {
      newb_tuple.is_tightened = true;
      newb_tuple.newb = newb;
      assert(EPSLE(lb, newb_tuple.newb));
      return newb_tuple;
   }

   return newb_tuple;
}

template<class datatype>
NewBoundTuple<datatype>
tightenVarLowerBound(const datatype coeff, const datatype slack, const datatype surplus, const int num_inf_contr,
                     const datatype lb, const datatype ub,
                     const bool isVarCont) {
   NewBoundTuple<datatype> newb_tuple = {false, lb}; // output
   datatype newb;

   if (num_inf_contr == 0) {
      newb = coeff > 0 ? surplus / coeff : slack / coeff;
      newb += ub;
   } else if (num_inf_contr == 1 && ISPOSINF(ub)) {
      newb = coeff > 0 ? surplus / coeff : slack / coeff;
   } else {
      return newb_tuple;
   }

   newb = adjustLowerBound(isVarCont, newb);

   if (isLbBetter(lb, ub, newb)) {
      newb_tuple.is_tightened = true;
      newb_tuple.newb = newb;
      assert(EPSLE(newb_tuple.newb, ub));
      return newb_tuple;
   }
   return newb_tuple;
}

template<class datatype>
bool canConsBeTightened(
        const datatype minact,
        const datatype maxact,
        const int numminactinf,
        const int nummaxactinf,
        const datatype lhs,
        const datatype rhs,
        const datatype maxactdelta
) {
   if (numminactinf > 1 && nummaxactinf > 1)
      return false;

   if (EPSLT(maxactdelta, GDP_INF)) {
      return !EPSLE(maxactdelta, min(rhs - minact, maxact - lhs));
   }

   return true;
}

template<class datatype>
bool isConsInfeasible(const datatype minactivity, const datatype maxactivity, const datatype rhs, const datatype lhs) {
   return (minactivity > rhs || maxactivity < lhs);
}

template<class datatype>
ActivitiesTuple computeActivities
        (
                const int considx,
                const int *col_indices,
                const int *row_indices,
                const datatype *vals,
                const datatype *ubs,
                const datatype *lbs
        ) {
   ActivitiesTuple actsTuple; // output

   double lb;
   double ub;
   double coeff;
   int val_idx;
   int var_idx;

   double maxactdelta = actsTuple.maxactdelta = 0.0;
   double minactivity = 0.0;
   double maxactivity = 0.0;
   int minact_inf = 0;
   int maxact_inf = 0;
   int n_vars_in_cons = row_indices[considx + 1] - row_indices[considx];

   FOLLOW_CONS_CALL(considx, printf("Printing constraint %d: ", considx));

   for (int var = 0; var < n_vars_in_cons; var++) {
      val_idx = row_indices[considx] + var;
      var_idx = col_indices[val_idx];

      coeff = vals[val_idx];
      lb = lbs[var_idx];
      ub = ubs[var_idx];

      FOLLOW_CONS_CALL(considx, printf("%9.2ex_%d [%9.2e, %9.2e] ", coeff, var_idx, lb, ub));

      maxactdelta = fabs(coeff) * (ub - lb);

      if (EPSGT(maxactdelta, actsTuple.maxactdelta))
         actsTuple.maxactdelta = maxactdelta;

      const int is_minac_inf = EPSGT(coeff, 0) ? ISNEGINF(lb) : ISPOSINF(ub);
      const int is_maxact_inf = EPSGT(coeff, 0) ? ISPOSINF(ub) : ISNEGINF(lb);
      minact_inf += is_minac_inf;
      maxact_inf += is_maxact_inf;

      if (is_minac_inf == 0) {
         minactivity += coeff > 0 ? coeff * lb : coeff * ub;
      }
      if (is_maxact_inf == 0) {
         maxactivity += coeff > 0 ? coeff * ub : coeff * lb;
      }
   }

   actsTuple.minact = minactivity;
   actsTuple.maxact = maxactivity;
   actsTuple.minact_inf = minact_inf;
   actsTuple.maxact_inf = maxact_inf;
   return actsTuple;
}

template<class datatype>
NewBounds<datatype> tightenVariable
        (
                const datatype coeff,
                const datatype lhs,
                const datatype rhs,
                const datatype minact,
                const datatype maxact,
                const int num_minact_inf,
                const int num_maxact_inf,
                const bool isVarCont,
                const datatype lb,
                const datatype ub
        ) {

   datatype slack = rhs - minact;
   datatype surplus = lhs - maxact;

   // initialize return data.
   NewBounds<datatype> newbds;
   newbds.lb = {false, lb};
   newbds.ub = {false, ub};

   if (coeff > 0.0) {
      if (EPSGT(coeff * (ub - lb), rhs - minact) && !ISPOSINF(rhs) && !ISNEGINF(minact)) {

         newbds.ub = tightenVarUpperBound(coeff, slack, surplus, num_minact_inf, lb, ub, isVarCont);
         // update data for lower bound tightening
//         if (newbds.ub.is_tightened) {
//            surplus = surplus + coeff * (ub - newbds.ub.newb);
//            ub = newbds.ub.newb;
//         }
      }

      if (EPSGT(coeff * (ub - lb), maxact - lhs) && !ISNEGINF(lhs) && !ISPOSINF(maxact)) {
         newbds.lb = tightenVarLowerBound(coeff, slack, surplus, num_maxact_inf, lb, ub, isVarCont);
      }
   } else {
      if (EPSGT(coeff * (lb - ub), rhs - minact) && !ISPOSINF(rhs) && !ISNEGINF(minact)) {

         newbds.lb = tightenVarLowerBound(coeff, slack, surplus, num_minact_inf, lb, ub, isVarCont);
         // update data for upper bound tightening
//         if (newbds.lb.is_tightened) {
//            surplus = surplus + coeff * (newbds.lb.newb - lb);
//            lb = newbds.lb.newb;
//         }
      }

      if (EPSGT(coeff * (lb - ub), maxact - lhs) && !ISNEGINF(lhs) && !ISPOSINF(maxact)) {
         newbds.ub = tightenVarUpperBound(coeff, slack, surplus, num_maxact_inf, lb, ub, isVarCont);
      }
   }
   return newbds;
}

template<class datatype>
void sequentialComputeActivities
        (
                const int n_cons,
                const int *col_indices,
                const int *row_indices,
                const datatype *vals,
                const datatype *ubs,
                const datatype *lbs,
                datatype *minacts,
                datatype *maxacts,
                int *minacts_inf,
                int *maxacts_inf,
                datatype *maxactdeltas
        ) {
   for (int considx = 0; considx < n_cons; considx++) {
      ActivitiesTuple activities = computeActivities(considx, col_indices, row_indices, vals, ubs, lbs);
      minacts[considx] = activities.minact;
      maxacts[considx] = activities.maxact;
      minacts_inf[considx] = activities.minact_inf;
      maxacts_inf[considx] = activities.maxact_inf;
      maxactdeltas[considx] = activities.maxactdelta;
   }
}

#endif