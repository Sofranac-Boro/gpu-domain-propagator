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
};
typedef struct ActivitiesTupleStruct ActivitiesTuple;

struct NewBoundTupleStruct {
    bool is_tightened;
    double newb;
};
typedef struct NewBoundTupleStruct NewBoundTuple;

struct NewBoundsStruct {
    NewBoundTuple lb;
    NewBoundTuple ub;
};
typedef struct NewBoundsStruct NewBounds;


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

   /* from SCIP, todo do we need this in the GPU version? */
   /* if lower bound is moved to 0 or higher, always accept bound change */
   //if (lb < 0.0 && newlb >= 0.0)
   //   return true;

   return EPSGT(newlb, lb);
}

template<class datatype>
bool isUbBetter(const datatype lb, const datatype ub, const datatype newub) {

   assert(EPSLE(lb, ub));

   /* from SCIP, todo do we need this in the GPU version? */
   /* if upper bound is moved to 0 or lower, always accept bound change */
   //if (ub > 0.0 && newub <= 0.0)
   //   return true;

   return EPSLT(newub, ub);
}

template<class datatype>
NewBoundTuple tightenVarUpperBound(const datatype coeff, const datatype slack, const datatype lb, const datatype ub,
                                   const bool isVarCont) {
   NewBoundTuple newb_tuple = {false, ub}; // output

   datatype newb = lb + (slack / fabs(coeff));
   newb = adjustUpperBound(isVarCont, newb);

   if (isUbBetter(lb, ub, newb)) {
      newb_tuple.is_tightened = true;
      newb_tuple.newb = newb;
      return newb_tuple;
   }
   return newb_tuple;
}

template<class datatype>
NewBoundTuple tightenVarLowerBound(const datatype coeff, const datatype surplus, const datatype lb, const datatype ub,
                                   const bool isVarCont) {
   NewBoundTuple newb_tuple = {false, lb}; // output

   datatype newb = ub - (surplus / fabs(coeff));
   newb = adjustLowerBound(isVarCont, newb);
   if (isLbBetter(lb, ub, newb)) {
      newb_tuple.is_tightened = true;
      newb_tuple.newb = newb;
      return newb_tuple;
   }
   return newb_tuple;
}

template<class datatype>
bool canConsBeTightened(const datatype slack, const datatype surplus, const datatype maxactdelta) {
   return !EPSLE(maxactdelta, MIN(slack, surplus));
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
   int n_vars_in_cons = row_indices[considx + 1] - row_indices[considx];

   for (int var = 0; var < n_vars_in_cons; var++) {
      val_idx = row_indices[considx] + var;
      var_idx = col_indices[val_idx];

      coeff = vals[val_idx];
      lb = lbs[var_idx];
      ub = ubs[var_idx];

      maxactdelta = fabs(coeff) * (ub - lb);

      if (EPSGT(maxactdelta, actsTuple.maxactdelta))
         actsTuple.maxactdelta = maxactdelta;
      if (considx == 7)
         printf("considx: %d, varidx: %d, coeff: %9.2e, lb: %9.2e, ub: %9.2e\n", considx, var_idx, coeff, lb, ub);

      minactivity += EPSGT(coeff, 0.0) ? coeff * lb : coeff * ub;
      maxactivity += EPSGT(coeff, 0.0) ? coeff * ub : coeff * lb;

   }
   actsTuple.minact = minactivity;
   actsTuple.maxact = maxactivity;

   return actsTuple;
}

template<class datatype>
NewBounds tightenVariable
        (
                const datatype coeff,
                const datatype lhs,
                const datatype rhs,
                const datatype minact,
                const datatype maxact,
                const bool isVarCont,
                const int var_idx,
                const int val_idx,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                const datatype *lbs,
                const datatype *ubs
        ) {

   datatype ub = ubs[var_idx];
   datatype lb = lbs[var_idx];
   datatype slack = rhs - minact;
   datatype surplus = maxact - lhs;

   // initialize return data.
   NewBounds newbds;
   newbds.lb = {false, lb};
   newbds.ub = {false, ub};


   if (EPSGT(coeff, 0.0)) {
      if (EPSGT(coeff * (ub - lb), slack) && EPSLT(rhs, GDP_INF) && EPSGT(minact, -GDP_INF)) {
         newbds.ub = tightenVarUpperBound(coeff, slack, lb, ub, isVarCont);
         // update data for lower bound tightening
         if (newbds.ub.is_tightened) {
            surplus = surplus - coeff * (ub - newbds.ub.newb);
            ub = newbds.ub.newb;
         }
      }

      if (EPSGT(coeff * (ub - lb), surplus) && EPSGT(lhs, -GDP_INF) && EPSLT(maxact, GDP_INF)) {
         newbds.lb = tightenVarLowerBound(coeff, surplus, lb, ub, isVarCont);
      }
   } else {
      if (EPSGT(coeff * (lb - ub), slack) && EPSLT(rhs, GDP_INF) && EPSGT(minact, -GDP_INF)) {
         newbds.lb = tightenVarLowerBound(coeff, slack, lb, ub, isVarCont);
         // update data for upper bound tightening
         if (newbds.lb.is_tightened) {
            surplus = surplus - coeff * (newbds.lb.newb - lb);
            lb = newbds.lb.newb;
         }
      }

      if (EPSGT(coeff * (lb - ub), surplus) && EPSGT(lhs, -GDP_INF) && EPSLT(maxact, GDP_INF)) {
         newbds.ub = tightenVarUpperBound(coeff, surplus, lb, ub, isVarCont);
      }
   }
   return newbds;
}

#endif