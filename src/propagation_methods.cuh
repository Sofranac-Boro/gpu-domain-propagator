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
NewBoundTuple tightenVarUpperBound(const datatype coeff, const datatype slack, const datatype surplus, const int num_inf_contr, const datatype lb, const datatype ub,
                                   const bool isVarCont) {

   NewBoundTuple newb_tuple = {false, ub}; // output
   datatype newb;

   if (num_inf_contr == 0)
   {
      newb = EPSGT(coeff, 0)? slack / coeff : surplus / coeff;
      newb += lb;
   }
   else if (num_inf_contr == 1 && EPSLE(lb, -GDP_INF))
   {
      newb = EPSGT(coeff, 0)? slack / coeff : surplus / coeff;
   }
   else
   {
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
NewBoundTuple tightenVarLowerBound(const datatype coeff, const datatype slack, const datatype surplus, const int num_inf_contr, const datatype lb, const datatype ub,
                                   const bool isVarCont) {
   NewBoundTuple newb_tuple = {false, lb}; // output
   datatype newb;


   if (num_inf_contr == 0)
   {
      newb = EPSGT(coeff, 0)? surplus / coeff : slack / coeff;
      newb += ub;
   }
   else if (num_inf_contr == 1 && EPSLE(lb, -GDP_INF))
   {
      newb = EPSGT(coeff, 0)? surplus / coeff : slack / coeff;
   }
   else
   {
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

      const int is_minac_inf = EPSGT(coeff, 0) ? EPSLE(lb, -GDP_INF) : EPSGE(ub, GDP_INF);
      const int is_maxact_inf = EPSGT(coeff, 0) ? EPSGE(ub, GDP_INF) : EPSLE(lb, -GDP_INF);
      minact_inf += is_minac_inf;
      maxact_inf += is_maxact_inf;

      if (is_minac_inf == 0)
      {
         minactivity += EPSGT(coeff, 0) ? coeff * lb : coeff * ub;
      }
      if (is_maxact_inf == 0)
      {
         maxactivity += EPSGT(coeff, 0) ? coeff * ub : coeff * lb;
      }
   }

   actsTuple.minact = minactivity;
   actsTuple.maxact = maxactivity;
   actsTuple.minact_inf = minact_inf;
   actsTuple.maxact_inf = maxact_inf;
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
                const int num_minact_inf,
                const int num_maxact_inf,
                const bool isVarCont,
                datatype lb,
                datatype ub
        ) {

   datatype slack = rhs - minact;
   datatype surplus = lhs - maxact;

   // initialize return data.
   NewBounds newbds;
   newbds.lb = {false, lb};
   newbds.ub = {false, ub};

   if (EPSGT(coeff, 0.0)) {
      if (EPSGT(coeff * (ub - lb), rhs - minact) && EPSLT(rhs, GDP_INF) && EPSGT(minact, -GDP_INF)) {

         newbds.ub = tightenVarUpperBound(coeff, slack, surplus, num_minact_inf, lb, ub, isVarCont);
         // update data for lower bound tightening
//         if (newbds.ub.is_tightened) {
//            surplus = surplus + coeff * (ub - newbds.ub.newb);
//            ub = newbds.ub.newb;
//         }
      }

      if (EPSGT(coeff * (ub - lb), maxact - lhs) && EPSGT(lhs, -GDP_INF) && EPSLT(maxact, GDP_INF)) {
         newbds.lb = tightenVarLowerBound(coeff, slack, surplus, num_maxact_inf, lb, ub, isVarCont);
      }
   } else {
      if (EPSGT(coeff * (lb - ub), rhs - minact) && EPSLT(rhs, GDP_INF) && EPSGT(minact, -GDP_INF)) {

         newbds.lb = tightenVarLowerBound(coeff, slack, surplus, num_minact_inf, lb, ub, isVarCont);
         // update data for upper bound tightening
//         if (newbds.lb.is_tightened) {
//            surplus = surplus + coeff * (newbds.lb.newb - lb);
//            lb = newbds.lb.newb;
//         }
      }

      if (EPSGT(coeff * (lb - ub), maxact - lhs) && EPSGT(lhs, -GDP_INF) && EPSLT(maxact, GDP_INF)) {
         newbds.ub = tightenVarUpperBound(coeff, slack, surplus, num_maxact_inf, lb, ub, isVarCont);
      }
   }
   return newbds;
}

#endif