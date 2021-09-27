#ifndef GPU_DOMAIN_PROPAGATOR_PROGRESS_FUNCTION_H
#define GPU_DOMAIN_PROPAGATOR_PROGRESS_FUNCTION_H

template<typename datatype>
struct ProgressMeasure {
    double P_fin;
    int P_inf;
};

// Normalizes the score to a value between 0 and 100
template<typename datatype>
__forceinline__ datatype normalizeTo0_100(
        const datatype val,
        const datatype max_val
) {
   return EPSGT(max_val, 0.0) ? (100.0 * datatype(val)) / datatype(max_val) : 0.0;
}

template<typename datatype>
void initMeasureData
        (
                const int n_cons,
                const int n_vars,
                const int nnz,
                const datatype *lbs,
                const datatype *ubs,
                const int *csr_col_indices,
                const int *csr_row_ptrs,
                const datatype *csr_vals,
                const datatype *lhss,
                const datatype *rhss,
                const GDP_VARTYPE *vartypes,
                const int *csc_row_indices,
                const int *csc_col_ptrs,
                datatype *lbs_weakest,
                datatype *ubs_weakest,
                datatype *lbs_limit,
                datatype *ubs_limit
        ) {
   memcpy(lbs_weakest, lbs, n_vars * sizeof(datatype));
   memcpy(ubs_weakest, ubs, n_vars * sizeof(datatype));

   printf("\n====   Running the weakest bounds procedure  ====");
   computeWeakestBounds<datatype>(n_cons, n_vars, csr_col_indices, csr_row_ptrs, csc_col_ptrs, csc_row_indices,
                                  csr_vals,
                                  lhss, rhss, lbs_weakest, ubs_weakest, vartypes);
   DEBUG_CALL(checkWeakestBoundsResult<datatype>(n_vars, lbs, ubs, lbs_weakest, ubs_weakest) );
   printf("====   end weakest bounds procedure  ====");
   // run sequnetial propagator to get limit bounds
   memcpy(lbs_limit, lbs, n_vars * sizeof(datatype));
   memcpy(ubs_limit, ubs, n_vars * sizeof(datatype));

   printf("\n\n==== propagator run for limit poitns ====");
   sequentialPropagate<datatype>(n_cons, n_vars, nnz, csr_col_indices, csr_row_ptrs, csr_vals, lhss, rhss, lbs_limit,
                                 ubs_limit, vartypes);
   printf("==== end limit points ====\n");
}

template<typename datatype>
ProgressMeasure<datatype>  calcProgressMeasureVar(
        const datatype lb,
        const datatype ub,
        const datatype lb_weakest,
        const datatype ub_weakest,
        const datatype lb_limit,
        const datatype ub_limit,
        const datatype lb_prevround, // need this for the infinite reductions measure
        const datatype ub_prevround // need this for the infinite reductions measure
) {
   ProgressMeasure<datatype> P_var = {.P_fin=0.0, .P_inf=0};

   //  lb <= lb_limit and  ub_limit <= ub
   // lb_weakest<= lb_limit and ub_limit <= ub_weakest
   assert(EPSGE(lb_limit, lb));
   assert(EPSLE(ub_limit, ub));
   assert(EPSLE(lb_weakest, lb_limit));
   assert(EPSGE(ub_weakest, ub_limit));
   // lb_weakest <= ub_initial and lb <= ub and lb_weakest <= ub_limit
   assert(EPSLE(lb_weakest, ub_weakest));
   assert(EPSLE(lb, ub));
   assert(EPSLE(lb_limit, ub_limit));

   // if limit bound is finite, so should be the weakest value.
   DEBUG_CALL(EPSGT(lb_limit, -GDP_INF) ? assert(EPSGT(lb_weakest, -GDP_INF)) : assert(true));
   DEBUG_CALL(EPSLT(ub_limit, GDP_INF) ? assert(EPSLT(ub_weakest, GDP_INF)) : assert(true));

   // if weakest bound is inf, it means it should never be possible to get a finite value for this bound
   DEBUG_CALL(EPSLE(lb_weakest, -GDP_INF) ? assert(EPSLE(lb, -GDP_INF)) : assert(true));
   DEBUG_CALL(EPSGE(ub_weakest, GDP_INF) ? assert(EPSGE(ub, GDP_INF)) : assert(true));

   // if weakest bound is finite, it should never be possible to get a "worse" finite value of the bound
   DEBUG_CALL(EPSGT(lb_weakest, -GDP_INF) && EPSGT(lb, -GDP_INF) ? assert(EPSGE(lb, lb_weakest)) : assert(true));
   DEBUG_CALL(EPSLT(ub_weakest, GDP_INF) && EPSLT(ub, GDP_INF) ? assert(EPSLE(ub, ub_weakest)) : assert(true));

   // measure contribution of finite domain reducitons
   datatype increment;
   // Checking that current bound is finite prevents numerical issues even though the math works in infinite space.
   // Do not try to compute case where start value is infinite. This means that limit value is also infinite, see above.
   // lb >= lb_weakest is reduntant here because of the assert but added for readibility as it is present in the paper.
   // if limit == weakest bound, no progress can be made.
   if (EPSGT(lb_weakest, -GDP_INF) && EPSGT(lb, -GDP_INF) && EPSGE(lb, lb_weakest) && !EPSEQ(lb_limit, lb_weakest)) {
      increment = (lb - lb_weakest) / (lb_limit - lb_weakest);
      assert(EPSLE(increment, 1.0));
      P_var.P_fin += increment;
   }
   // Upper bound
   if (EPSLT(ub_weakest, GDP_INF) && EPSLT(ub, GDP_INF) && EPSLE(ub, ub_weakest) && !(EPSEQ(ub_weakest, ub_limit))) {
      increment = (ub_weakest - ub) / (ub_weakest - ub_limit);
      assert(EPSLE(increment, 1.0));
      P_var.P_fin += increment;
   }

   // measure contribution to infinite-finite progress
   if (EPSLE(lb_prevround, -GDP_INF) && EPSGT(lb, -GDP_INF)) {
      P_var.P_inf += 1;
   }

   if (EPSGE(ub_prevround, GDP_INF) && EPSLT(ub, GDP_INF)) {
      P_var.P_inf+=1;
   }

   return P_var;
}

template<typename datatype>
ProgressMeasure<datatype> calcMaxMeasureValues
        (
                const int n_vars,
                const datatype *lbs_input,
                const datatype *ubs_input,
                const datatype *lbs_weakest,
                const datatype *ubs_weakest,
                const datatype *lbs_limit,
                const datatype *ubs_limit
        ) {

   ProgressMeasure<datatype> P_max = {.P_fin=0.0, .P_inf=0};

   for (int i = 0; i < n_vars; i++) {
      if (EPSGT(lbs_limit[i], -GDP_INF) && !EPSEQ(lbs_limit[i], lbs_weakest[i])) {
         assert(EPSLE(lbs_weakest[i], lbs_limit[i]));
         P_max.P_fin += 1.0;
      }

      if (EPSLT(ubs_limit[i], GDP_INF) && !EPSEQ(ubs_limit[i], ubs_weakest[i])) {
         assert(EPSGE(ubs_weakest[i], ubs_limit[i]));
         P_max.P_fin += 1.0;
      }

      if (EPSGT(lbs_limit[i], -GDP_INF) && EPSLE(lbs_input[i], -GDP_INF)) {
         P_max.P_inf += 1;
      }
      if (EPSLT(ubs_limit[i], GDP_INF) && EPSGE(ubs_input[i], GDP_INF)) {
         P_max.P_inf += 1;
      }
   }
   VERBOSE_CALL_2(printf("\nMaximum Measure values: P_fin=%.2f, P_inf=%d\n", P_max.P_fin, P_max.P_inf));
   return P_max;
}

template<typename datatype>
int measureAndPrintProgress(
        const int round,
        const int n_vars,
        const datatype *lbs,
        const datatype *ubs,
        const datatype *lbs_weakest,
        const datatype *ubs_weakest,
        const datatype *lbs_limit,
        const datatype *ubs_limit,
        const datatype *lbs_prev,
        const datatype *ubs_prev,
        const ProgressMeasure<datatype> P_max,
        const int P_inf_total
        )
{
   ProgressMeasure<datatype> P = {.P_fin = 0.0, .P_inf = 0};

   for (int varidx = 0; varidx < n_vars; varidx++) {
      const ProgressMeasure<datatype> P_tmp = calcProgressMeasureVar<datatype>(lbs[varidx], ubs[varidx], lbs_weakest[varidx], ubs_weakest[varidx],
                                    lbs_limit[varidx], ubs_limit[varidx], lbs_prev[varidx], ubs_prev[varidx]);
      P.P_fin += P_tmp.P_fin;
      P.P_inf += P_tmp.P_inf;

      assert(EPSLE(P.P_fin, P_max.P_fin) && EPSLE(P.P_inf, P_max.P_inf));
   }

   printf("round=%d, P_fin=%.10f, P_inf=%.2f\n", round, normalizeTo0_100<datatype>(P.P_fin, P_max.P_fin), normalizeTo0_100<datatype>(P.P_inf + P_inf_total, P_max.P_inf));
   return P.P_inf;
}


#endif //GPU_DOMAIN_PROPAGATOR_PROGRESS_FUNCTION_H
