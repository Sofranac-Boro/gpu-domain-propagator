//
// Created by boro on 9/24/21.
//



#ifndef GPU_DOMAIN_PROPAGATOR_TEST_WEAKEST_BOUNDS_CUH
#define GPU_DOMAIN_PROPAGATOR_TEST_WEAKEST_BOUNDS_CUH

#include "../../src/progressMeasureMethods/weakest_bounds.h"

TEST_CASE("Weakest bounds example 1") {
   WeakestBoundsExample1<double> p;
//   AchterbergExample<double> p;

   p.fill_csc_matrix();

   computeWeakestBounds<double>(p.n_cons, p.n_vars, p.csr_vals, p.csr_col_indices, p.csr_row_ptrs, p.csc_vals, p.csc_col_ptrs, p.csc_row_indices,
                                  p.lhss, p.rhss, p.lbs, p.ubs, p.vartypes);


  p.compareAnalyticalSolution();

}

#endif //GPU_DOMAIN_PROPAGATOR_TEST_WEAKEST_BOUNDS_CUH
