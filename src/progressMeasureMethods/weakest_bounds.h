#ifndef GPU_DOMAIN_PROPAGATOR_WEAKEST_BOUNDS_H
#define GPU_DOMAIN_PROPAGATOR_WEAKEST_BOUNDS_H

#include "../propagators/sequential_propagator.h"
#include <vector>
#include <iostream>
#include <exception>
#include "../misc.h"

using namespace std;

template<typename datatype>
struct NewWeakestBound {
    int bd_idx;
    char type;
    datatype newb;
    bool is_valid= true; // sometimes the bound should not be applied. E.g. to init recursion
};

template<typename datatype>
struct ConsIterator {
    const datatype* vals;
    const int* vars;
    const int size;
};

enum ActivityType {
    MAXACTIVITY = 0,
    MINACTIVITY = 1,
};
typedef enum ActivityType ACTIVITY_TYPE;

template<typename datatype>
void checkWeakestBoundsResult(
        const int n_vars,
        const datatype* lbs_orig,
        const datatype* ubs_orig,
        const datatype* lbs_w,
        const datatype* ubs_w
)
{
   for (int j=0; j<n_vars; j++)
   {
      // if weakest is inf, then original also has to be inf
      // if weakest not inf and orig not inf, then they have to be equal

      if (EPSLE(lbs_w[j], -GDP_INF))
      {
         assert(EPSLE(lbs_orig[j], -GDP_INF));
      } else {
         if (EPSGT(lbs_orig[j], -GDP_INF))
         {
            assert(EPSEQ(lbs_w[j], lbs_orig[j]));
         }
      }

      if (EPSGE(ubs_w[j], GDP_INF))
      {
         assert(EPSGE(ubs_orig[j], GDP_INF));
      } else {
         if (EPSLT(ubs_orig[j], GDP_INF))
         {
            assert(EPSEQ(ubs_w[j], ubs_orig[j]));
         }
      }
   }
}

/* METHODS FOR THE OLD WEKAEST BOUNDS PROCEDURE FROM THE CP2021 PAPER */

template<class datatype>
NewBoundTuple<datatype>
tightenVarWeakUpperBound(const datatype coeff, const datatype slack, const datatype surplus, const int num_inf_contr,
                     const datatype lb, const datatype ub,
                     const bool isVarCont) {

   NewBoundTuple<datatype> newb_tuple = {false, ub}; // output
   datatype newb;

   if (num_inf_contr == 0) {
      newb = EPSGT(coeff, 0) ? slack / coeff : surplus / coeff;
      newb += lb;
   } else if (num_inf_contr == 1 && EPSLE(lb, -GDP_INF)) {
      newb = EPSGT(coeff, 0) ? slack / coeff : surplus / coeff;
   } else {
      return newb_tuple;
   }

   newb = adjustUpperBound(isVarCont, newb);

   if (EPSLT(newb, GDP_INF) && (isUbWorse(lb, ub, newb) || EPSGE(ub, GDP_INF))) {
      newb_tuple.is_tightened = true;
      newb_tuple.newb = newb;
      assert(EPSLE(lb, newb_tuple.newb));
      return newb_tuple;
   }

   return newb_tuple;
}


template<class datatype>
NewBoundTuple<datatype>
tightenVarWeakLowerBound(const datatype coeff, const datatype slack, const datatype surplus, const int num_inf_contr,
                     const datatype lb, const datatype ub,
                     const bool isVarCont) {
   NewBoundTuple<datatype> newb_tuple = {false, lb}; // output
   datatype newb;


   if (num_inf_contr == 0) {
      newb = EPSGT(coeff, 0) ? surplus / coeff : slack / coeff;
      newb += ub;
   } else if (num_inf_contr == 1 && EPSLE(lb, -GDP_INF)) {
      newb = EPSGT(coeff, 0) ? surplus / coeff : slack / coeff;
   } else {
      return newb_tuple;
   }

   newb = adjustLowerBound(isVarCont, newb);
   if (EPSGT(newb, -GDP_INF) && (isLbWorse(lb, ub, newb) || EPSLE(lb, -GDP_INF))) {
      newb_tuple.is_tightened = true;
      newb_tuple.newb = newb;
      assert(EPSLE(newb_tuple.newb, ub));
      return newb_tuple;
   }
   return newb_tuple;
}

template<class datatype>
NewBounds<datatype> tightenWeakestVariable
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

   if (EPSGT(coeff, 0.0)) {
      if (EPSLT(rhs, GDP_INF) && EPSGT(minact, -GDP_INF)) {
         newbds.ub = tightenVarWeakUpperBound(coeff, slack, surplus, num_minact_inf, lb, ub, isVarCont);
      }

      if (EPSGT(lhs, -GDP_INF) && EPSLT(maxact, GDP_INF)) {
         newbds.lb = tightenVarWeakLowerBound(coeff, slack, surplus, num_maxact_inf, lb, ub, isVarCont);
      }
   } else {
      if (EPSLT(rhs, GDP_INF) && EPSGT(minact, -GDP_INF)) {

         newbds.lb = tightenVarWeakLowerBound(coeff, slack, surplus, num_minact_inf, lb, ub, isVarCont);
      }

      if (EPSGT(lhs, -GDP_INF) && EPSLT(maxact, GDP_INF)) {
         newbds.ub = tightenVarWeakUpperBound(coeff, slack, surplus, num_maxact_inf, lb, ub, isVarCont);
      }
   }
   return newbds;
}

template<class datatype>
bool preprocessorPropagationRound
        (
                const int n_cons,
                const int n_vars,
                const int *col_indices,
                const int *row_indices,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                const datatype *vals,
                const datatype *lhss,
                const datatype *rhss,
                datatype *lbs,
                datatype *ubs,
                const GDP_VARTYPE *vartypes,
                datatype *minacts,
                datatype *maxacts,
                int *minacts_inf,
                int *maxacts_inf,
                datatype *maxactdeltas,
                int *consmarked,
                const bool* lbs_inf,
                const bool* ubs_inf
        ) {

   datatype coeff;
   bool isVarCont;
   datatype rhs;
   datatype lhs;

   bool change_found = false;
   int val_idx;
   int varidx;

   int *consmarked_nextround = (int *) calloc(n_cons, sizeof(int));

   for (int considx = 0; considx < n_cons; considx++) {
      if (consmarked[considx] == 1 && (minacts_inf[considx] <= 1 || maxacts_inf[considx] <= 1)) {

         //DEBUG_CALL( printf("cons %d: minact: %.5f, maxact: %.5f\n", considx, minacts[considx], maxacts[considx]) );
            rhs = rhss[considx];
            lhs = lhss[considx];
            int num_vars_in_cons = row_indices[considx + 1] - row_indices[considx];

            for (int var = 0; var < num_vars_in_cons; var++) {
               val_idx = row_indices[considx] + var;
               varidx = col_indices[val_idx];
               coeff = vals[val_idx];

               isVarCont = vartypes[varidx] == GDP_CONTINUOUS;

               NewBounds<datatype> newbds = tightenWeakestVariable<datatype>
                       (
                               coeff, lhs, rhs, minacts[considx], maxacts[considx], minacts_inf[considx],
                               maxacts_inf[considx], isVarCont, lbs[varidx], ubs[varidx]
                       );

               // if the new bound is finite
               if (newbds.lb.is_tightened && EPSGT(newbds.lb.newb, -GDP_INF) && lbs_inf[varidx]) {
                  // it could happen that some other constraint in the system already found a finite tightening for this var.
                  // In this case, only update if the new finite bound is "worse"
                  if (EPSLE(lbs[varidx], -GDP_INF) ||
                      (EPSGT(lbs[varidx], -GDP_INF) && EPSLT(newbds.lb.newb, lbs[varidx]))) {
                     FOLLOW_VAR_CALL(varidx,
                                     printf("preprocessor lb change found: varidx: %7d, oldlb: %9.2e, newlb: %9.2e\n",
                                            varidx, lbs[varidx],
                                            newbds.lb.newb));

                     lbs[varidx] = newbds.lb.newb;
                     change_found = true;
                     markConstraints(varidx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
                  }
               }

               // if the new bound is finite
               if (newbds.ub.is_tightened && EPSLT(newbds.ub.newb, GDP_INF) && ubs_inf[varidx]) {
                  // it could happen that some other constraint in the system already found a finite tightening for this var.
                  // In this case, only update if the new finite bound is "worse"
                  if (EPSGE(ubs[varidx], GDP_INF) ||
                      (EPSLT(ubs[varidx], GDP_INF) && EPSGT(newbds.ub.newb, ubs[varidx]))) {
                     FOLLOW_VAR_CALL(varidx,
                                     printf("preprocessor ub change found: varidx: %7d, oldub: %9.2e, newub: %9.2e\n",
                                            varidx, ubs[varidx],
                                            newbds.ub.newb));
                     ubs[varidx] = newbds.ub.newb;
                     change_found = true;
                     markConstraints(varidx, csc_col_ptrs, csc_row_indices, consmarked_nextround);
                  }
               }
            }
         }
      }

   // copy data from consmarked_nextround into consmarked
   memcpy(consmarked, consmarked_nextround, n_cons * sizeof(int));
   free(consmarked_nextround);
   return change_found;
}

template<class datatype>
void executeWeakestBounds
        (
                const int n_cons,
                const int n_vars,
                const int *col_indices,
                const int *row_indices,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                const datatype *vals,
                const datatype *lhss,
                const datatype *rhss,
                datatype *lbs,
                datatype *ubs,
                const GDP_VARTYPE *vartypes
        ) {

   DEBUG_CALL(checkInput<datatype>(n_cons, n_vars, row_indices[n_cons], vals, lhss, rhss, lbs, ubs, vartypes));

   datatype *minacts = (datatype *) calloc(n_cons, sizeof(datatype));
   datatype *maxacts = (datatype *) calloc(n_cons, sizeof(datatype));
   int *minacts_inf = (int *) calloc(n_cons, sizeof(int));
   int *maxacts_inf = (int *) calloc(n_cons, sizeof(int));
   datatype *maxactdeltas = (datatype *) calloc(n_cons, sizeof(datatype));
   int *consmarked = (int *) calloc(n_cons, sizeof(int));

   bool* lbs_inf = (bool*) SAFEMALLOC(n_vars*sizeof(bool));
   bool* ubs_inf = (bool*) SAFEMALLOC(n_vars*sizeof(bool));

   // all cons marked for propagation in the first round
   for (int i = 0; i < n_cons; i++)
      consmarked[i] = 1;

   for (int j=0; j<n_vars; j++)
   {
      lbs_inf[j] = EPSLE(lbs[j], -GDP_INF);
      ubs_inf[j] = EPSGE(ubs[j], GDP_INF);
   }

#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif

   VERBOSE_CALL(printf("\ncpu_seq_dis execution start... Params: MAXNUMROUNDS: %d", MAX_NUM_ROUNDS));

   bool change_found = true;
   int prop_round;
   for (prop_round = 1; prop_round <= MAX_NUM_ROUNDS && change_found; prop_round++)  // maxnumrounds = 100
   {
      VERBOSE_CALL_2(printf("\nPropagation round: %d", prop_round));

      sequentialComputeActivities<datatype>(n_cons, col_indices, row_indices, vals, ubs, lbs, minacts, maxacts,
                                            minacts_inf, maxacts_inf,
                                            maxactdeltas);

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("preprocessor varidx: %7d bounds before round: %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR,
                             prop_round, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));

      change_found = preprocessorPropagationRound<datatype>
              (
                      n_cons, n_vars, col_indices, row_indices, csc_col_ptrs, csc_row_indices, vals, lhss, rhss,
                      lbs, ubs, vartypes, minacts, maxacts, minacts_inf, maxacts_inf,
                      maxactdeltas, consmarked, lbs_inf, ubs_inf
              );

      FOLLOW_VAR_CALL(FOLLOW_VAR,
                      printf("preprocessor varidx: %7d bounds after round: %7d: lb: %9.2e, ub: %9.2e\n", FOLLOW_VAR,
                             prop_round, lbs[FOLLOW_VAR],
                             ubs[FOLLOW_VAR]));
   }

   VERBOSE_CALL(printf("\ncpu_seq_dis propagation done. Num rounds: %d\n", prop_round - 1));
   VERBOSE_CALL(measureTime("cpu_seq_dis", start, std::chrono::steady_clock::now()));

   free(minacts);
   free(maxacts);
   free(minacts_inf);
   free(maxacts_inf);
   free(maxactdeltas);
   free(consmarked);
   free(lbs_inf);
   free(ubs_inf);
}

/* METHODS FOR THE NEW WEAKEST BOUNDS PROCEDURE */

template<typename datatype>
class ConstProbData{
public:
    int n_vars;
    int n_cons;
    const datatype* vals;
    const int* col_indices;
    const int* row_ptrs;
    const datatype* lhss;
    const datatype* rhss;
    const GDP_VARTYPE* vartypes;

    ConstProbData(const int n_cons_, const int n_vars_, const datatype* vals_, const int* col_indices_, const int* row_ptrs_, const datatype* lhss_, const datatype* rhss_, const GDP_VARTYPE* vartypes_) {
       n_cons = n_cons_;
       n_vars = n_vars_;
       vals = vals_;
       col_indices = col_indices_;
       row_ptrs = row_ptrs_;
       lhss = lhss_;
       rhss = rhss_;
       vartypes = vartypes_;
    }

    ConsIterator<datatype> getConsIterator(const int cons) const
    {
       ConsIterator<datatype> it = {
               .vals = &vals[row_ptrs[cons]],
               .vars = &col_indices[row_ptrs[cons]],
               .size = row_ptrs[cons+1] - row_ptrs[cons]
       };
       return it;
    }
   };

template<typename datatype>
class Bounds {
public:
    int n_vars;
    vector<datatype> lbs;
    vector<datatype> ubs;

    Bounds(const int n_vars_, const datatype* lbs_, const datatype* ubs_)
    {
       n_vars = n_vars_;

       lbs.reserve(n_vars);
       ubs.reserve(n_vars);

       lbs.insert(lbs.end(), &lbs_[0], &lbs_[n_vars_]);
       ubs.insert(ubs.end(), &ubs_[0], &ubs_[n_vars_]);
    }

    Bounds(const Bounds& bds1)
    {
       n_vars = bds1.n_vars;
       lbs = bds1.lbs;
       ubs = bds1.ubs;
    }

    void updateWeakestBound(NewWeakestBound<datatype> new_wbd)
    {
       if (new_wbd.is_valid) {
          const int j = new_wbd.bd_idx;
          const bool is_lower = new_wbd.type == 'l';
          const datatype newb = new_wbd.newb;

          assert((!is_lower && EPSLT(newb, GDP_INF) || (is_lower && EPSGT(newb, -GDP_INF))));

          // old is infinite and new finite OR old is finite and new is weaker
          if (is_lower && ( EPSLE(lbs[j], -GDP_INF) || ( EPSGT(lbs[j], -GDP_INF) && EPSLT(newb, lbs[j])) ) ) {
             lbs[j] = newb;
          }
          if (!is_lower && ( EPSGE(ubs[j], GDP_INF) || ( EPSLT(ubs[j], GDP_INF) && EPSGT(new_wbd.newb, ubs[j]) ) )) {
             ubs[new_wbd.bd_idx] = new_wbd.newb;
          }
       }
    }
};

template<typename datatype>
class Activities{
public:
    vector<datatype> minacts;
    vector<datatype> maxacts;
    vector<int> minacts_inf;
    vector<int> maxacts_inf;

    explicit Activities(const int n_cons)
    {
       minacts.resize(n_cons);
       maxacts.resize(n_cons);
       minacts_inf.resize(n_cons);
       maxacts_inf.resize(n_cons);
    }

    Activities(const Activities& acts1)
    {
       minacts = acts1.minacts;
       maxacts = acts1.maxacts;
       minacts_inf = acts1.minacts_inf;
       maxacts_inf = acts1.maxacts_inf;
    }

    void updateActivities(const ConstProbData<datatype>& p, const Bounds<datatype>& bds)
    {
       for (int considx = 0; considx < p.n_cons; considx++) {
          ActivitiesTuple activities = computeActivities(considx, p.col_indices, p.row_ptrs, p.vals, bds.ubs.data(), bds.lbs.data());
          minacts[considx] = activities.minact;
          maxacts[considx] = activities.maxact;
          minacts_inf[considx] = activities.minact_inf;
          maxacts_inf[considx] = activities.maxact_inf;
       }
    }

    void printActivities()
    {
       printVector(minacts, "minacts");
       printVector(maxacts, "maxacts");
    }
};

template<typename datatype>
NewWeakestBound<datatype> computeInfiniteReduction(const int cons, const ConstProbData<datatype>& p, const Bounds<datatype>& bds, const Activities<datatype>& acts, ACTIVITY_TYPE type)
{
   NewWeakestBound<datatype> wbd;

   const ConsIterator<datatype> it = p.getConsIterator(cons);
   for (int i = 0; i < it.size; i++)
   {
      const datatype coeff = it.vals[i];
      const int var_idx = it.vars[i];
      const datatype lb = bds.lbs[var_idx];
      const datatype ub = bds.ubs[var_idx];

      const bool is_minact = type == MINACTIVITY;
      const bool is_coeff_neg = EPSLT(coeff, 0.0);

      // check if this variable is the infinity contribution in this constraint
      bool is_inf_contr;
      if (is_minact)
      {
         is_inf_contr = EPSGT(coeff, 0) ? EPSLE(lb, -GDP_INF) : EPSGE(ub, GDP_INF);
      }
      else
      {
         is_inf_contr = EPSGT(coeff, 0) ? EPSGE(ub, GDP_INF) : EPSLE(lb, -GDP_INF);
      }

      if (is_inf_contr)
      {
         datatype slack = p.rhss[cons] - acts.minacts[cons];
         datatype surplus = p.lhss[cons] - acts.maxacts[cons];

         if ((( is_minact && is_coeff_neg ) || ( !is_minact && !is_coeff_neg )) && EPSLE(lb, -GDP_INF))
         {
            // lower bound is the infinity contributor

            assert(
                    (EPSGT(coeff, 0.0) && EPSGT(p.lhss[cons], -GDP_INF) && EPSLT(acts.maxacts[cons], GDP_INF)) ||
                    (EPSLT(coeff, 0.0) && EPSLT(p.rhss[cons], GDP_INF) && EPSGT(acts.minacts[cons], -GDP_INF))
                    );

            datatype newlb = EPSGT(coeff, 0) ? surplus / coeff : slack / coeff;
            newlb = adjustLowerBound(p.vartypes[var_idx] == GDP_CONTINUOUS, newlb);
            assert(EPSLE(lb, -GDP_INF) && EPSGT(newlb, -GDP_INF));
            assert(EPSLE(newlb, ub));
            wbd.newb = newlb;
            wbd.type = 'l';
            wbd.bd_idx = var_idx;
            return wbd;
         }
         else  if ((( !is_minact && is_coeff_neg ) || ( is_minact && !is_coeff_neg )) && EPSGE(ub, GDP_INF)) {
            // upper bound is the infinity contributor
            assert(
                    (EPSGT(coeff, 0.0) &&  EPSLT(p.rhss[cons], GDP_INF) && EPSGT(acts.minacts[cons], -GDP_INF) ) ||
                    (EPSLT(coeff, 0.0) && EPSGT(p.lhss[cons], -GDP_INF) && EPSLT(acts.maxacts[cons], GDP_INF) )
            );

            datatype newub = EPSGT(coeff, 0) ? slack / coeff : surplus / coeff;
            newub = adjustUpperBound(p.vartypes[var_idx] == GDP_CONTINUOUS, newub);
            assert(EPSGE(ub, GDP_INF) && EPSLT(newub, GDP_INF));
            assert(EPSLE(lb, newub));
            wbd.newb = newub;
            wbd.type = 'u';
            wbd.bd_idx = var_idx;
            return wbd;
         }
      }
   }
   // no tightening found.
   wbd.is_valid = false;
   return wbd;
}


template<typename datatype>
void weakestBoundsAlgorithm(const ConstProbData<datatype>& p, Bounds<datatype>& global_wbds, Bounds<datatype> bds, Activities<datatype> acts, NewWeakestBound<datatype> new_wbd)
{
   // todo cons marking, after correctness has been proven
   bds.updateWeakestBound(new_wbd);
   global_wbds.updateWeakestBound(new_wbd);

   acts.updateActivities(p, bds);


   // check for cons with exactly one infinity contribution in the minimum or maximum activity
   for (int cons=0; cons < p.n_cons; cons++)
   {


      if (acts.minacts_inf[cons] == 1 && EPSLT(p.rhss[cons], GDP_INF))
      {
         // each of these becomes a new subproblem
         NewWeakestBound<datatype> wbd = computeInfiniteReduction(cons, p, bds, acts, MINACTIVITY);
         if (wbd.is_valid)
         {
            weakestBoundsAlgorithm(p,global_wbds,bds,acts,wbd);
         }

      }

      if (acts.maxacts_inf[cons] == 1 && EPSGT(p.lhss[cons], -GDP_INF))
      {
         // each of these becomes a new subproblem
         NewWeakestBound<datatype> wbd1 = computeInfiniteReduction(cons, p, bds, acts, MAXACTIVITY);
         if (wbd1.is_valid)
         {
            weakestBoundsAlgorithm(p,global_wbds,bds,acts,wbd1);
         }
      }
   }
}

template<class datatype>
void computeWeakestBounds
        (
                const int n_cons,
                const int n_vars,
                const int *col_indices,
                const int *row_ptrs,
                const int *csc_col_ptrs,
                const int *csc_row_indices,
                const datatype *vals,
                const datatype *lhss,
                const datatype *rhss,
                datatype *lbs,
                datatype *ubs,
                const GDP_VARTYPE *vartypes
        ) {
   const ConstProbData<datatype> p(n_cons, n_vars, vals, col_indices, row_ptrs, lhss, rhss, vartypes);
   Bounds<datatype> bds(n_vars, lbs, ubs);
   Activities<datatype> acts(n_cons);

   // need a bound to start the recursive calls
   NewWeakestBound<datatype> wbd;
   wbd.is_valid=false;
   // copy of bounds to keep best (weakest) solutions
   Bounds<datatype> global_wbds = bds;
#ifdef VERBOSE
   auto start = std::chrono::steady_clock::now();
#endif
   weakestBoundsAlgorithm<datatype>(p, global_wbds, bds, acts, wbd);

   VERBOSE_CALL(printf("\nweakest_bounds procedure done.\n"));
   VERBOSE_CALL(measureTime("weakest_bounds", start, std::chrono::steady_clock::now()));

   // update the problem:
   for (int i=0; i<n_vars; i++)
   {
      lbs[i] = global_wbds.lbs[i];
      ubs[i] = global_wbds.ubs[i];
   }
}
#endif //GPU_DOMAIN_PROPAGATOR_WEAKEST_BOUNDS_H
