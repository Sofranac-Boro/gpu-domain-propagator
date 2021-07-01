#ifndef GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H
#define GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H

#include "../propagators/sequential_propagator.h"
#include <vector>

using namespace std;

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
void executePreprocessor
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
template<typename datatype>
class Mat;

template<typename datatype>
class Bounds {
private:
    vector<bool> active_lbs;
    vector<bool> active_ubs;

public:
    int n_vars;
    datatype* lbs;
    datatype* ubs;
    const GDP_VARTYPE* vartypes;

    Bounds(const int n_vars_, datatype* lbs_, datatype* ubs_, const GDP_VARTYPE *vartypes_)
    {
       n_vars = n_vars_;
       lbs=lbs_;
       ubs=ubs_;
       vartypes = vartypes_;

       active_lbs.reserve(n_vars_);
       active_ubs.reserve(n_vars_);

       fill(active_lbs.begin(),active_lbs.end(), false);
       fill(active_ubs.begin(),active_ubs.end(), false);
    }

    bool is_lb_inf(int var)
    {
       return EPSLE(lbs[var], -GDP_INF);
    }

    bool is_ub_inf(int var)
    {
       return EPSGE(ubs[var], GDP_INF);
    }

    bool is_lb_active(int var)
    {
       return (is_lb_inf(var) || active_lbs[var]);
    }

    bool is_ub_active(int var)
    {
       return (is_ub_inf(var) || active_ubs[var]);
    }

    void activate_lb(int var)
    {
       active_lbs[var] = true;
    }

    void activate_ub(int var)
    {
       active_ubs[var] = true;
    }

    void tightenVariable
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
            ) const
    {
       datatype slack = rhs - minact;
       datatype surplus = lhs - maxact;

       if (EPSGT(coeff, 0.0))
       {
         if (EPSLT(rhs, GDP_INF) && EPSGT(minact, -GDP_INF))
         {

         }
       }
    }
};

template<typename datatype>
class Mat {
public:
       int n_vars;
       int n_cons;
       int nnz;
       const datatype* vals;
      const int* col_indices;
    const int* row_ptrs;
    const datatype* lhss;
    const datatype* rhss;
    int max_n_vars = 0;

    vector<datatype> minacts;
    vector<datatype> maxacts;
    vector<int> minacts_inf;
    vector<int> maxacts_inf;
    vector<datatype> maxactdeltas;

    Mat(const int nnz_, const int n_cons_, const int n_vars_, const datatype* vals_, const int* col_indices_, const int* row_ptrs_, const datatype* lhss_, const datatype* rhss_){
       n_cons = n_cons_;
       n_vars=n_vars_;
       nnz = nnz_;
       vals = vals_;
       col_indices = col_indices_;
       row_ptrs = row_ptrs_;
       lhss=lhss_;
       rhss=rhss_;

       for(int i=0; i<n_cons; i++)
       {
          if (row_ptrs[i+1] - row_ptrs[i] > max_n_vars)
          {
             max_n_vars = row_ptrs[i+1] - row_ptrs[i];
          }
       }

       minacts.resize(n_cons_);
       maxacts.resize(n_cons_);
       minacts_inf.resize(n_cons_);
       maxacts_inf.resize(n_cons_);
       maxactdeltas.resize(n_cons_);
    }

    void compute_activities(Bounds<datatype>& bds)
    {
       sequentialComputeActivities<datatype>(n_cons, col_indices, row_ptrs, vals, bds.ubs, bds.lbs, minacts.data(), maxacts.data(),
                                             minacts_inf.data(), maxacts_inf.data(),
                                             maxactdeltas.data());
    }

    bool contains_inf_bd(int cons, Bounds<datatype>& bds)
    {
       for (int i= row_ptrs[cons]; i<row_ptrs[cons+1]; i++)
       {
          int var = col_indices[i];
          if (bds.is_lb_inf(var) || bds.is_ub_inf(var))
          {
             return true;
          }
       }
       return false;
    }

    bool inf_tight_possible(int cons)
    {
       return (minacts_inf[cons] <= 1 && EPSLT(rhss[cons], GDP_INF) || (maxacts_inf[cons] <= 1 && EPSGT(lhss[cons], -GDP_INF)));
    }

    datatype tightenCons(
            int cons,
            Bounds<datatype>& bds
    ) const
    {
       const int str = row_ptrs[cons];
       const int end = row_ptrs[cons+1];

       for (int validx=str; validx<end; validx++)
       {
          const int var = col_indices[validx];
          if (bds.is_lb_active(var))
          {

          }
       }
    }
};



template<typename datatype>
class Preprocessor {
public:
    vector<int> active_cons;
    vector<int> processed_lbs;
    vector<int> processed_ubs;


    Preprocessor(const int nnz, const int n_cons, const int n_vars, Mat<datatype>& mat_, Bounds<datatype>& bds_){
       active_cons.reserve(n_cons);
       processed_lbs.reserve(n_vars);
       processed_ubs.reserve(n_vars);

       for (int i=0; i<n_cons; i++)
       {
          if (mat_.contains_inf_bd(i, bds_))
          {
             active_cons.push_back(i);
          }
       }
       cout << "size of active cons: " << active_cons.size() << " out of total: " << n_cons << endl;
    }


    void weakest_bounds(Mat<datatype>& mat, Bounds<datatype>& bds)
    {
      // only cons that have at least one infinity are activated in the constructor.
       mat.compute_activities(bds);
       for (auto cons: active_cons)
       {
          if (mat.inf_tight_possible(cons))
          {



          }
//
       }
    }

};

template<class datatype>
void executePreprocessorNew
        (
                const int nnz,
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
   printf("\nNew preprocessor!\n\n");

   vector<int> processed_lbs(n_vars);
   vector<int> processed_ubs(n_vars);
   vector<int> active_cons(n_cons);

   Bounds<datatype> bds(n_vars,lbs,ubs,vartypes);
   Mat<datatype> mat(nnz,n_cons,n_vars,vals,col_indices,row_indices, lhss, rhss);
   Preprocessor<datatype> pp(nnz,n_cons,n_vars,mat, bds);

   pp.weakest_bounds(mat, bds);

   return;
}

#endif //GPU_DOMAIN_PROPAGATOR_PREPROCESSOR_H
