#ifndef __GPUPROPAGATOR_PROPMETHODS_CUH__
#define __GPUPROPAGATOR_PROPMETHODS_CUH__

#include <math.h>       /* fabs */
#include <omp.h>


struct ActivitiesTupleStruct{
   double minact;
   double maxact;
   double maxactdelta;
};
typedef struct ActivitiesTupleStruct ActivitiesTuple;

struct NewBoundTupleStruct{
    bool is_tightened;
    double newb;
};
typedef struct NewBoundTupleStruct NewBoundTuple;


void markConstraints
(
    const int var_idx,
    const int* csc_col_ptrs, 
    const int* csc_row_indices,
    int* consmarked  
);

template<class datatype>
datatype adjustUpperBound
(
    const bool isVarContinuous,
    const datatype ub
)
{
    if (!isVarContinuous)
        return floor(ub);
    return ub;
}

template<class datatype>
datatype adjustLowerBound(const bool isVarContinuous, const datatype lb)
{
    if (!isVarContinuous)
        return ceil(lb);
    return lb;
}

template<class datatype>
bool isLbBetter(const datatype lb, const datatype ub, const datatype newlb)
{ 
    assert(lb <= ub);

    /* if lower bound is moved to 0 or higher, always accept bound change */
    if( lb < 0.0 && newlb >= 0.0 )
        return true;

    return (newlb > lb);
}

template<class datatype>
bool isUbBetter(const datatype lb, const datatype ub, const datatype newub)
{

    assert(lb <= ub);

    /* if upper bound is moved to 0 or lower, always accept bound change */
    if( ub > 0.0 && newub <= 0.0 )
        return true;

    return (newub < ub);
}

template<class datatype>
NewBoundTuple tightenVarUpperBound(const datatype coeff, const datatype slack, const datatype lb, const datatype ub, const bool isVarCont)
{
    NewBoundTuple newb_tuple = {false, ub}; // output

    datatype newb = lb + (slack / fabs(coeff));    
    newb = adjustUpperBound(isVarCont, newb);

    if (isUbBetter(lb, ub, newb))
    {
        newb_tuple.is_tightened = true; newb_tuple.newb = newb;
        return newb_tuple;
    }
    return newb_tuple;
}

template<class datatype>
NewBoundTuple tightenVarLowerBound(const datatype coeff, const datatype surplus, const datatype lb, const datatype ub, const bool isVarCont)
{
    NewBoundTuple newb_tuple = {false, lb}; // output
    
    datatype newb = ub - (surplus / fabs(coeff));
    newb = adjustLowerBound(isVarCont, newb);
    if (isLbBetter(lb, ub, newb))
    {
        newb_tuple.is_tightened = true; newb_tuple.newb = newb;
        return newb_tuple;
    }
    return newb_tuple;
}

template<class datatype>
bool canConsBeTightened(const datatype slack, const datatype surplus, const datatype maxactdelta)
{
        return !EPSLE(maxactdelta, MIN(slack, surplus), 1e-6); //todo epsilon
}

template<class datatype>
bool isConsInfeasible(const datatype minactivity, const datatype maxactivity, const datatype rhs, const datatype lhs)
{
    return (minactivity > rhs || maxactivity < lhs);
}

template<class datatype>
ActivitiesTuple computeActivities
(
    const int considx,
    const int* col_indices,
    const int* row_indices,
    const datatype* vals,
    const datatype* ubs,
    const datatype* lbs
)
{
    ActivitiesTuple actsTuple; // output

    double lb;
    double ub;
    double coeff;
    int val_idx;
    int var_idx;

    double maxactdelta = actsTuple.maxactdelta = 0.0;
    double minactivity = 0.0;
    double maxactivity = 0.0;
    int n_vars_in_cons = row_indices[considx+1] - row_indices[considx];
    
    for( int var = 0; var < n_vars_in_cons; var++)
    {   
        val_idx = row_indices[considx] + var;
        var_idx = col_indices[val_idx];              
        
        coeff     = vals[val_idx];
        lb        = lbs[var_idx];
        ub        = ubs[var_idx];

        maxactdelta = fabs(coeff) * (ub - lb);

        if( maxactdelta > actsTuple.maxactdelta )
            actsTuple.maxactdelta = maxactdelta;

        minactivity += coeff > 0? coeff*lb : coeff*ub;
        maxactivity += coeff > 0? coeff*ub : coeff*lb;

    }
    actsTuple.minact = minactivity;
    actsTuple.maxact = maxactivity;

    return actsTuple;
}

template <class datatype>
bool tightenVariable
(
    const datatype coeff,
    const datatype lhs,
    const datatype rhs,
    const datatype minact,
    const datatype maxact,
    const bool isVarCont,
    const int var_idx,
    const int val_idx,
    const int* csc_col_ptrs,
    const int* csc_row_indices,
    int* consmarked,
    datatype* lbs,
    datatype* ubs
)
{
    datatype ub = ubs[var_idx];
    datatype lb = lbs[var_idx];
    datatype slack = rhs - minact;
    datatype surplus = maxact - lhs;
    bool change_found = false;
    NewBoundTuple newb_tuple;

    if (coeff > 0.0)
    {
        if (coeff * (ub-lb) > slack && rhs < 1e20 && minact > -1e20)
        {
            newb_tuple = tightenVarUpperBound(coeff, slack, lb, ub, isVarCont);
            if (newb_tuple.is_tightened)
            {
                markConstraints(var_idx, csc_col_ptrs, csc_row_indices, consmarked);
                ubs[var_idx] = newb_tuple.newb;
                // update surplus and ub for tightening the lower bound
                surplus = surplus - coeff*(ub - newb_tuple.newb);
                ub = newb_tuple.newb;
                change_found = true;
            }
        }

        if ( coeff * (ub-lb) > surplus && lhs > -1e20 && maxact < 1e20)
        {
            newb_tuple = tightenVarLowerBound(coeff, surplus, lb, ub, isVarCont);
            if (newb_tuple.is_tightened)
            {
                markConstraints(var_idx, csc_col_ptrs, csc_row_indices, consmarked);
                lbs[var_idx] = newb_tuple.newb;
                change_found = true;
            }  
        }
    }
    else
    {
        if (coeff * (lb - ub) > slack && rhs < 1e20 && minact > -1e20)
        {
            newb_tuple = tightenVarLowerBound(coeff, slack, lb, ub, isVarCont);
            if (newb_tuple.is_tightened)
            {
                markConstraints(var_idx, csc_col_ptrs, csc_row_indices, consmarked);
                lbs[var_idx] = newb_tuple.newb;
                surplus = surplus - coeff*(newb_tuple.newb - lb);
                lb = newb_tuple.newb;
                change_found = true;
            }
        }

        if (coeff * (lb - ub) > surplus && lhs > -1e20 && maxact < 1e20)
        {
            newb_tuple = tightenVarUpperBound(coeff, surplus, lb, ub, isVarCont);
            if (newb_tuple.is_tightened)
            {
                markConstraints(var_idx, csc_col_ptrs, csc_row_indices, consmarked);
                ubs[var_idx] = newb_tuple.newb;
                change_found = true;
            }
        }
    }
    return change_found;
}

#endif