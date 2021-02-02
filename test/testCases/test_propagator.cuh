
#include "../../src/propagation_methods.cuh"

TEST_CASE( "Adjusting variable upper bound")
{
double bound;
bool isVarContinuous;

bound = 3.7;
isVarContinuous = true;
double adjusted_bound = adjustUpperBound(isVarContinuous, bound);
assertDoubleEPSEQ(adjusted_bound, bound
);

bound = -3.7;
isVarContinuous = false;
adjusted_bound = adjustUpperBound(isVarContinuous, bound);
assertDoubleEPSEQ(adjusted_bound,
-4.0);
}

TEST_CASE( "Adjusting variable lower bound")
{
double bound;
bool isVarContinuous;

bound = -3.7;
isVarContinuous = true;
double adjusted_bound = adjustLowerBound(isVarContinuous, bound);
assertDoubleEPSEQ(adjusted_bound, bound
);

bound = 3.7;
isVarContinuous = false;
adjusted_bound = adjustLowerBound(isVarContinuous, bound);
assertDoubleEPSEQ(adjusted_bound,
4.0);
}

TEST_CASE( "Is upper  bound better test")
{
double oldlb;
double oldub;
double newub;

oldlb = -0.5;
oldub = 3.0;
newub = 1.5;
REQUIRE(isUbBetter(oldlb, oldub, newub)
);

oldlb = -0.5;
oldub = 3.0;
newub = 3.2;
REQUIRE_FALSE( isUbBetter(oldlb, oldub, newub)
);

oldlb = -0.5;
oldub = 0.001;
newub = -0.001;
REQUIRE( isUbBetter(oldlb, oldub, newub)
);

oldlb = -3.5;
oldub = -1.0;
newub = -2.0;
REQUIRE( isUbBetter(oldlb, oldub, newub)
);
}

TEST_CASE( "Is lower  bound better test")
{
double oldlb;
double oldub;
double newlb;

oldlb = -1.0;
oldub = 1.0;
newlb = -1.5;
REQUIRE_FALSE( isLbBetter(oldlb, oldub, newlb)
);

oldlb = -1.0;
oldub = 1.0;
newlb = -0.5;
REQUIRE( isLbBetter(oldlb, oldub, newlb)
);

oldlb = -0.001;
oldub = 1.0;
newlb = 0.001;
REQUIRE( isLbBetter(oldlb, oldub, newlb)
);

oldlb = 0.5;
oldub = 1.0;
newlb = 0.9;
REQUIRE( isLbBetter(oldlb, oldub, newlb)
);
}

TEST_CASE( "can cons be tightened test")
{

double slack = 9.1;
double surplus = 10.8;
double maxactdelta = 3.8;
REQUIRE_FALSE( canConsBeTightened(slack, surplus, maxactdelta)
);

slack = 2.0;
surplus = 3.8;
maxactdelta = 2.8;
REQUIRE( canConsBeTightened(slack, surplus, maxactdelta)
);
}

TEST_CASE( "is cons infeasible test")
{

double minact = 9.1;
double maxact = 10.8;
double lhs = 3.8;
double rhs = 3.2;
REQUIRE( isConsInfeasible(minact, maxact, rhs, lhs)
);

minact = -2;
maxact = 10;
lhs = -3.8;
rhs = 3.2;
REQUIRE_FALSE( isConsInfeasible(minact, maxact, rhs, lhs)
);
}

TEST_CASE( "tighten variable upper bound test 1")
{
double coeff;
double slack;
double lb;
double ub;
bool isVarCont;

coeff = 1.0;
slack = 13.0;
lb = -10.0;
ub = 10.0;
isVarCont = true;
int num_inf_contr = 0;
NewBoundTuple res = tightenVarUpperBound(coeff, slack, num_inf_contr, lb, ub, isVarCont);
REQUIRE( res
.is_tightened );
assertDoubleEPSEQ(res
.newb, 3.0);
}

TEST_CASE( "tighten variable upper bound test 2")
{
double coeff;
double slack;
double lb;
double ub;
bool isVarCont;

coeff = -10.0;
slack = 45.0;
lb = 1.0;
ub = 6.0;
isVarCont = false;
   int num_inf_contr = 0;
NewBoundTuple res = tightenVarUpperBound(coeff, slack, num_inf_contr, lb, ub, isVarCont);
REQUIRE( res
.is_tightened );
assertDoubleEPSEQ(res
.newb, 5.0);

isVarCont = true;
res = tightenVarUpperBound(coeff, slack, num_inf_contr, lb, ub, isVarCont);
REQUIRE( res
.is_tightened );
assertDoubleEPSEQ(res
.newb, 5.5);
}

TEST_CASE( "tighten variable lower bound test 1")
{
double coeff;
double slack;
double lb;
double ub;
bool isVarCont;

coeff = 10;
slack = 37.0;
lb = -2.0;
ub = 7.0;
isVarCont = true;
   int num_inf_contr = 0;
NewBoundTuple res = tightenVarLowerBound(coeff, slack, num_inf_contr, lb, ub, isVarCont);
REQUIRE( res
.is_tightened );
assertDoubleEPSEQ(res
.newb, 3.3);

isVarCont = false;
res = tightenVarLowerBound(coeff, slack, num_inf_contr, lb, ub, isVarCont);
REQUIRE( res
.is_tightened );
assertDoubleEPSEQ(res
.newb, 4.0);
}

TEST_CASE( "tighten variable lower bound test 2")
{
double coeff;
double slack;
double lb;
double ub;
bool isVarCont;

coeff = -10.5;
slack = 10.5;
lb = -7.5;
ub = -3.0;
isVarCont = true;
   int num_inf_contr = 0;
NewBoundTuple res = tightenVarLowerBound(coeff, slack, num_inf_contr, lb, ub, isVarCont);
REQUIRE( res
.is_tightened );
assertDoubleEPSEQ(res
.newb, -4.0);

isVarCont = false;
res = tightenVarLowerBound(coeff, slack, num_inf_contr, lb, ub, isVarCont);
REQUIRE( res
.is_tightened );
assertDoubleEPSEQ(res
.newb, -4.0);
}


