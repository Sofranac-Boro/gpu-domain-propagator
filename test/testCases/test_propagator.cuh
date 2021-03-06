
#include "../../src/propagation_methods.cuh"

TEST_CASE("Adjusting variable upper bound")
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

TEST_CASE("Adjusting variable lower bound")
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

TEST_CASE("Is upper  bound better test")
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
   REQUIRE_FALSE(isUbBetter(oldlb, oldub, newub)
   );

   oldlb = -0.5;
   oldub = 0.001;
   newub = -0.001;
   REQUIRE(isUbBetter(oldlb, oldub, newub)
   );

   oldlb = -3.5;
   oldub = -1.0;
   newub = -2.0;
   REQUIRE(isUbBetter(oldlb, oldub, newub)
   );
}

TEST_CASE("Is lower  bound better test")
{
   double oldlb;
   double oldub;
   double newlb;

   oldlb = -1.0;
   oldub = 1.0;
   newlb = -1.5;
   REQUIRE_FALSE(isLbBetter(oldlb, oldub, newlb)
   );

   oldlb = -1.0;
   oldub = 1.0;
   newlb = -0.5;
   REQUIRE(isLbBetter(oldlb, oldub, newlb)
   );

   oldlb = -0.001;
   oldub = 1.0;
   newlb = 0.001;
   REQUIRE(isLbBetter(oldlb, oldub, newlb)
   );

   oldlb = 0.5;
   oldub = 1.0;
   newlb = 0.9;
   REQUIRE(isLbBetter(oldlb, oldub, newlb)
   );
}

TEST_CASE("can cons be tightened test")
{
   double rhs = 9.1;
   double minact = 0;
   double maxact = 10.8;
   double lhs = 0;
   double maxactdelta = 3.8;
   REQUIRE_FALSE(canConsBeTightened(minact, maxact, 0, 0, lhs, rhs, maxactdelta)
   );

   rhs = 2.0;
   maxact = 3.8;
   maxactdelta = 2.8;
   REQUIRE(canConsBeTightened(minact, maxact, 0, 0, lhs, rhs, maxactdelta)
   );
}

TEST_CASE("is cons infeasible test")
{

   double minact = 9.1;
   double maxact = 10.8;
   double lhs = 3.8;
   double rhs = 3.2;
   REQUIRE(isConsInfeasible(minact, maxact, rhs, lhs)
   );

   minact = -2;
   maxact = 10;
   lhs = -3.8;
   rhs = 3.2;
   REQUIRE_FALSE(isConsInfeasible(minact, maxact, rhs, lhs)
   );
}

TEST_CASE("tighten variable upper bound test 1")
{
   double coeff;
   double slack;
   double surplus;
   double lb;
   double ub;
   bool isVarCont;

   coeff = 1.0;
   slack = 13.0;
   surplus = GDP_INF;
   lb = -10.0;
   ub = 10.0;
   isVarCont = true;
   int num_inf_contr = 0;
   NewBoundTuple<double> res = tightenVarUpperBound(coeff, slack, surplus, num_inf_contr, lb, ub, isVarCont);
   REQUIRE(res
                   .is_tightened);
   assertDoubleEPSEQ(res
                             .newb, 3.0);
}

TEST_CASE("tighten variable upper bound test 2")
{
   double coeff;
   double slack;
   double surplus;
   double lb;
   double ub;
   bool isVarCont;
   int num_inf_contr;

   coeff = -2.0;
   slack = 45.0;
   surplus = -109;
   lb = 0;
   ub = 100;
   isVarCont = false;
   num_inf_contr = 0;
   NewBoundTuple<double> res = tightenVarUpperBound(coeff, slack, surplus, num_inf_contr, lb, ub, isVarCont);
   REQUIRE(res
                   .is_tightened);
   assertDoubleEPSEQ(res
                             .newb, 54);

   isVarCont = true;
   res = tightenVarUpperBound(coeff, slack, surplus, num_inf_contr, lb, ub, isVarCont);
   REQUIRE(res
                   .is_tightened);
   assertDoubleEPSEQ(res
                             .newb, 54.5);
}

TEST_CASE("tighten variable lower bound test 1")
{
   double coeff;
   double slack;
   double surplus;
   double lb;
   double ub;
   bool isVarCont;

   coeff = 1;
   slack = GDP_INF;
   surplus = -5.1;
   lb = -100.0;
   ub = 10;
   isVarCont = true;
   int num_inf_contr = 0;
   NewBoundTuple<double> res = tightenVarLowerBound(coeff, slack, surplus, num_inf_contr, lb, ub, isVarCont);
   REQUIRE(res
                   .is_tightened);
   assertDoubleEPSEQ(res
                             .newb, 4.9);

   isVarCont = false;
   res = tightenVarLowerBound(coeff, slack, surplus, num_inf_contr, lb, ub, isVarCont);
   REQUIRE(res
                   .is_tightened);
   assertDoubleEPSEQ(res
                             .newb, 5.0);
}

TEST_CASE("tighten variable lower bound test 2")
{
   double coeff;
   double slack;
   double surplus;
   double lb;
   double ub;
   bool isVarCont;

   coeff = -10.5;
   slack = 10.5;
   surplus = 0.0;
   lb = -7.5;
   ub = -3.0;
   isVarCont = true;
   int num_inf_contr = 0;
   NewBoundTuple<double> res = tightenVarLowerBound(coeff, slack, surplus, num_inf_contr, lb, ub, isVarCont);
   REQUIRE(res
                   .is_tightened);
   assertDoubleEPSEQ(res
                             .newb, -4.0);

   isVarCont = false;
   res = tightenVarLowerBound(coeff, slack, surplus, num_inf_contr, lb, ub, isVarCont);
   REQUIRE(res
                   .is_tightened);
   assertDoubleEPSEQ(res
                             .newb, -4.0);
}


