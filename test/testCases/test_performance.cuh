
TEST_CASE("Generalized CFP performance with analytical solution", "[endtoend]") {
   int size = 3000;
   Tester<double> tester;
   GeneralizedCFP<double> ts_hyb(size);
   GeneralizedCFP<double> ts_seq(size);

   tester.
           executeGPUReduction(ts_hyb);
   tester.
           executeSequentialPropagator(ts_seq);

   tester.
           checkSolution(ts_hyb);
   tester.
           checkSolution(ts_seq);
}


TEST_CASE("ATM Synthetic example performance test", "[performance]") {

// std::vector<int> sizes{10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000, 5120000, 10240000};
   Tester<double> tester;
   std::vector<int> sizes{10000};

   for (
           int i = 0;
           i < sizes.

                   size();

           i++) {
      printf("\n\n== running tests for size: %d==\n", sizes[i]);

      ATMSynthetic<double> ts_1(sizes[i]);
      ATMSynthetic<double> ts_2(sizes[i]);
      printf("Num non-zeros: %d", ts_1.nnz);
      printf("vartype at 9012: %d\n", ts_1.vartypes[9012]);

      tester.
              executeSequentialPropagator(ts_1);
      tester.
              executeFullOMPPropagator(ts_2);
      ts_1.
              compareSolutions(ts_2);
      ts_2.

              resetProblem();

      tester.
              executeAtomicGPUPropagator(ts_2);
      ts_1.
              compareSolutions(ts_2);
      ts_2.

              resetProblem();

      tester.
              executeGPUReduction(ts_2);
      ts_1.
              compareSolutions(ts_2);
      ts_2.

              resetProblem();

      tester.
              executeFullOMPPropagator(ts_2);
      ts_1.
              compareSolutions(ts_2);
   }
}

