
TEST_CASE( "Generalized CFP performance with analytical solution", "[endtoend]" ) {
    int size = 3000;
    Tester<double> tester;
    GeneralizedCFP<double> ts_hyb(size);
    GeneralizedCFP<double> ts_seq(size);

    tester.executeFullGPUPropagator(ts_hyb);
    tester.executeSequentialPropagator(ts_seq);
    
    tester.checkSolution(ts_hyb);
    tester.checkSolution(ts_seq);
}


TEST_CASE( "ATM Synthitic example performance test", "[performance]" ) {

    // std::vector<int> sizes{10000, 20000, 40000, 80000, 160000, 320000, 640000, 1280000, 2560000, 5120000, 10240000};
    Tester<double> tester;
    std::vector<int> sizes{10000};

    for (int i=0; i< sizes.size(); i++)
    {
        printf("\n\n== running tests for size: %d==\n", sizes[i]);

        ATMSynthetic<double> ts_1(sizes[i]);
        ATMSynthetic<double> ts_2(sizes[i]);
        printf("Num non-zeros: %d", ts_1.nnz);

        tester.executeSequentialPropagator(ts_1);
        tester.executeFullOMPPropagator(ts_2);
        ts_1.compareSolutions(ts_2);
        ts_2.resetProblem();

        tester.executeAtomicGPUPropagator(ts_2);
        ts_1.compareSolutions(ts_2);
        ts_2.resetProblem();

        tester.executeFullGPUPropagator(ts_2);
        ts_1.compareSolutions(ts_2);
        ts_2.resetProblem();

        tester.executeFullOMPPropagator(ts_2);
        ts_1.compareSolutions(ts_2);
    }
}

