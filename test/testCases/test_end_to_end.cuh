
TEST_CASE( "End to End Achterberg example", "[endtoend]" ) {
    
    Tester<double> tester;
    AchterbergExample<double> ts_hyb;
    AchterbergExample<double> ts_seq;
    AchterbergExample<double> ts_full;

    printf("running omp\n");
    tester.executeFullOMPPropagator(ts_hyb);
    tester.checkSolution(ts_hyb);

    printf("running seq\n");
    tester.executeSequentialPropagator(ts_seq);
    tester.checkSolution(ts_seq);

    printf("running full\n");
    tester.executeFullGPUPropagator(ts_full);
    tester.checkSolution(ts_full);
}   


TEST_CASE( "End to End Test Savelsbergh example 1", "[endtoend]" ) {
    Tester<double> tester;
    SavelsberghExample1<double> ts_seq;
    SavelsberghExample1<double> ts_full;

    printf("running seq\n");
    tester.executeSequentialPropagator(ts_seq);
    tester.checkSolution(ts_seq);

    printf("running full\n");
    tester.executeFullGPUPropagator(ts_full);
    tester.checkSolution(ts_full);
}


TEST_CASE( "End to End Test Savelsbergh CFLP", "[endtoend]" ) {
    Tester<double> tester;
    SavelsberghCFLP<double> ts_seq;
    SavelsberghCFLP<double> ts_full;

    printf("running seq\n");
    tester.executeSequentialPropagator(ts_seq);
    tester.checkSolution(ts_seq);

    printf("running full\n");
    tester.executeFullGPUPropagator(ts_full);
    tester.checkSolution(ts_full);
}