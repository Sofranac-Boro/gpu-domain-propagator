
template <template<class> class Problem, class datatype>
void runAllAlgsAnalyticalSol()
{
        Tester<datatype> tester;
        Problem<datatype> ts_seq;
        Problem<datatype> ts_omp;
        Problem<datatype> ts_gpu_r;
        Problem<datatype> ts_gpu_a;

        printf("running cpu_seq");
        tester.executeSequentialPropagator(ts_seq);
        tester.checkSolution(ts_seq);

        printf("running cpu_omp");
        tester.executeFullOMPPropagator(ts_omp);
        tester.checkSolution(ts_omp);

        printf("running gpu_reduction");
        tester.executeGPUReduction(ts_gpu_r);
        tester.checkSolution(ts_gpu_r);

        printf("running gpu_atomic");
        tester.executeAtomicGPUPropagator(ts_gpu_a);
        tester.checkSolution(ts_gpu_a);
}

TEST_CASE( "End to End Achterberg example", "[endtoend]" ) {
    printf("Running Achterberg example\n");
    runAllAlgsAnalyticalSol<AchterbergExample, double>();
}

TEST_CASE( "End to End Test Savelsbergh example 1", "[endtoend]" ) {
    printf("Running SavelsberghExample1\n");
    runAllAlgsAnalyticalSol<SavelsberghExample1, double>();
}

TEST_CASE( "End to End Test Savelsbergh CFLP", "[endtoend]" ) {
    printf("Running SavelsberghCFLP\n");
    runAllAlgsAnalyticalSol<SavelsberghCFLP, double>();
}

TEST_CASE( "End to End Test Two Lines Example", "[endtoend]" ) {
    printf("Running TwoLinesExample\n");
    runAllAlgsAnalyticalSol<TwoLinesExample, double>();
}




