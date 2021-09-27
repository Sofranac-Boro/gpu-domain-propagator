#define CATCH_CONFIG_MAIN  // This tells Catch to provide a main() - only do this in one cpp file

#include "../lib/catch.hpp"
#include "test_infra.cuh"
#include "test_setups.cuh"
#include "tester.cuh"

// Include all test cases from the header files

#include "testCases/test_end_to_end.cuh"
#include "testCases/test_performance.cuh"
#include "testCases/test_propagator.cuh"
#include "testCases/test_components.cuh"
#include "testCases/test_weakest_bounds.cuh"



