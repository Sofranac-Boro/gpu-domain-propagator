#include "test_infra.cuh"

void assertDoubleEPSEQ(double val1, double val2, double eps) {
   REQUIRE(EPSEQ(val1, val2, eps));
}

double getRand10(double seed) {
   srand(seed);
   double value = ((double) rand() / (RAND_MAX));
   value = int(seed) % 2 == 0 ? value : -value;
   return value;
}

double getRand10pos(double seed) {
   srand(seed);
   double value = ((double) rand() / (RAND_MAX));
   return value;
}

