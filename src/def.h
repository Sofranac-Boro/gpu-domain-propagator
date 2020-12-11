//
// Created by boro on 14.10.20.
//

#ifndef GPU_DOMAIN_PROPAGATOR_RETCODES_H
#define GPU_DOMAIN_PROPAGATOR_RETCODES_H

#include <stdio.h>
#include "params.h"

#ifdef __cplusplus
extern "C" {
#endif

/** return codes for GDP methods: non-positive return codes are errors */
enum GDP_Retcode {
    GDP_OKAY = +1,       /**< normal termination */
    GDP_ERROR = 0,       /**< unspecified error */
    GDP_NOTIMPLEMENTED = -18        /**< function not implemented */
};
typedef enum GDP_Retcode GDP_RETCODE;           /**< return code for GDP method */

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

enum GDP_Vartype {
    GDP_BINARY = 0,
    GDP_INTEGER = 1,
    GDP_CONTINUOUS = 3
};
typedef enum GDP_Vartype GDP_VARTYPE;

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus
extern "C" {
#endif

/** return codes for GDP methods: non-positive return codes are errors */
enum GDP_algorithm {
    CPU_SEQ_DOUBLE = 0,
    CPU_OMP_DOUBLE = 1,
    GPU_ATOMIC_DOUBLE = 2,
    GPU_REDUCTION_DOUBLE = 3,
    CPU_SEQ_DISJOINT_DOUBLE = 4
};
typedef enum GDP_algorithm GDP_ALGORITHM;

#ifdef __cplusplus
}
#endif

#define REALABS(x)          ( fabs(x) )
#define EPSLE(x, y)    ((x)-(y) <= (GDP_EPS))
#define EPSLT(x, y)    ((x)-(y) < -(GDP_EPS))
#define EPSGE(x, y)    ((x)-(y) >= -(GDP_EPS))
#define EPSGT(x, y)    ((x)-(y) > (GDP_EPS))
#define EPSEQ(x, y)      ( REALABS((x)-(y)) <= (GDP_EPS) )
#define MAX(x, y)      ((x) >= (y) ? (x) : (y))
#define MIN(x, y)      ((x) <= (y) ? (x) : (y))
#define EPSFLOOR(x)   (floor((x)+(GDP_EPS)))
#define EPSCEIL(x)    (ceil((x)-(GDP_EPS)))

inline static void *safe_malloc(size_t n, unsigned long line) {
   void *p = malloc(n);
   if (!p) {
      fprintf(stderr, "[%s:%lu]Out of memory(%lu bytes)\n",
              __FILE__, line, (unsigned long) n);
      exit(EXIT_FAILURE);
   }
   return p;
}

#ifdef VERBOSE
#define VERBOSE_CALL(ans) {(ans);}
#else
#define VERBOSE_CALL(ans) do { } while(0)
#endif

#if VERBOSE >= 2
#define VERBOSE_CALL_2(ans) {(ans);}
#else
#define VERBOSE_CALL_2(ans) do { } while(0)
#endif

#ifdef DEBUG
#define SAFEMALLOC(n) safe_malloc(n, __LINE__)
#define DEBUG_CALL(ans) {(ans);}
#else
#define SAFEMALLOC(n) malloc(n)
#define DEBUG_CALL(ans) do { } while(0)
#endif

#ifdef FOLLOW_VAR
#define FOLLOW_VAR_CALL(varidx, ans) varidx==FOLLOW_VAR? (ans) : printf("")
#else
#define FOLLOW_VAR_CALL(varidx, ans) do { } while(0)
#endif

#ifdef FOLLOW_CONS
#define FOLLOW_CONS_CALL(considx, ans) considx==FOLLOW_CONS? (ans) : printf("")
#else
#define FOLLOW_CONS_CALL(considx, ans) do { } while(0)
#endif

#ifdef CALC_PROGRESS_REL
#define CALC_PROGRESS_REL_CALL(ans) {(ans);}
#else
#define CALC_PROGRESS_REL_CALL(ans) do { } while(0)
#endif

#ifdef CALC_PROGRESS_ABS
#define CALC_PROGRESS_ABS_CALL(ans) {(ans);}
#else
#define CALC_PROGRESS_ABS_CALL(ans) do { } while(0)
#endif

#endif //GPU_DOMAIN_PROPAGATOR_RETCODES_H
