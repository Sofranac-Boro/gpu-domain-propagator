//
// Created by boro on 14.10.20.
//

#ifndef GPU_DOMAIN_PROPAGATOR_RETCODES_H
#define GPU_DOMAIN_PROPAGATOR_RETCODES_H

#ifdef __cplusplus
extern "C" {
#endif

/** return codes for GDP methods: non-positive return codes are errors */
enum GDP_Retcode
{
    GDP_OKAY               =  +1,       /**< normal termination */
    GDP_ERROR              =   0,       /**< unspecified error */
    GDP_NOTIMPLEMENTED     = -18        /**< function not implemented */
};
typedef enum GDP_Retcode GDP_RETCODE;           /**< return code for GDP method */

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** return codes for GDP methods: non-positive return codes are errors */
enum GDP_algorithm
{
    CPU_SEQ_DOUBLE         = 0,
    CPU_OMP_DOUBLE         = 1,
    GPU_ATOMIC_DOUBLE      = 2,
    GPU_REDUCTION_DOUBLE   = 3,
    CPU_SEQ_DISJOINT_DOUBLE = 4
};
typedef enum GDP_algorithm GDP_ALGORITHM;

#ifdef __cplusplus
}
#endif

#endif //GPU_DOMAIN_PROPAGATOR_RETCODES_H
