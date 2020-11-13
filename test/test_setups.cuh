#ifndef __GPUPROPAGATOR_TESTSETUP_CUH__
#define __GPUPROPAGATOR_TESTSETUP_CUH__

#include "../src/misc.h"
#include "../src/commons.cuh"
#include "test_infra.cuh"
#include "../src/params.h"

template <typename datatype>
class TestSetup{
public:
    int n_cons;
    int n_vars;
    int nnz;
    datatype* csr_vals;
    int* csr_col_indices; 
    int* csr_row_ptrs; 
    datatype* csc_vals;
    int* csc_col_ptrs;
    int* csc_row_indices;
    datatype* lhss; 
    datatype* rhss; 
    datatype* maxacts; 
    datatype* minacts; 
    datatype* maxactdeltas; 
    datatype* lbs;
    datatype* ubs;
    int* vartypes;
    int* consmarked;
    datatype* lbs_analytical;
    datatype* ubs_analytical;

    void printBounds(int len)
    {
        if (len < 0)
        {
            len = n_vars;
        }
        
        printArray<double>(ubs, len, "ubs");
        printArray<double>(lbs, len, "lbs");
    }

    void compareAnalyticalSolution()
    {
        printf("Comparing analytical solutions\n");
        compareArrays<datatype>(n_vars, ubs, ubs_analytical, TEST_EPS, "ubs");
        compareArrays<datatype>(n_vars, lbs, lbs_analytical, TEST_EPS, "lbs");
    }

    void compareSolutions(TestSetup ts_ref)
    {
        printf("Comparing solutions\n");
        REQUIRE(n_vars == ts_ref.n_vars);
        compareArrays<datatype>(n_vars, ubs, ts_ref.ubs, TEST_EPS, "ubs");
        compareArrays<datatype>(n_vars, lbs, ts_ref.lbs, TEST_EPS, "lbs");
    }

    void printConsMatrix()
    {
        printf("\n");
        for (int cons=0; cons<n_cons; cons++)
        {
            std::vector<datatype> dense_cons(n_vars, 0);
            //fill in the non-zeros
            for (int j=csr_row_ptrs[cons]; j<csr_row_ptrs[cons+1]; j++)
            {
                int varidx = csr_col_indices[j];
                dense_cons[varidx] = csr_vals[j];
    
            }
            printf("lhs: % 06.2f, rhs: % 06.2f", lhss[cons], rhss[cons]);
            printArray<datatype>(dense_cons.data(), n_vars, "");
        }
        printf("\n");
    }

protected:
    void allocMem(int n_cons, int n_vars, int nnz)
    {
        csr_vals        = (datatype*) SAFEMALLOC( nnz        * sizeof( datatype ));
        csr_col_indices = (int*)      SAFEMALLOC( nnz        * sizeof( int      ));
        csr_row_ptrs    = (int*)      SAFEMALLOC( (n_cons+1) * sizeof( int      ));
        csc_vals        = (datatype*) SAFEMALLOC( nnz        * sizeof( datatype ));
        csc_col_ptrs    = (int*)      SAFEMALLOC( (n_vars+1) * sizeof( int      ));
        csc_row_indices = (int*)      SAFEMALLOC( nnz        * sizeof( int      ));
        lhss            = (datatype*) SAFEMALLOC( n_cons     * sizeof( datatype ));
        rhss            = (datatype*) SAFEMALLOC( n_cons     * sizeof( datatype ));
        maxacts         = (datatype*) SAFEMALLOC( n_cons     * sizeof( datatype ));
        minacts         = (datatype*) SAFEMALLOC( n_cons     * sizeof( datatype ));
        maxactdeltas    = (datatype*) SAFEMALLOC( n_cons     * sizeof( datatype ));
        lbs             = (datatype*) SAFEMALLOC( n_vars     * sizeof( datatype ));
        ubs             = (datatype*) SAFEMALLOC( n_vars     * sizeof( datatype ));
        vartypes        = (int*)      SAFEMALLOC( n_vars     * sizeof( int      ));
        consmarked      = (int*)      SAFEMALLOC( n_cons     * sizeof( int      ));
        lbs_analytical  = (datatype*) SAFEMALLOC( n_vars     * sizeof( datatype ));
        ubs_analytical  = (datatype*) SAFEMALLOC( n_vars     * sizeof( datatype ));
    }

    void freeMem()
    {
        free( csr_vals       ); 
        free( csr_col_indices); 
        free( csr_row_ptrs   ); 
        free( csc_vals       ); 
        free( csc_col_ptrs   ); 
        free( csc_row_indices); 
        free( lhss           ); 
        free( rhss           ); 
        free( maxacts        ); 
        free( minacts        ); 
        free( maxactdeltas   ); 
        free( lbs            ); 
        free( ubs            ); 
        free( vartypes       ); 
        free( consmarked     ); 
        free( lbs_analytical ); 
        free( ubs_analytical ); 
    }
};


// Two intersecting lines. Infinite convergence, however due to floating point arithmetic will converge.
// 0.5x + y = 1 and x - y = 1.
// Analytical solution is a point [1.33333..., 0.33333....]
template <typename datatype>
class TwoLinesExample: public TestSetup<datatype> {

public:
    TwoLinesExample(): TestSetup<datatype>()
    {

        this->n_cons = 2;
        this->n_vars = 2;
        this->nnz = 4;

        this->allocMem(this->n_cons, this->n_vars, this->nnz);
        initProblem();
        initAnalyticalSolution();
    }

    ~TwoLinesExample()
    {
        this->freeMem();
    }

    void initProblem()
    {

        int m = 2; // number of constraints
        int n = 2; // number of variables
        int nnz = 4; // number of non zeros in the A matrix

        // Initialize problem
        std::vector<double> vals{0.5, 1, 1, -1}; // dim nnz
        std::vector<int> col_indices{0, 1, 0, 1}; // dim nnz
        std::vector<int> row_indices{0, 2, 4}; // dim m+1

        std::vector<double> lhss{1, 1}; // dim m
        std::vector<double> rhss{1, 1}; // dim m

        std::vector<double> lbs{-2, -2}; // dim n
        std::vector<double> ubs{2, 2}; // dim n
        std::vector<int> vartypes{3, 3}; // dim n
        std::vector<int> consmarked(m, 1);

        memcpy(this->csr_vals       , vals.data()       , nnz   * sizeof( datatype ));
        memcpy(this->csr_col_indices, col_indices.data(), nnz   * sizeof( int      ));
        memcpy(this->csr_row_ptrs   , row_indices.data(), (m+1) * sizeof( int      ));
        memcpy(this->lhss           , lhss.data()       , m     * sizeof( datatype ));
        memcpy(this->rhss           , rhss.data()       , m     * sizeof( datatype ));
        memcpy(this->lbs            , lbs.data()        , n     * sizeof( datatype ));
        memcpy(this->ubs            , ubs.data()        , n     * sizeof( datatype ));
        memcpy(this->vartypes       , vartypes.data()   , n     * sizeof( int      ));
        memcpy(this->consmarked     , consmarked.data() , m     * sizeof( int      ));
   }

   void initAnalyticalSolution()
   {
        std::vector<double> ubs_analytical{1.333333333954215, 0.33333333395421505};
        std::vector<double> lbs_analytical{1.33333333209157, 0.3333333330228925};
        memcpy(this->lbs_analytical , lbs_analytical.data(), this->n_vars * sizeof( datatype ));
        memcpy(this->ubs_analytical , ubs_analytical.data(), this->n_vars * sizeof( datatype ));
   }

};

//from T. Achterberg's Thesis, page 172
template <typename datatype>
class AchterbergExample: public TestSetup<datatype> {
   
public:  
    AchterbergExample(): TestSetup<datatype>()
    {

        this->n_cons = 11;
        this->n_vars = 12;
        this->nnz = 52;

        this->allocMem(this->n_cons, this->n_vars, this->nnz);
        initProblem();
        initAnalyticalSolution();
    }

    ~AchterbergExample()
    {
        this->freeMem();
    }

    void initProblem()
    {      
        int m = 11; // number of constraints
        int n = 12; // number of variables
        int nnz = 52; // number of non zeros in the A matrix
        
        datatype tmp_vals[]   = {2, 3, 2, 9, -1, -2, -3, 5, -3, -3, 9, -2, 9, -1, 2, -4, -7, 2, 5, -2, -1,  5,  4, -5, 1, -1,  1, -2,  1, -1, -2, 1, -2, 4,
            2, -1,  3, -2, -1,  5,  1, 2, -6, -2, -2,  1,  1,  1,  2, -2, 2, -2}; // dim nnz

        int tmp_col_indices[] = {0, 7, 8, 1, 7, 8, 1, 2, 3, 1, 3, 9, 4, 8, 9, 5, 6, 9, 6, 8, 4, 6, 8, 9, 0, 1, 2, 4, 5, 7, 8,  9, 10, 11,
        1, 3, 4, 5, 7, 8, 9, 10, 11, 0, 2, 3, 4, 7, 8, 9, 10, 11}; // dim nnz

        int      tmp_row_indices[]   = {0, 3, 6, 9, 12, 15, 18, 20, 24, 34, 43, nnz}; // dim m+1
        datatype tmp_lhss[]          = {-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100}; // dim m
        datatype tmp_rhss[]          = {9, 0, 4, 6, 8, 3, 2, 2, 1, 2 ,1}; // dim m
        datatype tmp_lbs[]           = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // dim n
        datatype tmp_ubs[]           = {1, 1, 1, 1, 1, 1, 1, 100, 100, 100, 100, 100}; // dim n
        int      tmp_vartypes[]      = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1}; // dim n
        int      tmp_consmarked[]    = {1};
 
        memcpy(this->csr_vals       , tmp_vals       , nnz   * sizeof( datatype )); 
        memcpy(this->csr_col_indices, tmp_col_indices, nnz   * sizeof( int      )); 
        memcpy(this->csr_row_ptrs   , tmp_row_indices, (m+1) * sizeof( int      )); 
        memcpy(this->lhss           , tmp_lhss       , m     * sizeof( datatype )); 
        memcpy(this->rhss           , tmp_rhss       , m     * sizeof( datatype )); 
        memcpy(this->lbs            , tmp_lbs        , n     * sizeof( datatype )); 
        memcpy(this->ubs            , tmp_ubs        , n     * sizeof( datatype )); 
        memcpy(this->vartypes       , tmp_vartypes   , n     * sizeof( int      )); 
        memcpy(this->consmarked     , tmp_consmarked , m     * sizeof( int      )); 
   }

   void initAnalyticalSolution()
   {
        datatype tmp_ubs_analytical[] = {1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 3.00, 4.00, 6.00, 23.00, 15.00};
        datatype tmp_lbs_analytical[] = {0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00};
        memcpy(this->lbs_analytical , tmp_lbs_analytical, this->n_vars * sizeof( datatype )); 
        memcpy(this->ubs_analytical , tmp_ubs_analytical, this->n_vars * sizeof( datatype )); 
   }
    
};


// from Savelsbergh Preprocessing and Probing Techniques for MIP problems, page 13
// https://www2.isye.gatech.edu/~ms79/software/ojoc6.pdf
template <typename datatype>
class SavelsberghExample1: public TestSetup<datatype> {
   
public:  
    SavelsberghExample1(): TestSetup<datatype>()
    {

        this->n_cons = 6;
        this->n_vars = 6;
        this->nnz = 12;

        this->allocMem(this->n_cons, this->n_vars, this->nnz);
        initProblem();
        initAnalyticalSolution();
    }

    ~SavelsberghExample1()
    {
        this->freeMem();
    }

    void initProblem()
    {      

        int m = 6; // number of constraints
        int n = 6; // number of variables
        int nnz = 12; // number of non zeros in the A matrix
    
        // Initialize problem
        std::vector<double> vals{1, 3, 1, 2, 2, 1, -15, 1, -20, 1, -5, 1}; // dim nnz
        std::vector<int> col_indices{3, 4, 3, 4, 3, 4, 0, 3, 1, 4, 2, 5}; // dim nnz
        std::vector<int> row_indices{0, 2, 4, 6, 8, 10, nnz}; // dim m+1
        
        std::vector<double> lhss{15, 10, 20, -GDP_INF, -GDP_INF, -GDP_INF}; // dim m
        std::vector<double> rhss{GDP_INF, GDP_INF, GDP_INF, 0, 0, 0}; // dim m
        
        std::vector<double> lbs{0, 0, 0, 0, 0, 0}; // dim n
        std::vector<double> ubs{1, 1, 1, GDP_INF, GDP_INF, GDP_INF}; // dim n
        std::vector<int> vartypes{0, 0, 0, 3, 3, 3}; // dim n
        std::vector<int> consmarked(m, 1);

        memcpy(this->csr_vals       , vals.data()       , nnz   * sizeof( datatype )); 
        memcpy(this->csr_col_indices, col_indices.data(), nnz   * sizeof( int      )); 
        memcpy(this->csr_row_ptrs   , row_indices.data(), (m+1) * sizeof( int      )); 
        memcpy(this->lhss           , lhss.data()       , m     * sizeof( datatype )); 
        memcpy(this->rhss           , rhss.data()       , m     * sizeof( datatype )); 
        memcpy(this->lbs            , lbs.data()        , n     * sizeof( datatype )); 
        memcpy(this->ubs            , ubs.data()        , n     * sizeof( datatype )); 
        memcpy(this->vartypes       , vartypes.data()   , n     * sizeof( int      )); 
        memcpy(this->consmarked     , consmarked.data() , m     * sizeof( int      )); 
   }

   void initAnalyticalSolution()
   {
        std::vector<double> ubs_analytical{1.00, 1.00, 1.00, 15.00, 20.00, 5.00};
        std::vector<double> lbs_analytical{0.00, 0.00, 0.00, 0.000, 0.000, 0.00};
        memcpy(this->lbs_analytical , lbs_analytical.data(), this->n_vars * sizeof( datatype )); 
        memcpy(this->ubs_analytical , ubs_analytical.data(), this->n_vars * sizeof( datatype )); 
   }  
};


// from Savelsbergh Preprocessing and Probing Techniques for MIP problems, page 17:
// instance of the capacitated facility location problem (CFLP)
// https://www2.isye.gatech.edu/~ms79/software/ojoc6.pdf
template <typename datatype>
class SavelsberghCFLP: public TestSetup<datatype> {
   
public:  
    SavelsberghCFLP(): TestSetup<datatype>()
    {

        this->n_cons = 7;
        this->n_vars = 16;
        this->nnz = 28;

        this->allocMem(this->n_cons, this->n_vars, this->nnz);
        initProblem();
        initAnalyticalSolution();
    }

    ~SavelsberghCFLP()
    {
        this->freeMem();
    }

    void initProblem()
    {      

        int m = 7; // number of constraints
        int n = 16; // number of variables
        int nnz = 28; // number of non zeros in the A matrix

        // Initialize problem
        std::vector<double> vals{1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,-100, 1,1,1,-100, 1,1,1,-100, 1,1,1,-100}; // dim nnz
        std::vector<int> col_indices{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}; // dim nnz
        std::vector<int> row_indices{0, 4, 8, 12, 16, 20, 24, nnz}; // dim m+1
        
        std::vector<double> lhss{80, 70, 40, -GDP_INF, -GDP_INF, -GDP_INF, -GDP_INF}; // dim m
        std::vector<double> rhss{80, 70, 40, 0, 0, 0, 0}; // dim m
        
        std::vector<double> lbs{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // dim n
        std::vector<double> ubs{GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, GDP_INF, 1, 1, 1, 1}; // dim n
        std::vector<int> vartypes{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0}; // dim n
        std::vector<int> consmarked(m, 1);

        memcpy(this->csr_vals       , vals.data()       , nnz   * sizeof( datatype )); 
        memcpy(this->csr_col_indices, col_indices.data(), nnz   * sizeof( int      )); 
        memcpy(this->csr_row_ptrs   , row_indices.data(), (m+1) * sizeof( int      )); 
        memcpy(this->lhss           , lhss.data()       , m     * sizeof( datatype )); 
        memcpy(this->rhss           , rhss.data()       , m     * sizeof( datatype )); 
        memcpy(this->lbs            , lbs.data()        , n     * sizeof( datatype )); 
        memcpy(this->ubs            , ubs.data()        , n     * sizeof( datatype )); 
        memcpy(this->vartypes       , vartypes.data()   , n     * sizeof( int      )); 
        memcpy(this->consmarked     , consmarked.data() , m     * sizeof( int      )); 
   }

   void initAnalyticalSolution()
   {
        std::vector<double> ubs_analytical{80, 80, 80, 80, 70, 70, 70, 70, 40, 40, 40, 40, 1, 1, 1, 1};
        std::vector<double> lbs_analytical{0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0, 0,  0,  0,  0};
        memcpy(this->lbs_analytical , lbs_analytical.data(), this->n_vars * sizeof( datatype )); 
        memcpy(this->ubs_analytical , ubs_analytical.data(), this->n_vars * sizeof( datatype )); 
   }
    
};


//pattern of nnz resembles the ATM instance from MIPlib. To be used for testing performance of propagation
template <typename datatype>
class ATMSynthetic: public TestSetup<datatype> {
   
public:  
    ATMSynthetic(int n): TestSetup<datatype>()
    {
        int n_cons = n;
        int n_vars = n;

        // int stride = int(sqrt(n) * 0.01) + 1;
        int stride = 4;
        int nnz_ub = n*stride + (n-stride)*stride;
        this->allocMem(n_cons, n_vars, nnz_ub);
        
        int row_ptrs_ctr = 0;
        int col_ind_ctr = 0;
        int vals_ctr = 0;

        int num_rows_ctr = 0;
        this->csr_row_ptrs[row_ptrs_ctr++] = num_rows_ctr;

        // the dense stride at the top
        for (int i=0; i<stride; i++)
        {
            num_rows_ctr += n_vars;
            this->csr_row_ptrs[row_ptrs_ctr++] = num_rows_ctr;
            for (int j=0; j<n_vars; j++)
            {
                // dense case A[i][j] = val
                this->csr_vals[vals_ctr++] = getRand10((i*5 + 9) * (j*3));
                this->csr_col_indices[col_ind_ctr++] = j;
            }
        }

        // rectangles around the diagonal
        for (int i=stride; i<n_cons; i+=stride)
        {
            for (int ii=0; ii<stride; ii++)
            {
                if (i+ii < n_cons)
                {
                    num_rows_ctr += MIN(stride, (n_cons-i));
                    this->csr_row_ptrs[row_ptrs_ctr++] = num_rows_ctr;
                }
                for (int jj=0; jj<stride; jj++)
                {
                    int x = i+ii;
                    int y = i+jj;
                    if (x<n_cons && y<n_vars)
                    {
                    // dense A[x][y] = 1.0
                        this->csr_vals[vals_ctr++] = getRand10(x*y*2);
                        this->csr_col_indices[col_ind_ctr++] = y;
                    }
                }
            }
        }
        
        int nnz = vals_ctr;
        REQUIRE( nnz == this->csr_row_ptrs[row_ptrs_ctr-1]);

        for (int i =0; i<n_vars; i++)
        {
            int seed = (i*3 + 26) * 6;
            double val = getRand10(seed);
            int type = i<n_cons*0.9? 3 : 1;

            this->lhss[i] = -REALABS(val);
            this->rhss[i] = REALABS(val)*2;
            this->lbs[i] = -getRand10pos(seed);
            this->ubs[i] = getRand10pos(seed);
            this->vartypes[i] = type;
            this->consmarked[i] = 1;
        }

        // Check that the data initialization is correct
        REQUIRE( vals_ctr == nnz );
        REQUIRE( col_ind_ctr == nnz );
        REQUIRE( row_ptrs_ctr == n_cons + 1 );
        REQUIRE( this->csr_row_ptrs[row_ptrs_ctr-1] == nnz );

        this->n_cons = n_cons;
        this->n_vars = n_vars;
        this->nnz = nnz;
    }

    ~ATMSynthetic()
    {
        this->freeMem();
    }


    void resetProblem()
    {
        for (int i =0; i<this->n_vars; i++)
        {
            int seed = (i*3 + 26) * 6;
            this->lbs[i] = -getRand10pos(seed);
            this->ubs[i] = getRand10pos(seed);
            this->consmarked[i] = 1;
        }
    }
};

/*
 This is generalized capacitated facility problem from Savelsbergh Preprocessing and Probing Techniques for MIP problems, page 17:
 Check the paper for analytical solution and problem definition
 demands for cons i == (i+1) % capacity
*/
template <typename datatype>
class GeneralizedCFP: public TestSetup<datatype> {

private:
    int _k; // num locations and demand points
    int capacity = 100; // production capacity of factories. All factories have the same capacity
   
public:  
    GeneralizedCFP(int k): TestSetup<datatype>()
    {
        _k = k;
        int n_vars = k*k + k;
        int n_cons = 2*k;
        int nnz = k*k + (k+1)*k;
        
        std::vector<double> vals;
        std::vector<int> col_indices; 
        std::vector<int> row_indices; 
        std::vector<double> lhss; 
        std::vector<double> rhss; 
        std::vector<double> varlbs;
        std::vector<double> varubs;
        std::vector<int> vartypes;
        std::vector<int> consmarked(n_cons, 1);
        
        for (int i=0; i<k; i++)
        {
            for (int idx=0; idx<k; idx++)
            {
                // dense: A[i][i*k + idx] = 1.0
                vals.push_back(1.0);
                col_indices.push_back(i*k + idx);
            }
        }

        for (int i=k; i<2*k; i++)
        {
            for (int idx=0; idx<k; idx++)
            {
                // dense: A[i][idx*k + i-k] = 1.0
                vals.push_back(1.0);
                col_indices.push_back(idx*k + i-k);

                if(idx == k-1){
                    // dense: A[i][idx*k + i] = -capacity
                    vals.push_back(-capacity);
                    col_indices.push_back(idx*k + i);
                }
            }
        }

        int current_row = 0;
        row_indices.push_back(current_row);
        for (int i=0; i<n_cons; i++)
        {
            current_row += i<k? k: k+1;
            row_indices.push_back(current_row);

            lhss.push_back( i<k? (i+1) % capacity : -GDP_INF );
            rhss.push_back( i<k? (i+1) % capacity : 0);
        }

        for (int i=0; i<n_vars; i++)
        {
            varlbs.push_back(0);
            varubs.push_back( i<k*k? GDP_INF : 1.0 );
            vartypes.push_back( i<k*k? 3 : 0 );
        }

        std::vector<int> csc_row_indices(nnz, 0);
        std::vector<int> csc_col_ptrs(n_vars + 1, 0); 

        // Check that the data initialization is correct
        REQUIRE( vals.size() == nnz );
        REQUIRE( col_indices.size() == nnz );
        REQUIRE( row_indices.size() == n_cons + 1 );
        REQUIRE( lhss.size() == n_cons );
        REQUIRE( rhss.size() == n_cons );
        REQUIRE( varlbs.size() == n_vars );
        REQUIRE( varubs.size() == n_vars );
        REQUIRE( vartypes.size() == n_vars );
        REQUIRE( row_indices[n_cons] == nnz );

        this->n_cons = n_cons;
        this->n_vars = n_vars;
        this->nnz = nnz;
        this->allocMem(n_cons, n_vars, nnz);
        
        memcpy(this->csr_vals       , vals.data()        , nnz        * sizeof( datatype )); 
        memcpy(this->csr_col_indices, col_indices.data() , nnz        * sizeof( int      )); 
        memcpy(this->csr_row_ptrs   , row_indices.data() , (n_cons+1) * sizeof( int      )); 
        memcpy(this->lhss           , lhss.data()        , n_cons     * sizeof( datatype )); 
        memcpy(this->rhss           , rhss.data()        , n_cons     * sizeof( datatype )); 
        memcpy(this->lbs            , varlbs.data()      , n_vars     * sizeof( datatype )); 
        memcpy(this->ubs            , varubs.data()      , n_vars     * sizeof( datatype )); 
        memcpy(this->vartypes       , vartypes.data()    , n_vars     * sizeof( int      )); 
        memcpy(this->consmarked     , consmarked.data()  , n_cons     * sizeof( int      ));
        
        initAnalyticalSolution();
    }

    ~GeneralizedCFP()
    {
        this->freeMem();
    }

    void initAnalyticalSolution()
    {
        // build analytical solution.
        std::vector<double> varlbs_analytical;
        std::vector<double> varubs_analytical;
        
        // general integer types. 
        for (int i=0; i<_k; i++)
        {
            for (int j=0; j<_k; j++)
            {
                varlbs_analytical.push_back(0.0);
                varubs_analytical.push_back((i+1) % capacity);
            }
        }

        // Rest k vars, binary
        for (int i=0; i<_k; i++)
        {
            varlbs_analytical.push_back(0.0);
            varubs_analytical.push_back(1.0);
        }

        memcpy(this->lbs_analytical , varlbs_analytical.data(), this->n_vars * sizeof( datatype )); 
        memcpy(this->ubs_analytical , varubs_analytical.data(), this->n_vars * sizeof( datatype )); 
    }


    void resetProblem()
    {

        std::vector<datatype> varlbs;
        std::vector<datatype> varubs;
        std::vector<int> consmarked(this->n_cons, 1);

        for (int i=0; i<this->n_vars; i++)
        {
            varlbs.push_back(0);
            varubs.push_back( i<_k*_k? GDP_INF : 1.0 );
        }

        memcpy(this->lbs            , varlbs.data()      , this->n_vars     * sizeof( datatype )); 
        memcpy(this->ubs            , varubs.data()      , this->n_vars     * sizeof( datatype ));
        memcpy(this->consmarked     , consmarked.data()  , this->n_cons     * sizeof( int      )); 
     
    }
};




template <typename datatype>
class ex1: public TestSetup<datatype> {
   
public:  
    ex1(): TestSetup<datatype>()
    {

        this->n_cons = 3;
        this->n_vars = 4;
        this->nnz = 9;

        this->allocMem(this->n_cons, this->n_vars, this->nnz);
        initProblem();
    }

    ~ex1()
    {
        this->freeMem();
    }

    void initProblem()
    {      
        int m = 3; // number of constraints
        int n = 4; // number of variables
        int nnz = 9; // number of non zeros in the A matrix
        
        datatype tmp_vals[]   = {-1.0, 1.0, 1.0, 10.0, 1.0, -3.0, 1.0, 1.0, -3.5}; // dim nnz

        int tmp_col_indices[] = {0, 1, 2, 3, 0, 1, 2, 1, 3}; // dim nnz

        int      tmp_row_indices[]   = {0, 4, 7, 9}; // dim m+1
        datatype tmp_lhss[]          = {-1e100, -1e100, 0.0}; // dim m
        datatype tmp_rhss[]          = {20.0, 30.0, 0.0}; // dim m
        datatype tmp_lbs[]           = {0.0, 0.0, 0.0, 2.0}; // dim n
        datatype tmp_ubs[]           = {40.0, 1.7976931348623157e+308, 1.7976931348623157e+308, 3.0}; // dim n
        int      tmp_vartypes[]      = {3,3,3,1}; // dim n
        int      tmp_consmarked[]    = {1, 1, 1};
 
        memcpy(this->csr_vals       , tmp_vals       , nnz   * sizeof( datatype )); 
        memcpy(this->csr_col_indices, tmp_col_indices, nnz   * sizeof( int      )); 
        memcpy(this->csr_row_ptrs   , tmp_row_indices, (m+1) * sizeof( int      )); 
        memcpy(this->lhss           , tmp_lhss       , m     * sizeof( datatype )); 
        memcpy(this->rhss           , tmp_rhss       , m     * sizeof( datatype )); 
        memcpy(this->lbs            , tmp_lbs        , n     * sizeof( datatype )); 
        memcpy(this->ubs            , tmp_ubs        , n     * sizeof( datatype )); 
        memcpy(this->vartypes       , tmp_vartypes   , n     * sizeof( int      )); 
        memcpy(this->consmarked     , tmp_consmarked , m     * sizeof( int      )); 
   }
    
};

#endif