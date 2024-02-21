//
// CSU33014 Lab 1
//

// Please examine version each of the following routines with names
// starting lab1. Where the routine can be vectorized, please
// complete the corresponding vectorized routine using SSE vector
// intrinsics.

// Note the restrict qualifier in C indicates that "only the pointer
// itself or a value directly derived from it (such as pointer + 1)
// will be used to access the object to which it points".


#include <immintrin.h>
#include <stdio.h>
#include <xmmintrin.h>

#include "lab1-code.h"

/****************  routine 0 *******************/

// Here is an example routine that should be vectorized
void lab1_routine0(float *restrict a, float *restrict b, float *restrict c) {
    for (int i = 0; i < 1024; i++ ) {
        a[i] = b[i] * c[i];
    }
}

// here is a vectorized solution for the example above
void lab1_vectorized0(float * restrict a, float * restrict b, float * restrict c) {
    __m128 a4, b4, c4;
    for (int i = 0; i < 1024; i = i+4 ) {
        b4 = _mm_loadu_ps(&b[i]);
        c4 = _mm_loadu_ps(&c[i]);
        a4 = _mm_mul_ps(b4, c4);
        _mm_storeu_ps(&a[i], a4);
    }
}

/***************** routine 1 *********************/

// in the following, size can have any positive value
float lab1_routine1(float * restrict a, float * restrict b,
                    int size) {
    float sum = 0.0;
  
    for ( int i = 0; i < size; i++ ) {
        sum = sum + a[i] * b[i];
    }
    return sum;
}

// insert vectorized code for routine1 here
float lab1_vectorized1(float * restrict a, float * restrict b, int size) {
    __m128 sum4 = _mm_setzero_ps();
    size_t i;
    for (i = 0; i < (size & ~0b11); i += 4) {
        __m128 a4 = _mm_loadu_ps(a + i);
        __m128 b4 = _mm_loadu_ps(b + i);
        __m128 p4 = _mm_mul_ps(a4, b4);
        sum4 = _mm_add_ps(sum4, p4);
    }
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float sum;
    _mm_store_ss(&sum, sum4);
    for (; i < size; i++) sum += a[i] * b[i];
    return sum;
}

/******************* routine 2 ***********************/

// in the following, size can have any positive value
void lab1_routine2(float * restrict a, float * restrict b, int size) {
    for ( int i = 0; i < size; i++ ) {
        a[i] = 1 - (1.0/(b[i]+1.0));
    }
}

// in the following, size can have any positive value
void lab1_vectorized2(float * restrict a, float * restrict b, int size) {
    __m128 one4 = _mm_set1_ps(1.0);
    size_t i;
    for (i = 0; i < (size & ~0b11); i += 4) {
        __m128 b4 = _mm_loadu_ps(b + i);
        b4 = _mm_add_ps(b4, one4);
        b4 = _mm_rcp_ps(b4);
        b4 = _mm_sub_ps(one4, b4);
        _mm_storeu_ps(a + i, b4);
    }
    for (; i < size; i++) a[i] = 1 - (1.0 / (b[i]+1.0));
}

/******************** routine 3 ************************/

// in the following, size can have any positive value
void lab1_routine3(float * restrict a, float * restrict b, int size) {
    for ( int i = 0; i < size; i++ ) {
        if ( a[i] < 0.0 ) {
            a[i] = b[i];
        }
    }
}

// in the following, size can have any positive value
void lab1_vectorized3(float * restrict a, float * restrict b, int size) {
    __m128 zero4 = _mm_setzero_ps();
    size_t i;
  
    for (i = 0; i < size & (~0b11); i += 4) {
        __m128 a4 = _mm_loadu_ps(a + i);
        __m128 b4 = _mm_loadu_ps(b + i);
        __m128 mask = _mm_cmplt_ps(a4, zero4);
        __m128 r4 = _mm_blendv_ps(b4, a4, mask);
        _mm_storeu_ps(a + i, a4);
    }
    for (; i < size; i++) if ( a[i] < 0.0 ) a[i] = b[i];
}

/********************* routine 4 ***********************/

// hint: one way to vectorize the following code might use
// vector shuffle operations
void lab1_routine4(float * restrict a, float * restrict b, float * restrict c) {
    for ( int i = 0; i < 2048; i = i+2  ) {
        a[i] = b[i]*c[i] - b[i+1]*c[i+1];
        a[i+1] = b[i]*c[i+1] + b[i+1]*c[i];
    }
}

// void lab1_vectorized4(float * restrict a, float * restrict b, float * restrict c) {
//     int i;
//     for(i=0; i<2048; i=i+4) {
//         __m128 b4 = _mm_loadu_ps(&(b[i]));
//         __m128 c4 = _mm_loadu_ps(&(c[i]));
// 
//         __m128 b4_b2b2b0b0 = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(2,2,0,0)); // b2 b2 b0 b0
//         __m128 b4_b3b3blbl = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(3,3,1,1)); // b3 b3 bl bl
//         __m128 c4_c1cOc3c2 = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(2,3,0,1)); // c2 c3 c0 cl
// 
//         __m128 firstMulOp = _mm_mul_ps(b4_b2b2b0b0, c4);
//         __m128 secondMulOp = _mm_mul_ps(b4_b3b3blbl, c4_c1cOc3c2);
// 
//         __m128 mask = _mm_setr_ps(-1, 0, -1, 0);
//         mask = _mm_cmplt_ps(mask, _mm_setzero_ps());
// 
//         __m128 subResult = _mm_sub_ps(firstMulOp, secondMulOp);
//         __m128 addResult = _mm_add_ps(firstMulOp, secondMulOp);
// 
//         __m128 tempResult = _mm_and_ps(mask, subResult);
//         __m128 a4 = _mm_andnot_ps(mask, addResult);
//         a4 = _mm_or_ps(a4, tempResult);
//         
//         _mm_storeu_ps(&(a[i]), a4);
//     }
// }

void lab1_vectorized4(float * restrict a, float * restrict b, float * restrict  c) {
    for ( int i = 0; i < 2048; i = i+4) {
        __m128 b4 = _mm_loadu_ps(b + i);
        __m128 c4 = _mm_loadu_ps(c + i);

        __m128 final;

        if (1) {
            __m128 u4 = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(2, 2, 0, 0));
            u4 = _mm_mul_ps(b4, u4);
            __m128 y1 = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 d4 = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(3, 3, 1, 1));
            d4 = _mm_mul_ps(y1, d4);
            final = _mm_addsub_ps(u4, d4);
        } else {
            __m128 p4x = _mm_mul_ps(b4, c4);
            p4x = _mm_hsub_ps(p4x, p4x);
            p4x = _mm_shuffle_ps(p4x, p4x, _MM_SHUFFLE(1, 1, 0, 0));
            __m128 p4y = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(2, 3, 0, 1));
            p4y = _mm_mul_ps(p4y, c4);
            p4y = _mm_hadd_ps(p4y, p4y);
            p4y = _mm_shuffle_ps(p4y, p4y, _MM_SHUFFLE(1, 1, 0, 0));
            p4y = _mm_blend_ps(p4x, p4y, 0b1010);
            final = p4y;
        }
        
        _mm_storeu_ps(a + i, final);
    }
}

/********************* routine 5 ***********************/

// in the following, size can have any positive value
void lab1_routine5(unsigned char * restrict a, unsigned char * restrict b, int size) {
    for ( int i = 0; i < size; i++ ) {
        a[i] = b[i];
    }
}

void lab1_vectorized5(unsigned char * restrict a, unsigned char * restrict b, int size) {
    size_t i;
    for (i = 0; i < (size & ~0b1111); i += 16) {
        __m128i b4 = _mm_loadu_si128(b + i);
        _mm_storeu_si128(a + i, b4);
    }
    for (; i < size; i++) a[i] = b[i];
}

/********************* routine 6 ***********************/

void lab1_routine6(float * restrict a, float * restrict b,
                   float * restrict c) {
    a[0] = 0.0;
    for ( int i = 1; i < 1023; i++ ) {
        float sum = 0.0;
        for ( int j = 0; j < 3; j++ ) {
            sum = sum +  b[i+j-1] * c[j];
        }
        a[i] = sum;
    }
    a[1023] = 0.0;
}

void format_vector(const char *name, __m128 v) {
    float r4[4];
    _mm_storeu_ps(r4, v);
    if (name != NULL)           printf("%s: ", name);
    for (int i = 0; i < 4; i++) printf("%12.3f,", r4[i]);
    printf("\n");
}

void lab1_vectorized6(float *restrict a, float *restrict b, float *restrict c) {
    __m128 c4 = _mm_set_ps(0.0, c[2], c[1], c[0]);
    a[0] = 0.0;
    int i;
    for (i = 1; i < 1020; i ++ ) {
        __m128 b4 = _mm_loadu_ps(b + i - 1);
        __m128 r4 = _mm_mul_ps(b4, c4);
        r4 = _mm_hadd_ps(r4, r4);
        r4 = _mm_hadd_ps(r4, r4);
        _mm_store_ss(a + i, r4);
    }
    for (; i < 1023; i++ ) {
        float sum = 0.0;
        for ( int j = 0; j < 3; j++ ) {
            sum = sum +  b[i+j-1] * c[j];
        }
        a[i] = sum;
    }
    a[1023] = 0.0;
}
