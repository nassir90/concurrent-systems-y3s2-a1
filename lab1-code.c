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
float lab1_vectorized1(float *restrict a, float *restrict b, int size) {
    size_t limit = size & ~0b11;
    float i4[4] = {0, 0, 0, 0};
    for (size_t j = limit; j < size; j++) {
        i4[j - limit] = a[j] * b[j];
    }
    __m128 sum4 = _mm_loadu_ps( i4);
    for (size_t i = 0; i < limit; i += 4) {
        __m128 a4 = _mm_loadu_ps(a + i);
        __m128 b4 = _mm_loadu_ps(b + i);
        __m128 p4 = _mm_mul_ps(a4, b4);
        sum4 = _mm_add_ps(sum4, p4);
    }
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    float sum;
    _mm_store_ss(&sum, sum4);
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
        __m128 d4 = _mm_add_ps(b4, one4);
        b4 = _mm_div_ps(b4, d4);
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
    for (i = 0; i < (size & ~0b11); i += 4) {
        __m128 a4 = _mm_loadu_ps(a + i);
        __m128 b4 = _mm_loadu_ps(b + i);
        __m128 mask = _mm_cmplt_ps(a4, zero4);
        __m128 r4 = _mm_blendv_ps(a4, b4, mask);
        _mm_storeu_ps(a + i, r4);
    }
    for (; i < size; i++) if (a[i] < 0.0) a[i] = b[i];
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

void lab1_vectorized4(float * restrict a, float * restrict b, float * restrict c) {
    for (int i = 0; i < 2048; i = i + 4) {
        __m128 b4 = _mm_loadu_ps(b + i);
        __m128 c4 = _mm_loadu_ps(c + i);

        __m128 bE = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(2, 2, 0, 0));
        bE = _mm_mul_ps(bE, c4);
        __m128 bO = _mm_shuffle_ps(b4, b4, _MM_SHUFFLE(3, 3, 1, 1));
        c4 = _mm_shuffle_ps(c4, c4, _MM_SHUFFLE(2, 3, 0, 1));
        bO = _mm_mul_ps(bO, c4);
        __m128 final = _mm_addsub_ps(bE, bO);
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
    int i = 0;
    __m128i
        *av = (__m128i_u *)a,
        *bv = (__m128i_u *)b;
    for (; i < (size & 0b1111); i += 16) {
        _mm_storeu_si128(av++, _mm_loadu_si128(bv++));
    }
    for (; i < size; i++) a[i] = b[i];
}

/********************* routine 6 ***********************/

void lab1_routine6(float * restrict a, float * restrict b,
                   float * restrict c) {
    a[0] = 0.0;
    int m = 0;
    for ( int i = 1; i < 1023; i++ ) {
        float sum = 0.0;
        for ( int j = 0; j < 3; j++ ) {
            sum = sum + b[i + j - 1] * c[j];
        }
        a[i] = sum;
    }
    a[1023] = 0.0;
}

void lab1_vectorized6(float *restrict a, float *restrict b, float *restrict c) {
    a[0] = 0.0;
    a[1023] = 0.0;
    
    __m128 c0 = _mm_set1_ps(c[0]);
    c0 = _mm_shuffle_ps(c0, c0, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 c1 = _mm_set1_ps(c[1]);
    c1 = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 c2 = _mm_set1_ps(c[2]);
    c2 = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(0, 0, 0, 0));
    
    int i = 1;

    for (; i < 1020; i += 4) {
        __m128 b1, b2, b3, sum;
        b1 = _mm_loadu_ps(b + i - 1);
        b1 = _mm_mul_ps(b1, c0);
        b2 = _mm_loadu_ps(b + i);
        b2 = _mm_mul_ps(b2, c1);
        b3 = _mm_loadu_ps(b + i + 1);
        b3 = _mm_mul_ps(b3, c2);
        sum = b1;
        sum = _mm_add_ps(sum, b2);
        sum = _mm_add_ps(sum, b3);
        _mm_storeu_ps(a + i, sum);
    }
    
    for (; i < 1023; i++ ) {
        float sum = 0.0;
        for ( int j = 0; j < 3; j++ ) {
            sum = sum +  b[i+j-1] * c[j];
        }
        a[i] = sum;
    }
}
