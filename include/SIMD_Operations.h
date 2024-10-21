#ifndef SIMD_OPERATIONS_H
#define SIMD_OPERATIONS_H
#include <immintrin.h>
#include <Aligned_Allocator.h>
#include <vector>

template <typename FType, typename IType>
FType process(const FType*& new_data_ptr, const FType*& new_centroids_ptr, const IType cols){

    std::size_t i = 0;

    #ifdef SIMD_256
    if constexpr (std::is_same<FType, float>::value)
    {   
        __m256 sum_vec = _mm256_setzero_ps();
        for (; i + 7 < cols; i += 8)
        {
            __m256 new_data_vec = _mm256_load_ps(new_data_ptr + i);
            __m256 new_centroid_vec = _mm256_load_ps(new_centroids_ptr + i);
            __m256 difference = _mm256_sub_ps(new_data_vec, new_centroid_vec);
            difference = _mm256_mul_ps(difference, difference);
            sum_vec = _mm256_add_ps(sum_vec, difference);

        }

    
        __m128 low = _mm256_castps256_ps128(sum_vec); // [a0, a1, a2, a3]
        __m128 high = _mm256_extractf128_ps(sum_vec, 1); // [a4, a5, a6, a7]
        __m128 sum_128 = _mm_add_ps(low, high); // [(a0 + a4), (a1+ a5), (a2 + a6), (a3 + a7)]
        sum_128 = _mm_hadd_ps(sum_128, sum_128); // [(a0 + a4 + a1 + a5), (a2 + a6 + a3 + a7), X, X] X = duplicate
        sum_128 = _mm_hadd_ps(sum_128, sum_128); // [(a0 + a4 +a1 +a5 +a2 + a6+ a3+ a7), X, X, X]
        FType distance = _mm_cvtss_f32(sum_128); // extract the first element which holds the sum of all values in the originial sum_vec

        // add now the remaining elements to the sum 
        for (; i < cols; ++i)
        {
            FType diff = new_data_ptr[i] - new_centroids_ptr[i];
            distance += diff * diff;
            
        }

        return distance; 

    }

    else if constexpr (std::is_same<FType, double>::value)
    {
        __m256d sum_vec = _mm256_setzero_pd();
        for (; i + 3 < cols; i += 4)
        {
            __m256d new_data_vec = _mm256_load_pd(new_data_ptr + i);
            __m256d new_centroid_vec = _mm256_load_pd(new_centroids_ptr + i);
            __m256d difference = _mm256_sub_pd(new_data_vec, new_centroid_vec);
            difference = _mm256_mul_pd(difference, difference);
            sum_vec = _mm256_add_pd(sum_vec, difference);

        }

        
        __m128d low = _mm256_castpd256_pd128(sum_vec); // [a0, a1, a2, a3]
        __m128d high = _mm256_extractf128_pd(sum_vec, 1); // [a4, a5, a6, a7]
        __m128d sum_128 = _mm_add_pd(low, high); // [(a0 + a4), (a1+ a5), (a2 + a6), (a3 + a7)]
        sum_128 = _mm_hadd_pd(sum_128, sum_128); // [(a0 + a4 + a1 + a5), (a2 + a6 + a3 + a7), X, X] X = duplicate
        sum_128 = _mm_hadd_pd(sum_128, sum_128); // [(a0 + a4 +a1 +a5 +a2 + a6+ a3+ a7), X, X, X]
        FType distance =_mm_cvtsd_f64(sum_128); // extract the first element which holds the sum of all values in the originial sum_vec

        // add now the remaining elements to the sum 
        for (; i < cols; ++i)
        {
            FType diff = new_data_ptr[i] - new_centroids_ptr[i];
            distance += diff * diff;
            
        }

        return distance; 
    }

    #elif defined(SIMD_512)
    if constexpr (std::is_same<FType, float>::value)
    {   
        __m512 sum_vec = _mm512_setzero_ps();
        for (; i + 15 < cols; i += 16)
        {
            __m512 new_data_vec = _mm512_load_ps(new_data_ptr + i);
            __m512 new_centroid_vec = _mm512_load_ps(new_centroids_ptr + i);
            __m512 difference = _mm512_sub_ps(new_data_vec, new_centroid_vec);
            difference = _mm512_mul_ps(difference, difference);
            sum_vec = _mm512_add_ps(sum_vec, difference);

        }

        FType distance = _mm512_reduce_add_ps(sum_vec); 

        // add now the remaining elements to the sum 
        for (; i < cols; ++i)
        {
            FType diff = new_data_ptr[i] - new_centroids_ptr[i];
            distance += diff * diff;
            
        }

        return distance; 

    }

    else if constexpr (std::is_same<FType, double>::value)
    {
        __m512d sum_vec = _mm512_setzero_pd();
        for (; i + 7 < cols; i += 8)
        {
            __m512d new_data_vec = _mm512_load_pd(new_data_ptr + i);
            __m512d new_centroid_vec = _mm512_load_pd(new_centroids_ptr + i);
            __m512d difference = _mm512_sub_pd(new_data_vec, new_centroid_vec);
            difference = _mm512_mul_pd(difference, difference);
            sum_vec = _mm512_add_pd(sum_vec, difference);

        }

         FType distance = _mm512_reduce_add_pd(sum_vec); 
        // add now the remaining elements to the sum 
        for (; i < cols; ++i)
        {
            FType diff = new_data_ptr[i] - new_centroids_ptr[i];
            distance += diff * diff;
            
        }

        return distance; 
    }
    #endif











}


#endif

// Implementation wihtout splitting the sum_vector at the beginning
/** 
// sum_vec [a0, a1, a2, a3, a4, a5, a6, a7] 
// sum_vec [b0, b1, b2, b3, b4, b5, b6, b7] same elements
// when using _mm256_hadd_ps each element from the original vector stays in it's lower and upper bit lane
__m256 temp = _mm256_hadd_ps(sum_vec, sum_vec); // [(a0 + a1), (a2 + a3), (b0+ b1), (b2 + b3), (a4 + a5), (a6 + a7), (b4 + b5), (b6 + b7)]
__m256 temp = _mm256_hadd_ps(temp, temp); // lower 128 bits: [(a0 + a1 + a2 + a3), (b0 + b1 + b2 + b3), (a0 + a1 + a2 + a3), (b0 + b1 + b2 + b3)]
                                        // upper 128 bits: [(a4 + a5 + a6 + a7), (b4 + b5 + b6 + b7), (a4 + a5 + a6 + a7), (b4 + b5 + b6 + b7)]
__m128 low = _mm256_castps256_ps128(temp); // load the lower 128 bit vector
__m128 high = _mm256_extractf128_ps(temp, 1); // load the upper 128 bit vector
__m128 sum_128 = _mm_add_ps(low, high); // [(a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7), (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7), (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7), (b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7)]
// the sum of all elements in the originla vector is now stored in the first 32 bits of the vector. Extract it 
FType distance = _mm_cvtss_f32(sum_128);
**/

