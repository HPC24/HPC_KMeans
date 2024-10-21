#ifndef ALIGNED_ALLOCATOR_H
#define ALIGNED_ALLOCATOR_H
#include <memory>
#include <cstdlib>   
#include <iostream>

#ifdef SIMD_512
constexpr std::size_t BYTE_ALIGNMENT = 64; // 64-byte alignment for AVX-512
#elif defined(SIMD_256)
constexpr std::size_t BYTE_ALIGNMENT = 32; // 32-byte alignment for AVX
#else
constexpr std::size_t BYTE_ALIGNMENT = 32; // Default 32-byte alignment
#endif

template<typename FType>
struct AlignedAllocator{

    using value_type = FType;

    AlignedAllocator() = default;

    template<typename U>
    AlignedAllocator(const AlignedAllocator<U>& other) noexcept {};

    FType* allocate(std::size_t n){

        if (n == 0) {
            return nullptr; // Return nullptr if no memory is needed
        }
        
        std::size_t bytes = n * sizeof(FType);
        void* p = nullptr;
        if (posix_memalign(&p, BYTE_ALIGNMENT, bytes) != 0)
        {
              throw std::bad_alloc();
        }
          
        else 
            return reinterpret_cast<FType*>(p);

    }

    void deallocate(FType* p, std::size_t) {
        free(p);
    }

    template <typename U>
    bool operator==(const AlignedAllocator<U>& other) {return true;}

    template <typename U>
    bool operator!=(const AlignedAllocator<U>& other) {return false;}

};




#endif