#ifndef TESTS_H
#define TESTS_H
#include <Aligned_Allocator.h>

#include <cstdint>
#include <vector>

void CheckLabels();
template <typename FType>
void CheckData(std::vector<std::vector<FType>>& data);

template <typename FType>
bool is_memory_aligned(const std::vector<FType, AlignedAllocator<FType>>& vec) {

        // get the pointer to the first addess if this is evenly divisable by the bit length of the type then all elements are aligned
        std::cout << "Using BYTE_ALIGNMENT:" << BYTE_ALIGNMENT << std::endl;
        const FType* data_ptr = vec.data(); 
        return reinterpret_cast<std::uintptr_t>(data_ptr) %  BYTE_ALIGNMENT == 0;

    }

#endif