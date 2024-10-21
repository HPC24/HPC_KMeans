#ifndef CONT_MEM_PARALLEL_KMEANS_H
#define CONT_MEM_PARALLEL_KMEANS_H

#include <vector>
#include <random>
#include <iostream>
#include <concepts>
#include <optional>
#include <Aligned_Allocator.h>

template <std::floating_point FType, std::integral IType = std::size_t>
class Parallel_KMeans {

public:

    const int n_cluster;
    const int max_iter;
    const double tol;
    std::optional<int> seed;
    int n_iter;
    double inertia;
    std::mt19937 gen;

    std::vector<FType, AlignedAllocator<FType>> centroids;
    std::vector<int> labels;

    Parallel_KMeans(const int n_cluster, const int max_iter, const double tol, std::optional<int> seed = std::nullopt);
    void fit(const std::vector<std::vector<FType>>& data);
    std::vector<int> predict(const std::vector<std::vector<FType>>& new_data);

private:
    



    void initializeCentroids(const std::vector<FType, AlignedAllocator<FType>>& data, const IType rows, const IType cols);
    void ReinitializeCentroids(const std::vector<FType, AlignedAllocator<FType>>& data, std::vector<FType, AlignedAllocator<FType>>& new_centroids, int cluster_idx, const IType rows, const IType cols);
    void assignCentroids(const std::vector<FType, AlignedAllocator<FType>>& data, const IType rows, const IType cols);
    void updateCentroids(const std::vector<FType, AlignedAllocator<FType>>& data, std::vector<FType, AlignedAllocator<FType>>& new_centroids, const IType rows, const IType cols);
    bool calculateChange(std::vector<FType, AlignedAllocator<FType>>& new_centroids, const IType cols);



};

#endif