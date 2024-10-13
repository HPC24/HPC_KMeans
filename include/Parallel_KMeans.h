#ifndef PARALLEL_KMEANS_H
#define PARALLEL_KMEANS_H

#include <vector>
#include <random>
#include <iostream>
#include <concepts>
#include <optional>

template <typename FType, typename IType = std::size_t>

class Parallel_KMeans {

public:

    const int n_cluster;
    const int max_iter;
    const double tol;
    std::optional<int> seed;
    int n_iter;
    double inertia;
    std::mt19937 gen;

    std::vector<std::vector<FType>> centroids;
    std::vector<int> labels;

    Parallel_KMeans(const int n_cluster, const int max_iter, const double tol, std::optional<int> seed = std::nullopt);
    void fit(const std::vector<std::vector<FType>>& data);
    std::vector<int> predict(std::vector<std::vector<FType>>& new_data);

private:
    



    void initializeCentroids(const std::vector<std::vector<FType>>& data);
    void ReinitializeCentroids(const std::vector<std::vector<FType>>& data, std::vector<std::vector<FType>>& new_centroids, int cluster_idx);
    void assignCentroids(const std::vector<std::vector<FType>>& data, const IType rows, const IType cols);
    void updateCentroids(const std::vector<std::vector<FType>>& data, std::vector<std::vector<FType>>& new_centroids, const IType rows, const IType cols);
    bool calculateChange(std::vector<std::vector<FType>>& new_centroids, const IType cols);



};

#endif