#ifndef KMEANS_H
#define KMEANS_H

#include <vector>
#include <random>
#include <iostream>
#include <concepts>
#include <optional>

template <typename FType, typename IType = std::size_t>

class KMeans {

public:

    const int n_cluster;
    const int max_iter;
    int n_iter;
    const double tol;
    double inertia; 
    std::optional<int> seed;
    std::mt19937 gen;
    

    std::vector<std::vector<FType>> centroids;
    std::vector<int> labels;

    // std::optional used to either initialize the seed with an int or generate it randomly if not specified
    KMeans(const int n_cluster, const int max_iter, const double tol, std::optional<int> seed = std::nullopt);
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