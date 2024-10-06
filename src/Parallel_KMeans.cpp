#include <Parallel_KMeans.h>
#include <iostream>
#include <random>
#include <concepts>
#include <omp.h>
#include <optional>

template <typename FType, typename IType>
Parallel_KMeans<FType, IType>::Parallel_KMeans(const int n_cluster, const int max_iter, const double tol, std::optional<int> seed)
    : n_cluster{n_cluster},
    max_iter{max_iter},
    tol{tol},
    n_iter{0},
    seed{seed},
    gen(seed.value_or(std::random_device{}()))
    {

        std::cout << "Initialized Parallel KMeans object" << std::endl;
        std::cout << "Number of clusters: " << this->n_cluster << std::endl;
        std::cout << "Max Iterations: " << this-> max_iter << std::endl;
        std::cout << "Tolerance: " << this->tol << std::endl;

        if (seed.has_value())
        {
            std::cout << "Seed: " << this->seed.value() << std::endl;
        }
        else if (!seed.has_value())
        {
            std::cout << "Seed was initialized randomly " << std::endl;
        }

    }

template <typename FType, typename IType>
void Parallel_KMeans<FType, IType>::initializeCentroids(const std::vector<std::vector<FType>>& data){

    // get the random initial centroids form the intial data
    std::uniform_int_distribution<> dist{0,  static_cast<int>(data.size() - 1)};
    
    // set the initial centroids as point form the data
    for (int i = 0; i < this->n_cluster; i++){

        this->centroids[i] = data[dist(this->gen)];
    }

}

template <typename FType, typename IType>
void Parallel_KMeans<FType, IType>::ReinitializeCentroids(const std::vector<std::vector<FType>>& data, std::vector<std::vector<FType>>& new_centroids, int cluster_idx){

    // get the random initial centroids form the intial data
    std::uniform_int_distribution<> dist{0,  static_cast<int>(data.size() - 1)};
    
    // set the initial centroids as point form the data
    new_centroids[cluster_idx] = data[dist(this->gen)];

}
template <typename FType, typename IType>
void Parallel_KMeans<FType, IType>::fit(const std::vector<std::vector<FType>>& data){

    // set the constant row and col size to determine later loop iterations
    const IType rows = data.size();
    const IType cols = data.empty() ? 0 : data[0].size();

    if (cols == 0)
    {
        std::cerr << "Data vector is empty" << std::endl;
    }

    // initialize the new centroids and the centroid member variables with 0's
    std::vector<std::vector<FType>> new_centroids(this->n_cluster, std::vector<FType>(data[0].size(), 0.0));
    this->centroids = new_centroids;

    #ifdef DEBUG
    std::cout << "Fit first call new_centroids" << std::endl;

    for (int i = 0; i < new_centroids.size(); ++i)
    {
        
        for (int j = 0; j < new_centroids[0].size(); ++j)
        {
            std::cout << new_centroids[i][j] << " ";
        }

        std::cout << std::endl;
    }
    #endif

    // initiaize the labels of the points
    std::vector<int> labels(rows, 0);
    this->labels = std::move(labels);

    initializeCentroids(data);
    IType iter = 1;

    for (iter; iter < this->max_iter + 1; ++iter){

        assignCentroids(data, rows, cols);
        updateCentroids(data, new_centroids, rows, cols);
        bool converged = calculateChange(new_centroids, cols);

        if (converged)
        {
            std::cout << "Centroid positions have not changed anymore after " << iter << " iterations " 
            << "within a tolerance of " << this->tol << std::endl;
            this->n_iter = iter;
            break;
            
        }
        else
        {
            this->centroids = new_centroids;
        }
        
    }

    if (iter == this->max_iter + 1)
    {
        std::cout << "Maximum number of iterations has been reached" << std::endl;
        std::cout << "Maximum number of iterations: " << this->max_iter << std::endl;
        this->n_iter = iter;
    }
 

}

template <typename FType, typename IType>
std::vector<int> Parallel_KMeans<FType, IType>::predict(std::vector<std::vector<FType>>& new_data){

    const int COLS = new_data.empty() ? 0: new_data[0].size();
    if (COLS == 0)
    {
        std::cerr << "Data vector is empty" << std::endl;
        return {};
    }

    std::vector<int> new_labels(new_data.size(), 0);

    # pragma omp parallel for default(none) shared(new_labels, new_data, COLS, centroids)
    for (int point = 0; point < new_data.size(); ++point)
    {
        FType min_distance = std::numeric_limits<FType>::max();
        int best_centroid_idx = 0;

        for (int centroid = 0; centroid < centroids.size(); ++centroid)
        {
            FType distance = 0;

            for (int col_idx = 0; col_idx < COLS; ++col_idx)
            {
                FType diff = new_data[point][col_idx] - centroids[centroid][col_idx];
                distance += diff * diff;
            }

            distance = std::sqrt(distance);

            if (distance < min_distance)
            {
                min_distance = distance;
                best_centroid_idx = centroid;
            }
        }

        new_labels[point] = best_centroid_idx;
    }

    return new_labels;

}

template <typename FType, typename IType>
void Parallel_KMeans<FType, IType>::assignCentroids(
    const std::vector<std::vector<FType>>& data, 
    IType rows, 
    IType cols
) {

    double inertia = 0; 
    int n_cluster = this->n_cluster;

    // find the new centroid for every data point
    #pragma omp parallel for default(none) shared(data, rows, cols, n_cluster, labels, centroids) reduction( +: inertia)
    for (IType point = 0; point < rows; ++point)
    {
        
        // resets for each data point to find it's min distance
        FType min_distance = std::numeric_limits<FType>::max();
        int best_centroid_idx = 0;

        for (int centroid_idx = 0; centroid_idx < n_cluster; ++centroid_idx){

                FType distance = 0;

            for (IType coord_idx = 0; coord_idx < cols; ++coord_idx){
                
                FType single_distance = data[point][coord_idx] - centroids[centroid_idx][coord_idx];
                distance += single_distance * single_distance;

            }

            // calcuate the sqrt of the distance later once min distance is found so save some computation time
            // fiding the min distance does not change 
            // distance = std::sqrt(distance);

            if (distance < min_distance){
                min_distance = distance;
                best_centroid_idx = centroid_idx;
            }

        }

        labels[point] = best_centroid_idx;
        inertia += std::sqrt(min_distance);
    }

    this->inertia = inertia;


    #ifdef DEBUG
    std::cout << "Assign Centroids Labels " << std::endl;

    for (int i = 0; i < this->labels.size(); ++i)
    {
        std::cout << this->labels[i] << " ";
    }

    std::cout << std::endl;
    #endif

}


template <typename FType, typename IType>
void Parallel_KMeans<FType, IType>::updateCentroids(
    const std::vector<std::vector<FType>>& data, 
    std::vector<std::vector<FType>>& new_centroids,
    const IType rows, 
    const IType cols)
    {


    // reset the centroids before each update
    for (auto& centroid: new_centroids)
    {
        std::fill(centroid.begin(), centroid.end(), 0.0);
    }

    std::vector<IType> counts(this->n_cluster, 0);
    const int n_clusters = this->n_cluster;

    #pragma omp parallel default(none) shared(counts, data, rows, cols, n_clusters, new_centroids, labels)
    {

    std::vector<int> counts_private (n_clusters, 0);
    std::vector<std::vector<FType>> new_centroids_partial(n_clusters, std::vector<FType>(cols, 0.0));

    #pragma omp for nowait
    for (IType point = 0; point < rows; ++point)
    {

            int cluster = labels[point];
            counts_private[cluster] += 1;
            
            for (IType col_idx = 0; col_idx < cols; ++col_idx)
            {
                
                new_centroids_partial[cluster][col_idx] += data[point][col_idx];

            }
            
    }

    #pragma omp critical
    {
        for (int centroid = 0; centroid < n_clusters; ++centroid)
        {
            counts[centroid] += counts_private[centroid];

            for (IType col_idx = 0; col_idx < cols; ++col_idx)
            {
                new_centroids[centroid][col_idx] += new_centroids_partial[centroid][col_idx];
            }
        }
    }




    }


    for (int cluster_idx = 0; cluster_idx < this->n_cluster; ++cluster_idx){

        if (counts[cluster_idx] > 0)
        {
            for (IType col_idx = 0; col_idx < cols; ++col_idx)
            {
                new_centroids[cluster_idx][col_idx] /= counts[cluster_idx];
            }
        }

        else
        {
             for (IType col_idx = 0; col_idx < cols; ++col_idx)
            {
                ReinitializeCentroids(data, new_centroids, cluster_idx);
            }
        }

    }
    
    #ifdef DEBUG
    std::cout << "Counts vector" << std::endl;

    for (int i = 0; i < counts.size(); ++i)
    {
        std::cout << counts[i] << " ";
    }

    std::cout << std::endl;

    std::cout << "Update Centroids new centroids end" << std::endl;

    for (int i = 0; i < new_centroids.size(); ++i)
    {
        
        for (int j = 0; j < new_centroids[0].size(); ++j)
        {
            std::cout << new_centroids[i][j] << " ";
        }

        std::cout << std::endl;
    }
    #endif


}


template <typename FType, typename IType>
bool Parallel_KMeans<FType, IType>::calculateChange(std::vector<std::vector<FType>>& new_centroids, const IType cols){

    #ifdef DEBUG
    std::cout << "Calculate change this centroids " << std::endl;

    for (int i = 0; i < this->centroids.size(); ++i)
    {
        
        for (int j = 0; j < this->centroids[0].size(); ++j)
        {
            std::cout << this->centroids[i][j] << " ";
        }

        std::cout << std::endl;
    }

     std::cout << "Calculate change new centroids " << std::endl;

    for (int i = 0; i < new_centroids.size(); ++i)
    {
        
        for (int j = 0; j < new_centroids[0].size(); ++j)
        {
            std::cout << new_centroids[i][j] << " ";
        }

        std::cout << std::endl;
    }
    #endif

    double epsilon = this->tol;
    bool converged = true;

    for (int centroid_idx = 0; centroid_idx < this->n_cluster; ++centroid_idx)
    {
        double centroids_difference = 0;

        for (IType col_idx = 0; col_idx < cols; ++col_idx)
        {
            double centroids_diff = this->centroids[centroid_idx][col_idx] - new_centroids[centroid_idx][col_idx];
            centroids_difference += centroids_diff * centroids_diff;

        }

        centroids_difference = std::sqrt(centroids_difference);

        if (centroids_difference >= epsilon)
        {

            converged = false;
            break;
        
        }


    }


    return converged;

}

 
template class Parallel_KMeans<float, std::size_t>;