#include <Cont_Mem_Parallel_KMeans.h>
#include <Aligned_Allocator.h>
#include <SIMD_Operations.h>
#include <Tests.h>

#include <iostream>
#include <random>
#include <concepts>
#include <omp.h>
#include <optional>


template <std::floating_point FType, std::integral IType>
Parallel_KMeans<FType, IType>::Parallel_KMeans(const int n_cluster, const int max_iter, const double tol, std::optional<int> seed)
    : n_cluster{n_cluster},
    max_iter{max_iter},
    tol{tol},
    seed{seed},
    n_iter{0},
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

template <std::floating_point FType, std::integral IType>
void Parallel_KMeans<FType, IType>::initializeCentroids(const std::vector<FType, AlignedAllocator<FType>>& data, const IType rows, const IType cols){

    // get the random initial centroids form the intial data
    std::uniform_int_distribution<> dist{0,  static_cast<int>(rows - 1)};
    
    // set the initial centroids as point form the data
    for (IType row = 0; row < n_cluster; ++row)
    {
                FType* centroid_ptr = &centroids[row * cols];
                const FType* data_ptr = &data[dist(gen) * cols];

            for (IType col = 0; col < cols; ++col)
            {
                centroid_ptr[col] = data_ptr[col];
            }
    }

}

template <std::floating_point FType, std::integral IType>
void Parallel_KMeans<FType, IType>::ReinitializeCentroids(
    const std::vector<FType, AlignedAllocator<FType>>& data, 
    std::vector<FType, AlignedAllocator<FType>>& new_centroids, 
    int cluster_idx,
    const IType rows, 
    const IType cols){

    // get the random initial centroids form the intial data
    std::uniform_int_distribution<> dist{0,  static_cast<int>(rows - 1)};
    
    FType* centroid_ptr = &new_centroids[cluster_idx * cols];
    const FType* data_ptr = &data[dist(gen) * cols];

    for (IType col = 0; col < cols; ++col)
    {
        centroid_ptr[col] = data_ptr[col];
    }
}

template <std::floating_point FType, std::integral IType>
void Parallel_KMeans<FType, IType>::fit(const std::vector<std::vector<FType>>& data){

    // set the constant row and col size to determine later loop iterations
    const IType rows = data.size();
    const IType cols = data.empty() ? 0 : data[0].size();

    if (cols == 0)
    {
        std::cerr << "Data vector is empty" << std::endl;
    }

    //std::cout << "Rows:" << rows << std::endl;
    //std::cout << "Cols:" << cols << std::endl;
    

    std::vector<FType, AlignedAllocator<FType>> new_data(rows * cols, 0);

    // fill the new flat array with values
    for (IType row = 0; row < rows; ++row)
    {
        FType* new_data_ptr = &new_data[row * cols];

        for (IType col = 0; col < cols; ++col)
        {
            new_data_ptr[col] = data[row][col];
        }
    }

    if (is_memory_aligned(new_data))
    {
        std::cout << "new_data is memory aligned" << std::endl;
    }
    else
    {
        std::cout << "new_data is NOT memory aligned" << std::endl;
    }

    //std::cout << "New_data expected size:" << rows * cols << std::endl;
    //std::cout << "Actual size:" << new_data.size() << std::endl;

    // initialize the new centroids and the centroid member variables with 0's
    std::vector<FType, AlignedAllocator<FType>> new_centroids(n_cluster * cols, 0);
    centroids = new_centroids;

    #ifdef DEBUG
    std::cout << "Fit first call new_centroids" << std::endl;

    for (IType i = 0; i < rows; ++i)
    {
        
        for (IType j = 0; j < cols; ++j)
        {
            std::cout << new_centroids[i * cols + col] << " ";
        }

        std::cout << std::endl;
    }
    #endif

    // initiaize the labels of the points
    std::vector<int> labels_new(rows, 0);
    this->labels = std::move(labels_new);

    initializeCentroids(new_data, rows, cols);
    int iter = 1;

    for (; iter < this->max_iter + 1; ++iter){

        assignCentroids(new_data, rows, cols);
        updateCentroids(new_data, new_centroids, rows, cols);
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

    if (iter == this->max_iter)
    {
        std::cout << "Maximum number of iterations has been reached" << std::endl;
        std::cout << "Maximum number of iterations: " << this->max_iter << std::endl;
        this->n_iter = iter;
    }
 

}

template <std::floating_point FType, std::integral IType>
std::vector<int> Parallel_KMeans<FType, IType>::predict(const std::vector<std::vector<FType>>& new_data){

    const IType ROWS = new_data.size();
    const int COLS = new_data.empty() ? 0: new_data[0].size();
    if (COLS == 0)
    {
        std::cerr << "Data vector is empty" << std::endl;
        return {};
    }

    std::vector<int> new_labels(ROWS, 0);

    # pragma omp parallel for default(none) shared(new_labels, new_data, COLS, ROWS, centroids, n_cluster)
    for (IType point = 0; point < ROWS; ++point)
    {
        FType min_distance = std::numeric_limits<FType>::max();
        int best_centroid_idx = 0;
        const FType* new_data_ptr = new_data[point].data();

        for (int centroid = 0; centroid < n_cluster; ++centroid)
        {
            FType distance = 0;
            const FType* centroids_ptr = &centroids[centroid * COLS];
            

            distance = process(new_data_ptr, centroids_ptr, COLS);

            // for (IType col_idx = 0; col_idx < COLS; ++col_idx)
            // {
                

            //     // FType diff = new_data[point][col_idx] - centroids_ptr[col_idx];
            //     // distance += diff * diff;
            // }

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

template <std::floating_point FType, std::integral IType>
void Parallel_KMeans<FType, IType>::assignCentroids(
    const std::vector<FType, AlignedAllocator<FType>>& data, 
    IType rows, 
    IType cols
) {

    double inertia_shared = 0; 

    // find the new centroid for every data point
    #pragma omp parallel for default(none) shared(data, rows, cols, n_cluster, labels, centroids) reduction( +: inertia_shared) schedule(static)
    for (IType point = 0; point < rows; ++point)
    {
        
        // resets for each data point to find it's min distance
        FType min_distance = std::numeric_limits<FType>::max();
        int best_centroid_idx = 0;
        // Each Thread gets a ptr to the point it is currently processing
        const FType* data_ptr = &data[point * cols];

        for (IType centroid_idx = 0; centroid_idx < n_cluster; ++centroid_idx){

                FType distance = 0;
                // each thread gets a ptr to the current centroid it is processing 
                const FType* centroid_ptr = &centroids[centroid_idx * cols];

                #if defined(SIMD_256) || defined(SIMD_512)
                distance = process(data_ptr, centroid_ptr, cols);
                #endif

            #ifdef NO_SIMD
            #pragma omp simd
            for (IType coord_idx = 0; coord_idx < cols; ++coord_idx)
            {

                 FType single_distance = data_ptr[coord_idx] - centroid_ptr[coord_idx];
                 distance += single_distance * single_distance;

            }
            #endif

            // calcuate the sqrt of the distance later once min distance is found so save some computation time
            // fiding the min distance does not change 
            // distance = std::sqrt(distance);

            if (distance < min_distance){
                min_distance = distance;
                best_centroid_idx = centroid_idx;
            }

        }

        labels[point] = best_centroid_idx;
        inertia_shared += std::sqrt(min_distance);
    }

    this->inertia = inertia_shared;


    #ifdef DEBUG
    std::cout << "Assign Centroids Labels " << std::endl;

    for (IType i = 0; i < labels.size(); ++i)
    {
        std::cout << labels[i] << " ";
    }

    std::cout << std::endl;
    #endif

}


template <std::floating_point FType, std::integral IType>
void Parallel_KMeans<FType, IType>::updateCentroids(
    const std::vector<FType, AlignedAllocator<FType>>& data, 
    std::vector<FType, AlignedAllocator<FType>>& new_centroids,
    const IType rows, 
    const IType cols)
    {


    // reset the centroids before each update
    std::fill(new_centroids.begin(), new_centroids.end(), 0.0);

    std::vector<int> counts(n_cluster, 0);
    const int n_clusters = n_cluster;

    #pragma omp parallel default(none) shared(counts, data, rows, cols, n_clusters, new_centroids, labels)
    {

    std::vector<int> counts_private (n_clusters, 0);
    std::vector<FType> new_centroids_partial(n_clusters * cols);

    #pragma omp for nowait
    for (IType point = 0; point < rows; ++point)
    {

            int cluster = labels[point];
            counts_private[cluster] += 1;
            FType* new_centroids_partial_ptr = &new_centroids_partial[cluster * cols];
            const FType* data_ptr = &data[point * cols];
            
            for (IType col_idx = 0; col_idx < cols; ++col_idx)
            {
                
                new_centroids_partial_ptr[col_idx] += data_ptr[col_idx];

            }
            
    }

    #pragma omp critical
    {
        for (int centroid = 0; centroid < n_clusters; ++centroid)
        {
            counts[centroid] += counts_private[centroid];
            FType* new_centroids_ptr = &new_centroids[centroid * cols];
            FType* new_centroids_partial_ptr = &new_centroids_partial[centroid * cols];

            for (IType col_idx = 0; col_idx < cols; ++col_idx)
            {
                new_centroids_ptr[col_idx] += new_centroids_partial_ptr[col_idx];
            }
        }
    }




    }


    for (int cluster_idx = 0; cluster_idx < this->n_cluster; ++cluster_idx){

        FType* new_centroids_ptr = &new_centroids[cluster_idx * cols];

        if (counts[cluster_idx] > 0)
        {
            for (IType col_idx = 0; col_idx < cols; ++col_idx)
            {
                new_centroids_ptr[col_idx] /= counts[cluster_idx];
            }
        }

        else
        {
            ReinitializeCentroids(data, new_centroids, cluster_idx, rows, cols);
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

    for (IType i = 0; i < new_centroids.size(); ++i)
    {
        
        for (IType j = 0; j < new_centroids[0].size(); ++j)
        {
            std::cout << new_centroids[i * cols + j] << " ";
        }

        std::cout << std::endl;
    }
    #endif


}


template <std::floating_point FType, std::integral IType>
bool Parallel_KMeans<FType, IType>::calculateChange(std::vector<FType, AlignedAllocator<FType>>& new_centroids, const IType cols){

    #ifdef DEBUG
    std::cout << "Calculate change this centroids " << std::endl;

    for (IType i = 0; i < rows; ++i)
    {
        
        for (IType j = 0; j < cols; ++j)
        {
            std::cout << this->centroids[i * cols + j] << " ";
        }

        std::cout << std::endl;
    }

     std::cout << "Calculate change new centroids " << std::endl;

    for (IType i = 0; i < rows; ++i)
    {
        
        for (IType j = 0; j < cols; ++j)
        {
            std::cout << new_centroids[i * cols + col] << " ";
        }

        std::cout << std::endl;
    }
    #endif

    double epsilon = this->tol;
    bool converged = true;

    for (int centroid_idx = 0; centroid_idx < this->n_cluster; ++centroid_idx)
    {
        double centroids_difference = 0;
        FType* centroids_ptr = &centroids[centroid_idx * cols];
        FType* new_centroids_ptr = &new_centroids[centroid_idx * cols];

        for (IType col_idx = 0; col_idx < cols; ++col_idx)
        {
            double centroids_diff = centroids_ptr[col_idx] - new_centroids_ptr[col_idx];
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
template class Parallel_KMeans<float, unsigned int>;
template class Parallel_KMeans<double, std::size_t>;
template class Parallel_KMeans<double, unsigned int>;

