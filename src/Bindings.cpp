#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Cont_Mem_Parallel_KMeans.h" 

#include <vector>
#include <random>
#include <iostream>
#include <concepts>
#include <optional>

namespace py = pybind11;

// Instantiate the template for specific types (e.g., double for FType and size_t for IType)
using ParallelKMeansDouble = Parallel_KMeans<double, std::size_t>;
using ParallelKMeansFloat = Parallel_KMeans<float, std::size_t>;

PYBIND11_MODULE(P_KMeansLib, m) {
    // Binding the Parallel_KMeans class with double precision (double, std::size_t)
    py::class_<ParallelKMeansDouble>(m, "Parallel_KMeans_Double")
        .def(py::init<const int, const int, const double, std::optional<int>>())  // Expose the constructor
        .def("fit", &ParallelKMeansDouble::fit)  // Bind the fit method
        .def("predict", &ParallelKMeansDouble::predict)  // Bind the predict method

        .def_readonly("n_cluster", &ParallelKMeansDouble::n_cluster)
        .def_readonly("max_iter", &ParallelKMeansDouble::max_iter)
        .def_readonly("tol", &ParallelKMeansDouble::tol)
        .def_readonly("n_iter", &ParallelKMeansDouble::n_iter)
        .def_readonly("inertia", &ParallelKMeansDouble::inertia)
        .def_readonly("labels", &ParallelKMeansDouble::labels);


    py::class_<ParallelKMeansFloat>(m, "Parallel_KMeans_Float")
        .def(py::init<const int, const int, const float, std::optional<int>>())  // Expose the constructor
        .def("fit", &ParallelKMeansFloat::fit)  // Bind the fit method
        .def("predict", &ParallelKMeansFloat::predict)  // Bind the predict method

        .def_readonly("n_cluster", &ParallelKMeansFloat::n_cluster)
        .def_readonly("max_iter", &ParallelKMeansFloat::max_iter)
        .def_readonly("tol", &ParallelKMeansFloat::tol)
        .def_readonly("n_iter", &ParallelKMeansFloat::n_iter)
        .def_readonly("inertia", &ParallelKMeansFloat::inertia)
        .def_readonly("labels", &ParallelKMeansFloat::labels);

}