/**
 * Python bindings for MLCS Algorithm C++ implementation using pybind11.
 * This file connects the C++ implementation to the Python codebase.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include "mlcs_algorithm.h"

namespace py = pybind11;

PYBIND11_MODULE(mlcs_cpp, m) {
    m.doc() = "MLCS Algorithm C++ implementation for Video Lecture Content Indexer";

    py::class_<MLCSAlgorithm>(m, "MLCSAlgorithm")
        .def(py::init<const std::string&>(), py::arg("language") = "en")
        .def("preprocess_text", &MLCSAlgorithm::preprocessText,
            py::arg("text"), py::arg("language") = "")
        .def("normalize_token", &MLCSAlgorithm::normalizeToken,
            py::arg("token"), py::arg("language") = "")
        .def("generate_variants", &MLCSAlgorithm::generateVariants,
            py::arg("text"))
        .def("match_variants", &MLCSAlgorithm::matchVariants,
            py::arg("text"), py::arg("target"))
        .def("find_mlcs", &MLCSAlgorithm::findMlcs,
            py::arg("sequences"), py::arg("min_length") = 2)
        .def("extract_concept_signature", &MLCSAlgorithm::extractConceptSignature,
            py::arg("concept_text"), py::arg("contexts"), py::arg("language") = "");
}
