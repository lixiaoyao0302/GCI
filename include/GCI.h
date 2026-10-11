#pragma once

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gci {

using Point = pcl::PointXYZ;
using PointCloud = pcl::PointCloud<Point>;
using PointCloudPtr = PointCloud::Ptr;
using PointCloudConstPtr = PointCloud::ConstPtr;

// Callers must explicitly supply valid options; zero initialization is not a preset.
struct Options {
    std::size_t max_correspondences{};
    std::size_t normal_neighbors{};
    std::size_t relation_degree{};
    // Maximum average degree of the relation graph used after exact edge discovery.
    // Zero preserves the full graph. The resulting edge count is at most
    // ceil(edge_degree_budget * N / 2).
    std::size_t edge_degree_budget{};
    std::size_t hypothesis_count{};  // Total candidate evaluation budget H.
    std::size_t selection_budget{};
    std::size_t surface_sample_count{};
    std::size_t mean_field_iterations{};
    std::size_t outer_iterations{};
    std::size_t pose_iterations{};
    double relation_threshold{};  // tau_g: compatible-edge threshold (length units).
    double relation_sigma{};      // sigma_g: pairwise Gaussian noise scale.
    double normal_sigma{};
    double tangent_normal_ratio{};
    double prior_inlier{};
    // Optional a_i in ORIGINAL correspondence order; empty uses prior_inlier.
    std::vector<double> correspondence_priors;
    double mean_field_damping{};  // Retained old probability; paper gamma = 1 - this value.
    double information_regularizer{};
    double rotation_tolerance{};
    double translation_tolerance{};
    double objective_tolerance{};
    double surface_sigma_multiplier{};
    double surface_score_weight{};
    bool use_graph_posterior{};
    bool use_balanced_inference{};
    bool use_information_design{};
    bool use_symmetric_surface{};
    unsigned int seed{};
    int normal_threads{};
    int omp_threads{};
};

struct CandidateDiagnostic {
    std::string source;
    std::vector<int> support_indices;
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    double probabilistic_score = 0.0;
    double selection_score = 0.0;
    bool selected_for_initialization = false;
};

struct Diagnostics {
    std::size_t input_correspondences = 0;
    std::size_t used_correspondences = 0;
    std::size_t relation_count = 0;
    std::size_t tested_hypotheses = 0;
    std::size_t selected_count = 0;
    std::size_t completed_outer_iterations = 0;
    double characteristic_length = 1.0;
    double final_elbo = 0.0;
    std::size_t information_rank = 0;
    std::size_t selected_information_rank = 0;
    double min_information_eigenvalue = 0.0;
    double information_condition_number = 0.0;
    double covariance_trace = 0.0;
    double selected_min_information_eigenvalue = 0.0;
    double selected_information_condition_number = 0.0;
    double selected_covariance_trace = 0.0;
    double preprocessing_seconds = 0.0;
    double graph_seconds = 0.0;
    double posterior_seconds = 0.0;
    double hypothesis_seconds = 0.0;
    double selection_seconds = 0.0;
    double pose_seconds = 0.0;
    double diagnostics_seconds = 0.0;
    double total_seconds = 0.0;
    bool converged = false;
    std::vector<int> retained_input_indices;
    std::vector<int> selected_indices;
    std::vector<double> posterior;
    std::vector<double> information_eigenvalues;
    std::vector<double> selected_information_eigenvalues;
    std::vector<double> objective_history;
    Eigen::Matrix4d initialization_transformation = Eigen::Matrix4d::Identity();
    std::vector<CandidateDiagnostic> candidates;
};

struct Result {
    Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
    Diagnostics diagnostics;
    bool success = false;
    std::string message;
};

// Estimate target <- source. Ground truth is intentionally absent from this API.
Result registerPointClouds(
    const PointCloudConstPtr& source_reference,
    const PointCloudConstPtr& target_reference,
    const PointCloudConstPtr& source_correspondences,
    const PointCloudConstPtr& target_correspondences,
    const Options& options);

void writeDiagnosticsJson(const std::string& path, const Result& result);

}  // namespace gci
