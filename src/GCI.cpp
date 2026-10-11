#include "GCI.h"

#include <pcl/features/normal_3d_omp.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace gci {
namespace {

using Vec3 = Eigen::Vector3d;
using Mat3 = Eigen::Matrix3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat6 = Eigen::Matrix<double, 6, 6>;
constexpr double kPi = 3.14159265358979323846;
constexpr double kTiny = 1e-12;

struct Relation {
    int i = 0;
    int j = 0;
    double discrepancy = 0.0;
    double log_inlier = 0.0;
    double log_outlier = 0.0;
};

struct State {
    std::vector<Vec3> source;
    std::vector<Vec3> target;
    std::vector<Vec3> source_normals;
    std::vector<Vec3> target_normals;
    std::vector<Relation> relations;
    std::vector<std::vector<int>> incident_relations;
    std::vector<double> prior;
    std::vector<Vec3> source_surface_samples;
    std::vector<Vec3> target_surface_samples;
    pcl::KdTreeFLANN<Point>::Ptr source_surface_tree;
    pcl::KdTreeFLANN<Point>::Ptr target_surface_tree;
    Vec3 source_center = Vec3::Zero();
    Vec3 target_center = Vec3::Zero();
    double length = 1.0;
    double residual_half_width = 1.0;
};

double clampProbability(double value) {
    return std::min(1.0 - 1e-9, std::max(1e-9, value));
}

double sigmoid(double value) {
    if (value >= 0.0) {
        const double e = std::exp(-std::min(value, 60.0));
        return 1.0 / (1.0 + e);
    }
    const double e = std::exp(std::max(value, -60.0));
    return e / (1.0 + e);
}

Mat3 skew(const Vec3& v) {
    Mat3 m;
    m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return m;
}

Mat3 expSO3(const Vec3& omega) {
    const double theta = omega.norm();
    if (theta < 1e-10) return Mat3::Identity() + skew(omega);
    return Eigen::AngleAxisd(theta, omega / theta).toRotationMatrix();
}

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    double result = values[middle];
    if (values.size() % 2 == 0) {
        const auto lower = std::max_element(values.begin(), values.begin() + middle);
        result = 0.5 * (result + *lower);
    }
    return result;
}

Vec3 coordinateMedian(const std::vector<Vec3>& points) {
    Vec3 result = Vec3::Zero();
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<double> values;
        values.reserve(points.size());
        for (const Vec3& point : points) values.push_back(point[axis]);
        result[axis] = median(std::move(values));
    }
    return result;
}

std::vector<int> retainedIndices(std::size_t count, std::size_t maximum) {
    const std::size_t used = maximum == 0 ? count : std::min(count, maximum);
    std::vector<int> indices;
    indices.reserve(used);
    if (used == count) {
        for (std::size_t i = 0; i < count; ++i) indices.push_back(static_cast<int>(i));
        return indices;
    }
    for (std::size_t i = 0; i < used; ++i) {
        const double position = (static_cast<double>(i) + 0.5) * static_cast<double>(count) /
            static_cast<double>(used);
        indices.push_back(std::min(static_cast<int>(count - 1), static_cast<int>(position)));
    }
    return indices;
}

std::vector<Vec3> extractPoints(const PointCloudConstPtr& cloud, const std::vector<int>& indices) {
    std::vector<Vec3> output;
    output.reserve(indices.size());
    for (int index : indices) {
        const Point& point = cloud->points[static_cast<std::size_t>(index)];
        if (!pcl::isFinite(point)) throw std::invalid_argument("Point cloud contains NaN or Inf");
        output.emplace_back(point.x, point.y, point.z);
    }
    return output;
}

PointCloudPtr extractCloud(const PointCloudConstPtr& cloud, const std::vector<int>& indices) {
    PointCloudPtr output(new PointCloud);
    output->reserve(indices.size());
    for (int index : indices) output->push_back(cloud->points[static_cast<std::size_t>(index)]);
    return output;
}

std::vector<Vec3> estimateNormals(
    const PointCloudConstPtr& reference,
    const PointCloudConstPtr& queries,
    const Options& options) {
    pcl::NormalEstimationOMP<Point, pcl::Normal> estimator;
    // Keep local covariance estimation bitwise-stable by default. Candidate
    // evaluation is the intended OpenMP hotspot; users may opt in to parallel
    // normals explicitly through normal_threads.
    estimator.setNumberOfThreads(options.normal_threads > 0 ? options.normal_threads : 1);
    pcl::search::KdTree<Point>::Ptr tree(new pcl::search::KdTree<Point>);
    estimator.setSearchMethod(tree);
    estimator.setInputCloud(queries);
    estimator.setSearchSurface(reference);
    estimator.setKSearch(static_cast<int>(std::max<std::size_t>(3, options.normal_neighbors)));
    pcl::PointCloud<pcl::Normal> normals;
    estimator.compute(normals);
    if (normals.size() != queries->size()) throw std::runtime_error("Normal estimation returned a wrong count");
    std::vector<Vec3> output(normals.size(), Vec3::UnitZ());
    for (std::size_t i = 0; i < normals.size(); ++i) {
        Vec3 normal(normals[i].normal_x, normals[i].normal_y, normals[i].normal_z);
        if (normal.allFinite() && normal.norm() > 1e-10) output[i] = normal.normalized();
    }
    return output;
}

double gaussianLog1D(double value, double sigma) {
    const double variance = sigma * sigma;
    return -0.5 * (value * value / variance + std::log(2.0 * kPi * variance));
}

Mat3 pointCovariance(const Vec3& normal, const Options& options) {
    const double normal_variance = options.normal_sigma * options.normal_sigma;
    const double tangent_sigma = options.normal_sigma * options.tangent_normal_ratio;
    const double tangent_variance = tangent_sigma * tangent_sigma;
    return tangent_variance * Mat3::Identity() +
        (normal_variance - tangent_variance) * (normal * normal.transpose());
}

Mat3 residualCovariance(const State& state, std::size_t i, const Mat3& rotation, const Options& options) {
    Mat3 covariance = rotation * pointCovariance(state.source_normals[i], options) * rotation.transpose() +
        pointCovariance(state.target_normals[i], options);
    covariance.diagonal().array() += std::max(1e-8, 0.01 * options.normal_sigma * options.normal_sigma);
    return covariance;
}

double gaussianLog3D(const Vec3& residual, const Mat3& covariance) {
    Eigen::LLT<Mat3> llt(covariance);
    if (llt.info() != Eigen::Success) return -1e12;
    const Vec3 whitened = llt.matrixL().solve(residual);
    const Mat3 lower = llt.matrixL().toDenseMatrix();
    const double logdet = 2.0 * lower.diagonal().array().log().sum();
    return -0.5 * (whitened.squaredNorm() + logdet + 3.0 * std::log(2.0 * kPi));
}

std::uint64_t edgeKey(int a, int b) {
    if (a > b) std::swap(a, b);
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32U) |
        static_cast<std::uint32_t>(b);
}

void addRelation(
    State& state,
    int i,
    int j,
    double sigma,
    double outlier_half_width,
    std::unordered_set<std::uint64_t>& seen) {
    if (i == j) return;
    if (i > j) std::swap(i, j);
    if (!seen.insert(edgeKey(i, j)).second) return;
    const double discrepancy = (state.source[i] - state.source[j]).norm() -
        (state.target[i] - state.target[j]).norm();
    Relation relation;
    relation.i = i;
    relation.j = j;
    relation.discrepancy = discrepancy;
    relation.log_inlier = gaussianLog1D(discrepancy, sigma);
    relation.log_outlier = -std::log(2.0 * outlier_half_width);
    const int index = static_cast<int>(state.relations.size());
    state.relations.push_back(relation);
    state.incident_relations[i].push_back(index);
    state.incident_relations[j].push_back(index);
}

void pruneRelationsToBudget(State& state, const Options& options) {
    if (options.edge_degree_budget == 0 || state.relations.empty()) return;
    const std::size_t n = state.source.size();
    const std::size_t max_edges = std::min(
        state.relations.size(), (options.edge_degree_budget * n + 1) / 2);
    if (state.relations.size() <= max_edges) return;

    // Rank an edge by its best endpoint-local rank. This deterministic
    // round-robin-like ordering gives every correspondence a chance to retain
    // strong evidence before assigning additional edges to already dense nodes.
    std::vector<std::size_t> best_local_rank(
        state.relations.size(), std::numeric_limits<std::size_t>::max());
    for (const auto& incident : state.incident_relations) {
        std::vector<int> ordered = incident;
        std::sort(ordered.begin(), ordered.end(), [&](int left, int right) {
            const Relation& a = state.relations[static_cast<std::size_t>(left)];
            const Relation& b = state.relations[static_cast<std::size_t>(right)];
            const double a_strength = std::abs(a.log_inlier - a.log_outlier);
            const double b_strength = std::abs(b.log_inlier - b.log_outlier);
            if (a_strength != b_strength) return a_strength > b_strength;
            return edgeKey(a.i, a.j) < edgeKey(b.i, b.j);
        });
        for (std::size_t rank = 0; rank < ordered.size(); ++rank) {
            const std::size_t index = static_cast<std::size_t>(ordered[rank]);
            best_local_rank[index] = std::min(best_local_rank[index], rank);
        }
    }
    std::vector<std::size_t> order(state.relations.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t left, std::size_t right) {
        if (best_local_rank[left] != best_local_rank[right])
            return best_local_rank[left] < best_local_rank[right];
        const Relation& a = state.relations[left];
        const Relation& b = state.relations[right];
        const double a_strength = std::abs(a.log_inlier - a.log_outlier);
        const double b_strength = std::abs(b.log_inlier - b.log_outlier);
        if (a_strength != b_strength) return a_strength > b_strength;
        return edgeKey(a.i, a.j) < edgeKey(b.i, b.j);
    });
    order.resize(max_edges);
    std::sort(order.begin(), order.end());

    std::vector<Relation> retained;
    retained.reserve(max_edges);
    state.incident_relations.assign(n, {});
    for (std::size_t old_index : order) {
        const Relation& relation = state.relations[old_index];
        const int new_index = static_cast<int>(retained.size());
        retained.push_back(relation);
        state.incident_relations[static_cast<std::size_t>(relation.i)].push_back(new_index);
        state.incident_relations[static_cast<std::size_t>(relation.j)].push_back(new_index);
    }
    state.relations = std::move(retained);
}

void buildSparseRelations(State& state, const Options& options) {
    const int n = static_cast<int>(state.source.size());
    state.incident_relations.assign(static_cast<std::size_t>(n), {});
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(static_cast<std::size_t>(n) * options.relation_degree * 2);
    // The compatibility set is sparse on real data (about 0.7--1.4% on WHU-TLS
    // Railway). Enumerating it exactly prevents a 1%-inlier cluster from vanishing
    // under uniform edge sampling, while storage and subsequent inference remain sparse.
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const double discrepancy = std::abs(
                (state.source[static_cast<std::size_t>(i)] - state.source[static_cast<std::size_t>(j)]).norm() -
                (state.target[static_cast<std::size_t>(i)] - state.target[static_cast<std::size_t>(j)]).norm());
            if (discrepancy <= options.relation_threshold)
                addRelation(state, i, j, options.relation_sigma, state.residual_half_width, seen);
        }
    }

    // Retain explicit incompatible evidence at every baseline scale. This keeps
    // the pairwise factor probabilistic instead of turning it into a hard clique.
    std::mt19937 generator(options.seed);
    std::uniform_int_distribution<int> distribution(0, n - 1);
    const int per_scale = std::max(2, static_cast<int>(options.relation_degree / 6));
    for (int i = 0; i < n; ++i) {
        std::array<std::vector<int>, 3> scale_candidates;
        std::unordered_set<int> local_seen;
        const int trials = std::max(180, per_scale * 60);
        for (int trial = 0; trial < trials; ++trial) {
            const int j = distribution(generator);
            if (j == i || !local_seen.insert(j).second) continue;
            const double discrepancy = std::abs(
                (state.source[static_cast<std::size_t>(i)] - state.source[static_cast<std::size_t>(j)]).norm() -
                (state.target[static_cast<std::size_t>(i)] - state.target[static_cast<std::size_t>(j)]).norm());
            if (discrepancy <= options.relation_threshold) continue;
            const double normalized_baseline =
                (state.source[static_cast<std::size_t>(i)] - state.source[static_cast<std::size_t>(j)]).norm() /
                state.length;
            const int scale = normalized_baseline <= 0.5 ? 0 : (normalized_baseline <= 1.5 ? 1 : 2);
            if (static_cast<int>(scale_candidates[static_cast<std::size_t>(scale)].size()) < per_scale)
                scale_candidates[static_cast<std::size_t>(scale)].push_back(j);
        }
        // Deterministic scan fills rare scale bins without constructing a dense graph.
        for (int offset = 1; offset < n; ++offset) {
            bool complete = true;
            for (const auto& candidates : scale_candidates)
                complete = complete && static_cast<int>(candidates.size()) >= per_scale;
            if (complete) break;
            const int j = (i + offset * 7919) % n;
            if (j == i || !local_seen.insert(j).second) continue;
            const double discrepancy = std::abs(
                (state.source[static_cast<std::size_t>(i)] - state.source[static_cast<std::size_t>(j)]).norm() -
                (state.target[static_cast<std::size_t>(i)] - state.target[static_cast<std::size_t>(j)]).norm());
            if (discrepancy <= options.relation_threshold) continue;
            const double normalized_baseline =
                (state.source[static_cast<std::size_t>(i)] - state.source[static_cast<std::size_t>(j)]).norm() /
                state.length;
            const int scale = normalized_baseline <= 0.5 ? 0 : (normalized_baseline <= 1.5 ? 1 : 2);
            if (static_cast<int>(scale_candidates[static_cast<std::size_t>(scale)].size()) < per_scale)
                scale_candidates[static_cast<std::size_t>(scale)].push_back(j);
        }
        for (const auto& candidates : scale_candidates) for (int j : candidates)
            addRelation(state, i, j, options.relation_sigma, state.residual_half_width, seen);
    }
    pruneRelationsToBudget(state, options);
    double relation_half_width = 3.0 * options.relation_sigma;
    for (const Relation& relation : state.relations)
        relation_half_width = std::max(relation_half_width, std::abs(relation.discrepancy));
    const double log_outlier = -std::log(2.0 * relation_half_width);
    for (Relation& relation : state.relations) relation.log_outlier = log_outlier;
}

std::vector<double> meanField(
    const State& state,
    const Mat3* rotation,
    const Vec3* translation,
    const Options& options,
    std::vector<double> posterior) {
    const std::size_t n = state.source.size();
    if (posterior.size() != n) posterior = state.prior;
    const double log_outlier_residual = -3.0 * std::log(2.0 * state.residual_half_width);
    // This is the derivative scale induced by the balanced variational
    // objective: average node evidence + average relation evidence. Without
    // it, a node's confidence grows merely because it belongs to a dense
    // accidental compatibility component.
    const double relation_message_scale = !options.use_graph_posterior ? 0.0 :
        (options.use_balanced_inference ? static_cast<double>(n) /
            static_cast<double>(std::max<std::size_t>(1, state.relations.size())) : 1.0);
    std::vector<double> updated(n, options.prior_inlier);
    for (std::size_t iteration = 0; iteration < options.mean_field_iterations; ++iteration) {
        double largest_change = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double prior = clampProbability(state.prior[i]);
            double eta = std::log(prior / (1.0 - prior));
            if (rotation && translation) {
                const Vec3 residual = (*rotation) * state.source[i] + (*translation) - state.target[i];
                eta += gaussianLog3D(residual, residualCovariance(state, i, *rotation, options)) -
                    log_outlier_residual;
            }
            for (int relation_index : state.incident_relations[i]) {
                const Relation& relation = state.relations[static_cast<std::size_t>(relation_index)];
                const int neighbor = relation.i == static_cast<int>(i) ? relation.j : relation.i;
                eta += relation_message_scale * posterior[static_cast<std::size_t>(neighbor)] *
                    (relation.log_inlier - relation.log_outlier);
            }
            const double raw = sigmoid(eta);
            updated[i] = options.mean_field_damping * posterior[i] +
                (1.0 - options.mean_field_damping) * raw;
            largest_change = std::max(largest_change, std::abs(updated[i] - posterior[i]));
        }
        posterior.swap(updated);
        if (largest_change < 1e-5) break;
    }
    return posterior;
}

double elbo(
    const State& state,
    const Mat3& rotation,
    const Vec3& translation,
    const Options& options,
    const std::vector<double>& posterior) {
    const double log_outlier_residual = -3.0 * std::log(2.0 * state.residual_half_width);
    double value = 0.0;
    for (std::size_t i = 0; i < state.source.size(); ++i) {
        const double p = clampProbability(posterior[i]);
        const double prior = clampProbability(state.prior[i]);
        const Vec3 residual = rotation * state.source[i] + translation - state.target[i];
        const double log_inlier = gaussianLog3D(residual, residualCovariance(state, i, rotation, options));
        value += p * (std::log(prior) + log_inlier) +
            (1.0 - p) * (std::log(1.0 - prior) + log_outlier_residual) -
            p * std::log(p) - (1.0 - p) * std::log(1.0 - p);
    }
    for (const Relation& relation : state.relations) {
        const double both = posterior[static_cast<std::size_t>(relation.i)] *
            posterior[static_cast<std::size_t>(relation.j)];
        value += both * relation.log_inlier + (1.0 - both) * relation.log_outlier;
    }
    return value;
}

double surfaceLogLikelihood(
    const State& state,
    const Mat3& rotation,
    const Vec3& centered_translation,
    const Options& options) {
    if (!state.target_surface_tree || !state.source_surface_tree ||
        state.source_surface_samples.empty() || state.target_surface_samples.empty()) return 0.0;
    const double sigma = std::max(options.relation_threshold,
        options.surface_sigma_multiplier * options.normal_sigma);
    const double inverse_variance = 1.0 / (sigma * sigma);
    double score = 0.0;
    std::vector<int> indices(1);
    std::vector<float> squared_distances(1);
    for (const Vec3& source_point : state.source_surface_samples) {
        const Vec3 transformed = state.target_center + centered_translation +
            rotation * (source_point - state.source_center);
        Point query;
        query.x = static_cast<float>(transformed.x());
        query.y = static_cast<float>(transformed.y());
        query.z = static_cast<float>(transformed.z());
        double normalized_squared_distance = 9.0;
        if (state.target_surface_tree->nearestKSearch(query, 1, indices, squared_distances) > 0)
            normalized_squared_distance = std::min(9.0,
                static_cast<double>(squared_distances[0]) * inverse_variance);
        score -= 0.5 * normalized_squared_distance;
    }
    // The reverse term uses the inverse rigid transform. Together the two
    // directions form one symmetric surface observation and prevent a pose
    // from gaining likelihood merely by sliding the source along a large plane.
    if (options.use_symmetric_surface) for (const Vec3& target_point : state.target_surface_samples) {
        const Vec3 transformed = state.source_center + rotation.transpose() *
            (target_point - state.target_center - centered_translation);
        Point query;
        query.x = static_cast<float>(transformed.x());
        query.y = static_cast<float>(transformed.y());
        query.z = static_cast<float>(transformed.z());
        double normalized_squared_distance = 9.0;
        if (state.source_surface_tree->nearestKSearch(query, 1, indices, squared_distances) > 0)
            normalized_squared_distance = std::min(9.0,
                static_cast<double>(squared_distances[0]) * inverse_variance);
        score -= 0.5 * normalized_squared_distance;
    }
    const std::size_t sample_count = state.source_surface_samples.size() +
        (options.use_symmetric_surface ? state.target_surface_samples.size() : 0);
    return score / static_cast<double>(sample_count);
}

double jointObjective(
    const State& state,
    const Mat3& rotation,
    const Vec3& centered_translation,
    const Options& options,
    const std::vector<double>& posterior) {
    const double log_outlier_residual = -3.0 * std::log(2.0 * state.residual_half_width);
    double node_evidence = 0.0;
    for (std::size_t i = 0; i < state.source.size(); ++i) {
        const double p = clampProbability(posterior[i]);
        const double prior = clampProbability(state.prior[i]);
        const Vec3 residual = rotation * state.source[i] + centered_translation - state.target[i];
        const double log_inlier = gaussianLog3D(
            residual, residualCovariance(state, i, rotation, options));
        node_evidence += p * (std::log(prior) + log_inlier) +
            (1.0 - p) * (std::log(1.0 - prior) + log_outlier_residual) -
            p * std::log(p) - (1.0 - p) * std::log(1.0 - p);
    }
    double relation_evidence = 0.0;
    for (const Relation& relation : state.relations) {
        const double both = posterior[static_cast<std::size_t>(relation.i)] *
            posterior[static_cast<std::size_t>(relation.j)];
        relation_evidence += both * relation.log_inlier +
            (1.0 - both) * relation.log_outlier;
    }
    if (!options.use_graph_posterior) relation_evidence = 0.0;
    const double probabilistic_evidence = options.use_balanced_inference
        ? node_evidence / static_cast<double>(std::max<std::size_t>(1, state.source.size())) +
            relation_evidence / static_cast<double>(std::max<std::size_t>(1, state.relations.size()))
        : (node_evidence + relation_evidence) / static_cast<double>(std::max<std::size_t>(
            1, state.source.size() + (options.use_graph_posterior ? state.relations.size() : 0)));
    return probabilistic_evidence +
        options.surface_score_weight * surfaceLogLikelihood(
            state, rotation, centered_translation, options);
}

double modeConditionedObjective(
    const State& state,
    const Mat3& rotation,
    const Vec3& centered_translation,
    const Mat3& mode_rotation,
    const Vec3& mode_translation,
    const Options& options,
    const std::vector<double>& posterior) {
    const Mat3 relative_rotation = rotation * mode_rotation.transpose();
    const double cosine = std::max(-1.0, std::min(1.0,
        0.5 * (relative_rotation.trace() - 1.0)));
    const double rotation_distance = std::acos(cosine);
    const double translation_distance = (centered_translation - mode_translation).norm();
    const double translation_sigma = std::max(options.relation_threshold,
        options.surface_sigma_multiplier * options.normal_sigma);
    const double rotation_sigma = std::max(kPi / 180.0, translation_sigma / state.length);
    const double mode_log_prior = -0.5 * (
        rotation_distance * rotation_distance / (rotation_sigma * rotation_sigma) +
        translation_distance * translation_distance / (translation_sigma * translation_sigma));
    return jointObjective(state, rotation, centered_translation, options, posterior) +
        mode_log_prior;
}

bool estimateRigid(
    const State& state,
    const std::vector<int>& indices,
    const std::vector<double>* weights,
    Mat3& rotation,
    Vec3& translation) {
    if (indices.size() < 3) return false;
    double weight_sum = 0.0;
    Vec3 source_center = Vec3::Zero();
    Vec3 target_center = Vec3::Zero();
    for (int index : indices) {
        const double weight = weights ? std::max(1e-9, (*weights)[static_cast<std::size_t>(index)]) : 1.0;
        weight_sum += weight;
        source_center += weight * state.source[static_cast<std::size_t>(index)];
        target_center += weight * state.target[static_cast<std::size_t>(index)];
    }
    if (weight_sum <= kTiny) return false;
    source_center /= weight_sum;
    target_center /= weight_sum;
    Mat3 covariance = Mat3::Zero();
    for (int index : indices) {
        const double weight = weights ? std::max(1e-9, (*weights)[static_cast<std::size_t>(index)]) : 1.0;
        covariance += weight * (state.source[static_cast<std::size_t>(index)] - source_center) *
            (state.target[static_cast<std::size_t>(index)] - target_center).transpose();
    }
    Eigen::JacobiSVD<Mat3> svd(covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
    if (svd.singularValues()[1] < 1e-10) return false;
    rotation = svd.matrixV() * svd.matrixU().transpose();
    if (rotation.determinant() < 0.0) {
        Mat3 correction = Mat3::Identity();
        correction(2, 2) = -1.0;
        rotation = svd.matrixV() * correction * svd.matrixU().transpose();
    }
    translation = target_center - rotation * source_center;
    return rotation.allFinite() && translation.allFinite();
}

std::vector<int> descendingIndices(const std::vector<double>& values, std::size_t limit) {
    std::vector<int> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), [&](int a, int b) {
        if (values[static_cast<std::size_t>(a)] != values[static_cast<std::size_t>(b)])
            return values[static_cast<std::size_t>(a)] > values[static_cast<std::size_t>(b)];
        return a < b;
    });
    if (indices.size() > limit) indices.resize(limit);
    return indices;
}

std::vector<int> selectLocalInformationSupport(
    const State& state,
    int seed,
    const std::vector<int>& candidates,
    const std::vector<double>& confidence,
    std::size_t budget) {
    std::vector<int> pool = candidates;
    pool.push_back(seed);
    std::sort(pool.begin(), pool.end());
    pool.erase(std::unique(pool.begin(), pool.end()), pool.end());
    if (pool.size() < 3) return {};
    Vec3 source_center = Vec3::Zero();
    Vec3 target_center = Vec3::Zero();
    for (int index : pool) {
        source_center += state.source[static_cast<std::size_t>(index)];
        target_center += state.target[static_cast<std::size_t>(index)];
    }
    source_center /= static_cast<double>(pool.size());
    target_center /= static_cast<double>(pool.size());
    std::vector<double> source_radii;
    std::vector<double> target_radii;
    for (int index : pool) {
        source_radii.push_back((state.source[static_cast<std::size_t>(index)] - source_center).norm());
        target_radii.push_back((state.target[static_cast<std::size_t>(index)] - target_center).norm());
    }
    const double source_scale = std::max(1e-9, median(std::move(source_radii)));
    const double target_scale = std::max(1e-9, median(std::move(target_radii)));
    std::vector<Vec6> source_atoms(pool.size());
    std::vector<Vec6> target_atoms(pool.size());
    for (std::size_t i = 0; i < pool.size(); ++i) {
        const std::size_t index = static_cast<std::size_t>(pool[i]);
        const Vec3 source_point = (state.source[index] - source_center) / source_scale;
        const Vec3 target_point = (state.target[index] - target_center) / target_scale;
        source_atoms[i].head<3>() = source_point.cross(state.source_normals[index]);
        source_atoms[i].tail<3>() = state.source_normals[index];
        target_atoms[i].head<3>() = target_point.cross(state.target_normals[index]);
        target_atoms[i].tail<3>() = state.target_normals[index];
    }
    auto matrixLogDet = [](const Mat6& matrix) {
        Eigen::LLT<Mat6> llt(matrix);
        if (llt.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
        return 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
    };
    Mat6 source_information = 1e-3 * Mat6::Identity();
    Mat6 target_information = source_information;
    std::vector<unsigned char> available(pool.size(), 1);
    std::vector<int> selected;
    const std::size_t target_count = std::min(budget, pool.size());
    while (selected.size() < target_count) {
        int best = -1;
        double best_gain = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (!available[i]) continue;
            const int index = pool[i];
            const double weight = index == seed ? 1.0 :
                std::max(1e-8, confidence[static_cast<std::size_t>(index)]);
            const double gain = 0.5 * (
                matrixLogDet(source_information + weight * source_atoms[i] * source_atoms[i].transpose()) -
                    matrixLogDet(source_information) +
                matrixLogDet(target_information + weight * target_atoms[i] * target_atoms[i].transpose()) -
                    matrixLogDet(target_information));
            if (gain > best_gain || (gain == best_gain && index == seed)) {
                best_gain = gain;
                best = static_cast<int>(i);
            }
        }
        if (best < 0) break;
        const int index = pool[static_cast<std::size_t>(best)];
        const double weight = index == seed ? 1.0 :
            std::max(1e-8, confidence[static_cast<std::size_t>(index)]);
        source_information += weight * source_atoms[static_cast<std::size_t>(best)] *
            source_atoms[static_cast<std::size_t>(best)].transpose();
        target_information += weight * target_atoms[static_cast<std::size_t>(best)] *
            target_atoms[static_cast<std::size_t>(best)].transpose();
        available[static_cast<std::size_t>(best)] = 0;
        selected.push_back(index);
    }
    return selected;
}

void initializePose(
    const State& state,
    const Options& options,
    const std::vector<double>& graph_posterior,
    Mat3& best_rotation,
    Vec3& best_translation,
    std::vector<double>& best_posterior,
    double& best_elbo,
    std::size_t& tested,
    std::vector<CandidateDiagnostic>& candidate_diagnostics) {
    const std::size_t pool_size = std::min<std::size_t>(state.source.size(), 256);
    const std::vector<int> pool = descendingIndices(graph_posterior, pool_size);
    std::vector<double> sampling_weights;
    sampling_weights.reserve(pool.size());
    for (int index : pool) sampling_weights.push_back(std::max(1e-6, graph_posterior[static_cast<std::size_t>(index)]));
    std::discrete_distribution<int> distribution(sampling_weights.begin(), sampling_weights.end());
    std::mt19937 generator(options.seed + 17U);

    const bool use_coherent_mode = state.source.size() < 280;
    const std::size_t fallback_budget = options.hypothesis_count > 1 ? 1 : 0;
    const std::size_t random_budget = std::min<std::size_t>(32, options.hypothesis_count / 8);
    const std::size_t coherent_budget = use_coherent_mode
        ? std::min<std::size_t>(128, options.hypothesis_count / 3) : 0;
    const std::size_t triangle_budget = options.hypothesis_count -
        fallback_budget - random_budget - coherent_budget;
    best_elbo = -std::numeric_limits<double>::infinity();
    double best_joint_score = -std::numeric_limits<double>::infinity();
    candidate_diagnostics.clear();
    auto recordCandidate = [&](const char* source, const std::vector<int>& support,
                               const Mat3& rotation, const Vec3& centered_translation,
                               double probabilistic_score, double selection_score) {
        CandidateDiagnostic diagnostic;
        diagnostic.source = source;
        diagnostic.support_indices = support;
        diagnostic.transformation.block<3, 3>(0, 0) = rotation;
        diagnostic.transformation.block<3, 1>(0, 3) =
            state.target_center + centered_translation - rotation * state.source_center;
        diagnostic.probabilistic_score = probabilistic_score;
        diagnostic.selection_score = selection_score;
        candidate_diagnostics.push_back(std::move(diagnostic));
    };
    tested = 0;
    // Enumerate triangles from the same compatible relation factors. Reservoir
    // sampling gives tiny but coherent inlier components a chance proportional
    // to their triangle support, unlike three independent node draws.
    std::vector<std::vector<int>> compatible_neighbors(state.source.size());
    std::unordered_set<std::uint64_t> compatible_edges;
    compatible_edges.reserve(state.relations.size());
    for (const Relation& relation : state.relations) {
        if (std::abs(relation.discrepancy) > options.relation_threshold) continue;
        compatible_neighbors[static_cast<std::size_t>(relation.i)].push_back(relation.j);
        compatible_neighbors[static_cast<std::size_t>(relation.j)].push_back(relation.i);
        compatible_edges.insert(edgeKey(relation.i, relation.j));
    }
    std::vector<std::array<int, 3>> triangle_reservoir;
    triangle_reservoir.reserve(triangle_budget);
    std::size_t triangle_count = 0;
    for (int i = 0; i < static_cast<int>(compatible_neighbors.size()); ++i) {
        auto& neighbors = compatible_neighbors[static_cast<std::size_t>(i)];
        std::sort(neighbors.begin(), neighbors.end());
        for (std::size_t a = 0; a < neighbors.size(); ++a) {
            const int j = neighbors[a];
            if (j <= i) continue;
            for (std::size_t b = a + 1; b < neighbors.size(); ++b) {
                const int k = neighbors[b];
                if (k <= j || compatible_edges.count(edgeKey(j, k)) == 0) continue;
                const std::array<int, 3> triangle{{i, j, k}};
                ++triangle_count;
                if (triangle_reservoir.size() < triangle_budget) {
                    triangle_reservoir.push_back(triangle);
                } else {
                    std::uniform_int_distribution<std::size_t> replacement(0, triangle_count - 1);
                    const std::size_t position = replacement(generator);
                    if (position < triangle_reservoir.size()) triangle_reservoir[position] = triangle;
                }
            }
        }
    }
    // Stream evaluations: only the best posterior is kept. Candidate
    // diagnostics contain poses/scores, not an H-by-N posterior cache.
    // Use a stable graph-only posterior for every candidate (the caller keeps
    // it separate from best_posterior).
    for (const auto& triple : triangle_reservoir) {
        const std::vector<int> indices{triple[0], triple[1], triple[2]};
        Mat3 rotation;
        Vec3 translation;
        if (!estimateRigid(state, indices, &graph_posterior, rotation, translation)) continue;
        std::vector<double> posterior = meanField(
            state, &rotation, &translation, options, graph_posterior);
        const double probabilistic_score = elbo(state, rotation, translation, options, posterior);
        // At a candidate's own center the mode-prior penalty is zero.
        const double score = jointObjective(state, rotation, translation, options, posterior);
        recordCandidate("compatible_triangle", indices, rotation, translation,
            probabilistic_score, score);
        ++tested;
        if (score > best_joint_score) {
            best_joint_score = score;
            best_elbo = probabilistic_score;
            best_rotation = rotation;
            best_translation = translation;
            best_posterior = std::move(posterior);
        }
    }

    // Seed-conditioned coherent modes complement uniform triangle sampling.
    // The second-order graph score exposes small true components that would
    // otherwise be overwhelmed by a much larger accidental compatibility set.
    std::vector<double> degree(state.source.size(), 0.0);
    std::vector<double> graph_score(state.source.size(), 0.0);
    for (std::size_t i = 0; i < compatible_neighbors.size(); ++i)
        degree[i] = static_cast<double>(compatible_neighbors[i].size());
    for (std::size_t i = 0; i < compatible_neighbors.size(); ++i) {
        double propagated = 0.0;
        for (int neighbor : compatible_neighbors[i])
            propagated += degree[static_cast<std::size_t>(neighbor)];
        graph_score[i] = degree[i] * std::max(1.0, propagated);
    }
    const std::vector<int> coherent_seeds = use_coherent_mode
        ? descendingIndices(graph_score, std::min(coherent_budget, graph_score.size()))
        : std::vector<int>{};
    for (int seed : coherent_seeds) {
        std::vector<std::pair<double, int>> ranked;
        const auto& seed_neighbors = compatible_neighbors[static_cast<std::size_t>(seed)];
        for (int neighbor : seed_neighbors) {
            int common = 0;
            for (int candidate : seed_neighbors)
                if (compatible_edges.count(edgeKey(neighbor, candidate)) != 0) ++common;
            if (common > 0) ranked.emplace_back(
                static_cast<double>(common) * std::sqrt(std::max(1.0,
                    graph_score[static_cast<std::size_t>(neighbor)])), neighbor);
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.first != b.first) return a.first > b.first;
            return a.second < b.second;
        });
        std::vector<int> candidates;
        std::vector<double> confidence(state.source.size(), 0.0);
        confidence[static_cast<std::size_t>(seed)] = 1.0;
        if (ranked.size() < 2) continue;
        if (ranked.size() > 64) ranked.resize(64);
        const double maximum = std::max(ranked.front().first, 1e-12);
        for (const auto& item : ranked) {
            candidates.push_back(item.second);
            confidence[static_cast<std::size_t>(item.second)] = item.first / maximum;
        }
        const std::vector<int> support = selectLocalInformationSupport(
            state, seed, candidates, confidence, 12);
        if (support.size() < 3) continue;
        Mat3 rotation;
        Vec3 translation;
        if (!estimateRigid(state, support, &confidence, rotation, translation)) continue;
        for (int refinement = 0; refinement < 2; ++refinement) {
            std::vector<int> consensus;
            for (std::size_t i = 0; i < state.source.size(); ++i)
                if ((rotation * state.source[i] + translation - state.target[i]).norm() <=
                    options.relation_threshold) consensus.push_back(static_cast<int>(i));
            if (consensus.size() < 3 ||
                !estimateRigid(state, consensus, nullptr, rotation, translation)) break;
        }
        std::vector<double> candidate_posterior = meanField(
            state, &rotation, &translation, options, graph_posterior);
        const double probabilistic_score = elbo(
            state, rotation, translation, options, candidate_posterior);
        const double score = jointObjective(
            state, rotation, translation, options, candidate_posterior);
        recordCandidate("coherent_seed", support, rotation, translation,
            probabilistic_score, score);
        ++tested;
        if (score > best_joint_score) {
            best_joint_score = score;
            best_elbo = probabilistic_score;
            best_rotation = rotation;
            best_translation = translation;
            best_posterior = std::move(candidate_posterior);
        }
    }
    generator.seed(options.seed + 17U);
    const std::size_t random_target = std::min(options.hypothesis_count - fallback_budget, tested + random_budget);
    const std::size_t max_attempts = std::max<std::size_t>(options.hypothesis_count * 500, 4000);
    for (std::size_t attempt = 0; attempt < max_attempts && tested < random_target; ++attempt) {
        std::array<int, 3> triple{{pool[static_cast<std::size_t>(distribution(generator))],
                                  pool[static_cast<std::size_t>(distribution(generator))],
                                  pool[static_cast<std::size_t>(distribution(generator))]}};
        if (triple[0] == triple[1] || triple[0] == triple[2] || triple[1] == triple[2]) continue;
        bool compatible = true;
        for (int a = 0; a < 3; ++a) for (int b = a + 1; b < 3; ++b) {
            const double discrepancy = std::abs(
                (state.source[static_cast<std::size_t>(triple[a])] - state.source[static_cast<std::size_t>(triple[b])]).norm() -
                (state.target[static_cast<std::size_t>(triple[a])] - state.target[static_cast<std::size_t>(triple[b])]).norm());
            if (discrepancy > 4.0 * options.relation_threshold) compatible = false;
        }
        if (!compatible) continue;
        Mat3 rotation;
        Vec3 translation;
        const std::vector<int> indices{triple[0], triple[1], triple[2]};
        if (!estimateRigid(state, indices, &graph_posterior, rotation, translation)) continue;
        std::vector<double> posterior = meanField(state, &rotation, &translation, options, graph_posterior);
        const double probabilistic_score = elbo(state, rotation, translation, options, posterior);
        const double score = jointObjective(state, rotation, translation, options, posterior);
        recordCandidate("weighted_random_triangle", indices, rotation, translation,
            probabilistic_score, score);
        ++tested;
        if (score > best_joint_score) {
            best_joint_score = score;
            best_elbo = probabilistic_score;
            best_rotation = rotation;
            best_translation = translation;
            best_posterior = std::move(posterior);
        }
    }

    // Deterministic fallback and an additional graph-weighted candidate.
    const std::vector<int> top = descendingIndices(graph_posterior, std::min<std::size_t>(64, state.source.size()));
    Mat3 rotation;
    Vec3 translation;
    if (tested < options.hypothesis_count && estimateRigid(state, top, &graph_posterior, rotation, translation)) {
        std::vector<double> posterior = meanField(state, &rotation, &translation, options, graph_posterior);
        const double probabilistic_score = elbo(state, rotation, translation, options, posterior);
        const double score = jointObjective(state, rotation, translation, options, posterior);
        recordCandidate("graph_weighted_fallback", top, rotation, translation,
            probabilistic_score, score);
        ++tested;
        if (score > best_joint_score) {
            best_joint_score = score;
            best_elbo = probabilistic_score;
            best_rotation = rotation;
            best_translation = translation;
            best_posterior = std::move(posterior);
        }
    }
    if (!std::isfinite(best_elbo)) throw std::runtime_error("No non-degenerate initialization hypothesis was generated");
    if (!candidate_diagnostics.empty()) {
        Eigen::Matrix4d selected = Eigen::Matrix4d::Identity();
        selected.block<3, 3>(0, 0) = best_rotation;
        selected.block<3, 1>(0, 3) =
            state.target_center + best_translation - best_rotation * state.source_center;
        double best_distance = std::numeric_limits<double>::infinity();
        std::size_t best_index = 0;
        for (std::size_t i = 0; i < candidate_diagnostics.size(); ++i) {
            const double distance = (candidate_diagnostics[i].transformation - selected).squaredNorm();
            if (distance < best_distance) {
                best_distance = distance;
                best_index = i;
            }
        }
        candidate_diagnostics[best_index].selected_for_initialization = true;
    }
}

Mat6 informationAtom(
    const State& state,
    std::size_t i,
    const Mat3& rotation,
    const Options& options) {
    Eigen::Matrix<double, 3, 6> jacobian;
    jacobian.leftCols<3>() = -skew(rotation * state.source[i]) / state.length;
    jacobian.rightCols<3>() = Mat3::Identity();
    const Mat3 covariance = residualCovariance(state, i, rotation, options);
    return jacobian.transpose() * covariance.inverse() * jacobian;
}

double logDet(const Mat6& matrix) {
    Eigen::LLT<Mat6> llt(matrix);
    if (llt.info() != Eigen::Success) return -std::numeric_limits<double>::infinity();
    return 2.0 * llt.matrixL().toDenseMatrix().diagonal().array().log().sum();
}

std::vector<int> selectByInformation(
    const State& state,
    const Mat3& rotation,
    const std::vector<double>& posterior,
    const Options& options) {
    const std::size_t n = state.source.size();
    const std::size_t budget = std::min(options.selection_budget, n);
    if (!options.use_information_design) return descendingIndices(posterior, budget);
    std::vector<Mat6> atoms(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(n); ++i)
        atoms[static_cast<std::size_t>(i)] = posterior[static_cast<std::size_t>(i)] *
            informationAtom(state, static_cast<std::size_t>(i), rotation, options);
    Mat6 information = options.information_regularizer * Mat6::Identity();
    std::vector<unsigned char> chosen(n, 0);
    std::vector<int> selected;
    selected.reserve(budget);
    std::vector<double> gains(n, -std::numeric_limits<double>::infinity());
    for (std::size_t step = 0; step < budget; ++step) {
        const double current = logDet(information);
        int best = -1;
        double best_gain = -std::numeric_limits<double>::infinity();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < static_cast<int>(n); ++i) {
            gains[static_cast<std::size_t>(i)] = chosen[static_cast<std::size_t>(i)]
                ? -std::numeric_limits<double>::infinity()
                : logDet(information + atoms[static_cast<std::size_t>(i)]) - current;
        }
        for (std::size_t i = 0; i < n; ++i) {
            if (gains[i] > best_gain) {
                best_gain = gains[i];
                best = static_cast<int>(i);
            }
        }
        if (best < 0) break;
        chosen[static_cast<std::size_t>(best)] = 1;
        selected.push_back(best);
        information += atoms[static_cast<std::size_t>(best)];
    }
    return selected;
}

bool optimizePose(
    const State& state,
    const std::vector<int>& selected,
    const std::vector<double>& posterior,
    const Options& options,
    const Mat3& mode_rotation,
    const Vec3& mode_translation,
    Mat3& rotation,
    Vec3& translation) {
    bool converged = false;
    for (std::size_t iteration = 0; iteration < options.pose_iterations; ++iteration) {
        Mat6 normal = 1e-9 * Mat6::Identity();
        Vec6 gradient = Vec6::Zero();
        for (int index_value : selected) {
            const std::size_t i = static_cast<std::size_t>(index_value);
            const Vec3 transformed = rotation * state.source[i];
            const Vec3 residual = transformed + translation - state.target[i];
            Eigen::Matrix<double, 3, 6> jacobian;
            jacobian.leftCols<3>() = -skew(transformed) / state.length;
            jacobian.rightCols<3>() = Mat3::Identity();
            const Mat3 precision = residualCovariance(state, i, rotation, options).inverse();
            const double weight = posterior[i];
            normal += weight * jacobian.transpose() * precision * jacobian;
            gradient += weight * jacobian.transpose() * precision * residual;
        }
        Eigen::LDLT<Mat6> solver(normal);
        if (solver.info() != Eigen::Success) break;
        Vec6 delta = -solver.solve(gradient);
        if (!delta.allFinite()) break;
        Vec3 rotation_delta = delta.head<3>() / state.length;
        Vec3 translation_delta = delta.tail<3>();
        const double angle = rotation_delta.norm();
        if (angle > 0.5) rotation_delta *= 0.5 / angle;
        const double displacement = translation_delta.norm();
        if (displacement > state.length) translation_delta *= state.length / displacement;
        const double current_objective = modeConditionedObjective(
            state, rotation, translation, mode_rotation, mode_translation, options, posterior);
        bool accepted = false;
        Mat3 accepted_rotation = rotation;
        Vec3 accepted_translation = translation;
        Vec3 accepted_rotation_delta = Vec3::Zero();
        Vec3 accepted_translation_delta = Vec3::Zero();
        double step = 1.0;
        for (int line_search = 0; line_search < 10; ++line_search) {
            const Vec3 trial_rotation_delta = step * rotation_delta;
            const Vec3 trial_translation_delta = step * translation_delta;
            const Mat3 trial_rotation = expSO3(trial_rotation_delta) * rotation;
            const Vec3 trial_translation = translation + trial_translation_delta;
            const double trial_objective = modeConditionedObjective(
                state, trial_rotation, trial_translation, mode_rotation, mode_translation,
                options, posterior);
            if (std::isfinite(trial_objective) && trial_objective >= current_objective - 1e-12) {
                accepted = true;
                accepted_rotation = trial_rotation;
                accepted_translation = trial_translation;
                accepted_rotation_delta = trial_rotation_delta;
                accepted_translation_delta = trial_translation_delta;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) break;
        rotation = accepted_rotation;
        translation = accepted_translation;
        if (accepted_rotation_delta.norm() < options.rotation_tolerance &&
            accepted_translation_delta.norm() < options.translation_tolerance) {
            converged = true;
            break;
        }
    }
    return converged;
}

void computeDiagnostics(
    const State& state,
    const Mat3& rotation,
    const std::vector<double>& posterior,
    const Options& options,
    const std::vector<int>& selected,
    Diagnostics& diagnostics) {
    Mat6 information = Mat6::Zero();
    for (std::size_t i = 0; i < state.source.size(); ++i)
        information += posterior[i] * informationAtom(state, i, rotation, options);
    Mat6 selected_information = Mat6::Zero();
    for (int index : selected) {
        const std::size_t i = static_cast<std::size_t>(index);
        selected_information += posterior[i] * informationAtom(state, i, rotation, options);
    }
    auto spectrum = [](const Mat6& data_information, std::vector<double>& values,
                       std::size_t& rank, double& minimum, double& condition, double& trace) {
        values.clear(); rank = 0;
        minimum = std::numeric_limits<double>::quiet_NaN();
        condition = trace = std::numeric_limits<double>::infinity();
        Eigen::SelfAdjointEigenSolver<Mat6> solver(data_information);
        if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) return;
        const auto eigenvalues = solver.eigenvalues();
        const double maximum = std::max(0.0, eigenvalues[5]);
        const double tolerance = 64.0 * 6.0 * std::numeric_limits<double>::epsilon() * maximum;
        for (int i = 0; i < 6; ++i) {
            // Remove only negative roundoff, never add a positive eigenvalue.
            const double value = std::max(0.0, eigenvalues[i]);
            values.push_back(value);
            if (value > tolerance) ++rank;
        }
        minimum = values.front();
        if (rank == 6) {
            condition = values.back() / values.front();
            trace = 0.0;
            for (double value : values) trace += 1.0 / value;
        }
        // Rank-deficient data leave uncertainty unbounded (infinity in API,
        // null in JSON). No ridge or truncated pseudoinverse is reported as
        // full six-DoF covariance information.
    };
    spectrum(information, diagnostics.information_eigenvalues,
        diagnostics.information_rank, diagnostics.min_information_eigenvalue,
        diagnostics.information_condition_number, diagnostics.covariance_trace);
    spectrum(selected_information, diagnostics.selected_information_eigenvalues,
        diagnostics.selected_information_rank, diagnostics.selected_min_information_eigenvalue,
        diagnostics.selected_information_condition_number, diagnostics.selected_covariance_trace);
}

void validateInputs(
    const PointCloudConstPtr& source_reference,
    const PointCloudConstPtr& target_reference,
    const PointCloudConstPtr& source_correspondences,
    const PointCloudConstPtr& target_correspondences,
    const Options& options) {
    if (!source_reference || !target_reference || !source_correspondences || !target_correspondences)
        throw std::invalid_argument("Null point cloud pointer");
    if (source_reference->empty() || target_reference->empty())
        throw std::invalid_argument("Reference point cloud is empty");
    if (source_correspondences->size() != target_correspondences->size())
        throw std::invalid_argument("Correspondence point clouds must have equal size");
    if (source_correspondences->size() < 3) throw std::invalid_argument("At least three correspondences are required");
    auto positive = [](double value) { return std::isfinite(value) && value > 0.0; };
    if (!positive(options.relation_threshold) || !positive(options.relation_sigma) ||
        !positive(options.normal_sigma) || !positive(options.information_regularizer) ||
        !positive(options.surface_sigma_multiplier) || !positive(options.rotation_tolerance) ||
        !positive(options.translation_tolerance) || !positive(options.objective_tolerance))
        throw std::invalid_argument("Threshold, noise scales, regularizer and tolerances must be finite and positive");
    if (!std::isfinite(options.tangent_normal_ratio) || options.tangent_normal_ratio <= 1.0)
        throw std::invalid_argument("Tangential noise must exceed normal noise");
    if (!std::isfinite(options.surface_score_weight) || options.surface_score_weight < 0.0)
        throw std::invalid_argument("Surface score weight must be finite and nonnegative");
    if (!std::isfinite(options.prior_inlier) || options.prior_inlier <= 0.0 || options.prior_inlier >= 1.0 ||
        !std::isfinite(options.mean_field_damping) || options.mean_field_damping < 0.0 || options.mean_field_damping >= 1.0)
        throw std::invalid_argument("Prior must be in (0,1); retained posterior damping must be in [0,1)");
    if (options.max_correspondences < 3 || options.normal_neighbors < 3 ||
        options.selection_budget < 3 || options.surface_sample_count < 1 || options.relation_degree < 1 ||
        options.hypothesis_count < 1 || options.mean_field_iterations < 1 ||
        options.outer_iterations < 1 || options.pose_iterations < 1 ||
        options.normal_threads < 0 || options.omp_threads < 0)
        throw std::invalid_argument("Invalid sampling, iteration or thread budget");
    if (!options.correspondence_priors.empty()) {
        if (options.correspondence_priors.size() != source_correspondences->size())
            throw std::invalid_argument("Per-correspondence prior count must match original correspondence count");
        for (double prior : options.correspondence_priors)
            if (!std::isfinite(prior) || prior <= 0.0 || prior >= 1.0)
                throw std::invalid_argument("Per-correspondence priors must be finite and in (0,1)");
    }
}

}  // namespace

Result registerPointClouds(
    const PointCloudConstPtr& source_reference,
    const PointCloudConstPtr& target_reference,
    const PointCloudConstPtr& source_correspondences,
    const PointCloudConstPtr& target_correspondences,
    const Options& options) {
    using Clock = std::chrono::steady_clock;
    Result result;
    const auto total_begin = Clock::now();
    try {
        const auto preprocessing_begin = Clock::now();
        validateInputs(source_reference, target_reference, source_correspondences, target_correspondences, options);
#ifdef _OPENMP
        if (options.omp_threads > 0) omp_set_num_threads(options.omp_threads);
#endif
        State state;
        result.diagnostics.input_correspondences = source_correspondences->size();
        result.diagnostics.retained_input_indices = retainedIndices(source_correspondences->size(), options.max_correspondences);
        state.source = extractPoints(source_correspondences, result.diagnostics.retained_input_indices);
        state.target = extractPoints(target_correspondences, result.diagnostics.retained_input_indices);
        PointCloudPtr source_queries = extractCloud(source_correspondences, result.diagnostics.retained_input_indices);
        PointCloudPtr target_queries = extractCloud(target_correspondences, result.diagnostics.retained_input_indices);
        state.source_normals = estimateNormals(source_reference, source_queries, options);
        state.target_normals = estimateNormals(target_reference, target_queries, options);
        state.source_center = coordinateMedian(state.source);
        state.target_center = coordinateMedian(state.target);
        for (Vec3& point : state.source) point -= state.source_center;
        for (Vec3& point : state.target) point -= state.target_center;
        std::vector<double> source_radii, target_radii;
        source_radii.reserve(state.source.size());
        target_radii.reserve(state.target.size());
        for (const Vec3& point : state.source) source_radii.push_back(point.norm());
        for (const Vec3& point : state.target) target_radii.push_back(point.norm());
        state.length = std::max(options.relation_sigma, std::max(
            median(std::move(source_radii)), median(std::move(target_radii))));
        state.residual_half_width = std::max(4.0 * state.length, 10.0 * options.relation_sigma);
        state.prior.assign(state.source.size(), clampProbability(options.prior_inlier));
        if (!options.correspondence_priors.empty())
            for (std::size_t i = 0; i < state.source.size(); ++i)
                state.prior[i] = clampProbability(options.correspondence_priors[
                    static_cast<std::size_t>(result.diagnostics.retained_input_indices[i])]);
        const std::vector<int> surface_indices = retainedIndices(
            source_reference->size(), options.surface_sample_count);
        state.source_surface_samples = extractPoints(source_reference, surface_indices);
        const std::vector<int> target_surface_indices = retainedIndices(
            target_reference->size(), options.surface_sample_count);
        state.target_surface_samples = extractPoints(target_reference, target_surface_indices);
        state.source_surface_tree.reset(new pcl::KdTreeFLANN<Point>);
        state.source_surface_tree->setInputCloud(source_reference);
        state.target_surface_tree.reset(new pcl::KdTreeFLANN<Point>);
        state.target_surface_tree->setInputCloud(target_reference);
        result.diagnostics.preprocessing_seconds =
            std::chrono::duration<double>(Clock::now() - preprocessing_begin).count();

        const auto graph_begin = Clock::now();
        buildSparseRelations(state, options);
        result.diagnostics.graph_seconds =
            std::chrono::duration<double>(Clock::now() - graph_begin).count();

        auto stage_begin = Clock::now();
        const std::vector<double> graph_posterior = meanField(state, nullptr, nullptr, options, state.prior);
        std::vector<double> posterior = graph_posterior;
        result.diagnostics.posterior_seconds +=
            std::chrono::duration<double>(Clock::now() - stage_begin).count();
        Mat3 rotation = Mat3::Identity();
        Vec3 centered_translation = Vec3::Zero();
        double objective = -std::numeric_limits<double>::infinity();
        stage_begin = Clock::now();
        initializePose(state, options, graph_posterior, rotation, centered_translation, posterior, objective,
            result.diagnostics.tested_hypotheses, result.diagnostics.candidates);
        result.diagnostics.hypothesis_seconds =
            std::chrono::duration<double>(Clock::now() - stage_begin).count();
        result.diagnostics.initialization_transformation.setIdentity();
        result.diagnostics.initialization_transformation.block<3, 3>(0, 0) = rotation;
        result.diagnostics.initialization_transformation.block<3, 1>(0, 3) =
            state.target_center + centered_translation - rotation * state.source_center;

        std::vector<int> selected;
        double current_joint_objective = jointObjective(
            state, rotation, centered_translation, options, posterior);
        Mat3 best_rotation = rotation;
        Vec3 best_centered_translation = centered_translation;
        const Mat3 mode_rotation = rotation;
        const Vec3 mode_centered_translation = centered_translation;
        std::vector<double> best_posterior = posterior;
        double best_joint_objective = current_joint_objective;
        for (std::size_t outer = 0; outer < options.outer_iterations; ++outer) {
            const Mat3 previous_rotation = rotation;
            const Vec3 previous_translation = centered_translation;
            const double previous_objective = current_joint_objective;
            stage_begin = Clock::now();
            std::vector<double> proposed_posterior = meanField(
                state, &rotation, &centered_translation, options, posterior);
            result.diagnostics.posterior_seconds +=
                std::chrono::duration<double>(Clock::now() - stage_begin).count();
            const double posterior_objective = modeConditionedObjective(
                state, rotation, centered_translation, mode_rotation, mode_centered_translation,
                options, proposed_posterior);
            if (posterior_objective + 1e-12 >= current_joint_objective) {
                posterior = std::move(proposed_posterior);
                current_joint_objective = posterior_objective;
            }
            stage_begin = Clock::now();
            selected = selectByInformation(state, rotation, posterior, options);
            result.diagnostics.selection_seconds +=
                std::chrono::duration<double>(Clock::now() - stage_begin).count();
            stage_begin = Clock::now();
            optimizePose(state, selected, posterior, options,
                mode_rotation, mode_centered_translation, rotation, centered_translation);
            result.diagnostics.pose_seconds +=
                std::chrono::duration<double>(Clock::now() - stage_begin).count();
            const double next_objective = modeConditionedObjective(
                state, rotation, centered_translation, mode_rotation, mode_centered_translation,
                options, posterior);
            result.diagnostics.objective_history.push_back(next_objective);
            result.diagnostics.completed_outer_iterations = outer + 1;
            if (next_objective > best_joint_objective) {
                best_joint_objective = next_objective;
                best_rotation = rotation;
                best_centered_translation = centered_translation;
                best_posterior = posterior;
            }
            const double objective_change = std::abs(next_objective - previous_objective);
            const double rotation_change = Eigen::AngleAxisd(rotation * previous_rotation.transpose()).angle();
            const double translation_change = (centered_translation - previous_translation).norm();
            current_joint_objective = next_objective;
            if (objective_change <= options.objective_tolerance *
                    (1.0 + std::abs(current_joint_objective)) &&
                rotation_change < options.rotation_tolerance &&
                translation_change < options.translation_tolerance) {
                result.diagnostics.converged = true;
                break;
            }
        }
        rotation = best_rotation;
        centered_translation = best_centered_translation;
        stage_begin = Clock::now();
        posterior = meanField(state, &rotation, &centered_translation, options, std::move(best_posterior));
        result.diagnostics.posterior_seconds +=
            std::chrono::duration<double>(Clock::now() - stage_begin).count();
        objective = elbo(state, rotation, centered_translation, options, posterior);
        stage_begin = Clock::now();
        selected = selectByInformation(state, rotation, posterior, options);
        result.diagnostics.selection_seconds +=
            std::chrono::duration<double>(Clock::now() - stage_begin).count();
        result.diagnostics.used_correspondences = state.source.size();
        result.diagnostics.relation_count = state.relations.size();
        result.diagnostics.selected_count = selected.size();
        result.diagnostics.characteristic_length = state.length;
        result.diagnostics.final_elbo = objective;
        result.diagnostics.posterior = posterior;
        result.diagnostics.selected_indices = selected;
        stage_begin = Clock::now();
        computeDiagnostics(state, rotation, posterior, options, selected, result.diagnostics);
        result.diagnostics.diagnostics_seconds =
            std::chrono::duration<double>(Clock::now() - stage_begin).count();

        const Vec3 translation = state.target_center + centered_translation - rotation * state.source_center;
        result.transformation.setIdentity();
        result.transformation.block<3, 3>(0, 0) = rotation;
        result.transformation.block<3, 1>(0, 3) = translation;
        result.success = result.transformation.allFinite() && !selected.empty();
        result.message = result.success ? "ok" : "Non-finite pose or empty selected set";
    } catch (const std::exception& error) {
        result.success = false;
        result.message = error.what();
    }
    result.diagnostics.total_seconds =
        std::chrono::duration<double>(Clock::now() - total_begin).count();
    return result;
}

void writeDiagnosticsJson(const std::string& path, const Result& result) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot open diagnostics file: " + path);
    output << std::setprecision(12);
    const Diagnostics& d = result.diagnostics;
    auto jsonNumber = [](double value) {
        if (!std::isfinite(value)) return std::string("null");
        std::ostringstream number;
        number << std::setprecision(12) << value;
        return number.str();
    };
    output << "{\n  \"method\": \"GCI\",\n"
           << "  \"success\": " << (result.success ? "true" : "false") << ",\n"
           << "  \"message\": \"" << result.message << "\",\n"
           << "  \"input_correspondences\": " << d.input_correspondences << ",\n"
           << "  \"used_correspondences\": " << d.used_correspondences << ",\n"
           << "  \"relation_count\": " << d.relation_count << ",\n"
           << "  \"tested_hypotheses\": " << d.tested_hypotheses << ",\n"
           << "  \"selected_count\": " << d.selected_count << ",\n"
           << "  \"outer_iterations\": " << d.completed_outer_iterations << ",\n"
           << "  \"converged\": " << (d.converged ? "true" : "false") << ",\n"
           << "  \"characteristic_length\": " << d.characteristic_length << ",\n"
           << "  \"final_elbo\": " << d.final_elbo << ",\n"
           << "  \"preprocessing_seconds\": " << d.preprocessing_seconds << ",\n"
           << "  \"graph_seconds\": " << d.graph_seconds << ",\n"
           << "  \"posterior_seconds\": " << d.posterior_seconds << ",\n"
           << "  \"hypothesis_seconds\": " << d.hypothesis_seconds << ",\n"
           << "  \"selection_seconds\": " << d.selection_seconds << ",\n"
           << "  \"pose_seconds\": " << d.pose_seconds << ",\n"
           << "  \"diagnostics_seconds\": " << d.diagnostics_seconds << ",\n"
           << "  \"instrumented_total_seconds\": " << d.total_seconds << ",\n"
           << "  \"min_information_eigenvalue\": " << jsonNumber(d.min_information_eigenvalue) << ",\n"
           << "  \"information_condition_number\": " << jsonNumber(d.information_condition_number) << ",\n"
           << "  \"covariance_trace\": " << jsonNumber(d.covariance_trace) << ",\n";
    output << "  \"information_rank\": " << d.information_rank << ",\n"
           << "  \"selected_information_rank\": " << d.selected_information_rank << ",\n";
    output << "  \"selected_min_information_eigenvalue\": "
           << jsonNumber(d.selected_min_information_eigenvalue) << ",\n"
           << "  \"selected_information_condition_number\": "
           << jsonNumber(d.selected_information_condition_number) << ",\n"
           << "  \"selected_covariance_trace\": " << jsonNumber(d.selected_covariance_trace) << ",\n";
    auto writeArray = [&](const char* name, const auto& values, bool comma) {
        output << "  \"" << name << "\": [";
        for (std::size_t i = 0; i < values.size(); ++i) {
            if (i) output << ", ";
            output << values[i];
        }
        output << "]" << (comma ? "," : "") << "\n";
    };
    writeArray("information_eigenvalues", d.information_eigenvalues, true);
    writeArray("selected_information_eigenvalues", d.selected_information_eigenvalues, true);
    writeArray("objective_history", d.objective_history, true);
    writeArray("retained_input_indices", d.retained_input_indices, true);
    writeArray("selected_indices", d.selected_indices, true);
    writeArray("posterior", d.posterior, false);
    output << "}\n";
}

}  // namespace gci
