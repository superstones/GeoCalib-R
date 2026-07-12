#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <regex>
#include <fstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cmath>
#include <set>
#include <deque>
#include <map>
#include <unordered_map>
#include <memory>
#include <sstream>
#include <limits>
#include <iomanip>
#include <cstdlib>
#include <cctype>
#include <cstdint>

// Third-party dependencies.
#include <pangolin/pangolin.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>

// GTSAM pose graph optimization
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>

// Local aliases.
namespace fs = std::filesystem;
using json = nlohmann::json;
using Eigen::Matrix4f;
using Eigen::Matrix3f;
using Eigen::Vector3f;
using Eigen::Vector4f;
using Eigen::Quaternionf;

constexpr double kPi = 3.14159265358979323846;

// ==========================================================================================
// Configuration and global parameters.
// ==========================================================================================
struct SupervisedPriorRecord {
    bool valid = false;
    double dx = 0.0;
    double dy = 0.0;
    double dyaw = 0.0;
    double sigma_x = 0.5;
    double sigma_y = 0.5;
    double sigma_yaw = 0.08;
    double confidence = 0.0;
};

struct Config {
    std::string PC_DATA_ROOT = "Dair-v2xdataset/output_cooperative_world_scenes";
    const std::string JSON_FILENAME = "fused_inference_results.json";
    std::string METRICS_CSV;
    std::string SUPERVISED_PRIOR_CSV;
    std::map<std::string, SupervisedPriorRecord> SUPERVISED_PRIORS;
    const float POINT_SIZE = 2.0f;
    float MATCH_DIST_THRESHOLD = 10.0f;
    const float MIN_Z = -2.0f;
    const float MAX_Z = 4.0f;
    const float GROUND_OFFSET = -1.65f;
    int DELAY_MS = 150;

    // Vehicle-side delay in frames.
    int VEH_DELAY_FRAMES = 20;
    int WINDOW_SIZE = 10;
    const double BOX_POINT_SIGMA = 0.6;
    const double BOX_CENTER_SIGMA = 0.8;
    const double BOX_YAW_SIGMA = 0.15;
    const double ODOM_TRANS_SIGMA = 0.4;
    const double ODOM_ROT_SIGMA = 0.08;
    const double VELOCITY_TRANS_SIGMA = 1.0;
    const double VELOCITY_ROT_SIGMA = 0.18;
    const double PRIOR_TRANS_SIGMA = 5.0;
    const double PRIOR_ROT_SIGMA = 0.8;
    const double LANDMARK_PRIOR_TRANS_SIGMA = 0.7;
    const double LANDMARK_PRIOR_YAW_SIGMA = 0.25;
    const double LANDMARK_PRIOR_YAW_LOOSE_SIGMA = 3.0;
    const double FIXED_LAG_ANCHOR_TRANS_SIGMA = 0.25;
    const double FIXED_LAG_ANCHOR_YAW_SIGMA = 0.05;
    double YAW_ADAPTIVE_GATE = 0.55;
    double YAW_REJECT_GATE = 3.2;
    double SIZE_RATIO_GATE = 0.55;
    double GRAPH_CONSISTENCY_GATE = 2.0;
    double GRAPH_MIN_INLIER_RATIO = 0.45;
    int GRAPH_MIN_CANDIDATES = 3;
    const double SCENE_MATCH_CANDIDATE_DIST = 20.0;
    double SCENE_MATCH_MIN_SCORE = 0.20;
    int MAX_SCENE_CANDIDATES = 80;
    int MAX_FRAME_OBSERVATIONS = 25;
    int REMATCH_ITERATIONS = 3;
    const double REMATCH_TRANS_EPS = 0.05;
    const double REMATCH_YAW_EPS = 0.01;
    int MIN_BOX_OBSERVATIONS = 2;
    int MIN_CENTER_ONLY_OBSERVATIONS = 2;
    double MAX_CORRECTION_TRANS = 4.0;
    double MAX_CORRECTION_YAW = 0.20;
    double MAX_CORRECTION_STEP_TRANS = 2.0;
    double MAX_CORRECTION_STEP_YAW = 0.08;
    const double SINGLE_MATCH_MAX_CORRECTION_TRANS = 1.2;
    const double SINGLE_MATCH_MAX_CORRECTION_YAW = 0.08;
    const double SINGLE_MATCH_MAX_STEP_TRANS = 0.5;
    const double SINGLE_MATCH_MAX_STEP_YAW = 0.04;
    double MAX_MEAN_GRAPH_ERROR = 3.0;
    const double REJECT_DECAY = 0.75;
    const int TRUST_REGION_LINE_STEPS = 4;
    const double TRUST_REGION_CENTER_TOL = 0.0;
    const double TRUST_REGION_YAW_WEIGHT = 0.25;
    bool STATIC_ICP_ENABLED = false;
    int STATIC_ICP_MAX_BOXES = 2;
    int STATIC_ICP_MAX_POINTS = 1200;
    int STATIC_ICP_ITERATIONS = 8;
    int STATIC_ICP_MIN_CORRESPONDENCES = 50;
    double STATIC_ICP_GRID_SIZE = 0.8;
    double STATIC_ICP_MAX_CORRESPONDENCE = 3.0;
    double STATIC_ICP_MAX_RMSE = 2.0;
    double STATIC_ICP_MAX_TRANS = 1.2;
    double STATIC_ICP_MAX_YAW = 0.10;
    double STATIC_ICP_BOX_MARGIN = 1.0;
    double STATIC_ICP_TRANS_SIGMA = 0.55;
    double STATIC_ICP_YAW_SIGMA = 0.08;
    double SUPERVISED_PRIOR_MIN_CONF = 0.10;
    double SUPERVISED_PRIOR_MAX_SIGMA_TRANS = 2.0;
    double SUPERVISED_PRIOR_MAX_SIGMA_YAW = 0.35;
    bool V2X_REAL_MODE = false;
};

// ==========================================================================================
// 2. GTSAM sliding-window pose graph backend
// ==========================================================================================
double wrapAngle(double angle) {
    while (angle > kPi) angle -= 2.0 * kPi;
    while (angle < -kPi) angle += 2.0 * kPi;
    return angle;
}

double wrapBoxYawResidual(double angle) {
    double wrapped = wrapAngle(angle);
    if (wrapped > kPi / 2.0) wrapped -= kPi;
    if (wrapped < -kPi / 2.0) wrapped += kPi;
    return wrapped;
}

double boxYawDistance(double a, double b) {
    return std::abs(wrapBoxYawResidual(a - b));
}

double boxYaw(const std::vector<Vector3f>& box) {
    if (box.size() != 8) return 0.0;
    Vector3f center = Vector3f::Zero();
    for (const auto& p : box) center += p;
    center /= static_cast<float>(box.size());
    double xx = 0.0;
    double xy = 0.0;
    double yy = 0.0;
    for (const auto& p : box) {
        const double dx = p.x() - center.x();
        const double dy = p.y() - center.y();
        xx += dx * dx;
        xy += dx * dy;
        yy += dy * dy;
    }
    return 0.5 * std::atan2(2.0 * xy, xx - yy);
}

std::vector<Vector3f> transformBox(const Matrix4f& transform, const std::vector<Vector3f>& box) {
    std::vector<Vector3f> out;
    out.reserve(box.size());
    for (const auto& p : box) {
        Vector4f hp = transform * Vector4f(p.x(), p.y(), p.z(), 1.0f);
        out.emplace_back(hp.x(), hp.y(), hp.z());
    }
    return out;
}

Vector3f boxCenter(const std::vector<Vector3f>& box) {
    Vector3f center = Vector3f::Zero();
    for (const auto& p : box) center += p;
    return box.empty() ? center : center / static_cast<float>(box.size());
}

std::vector<int> bestCornerPermutation(const std::vector<Vector3f>& source, const std::vector<Vector3f>& target) {
    std::vector<int> best = {0, 1, 2, 3, 4, 5, 6, 7};
    if (source.size() != 8 || target.size() != 8) return best;

    const std::vector<std::vector<int>> bases = {
        {0, 1, 2, 3},
        {0, 3, 2, 1},
    };
    double best_cost = std::numeric_limits<double>::max();
    for (const auto& base : bases) {
        for (int shift = 0; shift < 4; ++shift) {
            std::vector<int> perm(8);
            for (int i = 0; i < 4; ++i) {
                int idx = base[(i + shift) % 4];
                perm[i] = idx;
                perm[i + 4] = idx + 4;
            }

            double cost = 0.0;
            for (int i = 0; i < 8; ++i) {
                cost += (source[perm[i]] - target[i]).squaredNorm();
            }
            if (cost < best_cost) {
                best_cost = cost;
                best = perm;
            }
        }
    }
    return best;
}

std::vector<Vector3f> reorderBox(const std::vector<Vector3f>& box, const std::vector<int>& perm) {
    if (box.size() != 8 || perm.size() != 8) return box;
    std::vector<Vector3f> reordered;
    reordered.reserve(8);
    for (int idx : perm) reordered.push_back(box[idx]);
    return reordered;
}

gtsam::Pose3 matrixToPose3(const Matrix4f& T) {
    Eigen::Matrix3d R = T.block<3,3>(0,0).cast<double>();
    Eigen::Vector3d t = T.block<3,1>(0,3).cast<double>();
    return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t.x(), t.y(), t.z()));
}

Matrix4f pose3ToMatrix(const gtsam::Pose3& pose) {
    Matrix4f T = Matrix4f::Identity();
    T.block<3,3>(0,0) = pose.rotation().matrix().cast<float>();
    T.block<3,1>(0,3) = pose.translation().cast<float>();
    return T;
}

gtsam::Pose2 matrixToPose2(const Matrix4f& T) {
    const double yaw = std::atan2(T(1, 0), T(0, 0));
    return gtsam::Pose2(T(0, 3), T(1, 3), yaw);
}

Matrix4f pose2ToMatrixLike(const Matrix4f& reference, const gtsam::Pose2& pose) {
    const gtsam::Pose2 ref_pose = matrixToPose2(reference);
    const double dyaw = wrapAngle(pose.theta() - ref_pose.theta());
    Eigen::AngleAxisf yaw_delta(static_cast<float>(dyaw), Vector3f::UnitZ());

    Matrix4f T = reference;
    T.block<3,3>(0,0) = yaw_delta.toRotationMatrix() * reference.block<3,3>(0,0);
    T(0, 3) = static_cast<float>(pose.x());
    T(1, 3) = static_cast<float>(pose.y());
    return T;
}

gtsam::Pose2 perturbPose2(const gtsam::Pose2& pose, int idx, double eps) {
    if (idx == 0) return gtsam::Pose2(pose.x() + eps, pose.y(), pose.theta());
    if (idx == 1) return gtsam::Pose2(pose.x(), pose.y() + eps, pose.theta());
    return gtsam::Pose2(pose.x(), pose.y(), wrapAngle(pose.theta() + eps));
}

Eigen::Vector3d planarBoxSize(const std::vector<Vector3f>& box) {
    if (box.size() != 8) return Eigen::Vector3d::Zero();

    const double yaw = boxYaw(box);
    const double c = std::cos(-yaw);
    const double s = std::sin(-yaw);
    double min_x = std::numeric_limits<double>::max();
    double min_y = std::numeric_limits<double>::max();
    double min_z = std::numeric_limits<double>::max();
    double max_x = -std::numeric_limits<double>::max();
    double max_y = -std::numeric_limits<double>::max();
    double max_z = -std::numeric_limits<double>::max();

    for (const auto& p : box) {
        const double x = c * p.x() - s * p.y();
        const double y = s * p.x() + c * p.y();
        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        min_z = std::min(min_z, static_cast<double>(p.z()));
        max_z = std::max(max_z, static_cast<double>(p.z()));
    }

    double a = max_x - min_x;
    double b = max_y - min_y;
    if (a < b) std::swap(a, b);
    return Eigen::Vector3d(a, b, max_z - min_z);
}

double maxSizeRatioError(const std::vector<Vector3f>& a, const std::vector<Vector3f>& b) {
    const Eigen::Vector3d sa = planarBoxSize(a);
    const Eigen::Vector3d sb = planarBoxSize(b);
    double err = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double denom = std::max(std::max(std::abs(sa(i)), std::abs(sb(i))), 1e-3);
        err = std::max(err, std::abs(sa(i) - sb(i)) / denom);
    }
    return err;
}

gtsam::Pose2 boxWorldPose2(const std::vector<Vector3f>& box) {
    const Vector3f center = boxCenter(box);
    return gtsam::Pose2(center.x(), center.y(), boxYaw(box));
}

std::vector<Eigen::Vector2d> boxOffsetsInPose2(const std::vector<Vector3f>& box, const gtsam::Pose2& pose) {
    std::vector<Eigen::Vector2d> offsets;
    offsets.reserve(box.size());
    const double c = std::cos(-pose.theta());
    const double s = std::sin(-pose.theta());
    for (const auto& p : box) {
        const double dx = p.x() - pose.x();
        const double dy = p.y() - pose.y();
        offsets.emplace_back(c * dx - s * dy, s * dx + c * dy);
    }
    return offsets;
}

struct BoxObservation {
    std::vector<Vector3f> veh_box_local;
    std::vector<Vector3f> inf_box_world;
    std::vector<Eigen::Vector2d> landmark_corner_offsets;
    std::string veh_track_id;
    std::string inf_track_id;
    std::string landmark_id;
    gtsam::Pose2 landmark_pose_world;
    Vector3f veh_center_world = Vector3f::Zero();
    Vector3f inf_center_world = Vector3f::Zero();
    double center_distance = 0.0;
    double yaw_diff = 0.0;
    double size_ratio_error = 0.0;
    double graph_inlier_ratio = 1.0;
    double scene_mean_error = 0.0;
    double sinkhorn_score = 1.0;
    double initial_agent_yaw = 0.0;
    double score = 1.0;
    int scene_inlier_count = 0;
    int frame_match_count = 0;
    bool center_only = false;
};

struct StaticIcpObservation {
    bool valid = false;
    gtsam::Pose2 pose_prior_world;
    int correspondences = 0;
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double dx = 0.0;
    double dy = 0.0;
    double dyaw = 0.0;
};

struct SupervisedPosePrior {
    bool valid = false;
    gtsam::Pose2 pose_prior_world;
    double sigma_x = 0.5;
    double sigma_y = 0.5;
    double sigma_yaw = 0.08;
    double confidence = 0.0;
};

struct FrameObservation {
    std::string frame_name;
    Matrix4f initial_vehicle_pose = Matrix4f::Identity();
    std::vector<BoxObservation> boxes;
    StaticIcpObservation static_icp;
    SupervisedPosePrior supervised_prior;
};

class PlanarBoxLandmarkFactor : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Pose2> {
    std::vector<Vector3f> veh_box_local_;
    std::vector<Eigen::Vector2d> landmark_corner_offsets_;
    Eigen::Vector2d veh_center_local_;
    double veh_yaw_local_ = 0.0;
    double fixed_agent_yaw_ = 0.0;
    bool center_only_ = false;

    gtsam::Vector computeError(const gtsam::Pose2& agent, const gtsam::Pose2& landmark) const {
        if (center_only_) {
            gtsam::Vector residual(2);
            const double c = std::cos(fixed_agent_yaw_);
            const double s = std::sin(fixed_agent_yaw_);
            const double center_x = agent.x() + c * veh_center_local_.x() - s * veh_center_local_.y();
            const double center_y = agent.y() + s * veh_center_local_.x() + c * veh_center_local_.y();
            residual(0) = center_x - landmark.x();
            residual(1) = center_y - landmark.y();
            return residual;
        }

        gtsam::Vector residual(17);
        for (size_t i = 0; i < 8; ++i) {
            const auto& p = veh_box_local_[i];
            const auto& o = landmark_corner_offsets_[i];
            const gtsam::Point2 veh_world = agent.transformFrom(gtsam::Point2(p.x(), p.y()));
            const gtsam::Point2 landmark_world = landmark.transformFrom(gtsam::Point2(o.x(), o.y()));
            residual(static_cast<int>(2 * i + 0)) = veh_world.x() - landmark_world.x();
            residual(static_cast<int>(2 * i + 1)) = veh_world.y() - landmark_world.y();
        }
        residual(16) = wrapBoxYawResidual(agent.theta() + veh_yaw_local_ - landmark.theta());
        return residual;
    }

    void numericalJacobian(
        const gtsam::Pose2& agent,
        const gtsam::Pose2& landmark,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2) const {
        const gtsam::Vector error = computeError(agent, landmark);
        const double eps = 1e-5;
        if (H1) {
            H1->resize(error.size(), 3);
            for (int i = 0; i < 3; ++i) {
                H1->col(i) =
                    (computeError(perturbPose2(agent, i, eps), landmark) -
                     computeError(perturbPose2(agent, i, -eps), landmark)) / (2.0 * eps);
            }
        }
        if (H2) {
            H2->resize(error.size(), 3);
            for (int i = 0; i < 3; ++i) {
                H2->col(i) =
                    (computeError(agent, perturbPose2(landmark, i, eps)) -
                     computeError(agent, perturbPose2(landmark, i, -eps))) / (2.0 * eps);
            }
        }
    }

public:
    PlanarBoxLandmarkFactor(
        gtsam::Key agent_key,
        gtsam::Key landmark_key,
        const BoxObservation& obs,
        const gtsam::SharedNoiseModel& noise)
        : gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Pose2>(noise, agent_key, landmark_key),
          veh_box_local_(obs.veh_box_local),
          landmark_corner_offsets_(obs.landmark_corner_offsets),
          fixed_agent_yaw_(obs.initial_agent_yaw),
          center_only_(obs.center_only) {
        const Vector3f c = boxCenter(obs.veh_box_local);
        veh_center_local_ = Eigen::Vector2d(c.x(), c.y());
        veh_yaw_local_ = boxYaw(obs.veh_box_local);
    }

    gtsam::Vector evaluateError(
        const gtsam::Pose2& agent,
        const gtsam::Pose2& landmark,
        gtsam::OptionalMatrixType H1 = nullptr,
        gtsam::OptionalMatrixType H2 = nullptr) const override {
        const gtsam::Vector error = computeError(agent, landmark);
        numericalJacobian(agent, landmark, H1, H2);
        return error;
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            std::make_shared<PlanarBoxLandmarkFactor>(*this));
    }
};

class ConstantVelocityFactor : public gtsam::NoiseModelFactor3<gtsam::Pose2, gtsam::Pose2, gtsam::Pose2> {
    gtsam::Vector computeError(
        const gtsam::Pose2& prev,
        const gtsam::Pose2& cur,
        const gtsam::Pose2& next) const {
        const gtsam::Pose2 d1 = prev.between(cur);
        const gtsam::Pose2 d2 = cur.between(next);
        gtsam::Vector residual(3);
        residual(0) = d2.x() - d1.x();
        residual(1) = d2.y() - d1.y();
        residual(2) = wrapAngle(d2.theta() - d1.theta());
        return residual;
    }

    void fillJacobian(
        const gtsam::Pose2& prev,
        const gtsam::Pose2& cur,
        const gtsam::Pose2& next,
        gtsam::OptionalMatrixType H1,
        gtsam::OptionalMatrixType H2,
        gtsam::OptionalMatrixType H3) const {
        const gtsam::Vector error = computeError(prev, cur, next);
        const double eps = 1e-5;
        if (H1) {
            H1->resize(error.size(), 3);
            for (int i = 0; i < 3; ++i) {
                H1->col(i) =
                    (computeError(perturbPose2(prev, i, eps), cur, next) -
                     computeError(perturbPose2(prev, i, -eps), cur, next)) / (2.0 * eps);
            }
        }
        if (H2) {
            H2->resize(error.size(), 3);
            for (int i = 0; i < 3; ++i) {
                H2->col(i) =
                    (computeError(prev, perturbPose2(cur, i, eps), next) -
                     computeError(prev, perturbPose2(cur, i, -eps), next)) / (2.0 * eps);
            }
        }
        if (H3) {
            H3->resize(error.size(), 3);
            for (int i = 0; i < 3; ++i) {
                H3->col(i) =
                    (computeError(prev, cur, perturbPose2(next, i, eps)) -
                     computeError(prev, cur, perturbPose2(next, i, -eps))) / (2.0 * eps);
            }
        }
    }

public:
    ConstantVelocityFactor(
        gtsam::Key prev_key,
        gtsam::Key cur_key,
        gtsam::Key next_key,
        const gtsam::SharedNoiseModel& noise)
        : gtsam::NoiseModelFactor3<gtsam::Pose2, gtsam::Pose2, gtsam::Pose2>(
              noise, prev_key, cur_key, next_key) {}

    gtsam::Vector evaluateError(
        const gtsam::Pose2& prev,
        const gtsam::Pose2& cur,
        const gtsam::Pose2& next,
        gtsam::OptionalMatrixType H1 = nullptr,
        gtsam::OptionalMatrixType H2 = nullptr,
        gtsam::OptionalMatrixType H3 = nullptr) const override {
        const gtsam::Vector error = computeError(prev, cur, next);
        fillJacobian(prev, cur, next, H1, H2, H3);
        return error;
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return std::static_pointer_cast<gtsam::NonlinearFactor>(
            std::make_shared<ConstantVelocityFactor>(*this));
    }
};

class SlidingWindowGtsamOptimizer {
    int window_size_;
    Config cfg_;
    std::deque<FrameObservation> frames_;
    std::map<std::string, gtsam::Pose2> optimized_cache_;
    std::map<std::string, gtsam::Pose2> landmark_cache_;
    double last_error_ = 0.0;
    double last_mean_error_ = 0.0;
    bool last_rejected_ = false;
    std::string last_reject_reason_;
    size_t last_landmark_count_ = 0;
    double last_frame_quality_ = 0.0;
    int reject_streak_ = 0;

    gtsam::SharedNoiseModel pose2Noise(double rot_sigma, double trans_sigma) const {
        gtsam::Vector3 sigmas;
        sigmas << trans_sigma, trans_sigma, rot_sigma;
        return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    }

    gtsam::SharedNoiseModel boxNoise(const BoxObservation& obs) const {
        const int dim = obs.center_only ? 2 : 17;
        const double weight = std::sqrt(observationQuality(obs));
        gtsam::Vector sigmas(dim);
        if (obs.center_only) {
            sigmas(0) = cfg_.BOX_CENTER_SIGMA / weight;
            sigmas(1) = cfg_.BOX_CENTER_SIGMA / weight;
        } else {
            for (int i = 0; i < 16; ++i) sigmas(i) = cfg_.BOX_POINT_SIGMA / weight;
            sigmas(16) = cfg_.BOX_YAW_SIGMA / weight;
        }
        auto gaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        return gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(1.0), gaussian);
    }

    gtsam::SharedNoiseModel landmarkPriorNoise(const BoxObservation& obs) const {
        const double scale = 1.0 / std::sqrt(observationQuality(obs));
        gtsam::Vector3 sigmas;
        sigmas << cfg_.LANDMARK_PRIOR_TRANS_SIGMA * scale,
                  cfg_.LANDMARK_PRIOR_TRANS_SIGMA * scale,
                  (obs.center_only ? cfg_.LANDMARK_PRIOR_YAW_LOOSE_SIGMA : cfg_.LANDMARK_PRIOR_YAW_SIGMA) * scale;
        return gtsam::noiseModel::Diagonal::Sigmas(sigmas);
    }

    gtsam::SharedNoiseModel staticIcpNoise(const StaticIcpObservation& obs) const {
        const double scale = std::clamp(obs.rmse / std::max(0.25, cfg_.STATIC_ICP_MAX_RMSE), 1.0, 2.5);
        gtsam::Vector3 sigmas;
        sigmas << cfg_.STATIC_ICP_TRANS_SIGMA * scale,
                  cfg_.STATIC_ICP_TRANS_SIGMA * scale,
                  cfg_.STATIC_ICP_YAW_SIGMA * scale;
        auto gaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        return gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(1.5), gaussian);
    }

    gtsam::SharedNoiseModel supervisedPriorNoise(const SupervisedPosePrior& prior) const {
        const double scale = 1.0 / std::sqrt(std::clamp(prior.confidence, 0.05, 1.0));
        gtsam::Vector3 sigmas;
        sigmas << std::max(0.05, prior.sigma_x * scale),
                  std::max(0.05, prior.sigma_y * scale),
                  std::max(0.01, prior.sigma_yaw * scale);
        auto gaussian = gtsam::noiseModel::Diagonal::Sigmas(sigmas);
        return gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Huber::Create(1.5), gaussian);
    }

    double observationQuality(const BoxObservation& obs) const {
        double quality = std::clamp(obs.score, 0.0, 1.0);
        quality *= std::clamp(obs.graph_inlier_ratio, 0.1, 1.0);
        quality *= std::clamp(obs.sinkhorn_score, 0.1, 1.0);
        quality *= std::exp(-obs.center_distance / std::max<double>(cfg_.MATCH_DIST_THRESHOLD, 1.0));
        quality *= std::exp(-obs.size_ratio_error / std::max(cfg_.SIZE_RATIO_GATE, 0.1));
        if (!obs.center_only) {
            quality *= std::exp(-obs.yaw_diff / std::max(cfg_.YAW_REJECT_GATE, 0.1));
        } else {
            quality *= 0.55;
        }
        if (obs.frame_match_count <= 1) quality *= 0.55;
        if (obs.scene_mean_error > 0.0) quality /= (1.0 + 0.25 * obs.scene_mean_error);
        return std::clamp(quality, 0.03, 1.0);
    }

    double frameQuality(const FrameObservation& frame) const {
        if (frame.boxes.empty()) return 0.0;
        double sum = 0.0;
        for (const auto& obs : frame.boxes) sum += observationQuality(obs);
        return sum / static_cast<double>(frame.boxes.size());
    }

    Eigen::Vector3d correctionFromInitial(const gtsam::Pose2& pose, const Matrix4f& initial) const {
        const gtsam::Pose2 init = matrixToPose2(initial);
        return Eigen::Vector3d(
            pose.x() - init.x(),
            pose.y() - init.y(),
            wrapAngle(pose.theta() - init.theta()));
    }

    gtsam::Pose2 poseWithCorrection(const Matrix4f& initial, const Eigen::Vector3d& corr) const {
        const gtsam::Pose2 init = matrixToPose2(initial);
        return gtsam::Pose2(
            init.x() + corr.x(),
            init.y() + corr.y(),
            wrapAngle(init.theta() + corr.z()));
    }

    gtsam::Pose2 fallbackLatestPose() const {
        const gtsam::Pose2 latest_initial = matrixToPose2(frames_.back().initial_vehicle_pose);
        if (frames_.size() < 2) return latest_initial;

        const auto& prev = frames_[frames_.size() - 2];
        auto cached_prev = optimized_cache_.find(prev.frame_name);
        if (cached_prev == optimized_cache_.end()) return latest_initial;

        Eigen::Vector3d prev_corr = correctionFromInitial(cached_prev->second, prev.initial_vehicle_pose);
        if (reject_streak_ > 1) prev_corr *= cfg_.REJECT_DECAY;
        return poseWithCorrection(frames_.back().initial_vehicle_pose, prev_corr);
    }

    double frameCenterError(const FrameObservation& frame, const gtsam::Pose2& pose) const {
        if (frame.boxes.empty()) return std::numeric_limits<double>::quiet_NaN();

        const Matrix4f pose_matrix = pose2ToMatrixLike(frame.initial_vehicle_pose, pose);
        double sum = 0.0;
        int count = 0;
        for (const auto& obs : frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
            const auto veh_box_world = transformBox(pose_matrix, obs.veh_box_local);
            sum += (boxCenter(veh_box_world) - boxCenter(obs.inf_box_world)).norm();
            ++count;
        }
        return count > 0 ? sum / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();
    }

    double framePlanarCenterError(const FrameObservation& frame, const gtsam::Pose2& pose) const {
        if (frame.boxes.empty()) return std::numeric_limits<double>::quiet_NaN();

        const Matrix4f pose_matrix = pose2ToMatrixLike(frame.initial_vehicle_pose, pose);
        double sum = 0.0;
        int count = 0;
        for (const auto& obs : frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
            const auto veh_box_world = transformBox(pose_matrix, obs.veh_box_local);
            const Vector3f veh_center = boxCenter(veh_box_world);
            const Vector3f inf_center = boxCenter(obs.inf_box_world);
            sum += std::hypot(veh_center.x() - inf_center.x(), veh_center.y() - inf_center.y());
            ++count;
        }
        return count > 0 ? sum / static_cast<double>(count) : std::numeric_limits<double>::quiet_NaN();
    }

    double selectionCenterError(const FrameObservation& frame, const gtsam::Pose2& pose) const {
        return framePlanarCenterError(frame, pose);
    }

    double frameResidualScore(const FrameObservation& frame, const gtsam::Pose2& pose) const {
        if (frame.boxes.empty()) return std::numeric_limits<double>::quiet_NaN();

        const Matrix4f pose_matrix = pose2ToMatrixLike(frame.initial_vehicle_pose, pose);
        double weighted_sum = 0.0;
        double weight_sum = 0.0;
        for (const auto& obs : frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
            const auto veh_box_world = transformBox(pose_matrix, obs.veh_box_local);
            double residual = (boxCenter(veh_box_world) - boxCenter(obs.inf_box_world)).norm();
            if (!obs.center_only) {
                residual += cfg_.TRUST_REGION_YAW_WEIGHT *
                    boxYawDistance(boxYaw(veh_box_world), boxYaw(obs.inf_box_world));
            }
            const double weight = observationQuality(obs);
            weighted_sum += weight * residual;
            weight_sum += weight;
        }
        return weight_sum > 0.0
            ? weighted_sum / weight_sum
            : std::numeric_limits<double>::quiet_NaN();
    }

    gtsam::Pose2 interpolateCorrection(
        const gtsam::Pose2& from,
        const gtsam::Pose2& to,
        double alpha) const {
        const auto& latest_frame = frames_.back();
        const Eigen::Vector3d from_corr = correctionFromInitial(from, latest_frame.initial_vehicle_pose);
        const Eigen::Vector3d to_corr = correctionFromInitial(to, latest_frame.initial_vehicle_pose);
        Eigen::Vector3d mixed;
        mixed.x() = from_corr.x() + alpha * (to_corr.x() - from_corr.x());
        mixed.y() = from_corr.y() + alpha * (to_corr.y() - from_corr.y());
        mixed.z() = wrapAngle(from_corr.z() + alpha * wrapAngle(to_corr.z() - from_corr.z()));
        return poseWithCorrection(latest_frame.initial_vehicle_pose, mixed);
    }

    gtsam::Pose2 centerRegistrationCandidate(const gtsam::Pose2& base) const {
        const auto& latest_frame = frames_.back();
        const Matrix4f base_matrix = pose2ToMatrixLike(latest_frame.initial_vehicle_pose, base);
        Eigen::Vector2d delta_sum = Eigen::Vector2d::Zero();
        double yaw_sum = 0.0;
        double weight_sum = 0.0;
        double yaw_weight_sum = 0.0;

        for (const auto& obs : latest_frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;

            const auto veh_box_world = transformBox(base_matrix, obs.veh_box_local);
            const Vector3f veh_center = boxCenter(veh_box_world);
            const Vector3f inf_center = boxCenter(obs.inf_box_world);
            const Eigen::Vector2d residual(
                static_cast<double>(inf_center.x() - veh_center.x()),
                static_cast<double>(inf_center.y() - veh_center.y()));
            const double residual_norm = residual.norm();
            const double robust = 1.0 / std::max(1.0, residual_norm / std::max(0.5, cfg_.BOX_CENTER_SIGMA));
            const double weight = observationQuality(obs) * robust;
            delta_sum += weight * residual;
            weight_sum += weight;

            if (!obs.center_only) {
                const double yaw_residual = wrapBoxYawResidual(boxYaw(obs.inf_box_world) - boxYaw(veh_box_world));
                yaw_sum += weight * yaw_residual;
                yaw_weight_sum += weight;
            }
        }

        if (weight_sum <= 1e-9) return base;

        Eigen::Vector2d delta = delta_sum / weight_sum;
        const double max_step = std::max(0.05, cfg_.MAX_CORRECTION_STEP_TRANS);
        const double delta_norm = delta.norm();
        if (delta_norm > max_step) delta *= max_step / delta_norm;

        double yaw_delta = 0.0;
        if (yaw_weight_sum > 1e-9) {
            yaw_delta = 0.5 * yaw_sum / yaw_weight_sum;
            yaw_delta = std::clamp(yaw_delta, -cfg_.MAX_CORRECTION_STEP_YAW, cfg_.MAX_CORRECTION_STEP_YAW);
        }

        return gtsam::Pose2(
            base.x() + delta.x(),
            base.y() + delta.y(),
            wrapAngle(base.theta() + yaw_delta));
    }

    gtsam::Pose2 refinedCenterRegistrationCandidate(const gtsam::Pose2& base, int iterations) const {
        gtsam::Pose2 pose = base;
        for (int i = 0; i < iterations; ++i) {
            pose = centerRegistrationCandidate(pose);
        }
        return pose;
    }

    bool boundedPlanarRegistrationCandidate(gtsam::Pose2& pose) const {
        const auto& latest_frame = frames_.back();
        if (!cfg_.V2X_REAL_MODE || latest_frame.boxes.empty()) return false;

        const gtsam::Pose2 initial = matrixToPose2(latest_frame.initial_vehicle_pose);
        const double initial_center = selectionCenterError(latest_frame, initial);
        if (!std::isfinite(initial_center)) return false;

        const Matrix3f ref_rotation = latest_frame.initial_vehicle_pose.block<3,3>(0,0);
        const Eigen::Vector2d init_t(initial.x(), initial.y());
        const double sparse_bonus = latest_frame.boxes.size() <= 4 ? 2.5 : 1.5;
        const double trans_gate = std::min(
            cfg_.MAX_CORRECTION_TRANS,
            std::max(0.8, initial_center + sparse_bonus));
        const double yaw_gate = cfg_.MAX_CORRECTION_YAW;
        const int yaw_steps = 41;

        double best_error = initial_center;
        gtsam::Pose2 best_pose = initial;
        bool found = false;

        for (int i = 0; i < yaw_steps; ++i) {
            const double alpha = yaw_steps == 1
                ? 0.0
                : static_cast<double>(i) / static_cast<double>(yaw_steps - 1);
            const double dyaw = -yaw_gate + 2.0 * yaw_gate * alpha;
            const double c = std::cos(dyaw);
            const double s = std::sin(dyaw);
            Matrix3f yaw_delta = Matrix3f::Identity();
            yaw_delta(0, 0) = static_cast<float>(c);
            yaw_delta(0, 1) = static_cast<float>(-s);
            yaw_delta(1, 0) = static_cast<float>(s);
            yaw_delta(1, 1) = static_cast<float>(c);
            const Matrix3f rotation = yaw_delta * ref_rotation;

            std::vector<Eigen::Vector2d> translations;
            std::vector<double> weights;
            translations.reserve(latest_frame.boxes.size());
            weights.reserve(latest_frame.boxes.size());
            for (const auto& obs : latest_frame.boxes) {
                if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
                const Vector3f src_center = boxCenter(obs.veh_box_local);
                const Vector3f dst_center = boxCenter(obs.inf_box_world);
                const Vector3f rotated = rotation * src_center;
                translations.emplace_back(
                    static_cast<double>(dst_center.x() - rotated.x()),
                    static_cast<double>(dst_center.y() - rotated.y()));
                weights.push_back(observationQuality(obs));
            }
            if (translations.empty()) continue;

            auto try_translation = [&](Eigen::Vector2d t) {
                Eigen::Vector2d corr = t - init_t;
                const double corr_norm = corr.norm();
                if (corr_norm > trans_gate) {
                    corr *= trans_gate / corr_norm;
                    t = init_t + corr;
                }

                const gtsam::Pose2 candidate(t.x(), t.y(), wrapAngle(initial.theta() + dyaw));
                const double candidate_error = selectionCenterError(latest_frame, candidate);
                if (std::isfinite(candidate_error) && candidate_error < best_error) {
                    best_error = candidate_error;
                    best_pose = candidate;
                    found = true;
                }
            };

            for (int weight_mode = 0; weight_mode < 2; ++weight_mode) {
                Eigen::Vector2d t_sum = Eigen::Vector2d::Zero();
                double weight_sum = 0.0;
                for (size_t j = 0; j < translations.size(); ++j) {
                    const double weight = weight_mode == 0 ? weights[j] : 1.0;
                    t_sum += weight * translations[j];
                    weight_sum += weight;
                }
                if (weight_sum > 1e-9) try_translation(t_sum / weight_sum);
            }

            std::vector<double> xs;
            std::vector<double> ys;
            xs.reserve(translations.size());
            ys.reserve(translations.size());
            for (const auto& t : translations) {
                xs.push_back(t.x());
                ys.push_back(t.y());
            }
            std::sort(xs.begin(), xs.end());
            std::sort(ys.begin(), ys.end());
            const size_t mid = xs.size() / 2;
            const Eigen::Vector2d median_t(
                xs.size() % 2 == 0 ? 0.5 * (xs[mid - 1] + xs[mid]) : xs[mid],
                ys.size() % 2 == 0 ? 0.5 * (ys[mid - 1] + ys[mid]) : ys[mid]);
            try_translation(median_t);

            if (translations.size() <= 8) {
                for (const auto& t : translations) try_translation(t);
            }
        }

        if (!found) return false;
        pose = best_pose;
        return true;
    }

    bool coarseBoxRegistrationCandidate(gtsam::Pose2& pose) const {
        const auto& latest_frame = frames_.back();
        if (latest_frame.boxes.size() < 2) return false;

        double weight_sum = 0.0;
        Eigen::Vector2d src_mean = Eigen::Vector2d::Zero();
        Eigen::Vector2d dst_mean = Eigen::Vector2d::Zero();
        struct Pair {
            Eigen::Vector2d src;
            Eigen::Vector2d dst;
            double weight = 0.0;
        };
        std::vector<Pair> pairs;
        pairs.reserve(latest_frame.boxes.size());

        for (const auto& obs : latest_frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
            const Vector3f src_center = boxCenter(obs.veh_box_local);
            const Vector3f dst_center = boxCenter(obs.inf_box_world);
            const double weight = observationQuality(obs);
            pairs.push_back({
                Eigen::Vector2d(src_center.x(), src_center.y()),
                Eigen::Vector2d(dst_center.x(), dst_center.y()),
                weight,
            });
            src_mean += weight * pairs.back().src;
            dst_mean += weight * pairs.back().dst;
            weight_sum += weight;
        }

        if (pairs.size() < 2 || weight_sum <= 1e-9) return false;
        if (pairs.size() == 2 &&
            ((pairs[0].src - pairs[1].src).norm() < 1.0 ||
             (pairs[0].dst - pairs[1].dst).norm() < 1.0)) {
            return false;
        }
        src_mean /= weight_sum;
        dst_mean /= weight_sum;

        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        for (const auto& pair : pairs) {
            H += pair.weight * (pair.src - src_mean) * (pair.dst - dst_mean).transpose();
        }

        Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
        if (R.determinant() < 0.0) {
            Eigen::Matrix2d V = svd.matrixV();
            V.col(1) *= -1.0;
            R = V * svd.matrixU().transpose();
        }
        const Eigen::Vector2d t = dst_mean - R * src_mean;
        pose = gtsam::Pose2(t.x(), t.y(), std::atan2(R(1, 0), R(0, 0)));
        return true;
    }

    bool sparseTranslationCandidate(const gtsam::Pose2& base, gtsam::Pose2& pose) const {
        const auto& latest_frame = frames_.back();
        if (latest_frame.boxes.empty() || latest_frame.boxes.size() > 2) return false;

        const double c = std::cos(base.theta());
        const double s = std::sin(base.theta());
        Eigen::Vector2d t_sum = Eigen::Vector2d::Zero();
        double weight_sum = 0.0;

        for (const auto& obs : latest_frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
            const Vector3f src_center = boxCenter(obs.veh_box_local);
            const Vector3f dst_center = boxCenter(obs.inf_box_world);
            const Eigen::Vector2d rotated(
                c * src_center.x() - s * src_center.y(),
                s * src_center.x() + c * src_center.y());
            const Eigen::Vector2d target(dst_center.x(), dst_center.y());
            const double weight = observationQuality(obs);
            t_sum += weight * (target - rotated);
            weight_sum += weight;
        }

        if (weight_sum <= 1e-9) return false;
        const Eigen::Vector2d t = t_sum / weight_sum;
        pose = gtsam::Pose2(t.x(), t.y(), base.theta());
        return true;
    }

    bool robustLowQualityTranslationCandidate(const gtsam::Pose2& base, gtsam::Pose2& pose) const {
        const auto& latest_frame = frames_.back();
        if (!cfg_.V2X_REAL_MODE || latest_frame.boxes.empty() || latest_frame.boxes.size() > 6) return false;

        struct TranslationVote {
            Eigen::Vector2d t;
            double weight = 0.0;
            double residual = 0.0;
        };
        std::vector<TranslationVote> votes;
        votes.reserve(latest_frame.boxes.size());

        const double c = std::cos(base.theta());
        const double s = std::sin(base.theta());
        Eigen::Vector2d weighted_mean = Eigen::Vector2d::Zero();
        double weight_sum = 0.0;
        for (const auto& obs : latest_frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
            const Vector3f src_center = boxCenter(obs.veh_box_local);
            const Vector3f dst_center = boxCenter(obs.inf_box_world);
            const Eigen::Vector2d rotated(
                c * src_center.x() - s * src_center.y(),
                s * src_center.x() + c * src_center.y());
            const Eigen::Vector2d target(dst_center.x(), dst_center.y());
            const double weight = observationQuality(obs);
            votes.push_back({target - rotated, weight, 0.0});
            weighted_mean += weight * votes.back().t;
            weight_sum += weight;
        }

        if (votes.empty() || weight_sum <= 1e-9) return false;
        weighted_mean /= weight_sum;
        for (auto& vote : votes) {
            vote.residual = (vote.t - weighted_mean).norm();
        }
        std::sort(votes.begin(), votes.end(), [](const TranslationVote& a, const TranslationVote& b) {
            return a.residual < b.residual;
        });

        const size_t keep = votes.size() >= 3 ? votes.size() - 1 : votes.size();
        Eigen::Vector2d trimmed_mean = Eigen::Vector2d::Zero();
        double trimmed_weight = 0.0;
        for (size_t i = 0; i < keep; ++i) {
            trimmed_mean += votes[i].weight * votes[i].t;
            trimmed_weight += votes[i].weight;
        }
        if (trimmed_weight <= 1e-9) return false;
        trimmed_mean /= trimmed_weight;

        pose = gtsam::Pose2(trimmed_mean.x(), trimmed_mean.y(), base.theta());
        const Eigen::Vector3d corr = correctionFromInitial(pose, latest_frame.initial_vehicle_pose);
        const double corr_trans = std::hypot(corr.x(), corr.y());
        const double base_center = frameCenterError(latest_frame, base);
        const double max_trans = std::max(cfg_.MAX_CORRECTION_TRANS, base_center + 2.5);
        return std::isfinite(base_center) && corr_trans <= max_trans;
    }

    std::vector<gtsam::Pose2> singleBoxStepCandidates(const gtsam::Pose2& base) const {
        const auto& latest_frame = frames_.back();
        const Matrix4f base_matrix = pose2ToMatrixLike(latest_frame.initial_vehicle_pose, base);
        std::vector<gtsam::Pose2> poses;
        poses.reserve(latest_frame.boxes.size());

        for (const auto& obs : latest_frame.boxes) {
            if (obs.veh_box_local.size() != 8 || obs.inf_box_world.size() != 8) continue;
            const auto veh_box_world = transformBox(base_matrix, obs.veh_box_local);
            const Vector3f src_center = boxCenter(veh_box_world);
            const Vector3f dst_center = boxCenter(obs.inf_box_world);
            Eigen::Vector2d delta(
                static_cast<double>(dst_center.x() - src_center.x()),
                static_cast<double>(dst_center.y() - src_center.y()));
            const double max_step = std::max(0.05, cfg_.MAX_CORRECTION_STEP_TRANS);
            const double delta_norm = delta.norm();
            if (delta_norm > max_step) delta *= max_step / delta_norm;

            double yaw_delta = 0.0;
            if (!obs.center_only) {
                yaw_delta = 0.5 * wrapBoxYawResidual(boxYaw(obs.inf_box_world) - boxYaw(veh_box_world));
                yaw_delta = std::clamp(yaw_delta, -cfg_.MAX_CORRECTION_STEP_YAW, cfg_.MAX_CORRECTION_STEP_YAW);
            }

            poses.emplace_back(
                base.x() + delta.x(),
                base.y() + delta.y(),
                wrapAngle(base.theta() + yaw_delta));
        }

        return poses;
    }

    gtsam::Pose2 selectTrustRegionPose(
        const gtsam::Pose2& optimized,
        bool& residual_guarded) const {
        residual_guarded = false;
        const auto& latest_frame = frames_.back();
        const gtsam::Pose2 initial = matrixToPose2(latest_frame.initial_vehicle_pose);
        const gtsam::Pose2 fallback = fallbackLatestPose();
        const double initial_center = selectionCenterError(latest_frame, initial);
        if (!std::isfinite(initial_center)) {
            if (latest_frame.static_icp.valid) return latest_frame.static_icp.pose_prior_world;
            if (latest_frame.supervised_prior.valid) return latest_frame.supervised_prior.pose_prior_world;
            return optimized;
        }

        struct Candidate {
            gtsam::Pose2 pose;
            double center = std::numeric_limits<double>::quiet_NaN();
            double score = std::numeric_limits<double>::quiet_NaN();
            bool full_update = false;
        };

        const double latest_quality = frameQuality(latest_frame);
        const bool low_quality_recovery =
            cfg_.V2X_REAL_MODE &&
            latest_frame.boxes.size() <= 4 &&
            latest_quality < 0.08 &&
            initial_center > 3.0;
        std::vector<Candidate> candidates;
        auto add_candidate = [&](const gtsam::Pose2& pose, bool full_update) {
            Candidate cand;
            cand.pose = pose;
            cand.center = selectionCenterError(latest_frame, pose);
            cand.score = frameResidualScore(latest_frame, pose);
            cand.full_update = full_update;
            if (std::isfinite(cand.center) && std::isfinite(cand.score)) {
                candidates.push_back(cand);
            }
        };

        add_candidate(initial, false);
        add_candidate(fallback, false);
        add_candidate(optimized, true);
        add_candidate(centerRegistrationCandidate(initial), false);
        add_candidate(centerRegistrationCandidate(fallback), false);
        add_candidate(centerRegistrationCandidate(optimized), false);
        if (initial_center > std::max(10.0, cfg_.MAX_CORRECTION_TRANS) || low_quality_recovery) {
            gtsam::Pose2 coarse;
            if (coarseBoxRegistrationCandidate(coarse)) {
                add_candidate(coarse, false);
                add_candidate(centerRegistrationCandidate(coarse), false);
            }
            gtsam::Pose2 sparse;
            if (sparseTranslationCandidate(initial, sparse)) {
                add_candidate(sparse, false);
            }
            gtsam::Pose2 robust_translation;
            if (robustLowQualityTranslationCandidate(initial, robust_translation)) {
                add_candidate(robust_translation, false);
                add_candidate(centerRegistrationCandidate(robust_translation), false);
            }
            if (robustLowQualityTranslationCandidate(fallback, robust_translation)) {
                add_candidate(robust_translation, false);
                add_candidate(centerRegistrationCandidate(robust_translation), false);
            }
            if (low_quality_recovery) {
                for (const auto& pose : singleBoxStepCandidates(initial)) {
                    add_candidate(pose, false);
                }
                for (const auto& pose : singleBoxStepCandidates(fallback)) {
                    add_candidate(pose, false);
                }
            }
        }
        if (!cfg_.V2X_REAL_MODE &&
            latest_frame.boxes.size() >= 2 &&
            latest_quality >= 0.05) {
            auto within_dair_registration_gate = [&](const gtsam::Pose2& pose) {
                const Eigen::Vector3d corr = correctionFromInitial(pose, latest_frame.initial_vehicle_pose);
                const double corr_trans = std::hypot(corr.x(), corr.y());
                const double corr_yaw = std::abs(corr.z());
                const double trans_gate = std::min(
                    cfg_.MAX_CORRECTION_TRANS,
                    std::max(1.0, initial_center + 1.5));
                return corr_trans <= trans_gate && corr_yaw <= cfg_.MAX_CORRECTION_YAW;
            };

            gtsam::Pose2 direct;
            if (coarseBoxRegistrationCandidate(direct) &&
                within_dair_registration_gate(direct)) {
                add_candidate(direct, false);
                add_candidate(refinedCenterRegistrationCandidate(direct, 2), false);
            }
        }
        if (cfg_.V2X_REAL_MODE) {
            auto within_v2x_registration_gate = [&](const gtsam::Pose2& pose) {
                const Eigen::Vector3d corr = correctionFromInitial(pose, latest_frame.initial_vehicle_pose);
                const double corr_trans = std::hypot(corr.x(), corr.y());
                const double corr_yaw = std::abs(corr.z());
                const double sparse_bonus = latest_frame.boxes.size() <= 4 ? 2.5 : 1.5;
                const double trans_gate = std::min(
                    cfg_.MAX_CORRECTION_TRANS,
                    std::max(0.8, initial_center + sparse_bonus));
                const double yaw_gate = cfg_.MAX_CORRECTION_YAW;
                return corr_trans <= trans_gate && corr_yaw <= yaw_gate;
            };

            gtsam::Pose2 direct;
            if (boundedPlanarRegistrationCandidate(direct) &&
                within_v2x_registration_gate(direct)) {
                add_candidate(direct, false);
                add_candidate(refinedCenterRegistrationCandidate(direct, 2), false);
            }
            if (latest_frame.boxes.size() >= 2 &&
                latest_quality >= 0.05 &&
                coarseBoxRegistrationCandidate(direct) &&
                within_v2x_registration_gate(direct)) {
                add_candidate(direct, false);
                add_candidate(centerRegistrationCandidate(direct), false);
                add_candidate(refinedCenterRegistrationCandidate(direct, 4), false);
            }
            if (latest_frame.boxes.size() == 1 &&
                sparseTranslationCandidate(initial, direct) &&
                within_v2x_registration_gate(direct)) {
                add_candidate(direct, false);
                add_candidate(centerRegistrationCandidate(direct), false);
                add_candidate(refinedCenterRegistrationCandidate(direct, 4), false);
            }
            if (latest_frame.boxes.size() >= 2 && latest_frame.boxes.size() <= 4) {
                for (const auto& pose : singleBoxStepCandidates(initial)) {
                    if (within_v2x_registration_gate(pose)) {
                        add_candidate(pose, false);
                        add_candidate(refinedCenterRegistrationCandidate(pose, 3), false);
                    }
                }
                for (const auto& pose : singleBoxStepCandidates(fallback)) {
                    if (within_v2x_registration_gate(pose)) {
                        add_candidate(pose, false);
                        add_candidate(refinedCenterRegistrationCandidate(pose, 3), false);
                    }
                }
            }
        }
        if (latest_frame.static_icp.valid) {
            add_candidate(latest_frame.static_icp.pose_prior_world, false);
            add_candidate(centerRegistrationCandidate(latest_frame.static_icp.pose_prior_world), false);
        }
        if (latest_frame.supervised_prior.valid) {
            add_candidate(latest_frame.supervised_prior.pose_prior_world, false);
            add_candidate(centerRegistrationCandidate(latest_frame.supervised_prior.pose_prior_world), false);
        }
        const int steps = std::max(1, cfg_.TRUST_REGION_LINE_STEPS);
        for (int i = 1; i < steps; ++i) {
            const double alpha = static_cast<double>(i) / static_cast<double>(steps);
            add_candidate(interpolateCorrection(initial, optimized, alpha), false);
            add_candidate(interpolateCorrection(fallback, optimized, alpha), false);
        }

        const double max_center = initial_center + cfg_.TRUST_REGION_CENTER_TOL;
        auto best = candidates.front();
        auto outside_correction_gate = [&](const Candidate& cand) {
            const Eigen::Vector3d corr = correctionFromInitial(cand.pose, latest_frame.initial_vehicle_pose);
            const double corr_trans = std::hypot(corr.x(), corr.y());
            const double corr_yaw = std::abs(corr.z());
            const double max_candidate_trans = std::max(cfg_.MAX_CORRECTION_TRANS, initial_center + 4.0);
            const double max_candidate_yaw = low_quality_recovery
                ? std::max(cfg_.MAX_CORRECTION_YAW, 0.45)
                : cfg_.MAX_CORRECTION_YAW;
            return corr_trans > max_candidate_trans || corr_yaw > max_candidate_yaw;
        };
        for (const auto& cand : candidates) {
            if (cand.center > max_center) continue;
            if (cand.center < best.center ||
                (std::abs(cand.center - best.center) < 1e-9 && cand.score < best.score)) {
                best = cand;
            }
        }

        if (cfg_.V2X_REAL_MODE && best.center > 1.5 && outside_correction_gate(best)) {
            bool found_guarded = false;
            Candidate guarded = best;
            for (const auto& cand : candidates) {
                if (cand.center > max_center || outside_correction_gate(cand)) continue;
                if (!found_guarded ||
                    cand.center < guarded.center ||
                    (std::abs(cand.center - guarded.center) < 1e-9 && cand.score < guarded.score)) {
                    guarded = cand;
                    found_guarded = true;
                }
            }
            if (found_guarded) best = guarded;
        }

        if (!best.full_update) {
            const double optimized_center = selectionCenterError(latest_frame, optimized);
            residual_guarded = std::isfinite(optimized_center) && optimized_center > max_center;
        }
        return best.pose;
    }

public:
    explicit SlidingWindowGtsamOptimizer(const Config& cfg)
        : window_size_(cfg.WINDOW_SIZE), cfg_(cfg) {}

    void reset() {
        frames_.clear();
        optimized_cache_.clear();
        landmark_cache_.clear();
        last_error_ = 0.0;
        last_mean_error_ = 0.0;
        last_rejected_ = false;
        last_reject_reason_.clear();
        last_landmark_count_ = 0;
        last_frame_quality_ = 0.0;
        reject_streak_ = 0;
    }

    void addFrame(const FrameObservation& frame) {
        frames_.push_back(frame);
        while (frames_.size() > static_cast<size_t>(window_size_)) {
            frames_.pop_front();
        }
    }

    void replaceLatestFrame(const FrameObservation& frame) {
        if (!frames_.empty()) frames_.back() = frame;
    }

    Matrix4f optimizeLatest() {
        if (frames_.empty()) return Matrix4f::Identity();

        gtsam::NonlinearFactorGraph graph;
        gtsam::Values initial_values;
        const auto prior_noise = pose2Noise(cfg_.PRIOR_ROT_SIGMA, cfg_.PRIOR_TRANS_SIGMA);
        const auto odom_noise = pose2Noise(cfg_.ODOM_ROT_SIGMA, cfg_.ODOM_TRANS_SIGMA);
        const auto velocity_noise = pose2Noise(cfg_.VELOCITY_ROT_SIGMA, cfg_.VELOCITY_TRANS_SIGMA);
        const auto fixed_lag_noise = pose2Noise(cfg_.FIXED_LAG_ANCHOR_YAW_SIGMA, cfg_.FIXED_LAG_ANCHOR_TRANS_SIGMA);
        std::map<std::string, size_t> landmark_indices;

        for (size_t i = 0; i < frames_.size(); ++i) {
            const auto& frame = frames_[i];
            const gtsam::Key key = gtsam::Symbol('x', i);
            auto cached = optimized_cache_.find(frame.frame_name);
            gtsam::Pose2 initial_pose = cached == optimized_cache_.end()
                ? matrixToPose2(frame.initial_vehicle_pose)
                : cached->second;
            initial_values.insert(key, initial_pose);

            graph.add(gtsam::PriorFactor<gtsam::Pose2>(
                key, matrixToPose2(frame.initial_vehicle_pose), prior_noise));
            if (i == 0 && cached != optimized_cache_.end() && frames_.size() >= static_cast<size_t>(window_size_)) {
                graph.add(gtsam::PriorFactor<gtsam::Pose2>(key, cached->second, fixed_lag_noise));
            }

            if (i > 0) {
                const auto prev_key = gtsam::Symbol('x', i - 1);
                const auto prev_pose = matrixToPose2(frames_[i - 1].initial_vehicle_pose);
                const auto cur_pose = matrixToPose2(frame.initial_vehicle_pose);
                graph.add(gtsam::BetweenFactor<gtsam::Pose2>(
                    prev_key, key, prev_pose.between(cur_pose), odom_noise));
            }
            if (i > 1) {
                graph.add(std::make_shared<ConstantVelocityFactor>(
                    gtsam::Symbol('x', i - 2), gtsam::Symbol('x', i - 1), key, velocity_noise));
            }
            if (frame.static_icp.valid && frame.boxes.empty()) {
                graph.add(gtsam::PriorFactor<gtsam::Pose2>(
                    key, frame.static_icp.pose_prior_world, staticIcpNoise(frame.static_icp)));
            }
            if (frame.supervised_prior.valid) {
                graph.add(gtsam::PriorFactor<gtsam::Pose2>(
                    key, frame.supervised_prior.pose_prior_world, supervisedPriorNoise(frame.supervised_prior)));
            }
        }

        for (size_t i = 0; i < frames_.size(); ++i) {
            const auto agent_key = gtsam::Symbol('x', i);
            for (const auto& obs : frames_[i].boxes) {
                if (obs.veh_box_local.size() != 8 ||
                    (!obs.center_only && obs.landmark_corner_offsets.size() != 8) ||
                    obs.landmark_id.empty()) {
                    continue;
                }

                auto [it, inserted] = landmark_indices.emplace(obs.landmark_id, landmark_indices.size());
                const gtsam::Key landmark_key = gtsam::Symbol('l', it->second);
                if (inserted) {
                    auto cached_landmark = landmark_cache_.find(obs.landmark_id);
                    initial_values.insert(
                        landmark_key,
                        cached_landmark == landmark_cache_.end() ? obs.landmark_pose_world : cached_landmark->second);
                }
                graph.add(gtsam::PriorFactor<gtsam::Pose2>(
                    landmark_key, obs.landmark_pose_world, landmarkPriorNoise(obs)));
                graph.add(std::make_shared<PlanarBoxLandmarkFactor>(
                    agent_key, landmark_key, obs, boxNoise(obs)));
            }
        }
        last_landmark_count_ = landmark_indices.size();
        last_frame_quality_ = frameQuality(frames_.back());

        gtsam::LevenbergMarquardtParams params;
        params.setMaxIterations(30);
        params.setVerbosityLM("SILENT");

        try {
            gtsam::LevenbergMarquardtOptimizer optimizer(graph, initial_values, params);
            gtsam::Values result = optimizer.optimize();
            last_error_ = graph.error(result);
            last_mean_error_ = last_landmark_count_ > 0
                ? last_error_ / static_cast<double>(last_landmark_count_)
                : 0.0;
            last_rejected_ = false;
            last_reject_reason_.clear();

            const gtsam::Pose2 latest = result.at<gtsam::Pose2>(gtsam::Symbol('x', frames_.size() - 1));
            const Eigen::Vector3d latest_corr = correctionFromInitial(latest, frames_.back().initial_vehicle_pose);
            const double corr_trans = std::hypot(latest_corr.x(), latest_corr.y());
            const double corr_yaw = std::abs(latest_corr.z());
            double max_corr_trans = cfg_.MAX_CORRECTION_TRANS;
            double max_corr_yaw = cfg_.MAX_CORRECTION_YAW;
            if (frames_.back().boxes.size() <= 1 || last_frame_quality_ < 0.18) {
                max_corr_trans = std::min(max_corr_trans, cfg_.SINGLE_MATCH_MAX_CORRECTION_TRANS);
                max_corr_yaw = std::min(max_corr_yaw, cfg_.SINGLE_MATCH_MAX_CORRECTION_YAW);
            }
            if (corr_trans > max_corr_trans || corr_yaw > max_corr_yaw) {
                last_rejected_ = true;
                last_reject_reason_ = "absolute_gate";
            }

            if (!last_rejected_ &&
                last_landmark_count_ > 0 &&
                last_mean_error_ > cfg_.MAX_MEAN_GRAPH_ERROR) {
                last_rejected_ = true;
                last_reject_reason_ = "error_gate";
            }

            if (!last_rejected_ && frames_.size() >= 2) {
                const auto& prev = frames_[frames_.size() - 2];
                auto cached_prev = optimized_cache_.find(prev.frame_name);
                if (cached_prev != optimized_cache_.end()) {
                    const Eigen::Vector3d prev_corr = correctionFromInitial(cached_prev->second, prev.initial_vehicle_pose);
                    const double step_trans = std::hypot(
                        latest_corr.x() - prev_corr.x(),
                        latest_corr.y() - prev_corr.y());
                    const double step_yaw = std::abs(wrapAngle(latest_corr.z() - prev_corr.z()));
                    double max_step_trans = cfg_.MAX_CORRECTION_STEP_TRANS;
                    double max_step_yaw = cfg_.MAX_CORRECTION_STEP_YAW;
                    if (frames_.back().boxes.size() <= 1 || last_frame_quality_ < 0.18) {
                        max_step_trans = std::min(max_step_trans, cfg_.SINGLE_MATCH_MAX_STEP_TRANS);
                        max_step_yaw = std::min(max_step_yaw, cfg_.SINGLE_MATCH_MAX_STEP_YAW);
                    }
                    if (step_trans > max_step_trans || step_yaw > max_step_yaw) {
                        last_rejected_ = true;
                        last_reject_reason_ = "continuity_gate";
                    }
                }
            }

            if (last_rejected_) {
                ++reject_streak_;
                bool residual_guarded = false;
                const gtsam::Pose2 fallback = selectTrustRegionPose(latest, residual_guarded);
                optimized_cache_[frames_.back().frame_name] = fallback;
                return pose2ToMatrixLike(frames_.back().initial_vehicle_pose, fallback);
            }

            bool residual_guarded = false;
            const gtsam::Pose2 selected_latest = selectTrustRegionPose(latest, residual_guarded);
            reject_streak_ = 0;
            for (size_t i = 0; i < frames_.size(); ++i) {
                optimized_cache_[frames_[i].frame_name] = result.at<gtsam::Pose2>(gtsam::Symbol('x', i));
            }
            optimized_cache_[frames_.back().frame_name] = selected_latest;
            for (const auto& [id, idx] : landmark_indices) {
                landmark_cache_[id] = result.at<gtsam::Pose2>(gtsam::Symbol('l', idx));
            }
            return pose2ToMatrixLike(frames_.back().initial_vehicle_pose, selected_latest);
        } catch (const std::exception& e) {
            std::cerr << "[GTSAM] optimization failed: " << e.what() << std::endl;
            last_rejected_ = true;
            last_reject_reason_ = "exception";
            ++reject_streak_;
            return frames_.back().initial_vehicle_pose;
        }
    }

    double lastError() const {
        return last_error_;
    }

    double lastMeanError() const {
        return last_mean_error_;
    }

    bool lastRejected() const {
        return last_rejected_;
    }

    const std::string& lastRejectReason() const {
        return last_reject_reason_;
    }

    size_t lastLandmarkCount() const {
        return last_landmark_count_;
    }

    double lastFrameQuality() const {
        return last_frame_quality_;
    }
};

// ==========================================================================================
// OpenGL helper functions.
// ==========================================================================================
Matrix4f invertTransformMatrix(const Matrix4f& T) {
    Matrix3f R = T.block<3,3>(0,0); Vector3f t = T.block<3,1>(0,3);
    Matrix3f Ri = R.transpose();
    Matrix4f Ti = Matrix4f::Identity(); Ti.block<3,3>(0,0) = Ri; Ti.block<3,1>(0,3) = -Ri * t;
    return Ti;
}

void glColorHeight(float z, float min_z, float max_z, bool is_veh) {
    float ratio = std::clamp((z - min_z) / (max_z - min_z), 0.0f, 1.0f);
    if (is_veh) glColor3f(0.1f, 0.4f + 0.6f * ratio, 0.1f);
    else glColor3f(0.8f, 0.1f + 0.4f * ratio, 0.1f);
}

void drawTrajectory(const std::vector<Vector3f>& pts, const pangolin::Colour& col, float lw) {
    if (pts.size() < 2) return;
    glColor4f(col.r, col.g, col.b, col.a); glLineWidth(lw);
    glBegin(GL_LINE_STRIP);
    for (const auto& p : pts) glVertex3f(p.x(), p.y(), p.z());
    glEnd();
}

void drawBox(const std::vector<Vector3f>& pts, const pangolin::Colour& col, float lw) {
    if (pts.size() != 8) return;
    const int c[12][2] = {{0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7}};
    glColor4f(col.r, col.g, col.b, col.a); glLineWidth(lw);
    for (auto& i : c) { pangolin::glDrawLine(pts[i[0]].x(), pts[i[0]].y(), pts[i[0]].z(), pts[i[1]].x(), pts[i[1]].y(), pts[i[1]].z()); }
}

void drawCoordinateFrame(const Matrix4f& pose, float size, float lw) {
    glLineWidth(lw);
    Vector4f o = pose * Vector4f(0,0,0,1), x = pose * Vector4f(size,0,0,1), y = pose * Vector4f(0,size,0,1), z = pose * Vector4f(0,0,size,1);
    glColor3f(1,0,0); pangolin::glDrawLine(o.x(),o.y(),o.z(),x.x(),x.y(),x.z());
    glColor3f(0,1,0); pangolin::glDrawLine(o.x(),o.y(),o.z(),y.x(),y.y(),y.z());
    glColor3f(0,0,1); pangolin::glDrawLine(o.x(),o.y(),o.z(),z.x(),z.y(),z.z());
}

Matrix4f jsonToMatrix4f(const json& raw) {
    Matrix4f T = Matrix4f::Identity();
    if (!raw.is_array() || raw.size() < 4) return T;
    for (int r = 0; r < 4; ++r) {
        if (!raw[r].is_array() || raw[r].size() < 4) continue;
        for (int c = 0; c < 4; ++c) T(r, c) = raw[r][c].get<float>();
    }
    return T;
}

std::vector<Vector3f> readBoxPoints(const json& obj) {
    std::vector<Vector3f> box;
    if (!obj.contains("3d_box_points_world") || !obj["3d_box_points_world"].is_array()) return box;
    for (const auto& p : obj["3d_box_points_world"]) {
        if (p.is_array() && p.size() >= 3) {
            box.emplace_back(p[0].get<float>(), p[1].get<float>(), p[2].get<float>());
        }
    }
    return box;
}

const json* findAnnotationByTrackId(const json& anns, const std::string& track_id) {
    if (!anns.is_array()) return nullptr;
    for (const auto& obj : anns) {
        if (obj.value("track_id", "") == track_id) return &obj;
    }
    return nullptr;
}

struct AnnotationBoxFeature {
    std::string track_id;
    std::vector<Vector3f> box_world;
    Vector3f center = Vector3f::Zero();
    double yaw = 0.0;
};

struct CandidateMatch {
    size_t veh_idx = 0;
    size_t inf_idx = 0;
    double center_distance = 0.0;
    double yaw_diff = 0.0;
    double size_ratio_error = 0.0;
    double scene_inlier_ratio = 1.0;
    double scene_mean_error = 0.0;
    double sinkhorn_score = 0.0;
    double score = 0.0;
    double hinted_score = 0.0;
    int scene_inlier_count = 0;
    bool hinted = false;
};

struct SceneMatchResult {
    std::vector<AnnotationBoxFeature> vehicle_boxes;
    std::vector<AnnotationBoxFeature> infrastructure_boxes;
    std::vector<CandidateMatch> matches;
};

std::string pairKey(const std::string& veh_id, const std::string& inf_id) {
    return veh_id + "\x1f" + inf_id;
}

std::vector<AnnotationBoxFeature> readAnnotationBoxFeatures(const json& anns, const Matrix4f& transform) {
    std::vector<AnnotationBoxFeature> out;
    if (!anns.is_array()) return out;
    for (const auto& obj : anns) {
        std::vector<Vector3f> box = transformBox(transform, readBoxPoints(obj));
        if (box.size() != 8) continue;

        AnnotationBoxFeature feature;
        feature.track_id = obj.value("track_id", "");
        feature.box_world = std::move(box);
        feature.center = boxCenter(feature.box_world);
        feature.yaw = boxYaw(feature.box_world);
        out.push_back(std::move(feature));
    }
    return out;
}

std::unordered_map<std::string, double> buildPairHintScores(const json& pairs) {
    std::unordered_map<std::string, double> hints;
    if (!pairs.is_array()) return hints;
    for (const auto& pair : pairs) {
        const std::string vid = pair.value("veh_track_id", "");
        const std::string iid = pair.value("inf_track_id", "");
        if (vid.empty() || iid.empty()) continue;
        hints[pairKey(vid, iid)] = std::clamp(pair.value("confidence", pair.value("score", 1.0)), 0.0, 1.0);
    }
    return hints;
}

Eigen::Vector2d transformPlanarPoint(const Vector3f& point, double yaw_delta, const Eigen::Vector2d& t) {
    const double c = std::cos(yaw_delta);
    const double s = std::sin(yaw_delta);
    return Eigen::Vector2d(
        c * point.x() - s * point.y() + t.x(),
        s * point.x() + c * point.y() + t.y());
}

void fillSceneConsistencyStats(
    CandidateMatch& cand,
    const std::vector<AnnotationBoxFeature>& vehicle_boxes,
    const std::vector<AnnotationBoxFeature>& infrastructure_boxes,
    const Config& cfg) {
    const auto& veh = vehicle_boxes[cand.veh_idx];
    const auto& inf = infrastructure_boxes[cand.inf_idx];
    const double yaw_delta = wrapBoxYawResidual(inf.yaw - veh.yaw);
    const double c = std::cos(yaw_delta);
    const double s = std::sin(yaw_delta);
    const Eigen::Vector2d t(
        inf.center.x() - (c * veh.center.x() - s * veh.center.y()),
        inf.center.y() - (s * veh.center.x() + c * veh.center.y()));

    int total = 0;
    int inliers = 0;
    double error_sum = 0.0;
    std::vector<bool> used_infra(infrastructure_boxes.size(), false);

    for (const auto& v_box : vehicle_boxes) {
        ++total;
        const Eigen::Vector2d predicted = transformPlanarPoint(v_box.center, yaw_delta, t);
        double best_dist = std::numeric_limits<double>::max();
        int best_idx = -1;
        for (size_t i = 0; i < infrastructure_boxes.size(); ++i) {
            if (used_infra[i]) continue;
            const auto& i_box = infrastructure_boxes[i];
            if (maxSizeRatioError(v_box.box_world, i_box.box_world) > cfg.SIZE_RATIO_GATE) continue;
            const double dx = predicted.x() - i_box.center.x();
            const double dy = predicted.y() - i_box.center.y();
            const double dist = std::hypot(dx, dy);
            if (dist < best_dist) {
                best_dist = dist;
                best_idx = static_cast<int>(i);
            }
        }

        if (best_idx >= 0 && best_dist <= cfg.GRAPH_CONSISTENCY_GATE) {
            used_infra[static_cast<size_t>(best_idx)] = true;
            ++inliers;
            error_sum += best_dist;
        }
    }

    cand.scene_inlier_count = inliers;
    cand.scene_inlier_ratio = total > 0 ? static_cast<double>(inliers) / static_cast<double>(total) : 1.0;
    cand.scene_mean_error = inliers > 0 ? error_sum / static_cast<double>(inliers) : cfg.SCENE_MATCH_CANDIDATE_DIST;
}

void applySinkhornConfidence(
    std::vector<CandidateMatch>& candidates,
    size_t vehicle_count,
    size_t infrastructure_count) {
    if (candidates.empty() || vehicle_count == 0 || infrastructure_count == 0) return;

    std::vector<std::vector<double>> affinity(vehicle_count, std::vector<double>(infrastructure_count, 0.0));
    for (const auto& cand : candidates) {
        affinity[cand.veh_idx][cand.inf_idx] = std::max(affinity[cand.veh_idx][cand.inf_idx], std::exp(4.0 * cand.score));
    }

    for (int iter = 0; iter < 10; ++iter) {
        for (auto& row : affinity) {
            double sum = 0.0;
            for (double v : row) sum += v;
            if (sum > 1e-9) {
                for (double& v : row) v /= sum;
            }
        }
        for (size_t col = 0; col < infrastructure_count; ++col) {
            double sum = 0.0;
            for (size_t row = 0; row < vehicle_count; ++row) sum += affinity[row][col];
            if (sum > 1e-9) {
                for (size_t row = 0; row < vehicle_count; ++row) affinity[row][col] /= sum;
            }
        }
    }

    for (auto& cand : candidates) {
        cand.sinkhorn_score = affinity[cand.veh_idx][cand.inf_idx];
        cand.score = std::clamp(0.75 * cand.score + 0.25 * cand.sinkhorn_score, 0.0, 1.0);
    }
}

bool allowV2XUnhintedCandidate(
    const CandidateMatch& cand,
    const Config& cfg,
    bool has_pair_hints) {
    if (!cfg.V2X_REAL_MODE || !has_pair_hints || cand.hinted) return true;

    const double min_score = std::max(0.18, 2.0 * cfg.SCENE_MATCH_MIN_SCORE);
    const int min_inliers = std::max(3, cfg.GRAPH_MIN_CANDIDATES);
    return cand.score >= min_score &&
           cand.center_distance <= cfg.MATCH_DIST_THRESHOLD &&
           cand.scene_inlier_count >= min_inliers &&
           cand.scene_mean_error <= cfg.GRAPH_CONSISTENCY_GATE;
}

SceneMatchResult buildSceneConsistentMatches(
    const json& veh_anns,
    const json& inf_anns,
    const json& pair_hints,
    const Matrix4f& vehicle_world_transform,
    const Config& cfg) {
    SceneMatchResult result;
    result.vehicle_boxes = readAnnotationBoxFeatures(veh_anns, vehicle_world_transform);
    result.infrastructure_boxes = readAnnotationBoxFeatures(inf_anns, Matrix4f::Identity());
    const auto hints = buildPairHintScores(pair_hints);

    struct PreliminaryCandidate {
        CandidateMatch cand;
        double score = 0.0;
    };

    std::vector<PreliminaryCandidate> preliminary;
    for (size_t v_idx = 0; v_idx < result.vehicle_boxes.size(); ++v_idx) {
        const auto& veh = result.vehicle_boxes[v_idx];
        for (size_t i_idx = 0; i_idx < result.infrastructure_boxes.size(); ++i_idx) {
            const auto& inf = result.infrastructure_boxes[i_idx];
            CandidateMatch cand;
            cand.veh_idx = v_idx;
            cand.inf_idx = i_idx;
            cand.center_distance = (veh.center - inf.center).norm();
            cand.yaw_diff = boxYawDistance(veh.yaw, inf.yaw);
            cand.size_ratio_error = maxSizeRatioError(veh.box_world, inf.box_world);
            auto hint = hints.find(pairKey(veh.track_id, inf.track_id));
            cand.hinted = hint != hints.end() || (!veh.track_id.empty() && veh.track_id == inf.track_id);
            cand.hinted_score = hint == hints.end() ? (cand.hinted ? 1.0 : 0.0) : hint->second;

            if (!cand.hinted &&
                (cand.center_distance > cfg.SCENE_MATCH_CANDIDATE_DIST ||
                 cand.yaw_diff > cfg.YAW_REJECT_GATE ||
                 cand.size_ratio_error > cfg.SIZE_RATIO_GATE)) {
                continue;
            }

            const double dist_score = std::exp(-cand.center_distance / std::max<double>(cfg.MATCH_DIST_THRESHOLD, 1.0));
            const double yaw_score = std::exp(-cand.yaw_diff / std::max(cfg.YAW_REJECT_GATE, 0.1));
            const double size_score = std::exp(-cand.size_ratio_error / std::max(cfg.SIZE_RATIO_GATE, 0.1));
            const double hint_score = cand.hinted ? std::max(0.45, cand.hinted_score) : 0.0;
            const double cheap_score = std::clamp(
                (0.55 * dist_score + 0.25 * size_score + 0.20 * hint_score) * yaw_score,
                0.0,
                1.0);
            preliminary.push_back({cand, cheap_score});
        }
    }

    std::sort(preliminary.begin(), preliminary.end(), [](const PreliminaryCandidate& a, const PreliminaryCandidate& b) {
        if (a.cand.hinted != b.cand.hinted) return a.cand.hinted;
        return a.score > b.score;
    });
    if (cfg.MAX_SCENE_CANDIDATES > 0 &&
        preliminary.size() > static_cast<size_t>(cfg.MAX_SCENE_CANDIDATES)) {
        size_t keep = static_cast<size_t>(cfg.MAX_SCENE_CANDIDATES);
        while (keep < preliminary.size() && preliminary[keep].cand.hinted) ++keep;
        preliminary.resize(keep);
    }

    std::vector<CandidateMatch> candidates;
    for (auto& prelim : preliminary) {
        CandidateMatch cand = prelim.cand;
        fillSceneConsistencyStats(cand, result.vehicle_boxes, result.infrastructure_boxes, cfg);
        if (!cand.hinted &&
            cand.scene_inlier_count < 2 &&
            cand.center_distance > cfg.MATCH_DIST_THRESHOLD) {
            continue;
        }
        const double dist_score = std::exp(-cand.center_distance / std::max<double>(cfg.MATCH_DIST_THRESHOLD, 1.0));
        const double yaw_score = std::exp(-cand.yaw_diff / std::max(cfg.YAW_REJECT_GATE, 0.1));
        const double size_score = std::exp(-cand.size_ratio_error / std::max(cfg.SIZE_RATIO_GATE, 0.1));
        const double scene_score = cand.scene_inlier_ratio / (1.0 + cand.scene_mean_error);
        cand.score = std::clamp(
            (0.55 * scene_score + 0.25 * dist_score + 0.20 * cand.hinted_score) * yaw_score * size_score,
            0.0,
            1.0);
        if (cand.hinted) cand.score = std::max(cand.score, 0.45 * cand.hinted_score);
        const bool low_conf_far_hint = cand.hinted &&
            cand.hinted_score < cfg.SCENE_MATCH_MIN_SCORE &&
            cand.center_distance > cfg.MATCH_DIST_THRESHOLD;
        if (low_conf_far_hint) continue;
        const bool trusted_hint = cand.hinted &&
            (cand.hinted_score >= cfg.SCENE_MATCH_MIN_SCORE ||
             cand.center_distance <= cfg.MATCH_DIST_THRESHOLD);
        if (!allowV2XUnhintedCandidate(cand, cfg, !hints.empty())) continue;
        if (cand.score >= cfg.SCENE_MATCH_MIN_SCORE || trusted_hint) candidates.push_back(cand);
    }

    applySinkhornConfidence(candidates, result.vehicle_boxes.size(), result.infrastructure_boxes.size());
    std::sort(candidates.begin(), candidates.end(), [](const CandidateMatch& a, const CandidateMatch& b) {
        return a.score > b.score;
    });

    std::vector<bool> used_vehicle(result.vehicle_boxes.size(), false);
    std::vector<bool> used_infra(result.infrastructure_boxes.size(), false);
    for (const auto& cand : candidates) {
        if (used_vehicle[cand.veh_idx] || used_infra[cand.inf_idx]) continue;
        const bool low_conf_far_hint = cand.hinted &&
            cand.hinted_score < cfg.SCENE_MATCH_MIN_SCORE &&
            cand.center_distance > cfg.MATCH_DIST_THRESHOLD;
        if (low_conf_far_hint) continue;
        const bool trusted_hint = cand.hinted &&
            (cand.hinted_score >= cfg.SCENE_MATCH_MIN_SCORE ||
             cand.center_distance <= cfg.MATCH_DIST_THRESHOLD);
        if (!allowV2XUnhintedCandidate(cand, cfg, !hints.empty())) continue;
        if (!trusted_hint && cand.score < cfg.SCENE_MATCH_MIN_SCORE) continue;
        used_vehicle[cand.veh_idx] = true;
        used_infra[cand.inf_idx] = true;
        result.matches.push_back(cand);
    }

    return result;
}

fs::path findSceneJson(const fs::path& scene_path, const Config& cfg) {
    const std::vector<std::string> candidates = {
        cfg.JSON_FILENAME,
        "fused_inference_results.json",
        "annotations_world.json",
        "modified_annotations_world.json"
    };
    for (const auto& name : candidates) {
        fs::path p = scene_path / name;
        if (fs::exists(p)) return p;
    }
    return {};
}

fs::path findPcdPath(const fs::path& scene_path, const json& anno, bool vehicle) {
    const char* key = vehicle ? "vehicle_pcd" : "infrastructure_pcd";
    if (anno.contains("metadata") && anno["metadata"].contains(key)) {
        fs::path p = scene_path / anno["metadata"][key].get<std::string>();
        if (fs::exists(p)) return p;
    }

    std::string sn = scene_path.filename().string();
    size_t dash = sn.find('-');
    if (dash != std::string::npos) {
        fs::path p = scene_path / ((vehicle ? sn.substr(0, dash) : sn.substr(dash + 1)) + "_world.pcd");
        if (fs::exists(p)) return p;
    }

    const std::vector<std::string> fallbacks = vehicle
        ? std::vector<std::string>{"vehicle_world.pcd", "veh_world.pcd"}
        : std::vector<std::string>{"infrastructure_world.pcd", "inf_world.pcd"};
    for (const auto& name : fallbacks) {
        fs::path p = scene_path / name;
        if (fs::exists(p)) return p;
    }
    return {};
}

std::vector<Vector3f> loadPcdXYZ(const fs::path& path) {
    std::vector<Vector3f> points;
    if (path.empty() || !fs::exists(path)) return points;

    std::ifstream file(path, std::ios::binary);
    if (!file) return points;

    std::string line;
    size_t point_count = 0;
    std::string data_mode;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        iss >> key;
        if (key == "POINTS") {
            iss >> point_count;
        } else if (key == "DATA") {
            iss >> data_mode;
            break;
        }
    }

    if (point_count == 0) return points;
    points.reserve(point_count);

    if (data_mode == "ascii") {
        float x, y, z;
        while (file >> x >> y >> z) points.emplace_back(x, y, z);
    } else if (data_mode == "binary") {
        for (size_t i = 0; i < point_count; ++i) {
            float xyz[3];
            file.read(reinterpret_cast<char*>(xyz), sizeof(xyz));
            if (!file) break;
            points.emplace_back(xyz[0], xyz[1], xyz[2]);
        }
    }
    return points;
}

std::int64_t planarGridKey(double x, double y, double grid_size) {
    const int ix = static_cast<int>(std::floor(x / grid_size));
    const int iy = static_cast<int>(std::floor(y / grid_size));
    return (static_cast<std::int64_t>(ix) << 32) ^ static_cast<unsigned int>(iy);
}

bool pointInsideExpandedBox2d(const Eigen::Vector2d& p, const std::vector<Vector3f>& box, double margin) {
    if (box.size() != 8) return false;
    const Vector3f center = boxCenter(box);
    const Eigen::Vector3d size = planarBoxSize(box);
    const double half_x = 0.5 * size.x() + margin;
    const double half_y = 0.5 * size.y() + margin;
    const double yaw = boxYaw(box);
    const double c = std::cos(-yaw);
    const double s = std::sin(-yaw);
    const double dx = p.x() - center.x();
    const double dy = p.y() - center.y();
    const double local_x = c * dx - s * dy;
    const double local_y = s * dx + c * dy;
    return std::abs(local_x) <= half_x && std::abs(local_y) <= half_y;
}

std::vector<Eigen::Vector2d> filterStaticPlanarPoints(
    const std::vector<Vector3f>& raw_points,
    const Matrix4f& transform,
    const std::vector<AnnotationBoxFeature>& dynamic_boxes,
    const Config& cfg) {
    struct Accum {
        Eigen::Vector2d sum = Eigen::Vector2d::Zero();
        int count = 0;
    };
    std::unordered_map<std::int64_t, Accum> grid;
    const double grid_size = std::max(0.1, cfg.STATIC_ICP_GRID_SIZE);

    for (const auto& raw : raw_points) {
        Vector4f hp = transform * Vector4f(raw.x(), raw.y(), raw.z(), 1.0f);
        if (!std::isfinite(hp.x()) || !std::isfinite(hp.y()) || !std::isfinite(hp.z())) continue;

        Eigen::Vector2d p(hp.x(), hp.y());
        bool dynamic = false;
        for (const auto& box : dynamic_boxes) {
            if (pointInsideExpandedBox2d(p, box.box_world, cfg.STATIC_ICP_BOX_MARGIN)) {
                dynamic = true;
                break;
            }
        }
        if (dynamic) continue;

        auto& cell = grid[planarGridKey(p.x(), p.y(), grid_size)];
        cell.sum += p;
        ++cell.count;
    }

    std::vector<Eigen::Vector2d> points;
    points.reserve(grid.size());
    for (const auto& [key, cell] : grid) {
        if (cell.count > 0) points.push_back(cell.sum / static_cast<double>(cell.count));
    }
    std::sort(points.begin(), points.end(), [](const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
        return a.x() == b.x() ? a.y() < b.y() : a.x() < b.x();
    });

    if (cfg.STATIC_ICP_MAX_POINTS > 0 &&
        points.size() > static_cast<size_t>(cfg.STATIC_ICP_MAX_POINTS)) {
        std::vector<Eigen::Vector2d> reduced;
        reduced.reserve(static_cast<size_t>(cfg.STATIC_ICP_MAX_POINTS));
        const size_t step = static_cast<size_t>(
            std::ceil(points.size() / static_cast<double>(cfg.STATIC_ICP_MAX_POINTS)));
        for (size_t i = 0; i < points.size() && reduced.size() < static_cast<size_t>(cfg.STATIC_ICP_MAX_POINTS); i += step) {
            reduced.push_back(points[i]);
        }
        points.swap(reduced);
    }
    return points;
}

struct IcpEstimate {
    bool valid = false;
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    Eigen::Vector2d t = Eigen::Vector2d::Zero();
    int correspondences = 0;
    double rmse = std::numeric_limits<double>::quiet_NaN();
};

using PlanarGridIndex = std::unordered_map<std::int64_t, std::vector<int>>;

PlanarGridIndex buildPlanarGridIndex(const std::vector<Eigen::Vector2d>& points, double grid_size) {
    PlanarGridIndex index;
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        index[planarGridKey(points[i].x(), points[i].y(), grid_size)].push_back(i);
    }
    return index;
}

bool findNearestGridPoint(
    const std::vector<Eigen::Vector2d>& target,
    const PlanarGridIndex& index,
    const Eigen::Vector2d& query,
    double grid_size,
    double max_dist,
    Eigen::Vector2d& nearest) {
    const int qx = static_cast<int>(std::floor(query.x() / grid_size));
    const int qy = static_cast<int>(std::floor(query.y() / grid_size));
    const int radius = std::max(1, static_cast<int>(std::ceil(max_dist / grid_size)));
    double best_sq = max_dist * max_dist;
    bool found = false;

    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            const std::int64_t key =
                (static_cast<std::int64_t>(qx + dx) << 32) ^ static_cast<unsigned int>(qy + dy);
            auto it = index.find(key);
            if (it == index.end()) continue;
            for (int idx : it->second) {
                const double dist_sq = (target[static_cast<size_t>(idx)] - query).squaredNorm();
                if (dist_sq < best_sq) {
                    best_sq = dist_sq;
                    nearest = target[static_cast<size_t>(idx)];
                    found = true;
                }
            }
        }
    }
    return found;
}

bool estimateRigid2D(
    const std::vector<Eigen::Vector2d>& source,
    const std::vector<Eigen::Vector2d>& target,
    Eigen::Matrix2d& R,
    Eigen::Vector2d& t) {
    if (source.size() != target.size() || source.size() < 3) return false;

    Eigen::Vector2d mean_source = Eigen::Vector2d::Zero();
    Eigen::Vector2d mean_target = Eigen::Vector2d::Zero();
    for (size_t i = 0; i < source.size(); ++i) {
        mean_source += source[i];
        mean_target += target[i];
    }
    mean_source /= static_cast<double>(source.size());
    mean_target /= static_cast<double>(target.size());

    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    for (size_t i = 0; i < source.size(); ++i) {
        H += (source[i] - mean_source) * (target[i] - mean_target).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d V = svd.matrixV();
    R = V * svd.matrixU().transpose();
    if (R.determinant() < 0.0) {
        V.col(1) *= -1.0;
        R = V * svd.matrixU().transpose();
    }
    t = mean_target - R * mean_source;
    return true;
}

IcpEstimate runStaticPlanarIcp(
    const std::vector<Eigen::Vector2d>& source,
    const std::vector<Eigen::Vector2d>& target,
    const Config& cfg) {
    IcpEstimate estimate;
    if (source.size() < static_cast<size_t>(cfg.STATIC_ICP_MIN_CORRESPONDENCES) ||
        target.size() < static_cast<size_t>(cfg.STATIC_ICP_MIN_CORRESPONDENCES)) {
        return estimate;
    }

    const double grid_size = std::max(0.1, cfg.STATIC_ICP_GRID_SIZE);
    PlanarGridIndex target_index = buildPlanarGridIndex(target, grid_size);

    Eigen::Matrix2d current_R = Eigen::Matrix2d::Identity();
    Eigen::Vector2d current_t = Eigen::Vector2d::Zero();
    
    std::vector<Eigen::Vector2d> current_source = source;
    
    int best_correspondences = 0;
    double best_rmse = std::numeric_limits<double>::quiet_NaN();

    for (int iter = 0; iter < cfg.STATIC_ICP_ITERATIONS; ++iter) {
        // Tighten the correspondence range over ICP iterations.
        double max_corr = std::max(0.2, cfg.STATIC_ICP_MAX_CORRESPONDENCE * (1.0 - 0.5 * iter / cfg.STATIC_ICP_ITERATIONS));
        
        struct Pair {
            Eigen::Vector2d s;
            Eigen::Vector2d t;
            double dist_sq;
        };
        std::vector<Pair> pairs;
        pairs.reserve(current_source.size());
        
        for (const auto& p : current_source) {
            Eigen::Vector2d nearest;
            if (findNearestGridPoint(target, target_index, p, grid_size, max_corr, nearest)) {
                pairs.push_back({p, nearest, (nearest - p).squaredNorm()});
            }
        }

        if (pairs.size() < static_cast<size_t>(cfg.STATIC_ICP_MIN_CORRESPONDENCES)) {
            break;
        }
        
        std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) {
            return a.dist_sq < b.dist_sq;
        });

        const size_t min_keep = static_cast<size_t>(cfg.STATIC_ICP_MIN_CORRESPONDENCES);
        const size_t trimmed_keep = static_cast<size_t>(std::ceil(0.8 * static_cast<double>(pairs.size())));
        const size_t keep_count = std::max(min_keep, std::min(trimmed_keep, pairs.size()));

        std::vector<Eigen::Vector2d> src_matched;
        std::vector<Eigen::Vector2d> tgt_matched;
        src_matched.reserve(keep_count);
        tgt_matched.reserve(keep_count);
        
        for (size_t i = 0; i < keep_count; ++i) {
            src_matched.push_back(pairs[i].s);
            tgt_matched.push_back(pairs[i].t);
        }
        
        Eigen::Matrix2d dR;
        Eigen::Vector2d dt;
        if (!estimateRigid2D(src_matched, tgt_matched, dR, dt)) {
            break;
        }

        double error_sq = 0.0;
        for (size_t i = 0; i < keep_count; ++i) {
            const Eigen::Vector2d fitted = dR * src_matched[i] + dt;
            error_sq += (tgt_matched[i] - fitted).squaredNorm();
        }
        double current_rmse = std::sqrt(error_sq / static_cast<double>(keep_count));
        if (!std::isfinite(current_rmse)) {
            break;
        }
        
        // Update the accumulated transform.
        current_R = dR * current_R;
        current_t = dR * current_t + dt;
        
        // Update source points for the next ICP iteration.
        for (auto& p : current_source) {
            p = dR * p + dt;
        }
        
        best_correspondences = static_cast<int>(keep_count);
        best_rmse = current_rmse;
        
        // Convergence check.
        if (dt.norm() < 1e-3 && std::abs(std::acos(std::clamp(0.5 * (dR.trace()), -1.0, 1.0))) < 1e-3) {
            break;
        }
    }

    const double trans = current_t.norm();
    const double dyaw = std::atan2(current_R(1, 0), current_R(0, 0));
    
    estimate.correspondences = best_correspondences;
    estimate.rmse = best_rmse;
    
    if (best_correspondences >= cfg.STATIC_ICP_MIN_CORRESPONDENCES &&
        std::isfinite(best_rmse) &&
        best_rmse <= cfg.STATIC_ICP_MAX_RMSE &&
        trans <= cfg.STATIC_ICP_MAX_TRANS &&
        std::abs(dyaw) <= cfg.STATIC_ICP_MAX_YAW) {
        estimate.valid = true;
        estimate.R = current_R;
        estimate.t = current_t;
        estimate.rmse = best_rmse;
    }
    return estimate;
}

StaticIcpObservation buildStaticIcpObservation(
    const fs::path& scene_path_inf,
    const fs::path& scene_path_veh,
    const json& anno_inf,
    const json& anno_veh,
    const std::vector<AnnotationBoxFeature>& vehicle_boxes,
    const std::vector<AnnotationBoxFeature>& infrastructure_boxes,
    const Matrix4f& vehicle_world_transform,
    const Matrix4f& base_vehicle_pose,
    const Config& cfg) {
    StaticIcpObservation obs;
    if (!cfg.STATIC_ICP_ENABLED) return obs;

    const fs::path vp = findPcdPath(scene_path_veh, anno_veh, true);
    const fs::path ip = findPcdPath(scene_path_inf, anno_inf, false);
    if (vp.empty() || ip.empty()) return obs;

    const std::vector<Vector3f> veh_raw = loadPcdXYZ(vp);
    const std::vector<Vector3f> inf_raw = loadPcdXYZ(ip);
    if (veh_raw.empty() || inf_raw.empty()) return obs;

    const std::vector<Eigen::Vector2d> veh_static =
        filterStaticPlanarPoints(veh_raw, vehicle_world_transform, vehicle_boxes, cfg);
    const std::vector<Eigen::Vector2d> inf_static =
        filterStaticPlanarPoints(inf_raw, Matrix4f::Identity(), infrastructure_boxes, cfg);

    const IcpEstimate icp = runStaticPlanarIcp(veh_static, inf_static, cfg);
    if (!icp.valid) return obs;

    const double dyaw = wrapAngle(std::atan2(icp.R(1, 0), icp.R(0, 0)));
    const gtsam::Pose2 delta(icp.t.x(), icp.t.y(), dyaw);
    obs.valid = true;
    obs.pose_prior_world = delta.compose(matrixToPose2(base_vehicle_pose));
    obs.correspondences = icp.correspondences;
    obs.rmse = icp.rmse;
    obs.dx = icp.t.x();
    obs.dy = icp.t.y();
    obs.dyaw = dyaw;
    return obs;
}

std::vector<fs::path> getSortedScenes(const std::string& root) {
    std::vector<fs::path> folders;
    if (!fs::exists(root)) return folders;
    for (const auto& e : fs::directory_iterator(root)) if (e.is_directory()) folders.push_back(e.path());
    std::sort(folders.begin(), folders.end(), [](const fs::path& a, const fs::path& b) {
        std::regex re("veh_(\\d+)-inf_(\\d+)"); std::smatch ma, mb;
        std::string na = a.filename().string(), nb = b.filename().string();
        if (std::regex_match(na, ma, re) && std::regex_match(nb, mb, re)) {
            int va = std::stoi(ma[1]), vb = std::stoi(mb[1]);
            return va != vb ? va < vb : std::stoi(ma[2]) < std::stoi(mb[2]);
        }
        return na < nb;
    });
    return folders;
}

std::string trimString(const std::string& value) {
    size_t first = 0;
    while (first < value.size() && std::isspace(static_cast<unsigned char>(value[first]))) ++first;
    size_t last = value.size();
    while (last > first && std::isspace(static_cast<unsigned char>(value[last - 1]))) --last;
    return value.substr(first, last - first);
}

std::string normalizePriorPath(std::string value) {
    std::replace(value.begin(), value.end(), '\\', '/');
    while (!value.empty() && value.back() == '/') value.pop_back();
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string supervisedPriorKey(const std::string& root, const std::string& scene) {
    return normalizePriorPath(root) + "|" + scene;
}

std::vector<std::string> parseCsvLine(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    bool quoted = false;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '"') {
            if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
                cell.push_back('"');
                ++i;
            } else {
                quoted = !quoted;
            }
        } else if (c == ',' && !quoted) {
            cells.push_back(cell);
            cell.clear();
        } else {
            cell.push_back(c);
        }
    }
    cells.push_back(cell);
    return cells;
}

double csvDouble(
    const std::map<std::string, std::string>& row,
    const std::string& key,
    double default_value = 0.0) {
    auto it = row.find(key);
    if (it == row.end() || it->second.empty()) return default_value;
    try {
        const double value = std::stod(it->second);
        return std::isfinite(value) ? value : default_value;
    } catch (...) {
        return default_value;
    }
}

std::map<std::string, SupervisedPriorRecord> loadSupervisedPriorCsv(const std::string& path) {
    std::map<std::string, SupervisedPriorRecord> priors;
    if (path.empty()) return priors;
    std::ifstream file(path);
    if (!file) {
        std::cerr << "[SupervisedPrior] CSV not found, disabled: " << path << std::endl;
        return priors;
    }

    std::string line;
    if (!std::getline(file, line)) return priors;
    std::vector<std::string> header = parseCsvLine(line);
    for (auto& name : header) name = trimString(name);

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::vector<std::string> cells = parseCsvLine(line);
        std::map<std::string, std::string> row;
        for (size_t i = 0; i < header.size() && i < cells.size(); ++i) {
            row[header[i]] = trimString(cells[i]);
        }

        const std::string root = row["root"];
        const std::string scene = row["scene"];
        if (root.empty() || scene.empty()) continue;

        SupervisedPriorRecord prior;
        prior.valid = true;
        prior.dx = csvDouble(row, "dx", csvDouble(row, "delta_x"));
        prior.dy = csvDouble(row, "dy", csvDouble(row, "delta_y"));
        prior.dyaw = csvDouble(row, "dyaw", std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(prior.dyaw)) {
            prior.dyaw = csvDouble(row, "yaw_rad", std::numeric_limits<double>::quiet_NaN());
        }
        if (!std::isfinite(prior.dyaw)) {
            prior.dyaw = csvDouble(row, "yaw_deg") * kPi / 180.0;
        }
        prior.sigma_x = std::max(0.01, csvDouble(row, "sigma_x", csvDouble(row, "sigma_trans", 0.5)));
        prior.sigma_y = std::max(0.01, csvDouble(row, "sigma_y", csvDouble(row, "sigma_trans", 0.5)));
        prior.sigma_yaw = std::max(0.005, csvDouble(row, "sigma_yaw", 0.08));
        prior.confidence = std::clamp(csvDouble(row, "confidence", 1.0), 0.0, 1.0);
        priors[supervisedPriorKey(root, scene)] = prior;
    }
    return priors;
}

SupervisedPosePrior lookupSupervisedPrior(
    const Config& cfg,
    const std::string& root,
    const std::string& scene,
    const Matrix4f& initial_pose) {
    SupervisedPosePrior out;
    if (cfg.SUPERVISED_PRIORS.empty()) return out;

    auto it = cfg.SUPERVISED_PRIORS.find(supervisedPriorKey(root, scene));
    if (it == cfg.SUPERVISED_PRIORS.end()) {
        it = cfg.SUPERVISED_PRIORS.find("|" + scene);
    }
    if (it == cfg.SUPERVISED_PRIORS.end() || !it->second.valid) return out;

    const auto& prior = it->second;
    const double sigma_trans = std::hypot(prior.sigma_x, prior.sigma_y);
    if (prior.confidence < cfg.SUPERVISED_PRIOR_MIN_CONF ||
        sigma_trans > cfg.SUPERVISED_PRIOR_MAX_SIGMA_TRANS ||
        prior.sigma_yaw > cfg.SUPERVISED_PRIOR_MAX_SIGMA_YAW) {
        return out;
    }

    const gtsam::Pose2 init = matrixToPose2(initial_pose);
    out.valid = true;
    out.pose_prior_world = gtsam::Pose2(
        init.x() + prior.dx,
        init.y() + prior.dy,
        wrapAngle(init.theta() + prior.dyaw));
    out.sigma_x = prior.sigma_x;
    out.sigma_y = prior.sigma_y;
    out.sigma_yaw = prior.sigma_yaw;
    out.confidence = prior.confidence;
    return out;
}

double observationSelectionScore(const BoxObservation& obs, const Config& cfg) {
    double quality = std::clamp(obs.score, 0.0, 1.0);
    quality *= std::clamp(obs.graph_inlier_ratio, 0.1, 1.0);
    quality *= std::clamp(obs.sinkhorn_score, 0.1, 1.0);
    quality *= std::exp(-obs.center_distance / std::max<double>(cfg.MATCH_DIST_THRESHOLD, 1.0));
    quality *= std::exp(-obs.size_ratio_error / std::max(cfg.SIZE_RATIO_GATE, 0.1));
    if (obs.center_only) quality *= 0.65;

    const Eigen::Vector3d size = planarBoxSize(obs.inf_box_world);
    const double footprint = std::max(0.0, size.x() * size.y());
    return quality * std::log1p(footprint) / (1.0 + 0.25 * std::max(0.0, obs.scene_mean_error));
}

void applyTopObservationFilter(std::vector<BoxObservation>& boxes, const Config& cfg) {
    if (cfg.MAX_FRAME_OBSERVATIONS <= 0 ||
        boxes.size() <= static_cast<size_t>(cfg.MAX_FRAME_OBSERVATIONS)) {
        return;
    }

    std::stable_sort(boxes.begin(), boxes.end(), [&](const BoxObservation& a, const BoxObservation& b) {
        return observationSelectionScore(a, cfg) > observationSelectionScore(b, cfg);
    });
    boxes.resize(static_cast<size_t>(cfg.MAX_FRAME_OBSERVATIONS));
}

bool applyGraphConsistencyFilter(std::vector<BoxObservation>& boxes, const Config& cfg) {
    if (boxes.size() == 2) {
        const double veh_dist = (boxes[0].veh_center_world - boxes[1].veh_center_world).norm();
        const double inf_dist = (boxes[0].inf_center_world - boxes[1].inf_center_world).norm();
        const double sparse_pair_gate = cfg.V2X_REAL_MODE
            ? std::min(cfg.GRAPH_CONSISTENCY_GATE, 1.75)
            : std::min(cfg.GRAPH_CONSISTENCY_GATE, 1.50);
        if (std::abs(veh_dist - inf_dist) > sparse_pair_gate) {
            boxes.clear();
            return true;
        }
    }

    if (boxes.size() < static_cast<size_t>(cfg.GRAPH_MIN_CANDIDATES)) return false;

    std::vector<BoxObservation> kept;
    kept.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        int total = 0;
        int inliers = 0;
        for (size_t j = 0; j < boxes.size(); ++j) {
            if (i == j) continue;
            const double veh_dist = (boxes[i].veh_center_world - boxes[j].veh_center_world).norm();
            const double inf_dist = (boxes[i].inf_center_world - boxes[j].inf_center_world).norm();
            if (veh_dist < 1e-3 && inf_dist < 1e-3) continue;
            ++total;
            if (std::abs(veh_dist - inf_dist) <= cfg.GRAPH_CONSISTENCY_GATE) ++inliers;
        }

        const double ratio = total > 0 ? static_cast<double>(inliers) / static_cast<double>(total) : 1.0;
        boxes[i].graph_inlier_ratio = ratio;
        if (ratio >= cfg.GRAPH_MIN_INLIER_RATIO) kept.push_back(boxes[i]);
    }

    if (kept.size() >= 2) {
        boxes.swap(kept);
    } else if (cfg.V2X_REAL_MODE && boxes.size() <= 2) {
        std::stable_sort(boxes.begin(), boxes.end(), [&](const BoxObservation& a, const BoxObservation& b) {
            return observationSelectionScore(a, cfg) > observationSelectionScore(b, cfg);
        });
        boxes.resize(1);
    }
    return false;
}

struct PlanarCenterFit {
    bool valid = false;
    Eigen::Matrix2d R = Eigen::Matrix2d::Identity();
    Eigen::Vector2d t = Eigen::Vector2d::Zero();
};

PlanarCenterFit fitPlanarCenters(const std::vector<BoxObservation>& boxes) {
    PlanarCenterFit fit;
    if (boxes.size() < 3) return fit;

    struct Pair2D {
        Eigen::Vector2d src;
        Eigen::Vector2d dst;
        double weight = 1.0;
    };

    std::vector<Pair2D> pairs;
    pairs.reserve(boxes.size());
    Eigen::Vector2d src_mean = Eigen::Vector2d::Zero();
    Eigen::Vector2d dst_mean = Eigen::Vector2d::Zero();
    double weight_sum = 0.0;
    for (const auto& obs : boxes) {
        if (!obs.veh_center_world.allFinite() || !obs.inf_center_world.allFinite()) continue;
        const double weight = std::max(0.03, std::clamp(obs.score, 0.0, 1.0));
        pairs.push_back({
            Eigen::Vector2d(obs.veh_center_world.x(), obs.veh_center_world.y()),
            Eigen::Vector2d(obs.inf_center_world.x(), obs.inf_center_world.y()),
            weight,
        });
        src_mean += weight * pairs.back().src;
        dst_mean += weight * pairs.back().dst;
        weight_sum += weight;
    }
    if (pairs.size() < 3 || weight_sum <= 1e-9) return fit;

    src_mean /= weight_sum;
    dst_mean /= weight_sum;
    Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
    for (const auto& pair : pairs) {
        H += pair.weight * (pair.src - src_mean) * (pair.dst - dst_mean).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix2d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix2d R = svd.matrixV() * svd.matrixU().transpose();
    if (R.determinant() < 0.0) {
        Eigen::Matrix2d V = svd.matrixV();
        V.col(1) *= -1.0;
        R = V * svd.matrixU().transpose();
    }

    fit.valid = true;
    fit.R = R;
    fit.t = dst_mean - R * src_mean;
    return fit;
}

void applyPlanarResidualOutlierFilter(std::vector<BoxObservation>& boxes, const Config& cfg) {
    if (boxes.size() < 3) return;

    const PlanarCenterFit fit = fitPlanarCenters(boxes);
    if (!fit.valid) return;

    std::vector<double> residuals;
    residuals.reserve(boxes.size());
    for (const auto& obs : boxes) {
        const Eigen::Vector2d src(obs.veh_center_world.x(), obs.veh_center_world.y());
        const Eigen::Vector2d dst(obs.inf_center_world.x(), obs.inf_center_world.y());
        residuals.push_back((fit.R * src + fit.t - dst).norm());
    }
    if (residuals.size() < 3) return;

    std::vector<double> sorted = residuals;
    std::sort(sorted.begin(), sorted.end());
    const double median = sorted[sorted.size() / 2];
    std::vector<double> deviations;
    deviations.reserve(sorted.size());
    for (double residual : residuals) deviations.push_back(std::abs(residual - median));
    std::sort(deviations.begin(), deviations.end());
    const double mad = deviations[deviations.size() / 2];
    const double adaptive_gate = median + std::max(0.35, 2.5 * mad);
    const double residual_gate = std::clamp(adaptive_gate, 0.75, 1.35);

    std::vector<BoxObservation> kept;
    kept.reserve(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (residuals[i] <= residual_gate) kept.push_back(boxes[i]);
    }

    if (kept.size() >= 2 && kept.size() < boxes.size()) {
        boxes.swap(kept);
    }
}

void applyHardCaseFilter(std::vector<BoxObservation>& boxes, const Config& cfg) {
    if (boxes.size() < static_cast<size_t>(cfg.MIN_BOX_OBSERVATIONS)) {
        boxes.clear();
        return;
    }

    int yaw_reliable = 0;
    for (const auto& obs : boxes) {
        if (!obs.center_only) ++yaw_reliable;
    }

    if (yaw_reliable == 0 && boxes.size() < static_cast<size_t>(cfg.MIN_CENTER_ONLY_OBSERVATIONS)) {
        boxes.clear();
    }
}

bool buildFrameObservation(
    const fs::path& scene_path_inf,
    const fs::path& scene_path_veh,
    const Config& cfg,
    FrameObservation& frame_obs,
    Matrix4f& initial_vehicle_pose,
    size_t& raw_pair_count,
    const Matrix4f* matching_vehicle_pose = nullptr) {
    fs::path jp_inf = findSceneJson(scene_path_inf, cfg);
    fs::path jp_veh = findSceneJson(scene_path_veh, cfg);
    if (jp_inf.empty() || jp_veh.empty()) return false;

    std::ifstream f_i(jp_inf);
    std::ifstream f_v(jp_veh);
    if (!f_i || !f_v) return false;

    json anno_inf = json::parse(f_i);
    json anno_veh = json::parse(f_v);
    if (!anno_inf.contains("transformation_matrices") ||
        !anno_veh.contains("transformation_matrices")) return false;

    Matrix4f T_init_current = jsonToMatrix4f(anno_inf["transformation_matrices"]["vehicle_lidar_to_world"]);
    Matrix4f T_init_delayed = jsonToMatrix4f(anno_veh["transformation_matrices"]["vehicle_lidar_to_world"]);
    Matrix4f T_match_current = matching_vehicle_pose ? *matching_vehicle_pose : T_init_current;
    Matrix4f T_motion_compensation = T_match_current * invertTransformMatrix(T_init_delayed);

    frame_obs = FrameObservation();
    frame_obs.frame_name = scene_path_inf.filename().string();
    frame_obs.initial_vehicle_pose = T_init_current;
    frame_obs.supervised_prior = lookupSupervisedPrior(
        cfg, cfg.PC_DATA_ROOT, frame_obs.frame_name, T_init_current);
    initial_vehicle_pose = T_init_current;
    const json empty_array = json::array();
    const json& pairs = anno_inf.contains("cooperative_pairs") ? anno_inf["cooperative_pairs"] : empty_array;
    const json& veh_anns = anno_veh.contains("vehicle_annotations_world") ? anno_veh["vehicle_annotations_world"] : empty_array;
    const json& inf_anns = anno_inf.contains("infrastructure_annotations_world") ? anno_inf["infrastructure_annotations_world"] : empty_array;
    raw_pair_count = pairs.size();

    SceneMatchResult scene_matches = buildSceneConsistentMatches(
        veh_anns, inf_anns, pairs, T_motion_compensation, cfg);
    if (raw_pair_count == 0) raw_pair_count = scene_matches.matches.size();

    for (const auto& match : scene_matches.matches) {
        const auto& v_feature = scene_matches.vehicle_boxes[match.veh_idx];
        const auto& i_feature = scene_matches.infrastructure_boxes[match.inf_idx];
        BoxObservation obs;
        std::vector<int> perm = bestCornerPermutation(v_feature.box_world, i_feature.box_world);
        obs.veh_box_local = reorderBox(transformBox(invertTransformMatrix(T_match_current), v_feature.box_world), perm);
        obs.inf_box_world = i_feature.box_world;
        obs.veh_track_id = v_feature.track_id;
        obs.inf_track_id = i_feature.track_id;
        obs.landmark_id = frame_obs.frame_name + ":" + (obs.inf_track_id.empty() ? obs.veh_track_id : obs.inf_track_id);
        obs.landmark_pose_world = boxWorldPose2(i_feature.box_world);
        obs.landmark_corner_offsets = boxOffsetsInPose2(i_feature.box_world, obs.landmark_pose_world);
        obs.veh_center_world = v_feature.center;
        obs.inf_center_world = i_feature.center;
        obs.center_distance = match.center_distance;
        obs.yaw_diff = match.yaw_diff;
        obs.size_ratio_error = match.size_ratio_error;
        obs.graph_inlier_ratio = match.scene_inlier_ratio;
        obs.scene_mean_error = match.scene_mean_error;
        obs.sinkhorn_score = match.sinkhorn_score;
        obs.scene_inlier_count = match.scene_inlier_count;
        obs.center_only = match.yaw_diff > cfg.YAW_ADAPTIVE_GATE;
        obs.initial_agent_yaw = matrixToPose2(T_init_current).theta();
        obs.score = match.score;
        frame_obs.boxes.push_back(obs);
    }
    const bool rejected_sparse_pair = applyGraphConsistencyFilter(frame_obs.boxes, cfg);
    applyPlanarResidualOutlierFilter(frame_obs.boxes, cfg);
    applyHardCaseFilter(frame_obs.boxes, cfg);
    for (auto& obs : frame_obs.boxes) {
        obs.frame_match_count = static_cast<int>(frame_obs.boxes.size());
    }
    applyTopObservationFilter(frame_obs.boxes, cfg);
    for (auto& obs : frame_obs.boxes) {
        obs.frame_match_count = static_cast<int>(frame_obs.boxes.size());
    }

    if (!rejected_sparse_pair &&
        cfg.STATIC_ICP_ENABLED &&
        matching_vehicle_pose == nullptr &&
        frame_obs.boxes.size() <= static_cast<size_t>(cfg.STATIC_ICP_MAX_BOXES)) {
        frame_obs.static_icp = buildStaticIcpObservation(
            scene_path_inf,
            scene_path_veh,
            anno_inf,
            anno_veh,
            scene_matches.vehicle_boxes,
            scene_matches.infrastructure_boxes,
            T_motion_compensation,
            T_match_current,
            cfg);
    }
    return true;
}

double poseDeltaTrans(const Matrix4f& a, const Matrix4f& b) {
    const Vector3f da = a.block<3,1>(0,3) - b.block<3,1>(0,3);
    return std::hypot(da.x(), da.y());
}

double poseDeltaYaw(const Matrix4f& a, const Matrix4f& b) {
    return std::abs(wrapAngle(matrixToPose2(a).theta() - matrixToPose2(b).theta()));
}

std::string csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string escaped = "\"";
    for (char c : value) {
        if (c == '"') escaped += "\"\"";
        else escaped += c;
    }
    escaped += "\"";
    return escaped;
}

void writeMetricValue(std::ostream& os, double value) {
    if (std::isnan(value)) os << "nan";
    else os << std::fixed << std::setprecision(6) << value;
}

void writeMetricsHeader(std::ostream& os) {
    os << "dataset,root,frame_index,scene,raw_pairs,accepted,center_only,landmarks,rematch,quality,"
       << "delta_x,delta_y,delta_z,delta_trans,yaw_deg,graph_error,mean_error,"
       << "init_center_error,opt_center_error,center_error_delta,"
       << "init_planar_center_error,opt_planar_center_error,planar_center_error_delta,"
       << "init_yaw_error_deg,opt_yaw_error_deg,yaw_error_delta,"
       << "icp_used,icp_pairs,icp_rmse,icp_dx,icp_dy,icp_yaw_deg,"
       << "supervised_used,supervised_confidence,supervised_sigma_trans,supervised_sigma_yaw,"
       << "rejected,reject_reason\n";
}

struct AlignmentMetrics {
    double init_center_error = std::numeric_limits<double>::quiet_NaN();
    double opt_center_error = std::numeric_limits<double>::quiet_NaN();
    double init_planar_center_error = std::numeric_limits<double>::quiet_NaN();
    double opt_planar_center_error = std::numeric_limits<double>::quiet_NaN();
    double init_yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
    double opt_yaw_error_deg = std::numeric_limits<double>::quiet_NaN();
};

std::vector<Vector3f> transformLocalVehicleBox(const Matrix4f& vehicle_pose, const BoxObservation& obs) {
    return transformBox(vehicle_pose, obs.veh_box_local);
}

AlignmentMetrics computeAlignmentMetrics(
    const FrameObservation& frame_obs,
    const Matrix4f& initial_pose,
    const Matrix4f& optimized_pose) {
    AlignmentMetrics metrics;
    if (frame_obs.boxes.empty()) return metrics;

    double init_center_sum = 0.0;
    double opt_center_sum = 0.0;
    double init_planar_center_sum = 0.0;
    double opt_planar_center_sum = 0.0;
    double init_yaw_sum = 0.0;
    double opt_yaw_sum = 0.0;
    int center_count = 0;
    int yaw_count = 0;
    for (const auto& obs : frame_obs.boxes) {
        const auto init_box = transformLocalVehicleBox(initial_pose, obs);
        const auto opt_box = transformLocalVehicleBox(optimized_pose, obs);
        if (init_box.size() != 8 || opt_box.size() != 8 || obs.inf_box_world.size() != 8) continue;

        const Vector3f init_center = boxCenter(init_box);
        const Vector3f opt_center = boxCenter(opt_box);
        const Vector3f inf_center = boxCenter(obs.inf_box_world);
        init_center_sum += (init_center - inf_center).norm();
        opt_center_sum += (opt_center - inf_center).norm();
        init_planar_center_sum += std::hypot(init_center.x() - inf_center.x(), init_center.y() - inf_center.y());
        opt_planar_center_sum += std::hypot(opt_center.x() - inf_center.x(), opt_center.y() - inf_center.y());
        ++center_count;

        if (!obs.center_only) {
            init_yaw_sum += boxYawDistance(boxYaw(init_box), boxYaw(obs.inf_box_world)) * 180.0 / kPi;
            opt_yaw_sum += boxYawDistance(boxYaw(opt_box), boxYaw(obs.inf_box_world)) * 180.0 / kPi;
            ++yaw_count;
        }
    }

    if (center_count > 0) {
        metrics.init_center_error = init_center_sum / static_cast<double>(center_count);
        metrics.opt_center_error = opt_center_sum / static_cast<double>(center_count);
        metrics.init_planar_center_error = init_planar_center_sum / static_cast<double>(center_count);
        metrics.opt_planar_center_error = opt_planar_center_sum / static_cast<double>(center_count);
    }
    if (yaw_count > 0) {
        metrics.init_yaw_error_deg = init_yaw_sum / static_cast<double>(yaw_count);
        metrics.opt_yaw_error_deg = opt_yaw_sum / static_cast<double>(yaw_count);
    }
    return metrics;
}

void writeMetricsRow(
    std::ostream& os,
    const std::string& dataset,
    const std::string& root,
    int frame_idx,
    const fs::path& scene_path,
    size_t raw_pair_count,
    const FrameObservation& frame_obs,
    size_t center_only_count,
    const SlidingWindowGtsamOptimizer& optimizer,
    int rematch_count,
    const Vector3f& delta_t,
    double yaw_deg,
    const AlignmentMetrics& alignment) {
    const double delta_trans = std::hypot(delta_t.x(), delta_t.y());
    os << csvEscape(dataset) << ','
       << csvEscape(root) << ','
       << frame_idx << ','
       << csvEscape(scene_path.filename().string()) << ','
       << raw_pair_count << ','
       << frame_obs.boxes.size() << ','
       << center_only_count << ','
       << optimizer.lastLandmarkCount() << ','
       << rematch_count << ',';
    writeMetricValue(os, optimizer.lastFrameQuality());
    os << ',';
    writeMetricValue(os, delta_t.x());
    os << ',';
    writeMetricValue(os, delta_t.y());
    os << ',';
    writeMetricValue(os, delta_t.z());
    os << ',';
    writeMetricValue(os, delta_trans);
    os << ',';
    writeMetricValue(os, yaw_deg);
    os << ',';
    writeMetricValue(os, optimizer.lastError());
    os << ',';
    writeMetricValue(os, optimizer.lastMeanError());
    os << ',';
    writeMetricValue(os, alignment.init_center_error);
    os << ',';
    writeMetricValue(os, alignment.opt_center_error);
    os << ',';
    writeMetricValue(os, alignment.init_center_error - alignment.opt_center_error);
    os << ',';
    writeMetricValue(os, alignment.init_planar_center_error);
    os << ',';
    writeMetricValue(os, alignment.opt_planar_center_error);
    os << ',';
    writeMetricValue(os, alignment.init_planar_center_error - alignment.opt_planar_center_error);
    os << ',';
    writeMetricValue(os, alignment.init_yaw_error_deg);
    os << ',';
    writeMetricValue(os, alignment.opt_yaw_error_deg);
    os << ',';
    writeMetricValue(os, alignment.init_yaw_error_deg - alignment.opt_yaw_error_deg);
    os << ','
       << (frame_obs.static_icp.valid ? 1 : 0) << ','
       << frame_obs.static_icp.correspondences << ',';
    writeMetricValue(os, frame_obs.static_icp.rmse);
    os << ',';
    writeMetricValue(os, frame_obs.static_icp.dx);
    os << ',';
    writeMetricValue(os, frame_obs.static_icp.dy);
    os << ',';
    writeMetricValue(os, frame_obs.static_icp.dyaw * 180.0 / kPi);
    os << ','
       << (frame_obs.supervised_prior.valid ? 1 : 0) << ',';
    writeMetricValue(os, frame_obs.supervised_prior.confidence);
    os << ',';
    writeMetricValue(
        os,
        std::hypot(frame_obs.supervised_prior.sigma_x, frame_obs.supervised_prior.sigma_y));
    os << ',';
    writeMetricValue(os, frame_obs.supervised_prior.sigma_yaw);
    os << ','
       << (optimizer.lastRejected() ? 1 : 0) << ','
       << csvEscape(optimizer.lastRejectReason())
       << '\n';
}

Matrix4f optimizeLatestWithRematching(
    SlidingWindowGtsamOptimizer& optimizer,
    const fs::path& scene_path_inf,
    const fs::path& scene_path_veh,
    const Config& cfg,
    FrameObservation& frame_obs,
    Matrix4f& initial_vehicle_pose,
    size_t& raw_pair_count,
    int& rematch_count) {
    rematch_count = 0;
    optimizer.addFrame(frame_obs);
    Matrix4f corrected = optimizer.optimizeLatest();

    for (int iter = 1; iter < cfg.REMATCH_ITERATIONS; ++iter) {
        FrameObservation rematched_obs;
        Matrix4f rematched_initial = Matrix4f::Identity();
        size_t rematched_raw_count = 0;
        if (!buildFrameObservation(
                scene_path_inf,
                scene_path_veh,
                cfg,
                rematched_obs,
                rematched_initial,
                rematched_raw_count,
                &corrected)) {
            break;
        }
        if (rematched_obs.boxes.empty()) break;
        if (!rematched_obs.static_icp.valid) rematched_obs.static_icp = frame_obs.static_icp;

        optimizer.replaceLatestFrame(rematched_obs);
        const Matrix4f next_corrected = optimizer.optimizeLatest();
        ++rematch_count;
        frame_obs = std::move(rematched_obs);
        initial_vehicle_pose = rematched_initial;
        raw_pair_count = std::max(raw_pair_count, rematched_raw_count);

        const double trans_delta = poseDeltaTrans(next_corrected, corrected);
        const double yaw_delta = poseDeltaYaw(next_corrected, corrected);
        corrected = next_corrected;
        if (trans_delta < cfg.REMATCH_TRANS_EPS && yaw_delta < cfg.REMATCH_YAW_EPS) break;
    }

    return corrected;
}

std::vector<std::string> readRootListFile(const std::string& path) {
    std::vector<std::string> roots;
    std::ifstream file(path);
    if (!file) return roots;

    std::string line;
    while (std::getline(file, line)) {
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char c) {
            return !std::isspace(c);
        }));
        line.erase(std::find_if(line.rbegin(), line.rend(), [](unsigned char c) {
            return !std::isspace(c);
        }).base(), line.end());
        if (!line.empty() && line[0] != '#') roots.push_back(line);
    }
    return roots;
}

int runHeadlessRoot(
    const Config& base_cfg,
    const std::string& root,
    const std::string& dataset_name,
    int max_frames,
    std::ostream* metrics_out) {
    Config cfg = base_cfg;
    cfg.PC_DATA_ROOT = root;
    auto scenes = getSortedScenes(cfg.PC_DATA_ROOT);
    if (scenes.empty()) {
        std::cerr << "No scenes found under root: " << cfg.PC_DATA_ROOT << std::endl;
        return 1;
    }

    std::cout << "[Startup] root=" << cfg.PC_DATA_ROOT
              << " scenes=" << scenes.size()
              << " delay=" << cfg.VEH_DELAY_FRAMES
              << " window=" << cfg.WINDOW_SIZE
              << " delay_ms=" << cfg.DELAY_MS
              << " match_dist=" << cfg.MATCH_DIST_THRESHOLD
              << " min_boxes=" << cfg.MIN_BOX_OBSERVATIONS
              << " max_candidates=" << cfg.MAX_SCENE_CANDIDATES
              << " max_obs=" << cfg.MAX_FRAME_OBSERVATIONS
              << " rematch_iters=" << cfg.REMATCH_ITERATIONS
              << " max_mean_error=" << cfg.MAX_MEAN_GRAPH_ERROR
              << " icp_static=" << (cfg.STATIC_ICP_ENABLED ? "on" : "off")
              << " supervised_priors=" << cfg.SUPERVISED_PRIORS.size()
              << " dataset=" << dataset_name
              << " mode=headless"
              << std::endl;
    std::cout << "[Startup] first_scene=" << scenes.front().string() << std::endl;

    SlidingWindowGtsamOptimizer optimizer(cfg);
    const int total_frames = max_frames > 0
        ? std::min(max_frames, static_cast<int>(scenes.size()))
        : static_cast<int>(scenes.size());

    for (int frame_idx = 0; frame_idx < total_frames; ++frame_idx) {
        int veh_idx = std::max(0, frame_idx - cfg.VEH_DELAY_FRAMES);
        FrameObservation frame_obs;
        Matrix4f T_init_current = Matrix4f::Identity();
        size_t raw_pair_count = 0;
        if (!buildFrameObservation(scenes[frame_idx], scenes[veh_idx], cfg, frame_obs, T_init_current, raw_pair_count)) {
            std::cout << "[Frame " << std::setw(3) << frame_idx << "] skipped: missing json or transform" << std::endl;
            continue;
        }

        int rematch_count = 0;
        Matrix4f T_corr = optimizeLatestWithRematching(
            optimizer,
            scenes[frame_idx],
            scenes[veh_idx],
            cfg,
            frame_obs,
            T_init_current,
            raw_pair_count,
            rematch_count);
        Vector3f delta_t = T_corr.block<3,1>(0,3) - T_init_current.block<3,1>(0,3);
        float yaw_deg = static_cast<float>(
            wrapAngle(matrixToPose2(T_corr).theta() - matrixToPose2(T_init_current).theta()) *
            180.0 / kPi);
        size_t center_only_count = 0;
        for (const auto& obs : frame_obs.boxes) if (obs.center_only) ++center_only_count;
        const AlignmentMetrics alignment = computeAlignmentMetrics(frame_obs, T_init_current, T_corr);

        std::cout << "[Frame " << std::setw(3) << frame_idx << "] "
                  << "raw_pairs=" << raw_pair_count
                  << " accepted=" << frame_obs.boxes.size()
                  << " landmarks=" << optimizer.lastLandmarkCount()
                  << " center_only=" << center_only_count
                  << " rematch=" << rematch_count
                  << " quality=" << std::fixed << std::setprecision(3) << optimizer.lastFrameQuality()
                  << " delta_t=[" << std::fixed << std::setprecision(3)
                  << delta_t.x() << ", " << delta_t.y() << ", " << delta_t.z() << "]"
                  << " yaw=" << std::fixed << std::setprecision(2) << yaw_deg
                  << " error=" << std::fixed << std::setprecision(3) << optimizer.lastError()
                  << " mean_error=" << std::fixed << std::setprecision(3) << optimizer.lastMeanError()
                  << " icp=" << (frame_obs.static_icp.valid ? "on" : "off")
                  << "(" << frame_obs.static_icp.correspondences << ","
                  << std::fixed << std::setprecision(3) << frame_obs.static_icp.rmse << ")"
                  << " supervised=" << (frame_obs.supervised_prior.valid ? "on" : "off")
                  << (optimizer.lastRejected() ? (" rejected=" + optimizer.lastRejectReason()) : "")
                  << std::endl;

        if (metrics_out) {
            writeMetricsRow(
                *metrics_out,
                dataset_name,
                cfg.PC_DATA_ROOT,
                frame_idx,
                scenes[frame_idx],
                raw_pair_count,
                frame_obs,
                center_only_count,
                optimizer,
                rematch_count,
                delta_t,
                yaw_deg,
                alignment);
        }
    }
    return 0;
}

// ==========================================================================================
// Main loop.
// ==========================================================================================
int main(int argc, char** argv) {
    Config cfg;
    bool headless = false;
    int max_frames = 0;
    bool v2x_real_mode = false;
    std::string root_list_path;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--root" || arg == "-r") && i + 1 < argc) {
            cfg.PC_DATA_ROOT = argv[++i];
        } else if (arg == "--root-list" && i + 1 < argc) {
            root_list_path = argv[++i];
        } else if (arg == "--delay" && i + 1 < argc) {
            cfg.VEH_DELAY_FRAMES = std::stoi(argv[++i]);
        } else if (arg == "--window" && i + 1 < argc) {
            cfg.WINDOW_SIZE = std::stoi(argv[++i]);
        } else if (arg == "--match-dist" && i + 1 < argc) {
            cfg.MATCH_DIST_THRESHOLD = std::stof(argv[++i]);
        } else if (arg == "--delay-ms" && i + 1 < argc) {
            cfg.DELAY_MS = std::stoi(argv[++i]);
        } else if (arg == "--metrics-csv" && i + 1 < argc) {
            cfg.METRICS_CSV = argv[++i];
        } else if (arg == "--supervised-prior-csv" && i + 1 < argc) {
            cfg.SUPERVISED_PRIOR_CSV = argv[++i];
        } else if (arg == "--headless") {
            headless = true;
        } else if (arg == "--v2x-real") {
            v2x_real_mode = true;
            cfg.V2X_REAL_MODE = true;
            cfg.SCENE_MATCH_MIN_SCORE = 0.08;
            cfg.MIN_BOX_OBSERVATIONS = 1;
            cfg.MIN_CENTER_ONLY_OBSERVATIONS = 1;
            cfg.GRAPH_MIN_CANDIDATES = 2;
            cfg.MAX_MEAN_GRAPH_ERROR = 3.0;
            cfg.MAX_CORRECTION_TRANS = 5.0;
            cfg.MAX_CORRECTION_YAW = 0.35;
            cfg.MAX_CORRECTION_STEP_TRANS = 2.0;
            cfg.MAX_CORRECTION_STEP_YAW = 0.15;
        } else if (arg == "--max-frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (arg == "--min-boxes" && i + 1 < argc) {
            cfg.MIN_BOX_OBSERVATIONS = std::stoi(argv[++i]);
        } else if (arg == "--min-center-only" && i + 1 < argc) {
            cfg.MIN_CENTER_ONLY_OBSERVATIONS = std::stoi(argv[++i]);
        } else if (arg == "--graph-min-candidates" && i + 1 < argc) {
            cfg.GRAPH_MIN_CANDIDATES = std::stoi(argv[++i]);
        } else if (arg == "--max-scene-candidates" && i + 1 < argc) {
            cfg.MAX_SCENE_CANDIDATES = std::stoi(argv[++i]);
        } else if (arg == "--max-frame-observations" && i + 1 < argc) {
            cfg.MAX_FRAME_OBSERVATIONS = std::stoi(argv[++i]);
        } else if (arg == "--graph-consistency-gate" && i + 1 < argc) {
            cfg.GRAPH_CONSISTENCY_GATE = std::stod(argv[++i]);
        } else if (arg == "--graph-min-inlier-ratio" && i + 1 < argc) {
            cfg.GRAPH_MIN_INLIER_RATIO = std::stod(argv[++i]);
        } else if (arg == "--scene-match-min-score" && i + 1 < argc) {
            cfg.SCENE_MATCH_MIN_SCORE = std::stod(argv[++i]);
        } else if (arg == "--yaw-adaptive-gate" && i + 1 < argc) {
            cfg.YAW_ADAPTIVE_GATE = std::stod(argv[++i]);
        } else if (arg == "--yaw-reject-gate" && i + 1 < argc) {
            cfg.YAW_REJECT_GATE = std::stod(argv[++i]);
        } else if (arg == "--size-ratio-gate" && i + 1 < argc) {
            cfg.SIZE_RATIO_GATE = std::stod(argv[++i]);
        } else if (arg == "--rematch-iterations" && i + 1 < argc) {
            cfg.REMATCH_ITERATIONS = std::stoi(argv[++i]);
        } else if (arg == "--max-mean-error" && i + 1 < argc) {
            cfg.MAX_MEAN_GRAPH_ERROR = std::stod(argv[++i]);
        } else if (arg == "--max-correction-trans" && i + 1 < argc) {
            cfg.MAX_CORRECTION_TRANS = std::stod(argv[++i]);
        } else if (arg == "--max-correction-yaw" && i + 1 < argc) {
            cfg.MAX_CORRECTION_YAW = std::stod(argv[++i]);
        } else if (arg == "--max-step-trans" && i + 1 < argc) {
            cfg.MAX_CORRECTION_STEP_TRANS = std::stod(argv[++i]);
        } else if (arg == "--max-step-yaw" && i + 1 < argc) {
            cfg.MAX_CORRECTION_STEP_YAW = std::stod(argv[++i]);
        } else if (arg == "--icp-static") {
            cfg.STATIC_ICP_ENABLED = true;
        } else if (arg == "--icp-max-boxes" && i + 1 < argc) {
            cfg.STATIC_ICP_MAX_BOXES = std::stoi(argv[++i]);
        } else if (arg == "--icp-max-points" && i + 1 < argc) {
            cfg.STATIC_ICP_MAX_POINTS = std::stoi(argv[++i]);
        } else if (arg == "--icp-iterations" && i + 1 < argc) {
            cfg.STATIC_ICP_ITERATIONS = std::stoi(argv[++i]);
        } else if (arg == "--icp-min-correspondences" && i + 1 < argc) {
            cfg.STATIC_ICP_MIN_CORRESPONDENCES = std::stoi(argv[++i]);
        } else if (arg == "--icp-grid" && i + 1 < argc) {
            cfg.STATIC_ICP_GRID_SIZE = std::stod(argv[++i]);
        } else if (arg == "--icp-max-correspondence" && i + 1 < argc) {
            cfg.STATIC_ICP_MAX_CORRESPONDENCE = std::stod(argv[++i]);
        } else if (arg == "--icp-max-rmse" && i + 1 < argc) {
            cfg.STATIC_ICP_MAX_RMSE = std::stod(argv[++i]);
        } else if (arg == "--icp-max-trans" && i + 1 < argc) {
            cfg.STATIC_ICP_MAX_TRANS = std::stod(argv[++i]);
        } else if (arg == "--icp-max-yaw" && i + 1 < argc) {
            cfg.STATIC_ICP_MAX_YAW = std::stod(argv[++i]);
        } else if (arg == "--icp-box-margin" && i + 1 < argc) {
            cfg.STATIC_ICP_BOX_MARGIN = std::stod(argv[++i]);
        } else if (arg == "--icp-trans-sigma" && i + 1 < argc) {
            cfg.STATIC_ICP_TRANS_SIGMA = std::stod(argv[++i]);
        } else if (arg == "--icp-yaw-sigma" && i + 1 < argc) {
            cfg.STATIC_ICP_YAW_SIGMA = std::stod(argv[++i]);
        } else if (arg == "--supervised-min-confidence" && i + 1 < argc) {
            cfg.SUPERVISED_PRIOR_MIN_CONF = std::stod(argv[++i]);
        } else if (arg == "--supervised-max-sigma-trans" && i + 1 < argc) {
            cfg.SUPERVISED_PRIOR_MAX_SIGMA_TRANS = std::stod(argv[++i]);
        } else if (arg == "--supervised-max-sigma-yaw" && i + 1 < argc) {
            cfg.SUPERVISED_PRIOR_MAX_SIGMA_YAW = std::stod(argv[++i]);
        }
    }
    cfg.VEH_DELAY_FRAMES = std::max(0, cfg.VEH_DELAY_FRAMES);
    cfg.WINDOW_SIZE = std::max(1, cfg.WINDOW_SIZE);
    cfg.DELAY_MS = std::max(1, cfg.DELAY_MS);
    cfg.MIN_BOX_OBSERVATIONS = std::max(1, cfg.MIN_BOX_OBSERVATIONS);
    cfg.MIN_CENTER_ONLY_OBSERVATIONS = std::max(1, cfg.MIN_CENTER_ONLY_OBSERVATIONS);
    cfg.GRAPH_MIN_CANDIDATES = std::max(1, cfg.GRAPH_MIN_CANDIDATES);
    cfg.MAX_SCENE_CANDIDATES = std::max(1, cfg.MAX_SCENE_CANDIDATES);
    cfg.MAX_FRAME_OBSERVATIONS = std::max(1, cfg.MAX_FRAME_OBSERVATIONS);
    cfg.GRAPH_CONSISTENCY_GATE = std::max(0.01, cfg.GRAPH_CONSISTENCY_GATE);
    cfg.GRAPH_MIN_INLIER_RATIO = std::clamp(cfg.GRAPH_MIN_INLIER_RATIO, 0.0, 1.0);
    cfg.SCENE_MATCH_MIN_SCORE = std::max(0.0, cfg.SCENE_MATCH_MIN_SCORE);
    cfg.YAW_ADAPTIVE_GATE = std::max(0.0, cfg.YAW_ADAPTIVE_GATE);
    cfg.YAW_REJECT_GATE = std::max(0.0, cfg.YAW_REJECT_GATE);
    cfg.SIZE_RATIO_GATE = std::max(0.0, cfg.SIZE_RATIO_GATE);
    cfg.REMATCH_ITERATIONS = std::max(1, cfg.REMATCH_ITERATIONS);
    cfg.STATIC_ICP_MAX_BOXES = std::max(0, cfg.STATIC_ICP_MAX_BOXES);
    cfg.STATIC_ICP_MAX_POINTS = std::max(100, cfg.STATIC_ICP_MAX_POINTS);
    cfg.STATIC_ICP_ITERATIONS = std::max(1, cfg.STATIC_ICP_ITERATIONS);
    cfg.STATIC_ICP_MIN_CORRESPONDENCES = std::max(3, cfg.STATIC_ICP_MIN_CORRESPONDENCES);
    cfg.STATIC_ICP_GRID_SIZE = std::max(0.1, cfg.STATIC_ICP_GRID_SIZE);
    cfg.STATIC_ICP_MAX_CORRESPONDENCE = std::max(0.2, cfg.STATIC_ICP_MAX_CORRESPONDENCE);
    cfg.STATIC_ICP_MAX_RMSE = std::max(0.1, cfg.STATIC_ICP_MAX_RMSE);
    cfg.STATIC_ICP_MAX_TRANS = std::max(0.05, cfg.STATIC_ICP_MAX_TRANS);
    cfg.STATIC_ICP_MAX_YAW = std::max(0.01, cfg.STATIC_ICP_MAX_YAW);
    cfg.STATIC_ICP_BOX_MARGIN = std::max(0.0, cfg.STATIC_ICP_BOX_MARGIN);
    cfg.STATIC_ICP_TRANS_SIGMA = std::max(0.05, cfg.STATIC_ICP_TRANS_SIGMA);
    cfg.STATIC_ICP_YAW_SIGMA = std::max(0.01, cfg.STATIC_ICP_YAW_SIGMA);
    cfg.SUPERVISED_PRIOR_MIN_CONF = std::clamp(cfg.SUPERVISED_PRIOR_MIN_CONF, 0.0, 1.0);
    cfg.SUPERVISED_PRIOR_MAX_SIGMA_TRANS = std::max(0.05, cfg.SUPERVISED_PRIOR_MAX_SIGMA_TRANS);
    cfg.SUPERVISED_PRIOR_MAX_SIGMA_YAW = std::max(0.01, cfg.SUPERVISED_PRIOR_MAX_SIGMA_YAW);
    cfg.SUPERVISED_PRIORS = loadSupervisedPriorCsv(cfg.SUPERVISED_PRIOR_CSV);
    if (!cfg.SUPERVISED_PRIORS.empty()) {
        std::cout << "[SupervisedPrior] loaded " << cfg.SUPERVISED_PRIORS.size()
                  << " records from " << cfg.SUPERVISED_PRIOR_CSV << std::endl;
    }
    max_frames = std::max(0, max_frames);

    if (headless) {
        std::ofstream metrics_file;
        std::ostream* metrics_out = nullptr;
        if (!cfg.METRICS_CSV.empty()) {
            fs::path metrics_path(cfg.METRICS_CSV);
            if (metrics_path.has_parent_path()) fs::create_directories(metrics_path.parent_path());
            metrics_file.open(metrics_path);
            if (!metrics_file) {
                std::cerr << "Failed to open metrics csv: " << metrics_path.string() << std::endl;
                return -1;
            }
            writeMetricsHeader(metrics_file);
            metrics_out = &metrics_file;
        }

        std::vector<std::string> roots;
        if (!root_list_path.empty()) {
            roots = readRootListFile(root_list_path);
            if (roots.empty()) {
                std::cerr << "No roots found in root list: " << root_list_path << std::endl;
                return -1;
            }
        } else {
            roots.push_back(cfg.PC_DATA_ROOT);
        }

        const std::string dataset_name = v2x_real_mode ? "v2x-real" : "dair-v2x";
        int status = 0;
        for (size_t root_idx = 0; root_idx < roots.size(); ++root_idx) {
            std::cout << "[Root " << (root_idx + 1) << "/" << roots.size() << "] "
                      << roots[root_idx] << std::endl;
            status |= runHeadlessRoot(cfg, roots[root_idx], dataset_name, max_frames, metrics_out);
        }
        return status == 0 ? 0 : -1;
    }

    auto scenes = getSortedScenes(cfg.PC_DATA_ROOT);
    if (scenes.empty()) { std::cerr << "No scenes found!" << std::endl; return -1; }
    std::cout << "[Startup] root=" << cfg.PC_DATA_ROOT
              << " scenes=" << scenes.size()
              << " delay=" << cfg.VEH_DELAY_FRAMES
              << " window=" << cfg.WINDOW_SIZE
              << " delay_ms=" << cfg.DELAY_MS
              << " match_dist=" << cfg.MATCH_DIST_THRESHOLD
              << " min_boxes=" << cfg.MIN_BOX_OBSERVATIONS
              << " max_obs=" << cfg.MAX_FRAME_OBSERVATIONS
              << " rematch_iters=" << cfg.REMATCH_ITERATIONS
              << " max_mean_error=" << cfg.MAX_MEAN_GRAPH_ERROR
              << " icp_static=" << (cfg.STATIC_ICP_ENABLED ? "on" : "off")
              << " supervised_priors=" << cfg.SUPERVISED_PRIORS.size()
              << (v2x_real_mode ? " dataset=v2x-real" : "")
              << (headless ? " mode=headless" : " mode=gui")
              << std::endl;
    std::cout << "[Startup] first_scene=" << scenes.front().string() << std::endl;

    if (headless) {
        std::ofstream metrics_file;
        std::ostream* metrics_out = nullptr;
        if (!cfg.METRICS_CSV.empty()) {
            fs::path metrics_path(cfg.METRICS_CSV);
            if (metrics_path.has_parent_path()) fs::create_directories(metrics_path.parent_path());
            metrics_file.open(metrics_path);
            if (!metrics_file) {
                std::cerr << "Failed to open metrics csv: " << metrics_path.string() << std::endl;
                return -1;
            }
            writeMetricsHeader(metrics_file);
            metrics_out = &metrics_file;
        }
        const std::string dataset_name = v2x_real_mode ? "v2x-real" : "dair-v2x";
        SlidingWindowGtsamOptimizer optimizer(cfg);
        const int total_frames = max_frames > 0
            ? std::min(max_frames, static_cast<int>(scenes.size()))
            : static_cast<int>(scenes.size());

        for (int frame_idx = 0; frame_idx < total_frames; ++frame_idx) {
            int veh_idx = std::max(0, frame_idx - cfg.VEH_DELAY_FRAMES);
            FrameObservation frame_obs;
            Matrix4f T_init_current = Matrix4f::Identity();
            size_t raw_pair_count = 0;
            if (!buildFrameObservation(scenes[frame_idx], scenes[veh_idx], cfg, frame_obs, T_init_current, raw_pair_count)) {
                std::cout << "[Frame " << std::setw(3) << frame_idx << "] skipped: missing json or transform" << std::endl;
                continue;
            }

            int rematch_count = 0;
            Matrix4f T_corr = optimizeLatestWithRematching(
                optimizer,
                scenes[frame_idx],
                scenes[veh_idx],
                cfg,
                frame_obs,
                T_init_current,
                raw_pair_count,
                rematch_count);
            Vector3f delta_t = T_corr.block<3,1>(0,3) - T_init_current.block<3,1>(0,3);
            float yaw_deg = static_cast<float>(
                wrapAngle(matrixToPose2(T_corr).theta() - matrixToPose2(T_init_current).theta()) *
                180.0 / kPi);
            size_t center_only_count = 0;
            for (const auto& obs : frame_obs.boxes) if (obs.center_only) ++center_only_count;
            const AlignmentMetrics alignment = computeAlignmentMetrics(frame_obs, T_init_current, T_corr);

            std::cout << "[Frame " << std::setw(3) << frame_idx << "] "
                      << "raw_pairs=" << raw_pair_count
                      << " accepted=" << frame_obs.boxes.size()
                      << " landmarks=" << optimizer.lastLandmarkCount()
                      << " center_only=" << center_only_count
                      << " rematch=" << rematch_count
                      << " quality=" << std::fixed << std::setprecision(3) << optimizer.lastFrameQuality()
                      << " delta_t=[" << std::fixed << std::setprecision(3)
                      << delta_t.x() << ", " << delta_t.y() << ", " << delta_t.z() << "]"
                      << " yaw=" << std::fixed << std::setprecision(2) << yaw_deg
                      << " error=" << std::fixed << std::setprecision(3) << optimizer.lastError()
                      << " mean_error=" << std::fixed << std::setprecision(3) << optimizer.lastMeanError()
                      << " icp=" << (frame_obs.static_icp.valid ? "on" : "off")
                      << "(" << frame_obs.static_icp.correspondences << ","
                      << std::fixed << std::setprecision(3) << frame_obs.static_icp.rmse << ")"
                      << " supervised=" << (frame_obs.supervised_prior.valid ? "on" : "off")
                      << (optimizer.lastRejected() ? (" rejected=" + optimizer.lastRejectReason()) : "")
                      << std::endl;

            if (metrics_out) {
                writeMetricsRow(
                    *metrics_out,
                    dataset_name,
                    cfg.PC_DATA_ROOT,
                    frame_idx,
                    scenes[frame_idx],
                    raw_pair_count,
                    frame_obs,
                    center_only_count,
                    optimizer,
                    rematch_count,
                    delta_t,
                    yaw_deg,
                    alignment);
            }
        }
        return 0;
    }

    std::cout << "[Startup] creating Pangolin window..." << std::endl;
    pangolin::CreateWindowAndBind("Cooperative Visualizer Pro", 1800, 1200);
    std::cout << "[Startup] Pangolin window created. initial_should_quit="
              << (pangolin::ShouldQuit() ? "true" : "false") << std::endl;
    glEnable(GL_DEPTH_TEST); glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // GUI panel.
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, 0.15);
    pangolin::Var<bool> ui_play("ui.Play", true, true);
    pangolin::Var<int> ui_delay("ui.DelayMS", cfg.DELAY_MS, 10, 2000);
    pangolin::Var<int> ui_traj_len("ui.TrajMaxLen", 150, 1, 1000);
    pangolin::Var<bool> ui_show_inf("ui.ShowInfBox", false, true);
    pangolin::Var<bool> ui_highlight_match("ui.HighlightMatch", true, true);
    pangolin::Var<bool> ui_reset_traj("ui.ClearTraj", false, false);

    pangolin::OpenGlRenderState s_cam(pangolin::ProjectionMatrix(1800,1200,900,900,900,600,0.1,10000), pangolin::ModelViewLookAt(0,-70,50,0,0,0,pangolin::AxisZ));
    pangolin::View& d_cam = pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.15, 1.0, -1.5).SetHandler(new pangolin::Handler3D(s_cam));

    std::vector<Vector3f> traj_orig_world, traj_corr_world;
    int frame_idx = 0;
    SlidingWindowGtsamOptimizer gtsam_optimizer(cfg);

    pangolin::RegisterKeyPressCallback(' ', [&](){ ui_play = !ui_play; });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_RIGHT, [&](){ ui_play = false; frame_idx = (frame_idx + 1) % (int)scenes.size(); });
    pangolin::RegisterKeyPressCallback(pangolin::PANGO_SPECIAL + pangolin::PANGO_KEY_LEFT, [&](){ ui_play = false; frame_idx = (frame_idx - 1 + (int)scenes.size()) % (int)scenes.size(); });

    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); glClearColor(0.05f, 0.05f, 0.07f, 1.0f);
        d_cam.Activate(s_cam);

        if (ui_reset_traj || (frame_idx == 0 && ui_play)) {
            traj_orig_world.clear();
            traj_corr_world.clear();
            gtsam_optimizer.reset();
            ui_reset_traj = false;
        }

        int veh_idx = std::max(0, frame_idx - cfg.VEH_DELAY_FRAMES);
        auto scene_path_inf = scenes[frame_idx];
        auto scene_path_veh = scenes[veh_idx];

        std::string sn_inf = scene_path_inf.filename().string();
        fs::path jp_inf = findSceneJson(scene_path_inf, cfg);
        fs::path jp_veh = findSceneJson(scene_path_veh, cfg);

        if (jp_inf.empty() || jp_veh.empty()) { frame_idx = (frame_idx + 1) % (int)scenes.size(); continue; }

        std::ifstream f_i(jp_inf); json anno_inf = json::parse(f_i);
        std::ifstream f_v(jp_veh); json anno_veh = json::parse(f_v);

        Matrix4f T_init_current = jsonToMatrix4f(anno_inf["transformation_matrices"]["vehicle_lidar_to_world"]);
        Matrix4f T_init_delayed = jsonToMatrix4f(anno_veh["transformation_matrices"]["vehicle_lidar_to_world"]);

        Matrix4f T_motion_compensation = T_init_current * invertTransformMatrix(T_init_delayed);

        fs::path vp = findPcdPath(scene_path_veh, anno_veh, true);
        fs::path ip = findPcdPath(scene_path_inf, anno_inf, false);
        std::vector<Vector3f> veh_pcd = loadPcdXYZ(vp);
        std::vector<Vector3f> inf_pcd = loadPcdXYZ(ip);

        std::set<std::string> matched_veh_ids, matched_inf_ids;
        struct MatchCenterPair { Vector3f v_c, i_c; };
        std::vector<MatchCenterPair> visual_match_lines;
        FrameObservation frame_obs;
        Matrix4f T_from_builder = Matrix4f::Identity();
        size_t raw_pair_count = 0;
        if (!buildFrameObservation(scene_path_inf, scene_path_veh, cfg, frame_obs, T_from_builder, raw_pair_count)) {
            frame_idx = (frame_idx + 1) % (int)scenes.size();
            continue;
        }
        T_init_current = T_from_builder;

        int rematch_count = 0;
        Matrix4f T_corr = optimizeLatestWithRematching(
            gtsam_optimizer,
            scene_path_inf,
            scene_path_veh,
            cfg,
            frame_obs,
            T_init_current,
            raw_pair_count,
            rematch_count);

        size_t center_only_count = 0;
        matched_veh_ids.clear();
        matched_inf_ids.clear();
        visual_match_lines.clear();
        for (const auto& obs : frame_obs.boxes) {
            matched_veh_ids.insert(obs.veh_track_id);
            matched_inf_ids.insert(obs.inf_track_id);
            visual_match_lines.push_back({obs.veh_center_world, obs.inf_center_world});
            if (obs.center_only) ++center_only_count;
        }

        Vector3f curr_t = T_corr.block<3,1>(0,3) - T_init_current.block<3,1>(0,3);
        float yaw_deg = static_cast<float>(
            wrapAngle(matrixToPose2(T_corr).theta() - matrixToPose2(T_init_current).theta()) *
            180.0 / kPi);

        const bool has_matches = !frame_obs.boxes.empty();
        std::cout << (has_matches ? "\033[1;32m" : "\033[1;33m")
                  << "[Frame " << std::setw(3) << frame_idx << "]\033[0m "
                  << "Pairs: " << raw_pair_count << " -> " << std::setw(2) << frame_obs.boxes.size()
                  << " | Landmarks: " << gtsam_optimizer.lastLandmarkCount()
                  << " | CenterOnly: " << center_only_count
                  << " | Rematch: " << rematch_count
                  << " | Quality: " << std::fixed << std::setprecision(3) << gtsam_optimizer.lastFrameQuality()
                  << " | "
                  << "GTSAM_Delta_T: [" << std::fixed << std::setprecision(3)
                  << curr_t.x() << ", " << curr_t.y() << ", " << curr_t.z() << "] | "
                  << "GTSAM_Yaw: " << std::fixed << std::setprecision(2) << std::setw(6) << yaw_deg << " deg | "
                  << "GraphError: " << std::fixed << std::setprecision(3) << gtsam_optimizer.lastError()
                  << " | MeanError: " << std::fixed << std::setprecision(3) << gtsam_optimizer.lastMeanError()
                  << " | ICP: " << (frame_obs.static_icp.valid ? "on" : "off")
                  << "(" << frame_obs.static_icp.correspondences << ","
                  << std::fixed << std::setprecision(3) << frame_obs.static_icp.rmse << ")"
                  << (gtsam_optimizer.lastRejected() ? (" | rejected=" + gtsam_optimizer.lastRejectReason()) : "")
                  << (has_matches ? "" : " | no box factors, odometry/prior only")
                  << std::endl;

        Matrix4f T_world_to_curr_veh = invertTransformMatrix(T_corr);

        if (ui_play) {
            Vector4f local_bot(0,0,cfg.GROUND_OFFSET,1);
            traj_orig_world.push_back((T_init_current * local_bot).head<3>());
            traj_corr_world.push_back((T_corr * local_bot).head<3>());
            while (traj_orig_world.size() > (size_t)ui_traj_len) { traj_orig_world.erase(traj_orig_world.begin()); traj_corr_world.erase(traj_corr_world.begin()); }
        }

        // Render frame.
        std::vector<Vector3f> t_o_v, t_c_v;
        for(auto& p : traj_orig_world) { Vector4f pv = T_world_to_curr_veh * Vector4f(p.x(),p.y(),p.z(),1.0f); t_o_v.emplace_back(pv.x(),pv.y(),pv.z()); }
        for(auto& p : traj_corr_world) { Vector4f pv = T_world_to_curr_veh * Vector4f(p.x(),p.y(),p.z(),1.0f); t_c_v.emplace_back(pv.x(),pv.y(),pv.z()); }
        drawTrajectory(t_o_v, pangolin::Colour(1,0.3,0.3,0.5), 1.0f);
        drawTrajectory(t_c_v, pangolin::Colour(0,1,1,1), 3.0f);

        Matrix4f T_axis = Matrix4f::Identity(); T_axis(2,3) = cfg.GROUND_OFFSET;
        drawCoordinateFrame(T_axis, 3.0f, 2.0f);

        auto drawPCD = [&](const std::vector<Vector3f>& cloud, bool is_veh) {
            glPointSize(cfg.POINT_SIZE); glBegin(GL_POINTS);
            for(auto& p : cloud) {
                Vector4f p_w = is_veh ? (T_motion_compensation * Vector4f(p.x(),p.y(),p.z(),1)) : Vector4f(p.x(),p.y(),p.z(),1);
                Vector4f pv = T_world_to_curr_veh * p_w;
                glColorHeight(p.z(), cfg.MIN_Z, cfg.MAX_Z, is_veh); glVertex3f(pv.x(), pv.y(), pv.z());
            }
            glEnd();
        };
        drawPCD(veh_pcd, true); drawPCD(inf_pcd, false);

        auto drawBoxes = [&](const json& anns, bool is_veh_source) {
            for (const auto& obj : anns) {
                if(!obj.contains("3d_box_points_world")) continue;
                std::vector<Vector3f> box_v;
                for(auto& p : obj["3d_box_points_world"]) {
                    Vector4f p_w = is_veh_source ? (T_motion_compensation * Vector4f(p[0],p[1],p[2],1)) : Vector4f(p[0],p[1],p[2],1);
                    Vector4f pv = T_world_to_curr_veh * p_w; box_v.emplace_back(pv.x(),pv.y(),pv.z());
                }
                std::string tid = obj.value("track_id", "");
                bool is_m = ui_highlight_match && (is_veh_source ? matched_veh_ids.count(tid) : matched_inf_ids.count(tid));
                pangolin::Colour col = is_m ? pangolin::Colour(0,1,0) : (is_veh_source ? pangolin::Colour(1,1,0) : pangolin::Colour(0.5,0.5,1));
                drawBox(box_v, col, is_m ? 4.0f : 2.0f);
            }
        };
        drawBoxes(anno_veh.value("vehicle_annotations_world", json::array()), true);
        if (ui_show_inf) drawBoxes(anno_inf.value("infrastructure_annotations_world", json::array()), false);

        if (ui_highlight_match && !visual_match_lines.empty()) {
            glLineWidth(1.0f); glColor3f(1.0f, 1.0f, 1.0f); glBegin(GL_LINES);
            for (auto& pair : visual_match_lines) {
                Vector4f v1 = T_world_to_curr_veh * Vector4f(pair.v_c.x(), pair.v_c.y(), pair.v_c.z(), 1.0f);
                Vector4f v2 = T_world_to_curr_veh * Vector4f(pair.i_c.x(), pair.i_c.y(), pair.i_c.z(), 1.0f);
                glVertex3f(v1.x(), v1.y(), v1.z()); glVertex3f(v2.x(), v2.y(), v2.z());
            }
            glEnd();
        }

        pangolin::FinishFrame();
        if (ui_play) {
            std::this_thread::sleep_for(std::chrono::milliseconds((int)ui_delay));
            frame_idx = (frame_idx + 1) % (int)scenes.size();
        }
    }
    return 0;
}
