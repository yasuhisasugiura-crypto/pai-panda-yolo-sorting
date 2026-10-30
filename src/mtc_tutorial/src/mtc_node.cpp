// Panda sorting demo (MoveIt Task Constructor) -- Stage A/B/C rebuild.
//   drinks (bottle / cup / wine glass / vase) -> LEFT tray  (+y)
//   food   (banana / apple / orange / box)    -> RIGHT tray (-y)
//
//   source_mode = fixed       : baseline. 6 configured objects on a fixed grid.
//   source_mode = detections  : build/update objects from
//                               /perception/detections_3d (vision_msgs).
//   autonomous  = true        : Stage C. Loop until every detected object is
//                               sorted; retry on failure, skip after retry_max.
// Topics out: /mtc/active_index, /mtc/cycle_count (Int32),
//             /mtc/active_class (String), /mtc/object_labels (MarkerArray)
// Topics in : /perception/detections_3d (vision_msgs/Detection3DArray)
//
// v0.4.1 fixes (see FIXES_ja.md):
//   * runDetections(): the result lock was held while calling publishMarkers()
//     and finish(), which lock specs_mu_ again -> guaranteed self-deadlock after
//     the first object in Stage B/C. The lock is now scoped.
//   * ensureInScene(): no longer touches specs_ without the mutex (data race with
//     onDetections() push_back), and now inserts every stable unprocessed object
//     (not only the target) so planning avoids the neighbours.
//   * tray capacity is checked before picking; a full tray skips/stops cleanly
//     instead of throwing out of run().
//   * finish(): active_/active_id_ are reset under the mutex (data race with
//     onDetections(), found with ThreadSanitizer).
//   * detect_wait_sec default 20 -> 60 s (matches the launch file).
//
// v0.4.2 fixes (see FIXES_ja.md):
//   * finish(): with 0 objects (nothing detected within detect_wait_sec) the state
//     was published as "DONE" although run() returned false. Now "INTERRUPTED".
//   * tray-full skip also clears the active highlight (active_ = -1).

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Geometry>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
// MoveIt: newer releases use .hpp, older Jazzy releases only ship .h
#if __has_include(<moveit/planning_scene_interface/planning_scene_interface.hpp>)
#include <moveit/planning_scene_interface/planning_scene_interface.hpp>
#else
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#endif
#include <moveit/task_constructor/solvers.h>
#include <moveit/task_constructor/stages.h>
#include <moveit/task_constructor/task.h>
#include <moveit_msgs/msg/attached_collision_object.hpp>
#include <moveit_msgs/msg/collision_object.hpp>
#include <moveit_msgs/msg/move_it_error_codes.hpp>
#include <moveit_msgs/msg/object_color.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit_task_constructor_msgs/action/execute_task_solution.hpp>
#include <moveit_msgs/srv/get_planning_scene.hpp>
#include <moveit_msgs/srv/apply_planning_scene.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace mtc = moveit::task_constructor;
using namespace std::chrono_literals;
using Prim = shape_msgs::msg::SolidPrimitive;

static std_msgs::msg::ColorRGBA rgba(double r, double g, double b, double a = 1.0)
{
  std_msgs::msg::ColorRGBA c;
  c.r = static_cast<float>(r);
  c.g = static_cast<float>(g);
  c.b = static_cast<float>(b);
  c.a = static_cast<float>(a);
  return c;
}

struct ObjSpec
{
  std::string id, cls;
  bool drink{false};
  int type{Prim::BOX};
  std::vector<double> dims{0.05, 0.05, 0.05};
  double cz{0.025};               // center height when standing on the table (z = 0)
  geometry_msgs::msg::Pose pose;  // current pose (source -> tray after success)
  std_msgs::msg::ColorRGBA color;
  bool placed{false};
  bool skipped{false};
  bool in_scene{false};
  geometry_msgs::msg::Pose scene_pose;  // pose last written to the PlanningScene
  int seen{0};
};

class MTCTaskNode
{
public:
  explicit MTCTaskNode(const rclcpp::NodeOptions& options)
  : node_(std::make_shared<rclcpp::Node>("mtc_node", options))
  {
    std::string csv = "bottle,cup,wine glass,banana,apple,orange";
    node_->get_parameter_or("object_classes", csv, csv);
    node_->get_parameter_or("velocity_scaling", vel_, 0.2);
    node_->get_parameter_or("acceleration_scaling", acc_, 0.2);
    node_->get_parameter_or("max_solutions", max_sol_, 5);
    node_->get_parameter_or("cycle_pause_sec", pause_, 1.0);
    node_->get_parameter_or("reset_demo_scene", reset_scene_, false);
    node_->get_parameter_or("startup_timeout_sec", startup_timeout_, 30.0);
    node_->get_parameter_or("grasp_offset_mode", grasp_offset_mode_, std::string("baseline"));
    if (grasp_offset_mode_ != "baseline" && grasp_offset_mode_ != "height")
      throw std::invalid_argument("grasp_offset_mode must be baseline|height");
    // Stage B/C parameters (defaults keep the baseline behaviour)
    node_->get_parameter_or("source_mode", source_mode_, std::string("fixed"));
    node_->get_parameter_or("autonomous", autonomous_, false);
    node_->get_parameter_or("detections_topic", det_topic_, std::string("/perception/detections_3d"));
    node_->get_parameter_or("retry_max", retry_max_, 3);
    node_->get_parameter_or("detect_wait_sec", detect_wait_, 60.0);
    node_->get_parameter_or("stable_frames", stable_need_, 1);
    if (!std::isfinite(vel_) || vel_ <= 0 || vel_ > 1 ||
        !std::isfinite(acc_) || acc_ <= 0 || acc_ > 1 ||
        !std::isfinite(pause_) || pause_ < 0 || max_sol_ < 1 || max_sol_ > 100 ||
        retry_max_ < 1 || retry_max_ > 10 ||
        !std::isfinite(detect_wait_) || detect_wait_ <= 0 || stable_need_ < 1 || stable_need_ > 1000 ||
        !std::isfinite(startup_timeout_) || startup_timeout_ <= 0) {
      throw std::invalid_argument("invalid scaling, pause, solution count, retry or timeout");
    }
    if (source_mode_ != "fixed" && source_mode_ != "detections")
      throw std::invalid_argument("source_mode must be fixed|detections");

    const auto state_qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local();
    idx_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/mtc/active_index", state_qos);
    cyc_pub_ = node_->create_publisher<std_msgs::msg::Int32>("/mtc/cycle_count", state_qos);
    cls_pub_ = node_->create_publisher<std_msgs::msg::String>("/mtc/active_class", state_qos);
    marker_pub_ =
      node_->create_publisher<visualization_msgs::msg::MarkerArray>("/mtc/object_labels", state_qos);

    if (source_mode_ == "fixed") {
      buildSpecs(csv);
    } else {
      // Detections drive the scene; subscribe with matching durability.
      det_sub_ = node_->create_subscription<vision_msgs::msg::Detection3DArray>(
        det_topic_, state_qos,
        [this](vision_msgs::msg::Detection3DArray::SharedPtr m) { onDetections(m); });
    }
    RCLCPP_INFO(node_->get_logger(), "mode=%s autonomous=%d objects=%zu (vel=%.2f acc=%.2f)",
                source_mode_.c_str(), autonomous_ ? 1 : 0, specs_.size(), vel_, acc_);
  }

  rclcpp::node_interfaces::NodeBaseInterface::SharedPtr getNodeBaseInterface()
  {
    return node_->get_node_base_interface();
  }

  bool run()
  {
    waitForBackend();
    setupScene();
    return (source_mode_ == "fixed") ? runFixedPass() : runDetections();
  }

private:
  static constexpr const char* kArm = "panda_arm";
  static constexpr const char* kHand = "hand";
  static constexpr const char* kHandFrame = "panda_hand";
  static constexpr double kTrayY = 0.40;

  // ------------------------------------------------------------------ helpers
  void pubI(const rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr& p, int v)
  {
    std_msgs::msg::Int32 m; m.data = v; p->publish(m);
  }

  static std::string under(std::string s)
  {
    for (auto& ch : s) if (ch == ' ') ch = '_';
    return s;
  }

  static std::string pad2(std::size_t v)
  {
    std::string s = std::to_string(v);
    return s.size() < 2 ? "0" + s : s;
  }

  static std::string lower(std::string s)
  {
    std::transform(s.begin(), s.end(), s.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (s == "wineglass" || s == "wine_glass") s = "wine glass";
    return s;
  }

  void classify(ObjSpec& s)
  {
    const std::string& c = s.cls;
    s.drink = (c == "bottle" || c == "cup" || c == "wine glass" || c == "vase");
    if (c == "bottle") {
      s.type = Prim::CYLINDER; s.dims = {0.18, 0.03}; s.color = rgba(0.1, 0.7, 0.2);
    } else if (c == "wine glass") {
      s.type = Prim::CYLINDER; s.dims = {0.16, 0.03}; s.color = rgba(0.7, 0.2, 0.8);
    } else if (c == "cup") {
      s.type = Prim::CYLINDER; s.dims = {0.09, 0.035}; s.color = rgba(0.1, 0.7, 0.8);
    } else if (c == "vase") {
      s.type = Prim::CYLINDER; s.dims = {0.15, 0.035}; s.color = rgba(0.3, 0.4, 0.9);
    } else if (c == "banana") {
      s.type = Prim::BOX; s.dims = {0.12, 0.04, 0.04}; s.color = rgba(0.95, 0.85, 0.1);
    } else if (c == "apple") {
      s.type = Prim::SPHERE; s.dims = {0.035}; s.color = rgba(0.85, 0.1, 0.1);
    } else if (c == "orange") {
      s.type = Prim::SPHERE; s.dims = {0.037}; s.color = rgba(1.0, 0.5, 0.0);
    } else {
      s.type = Prim::BOX; s.dims = {0.05, 0.05, 0.05}; s.color = rgba(0.6, 0.6, 0.6);
    }
    if (s.type == Prim::CYLINDER) s.cz = s.dims[0] / 2.0;
    else if (s.type == Prim::SPHERE) s.cz = s.dims[0];
    else s.cz = s.dims[2] / 2.0;
  }

  // source grid (fixed mode): 2 rows x 4 columns
  geometry_msgs::msg::Pose sourcePose(std::size_t i, double cz)
  {
    static const double xs[2] = {0.36, 0.52};
    static const double ys[4] = {-0.21, -0.07, 0.07, 0.21};
    geometry_msgs::msg::Pose p;
    p.orientation.w = 1.0;
    p.position.x = xs[(i / 4) % 2];
    p.position.y = ys[i % 4];
    p.position.z = cz;
    return p;
  }

  geometry_msgs::msg::Pose slotPose(bool drink, double cz) const
  {
    static const double sx[4] = {0.56, 0.44, 0.32, 0.20};
    const int n = drink ? left_n_ : right_n_;
    geometry_msgs::msg::Pose p;
    p.orientation.w = 1.0;
    if (n < 0 || n >= 4) throw std::runtime_error("tray capacity exceeded");
    p.position.x = sx[n];
    p.position.y = drink ? kTrayY : -kTrayY;
    p.position.z = cz;
    return p;
  }

  void buildSpecs(const std::string& csv)
  {
    std::size_t start = 0, idx = 0;
    while (start <= csv.size()) {
      const std::size_t comma = csv.find(',', start);
      const std::string tok =
        csv.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
      const std::size_t a = tok.find_first_not_of(" \t");
      const std::size_t b = tok.find_last_not_of(" \t");
      if (a != std::string::npos) {
        ObjSpec s;
        s.cls = lower(tok.substr(a, b - a + 1));
        const std::vector<std::string> supported =
          {"bottle", "cup", "wine glass", "vase", "banana", "apple", "orange", "box"};
        if (std::find(supported.begin(), supported.end(), s.cls) == supported.end())
          throw std::invalid_argument("unsupported class: " + s.cls);
        classify(s);
        s.id = "sorting_demo_obj" + pad2(idx) + "_" + under(s.cls);
        s.pose = sourcePose(idx, s.cz);
        specs_.push_back(s);
        ++idx;
        if (idx > 8) throw std::invalid_argument("at most 8 objects supported");
      }
      if (comma == std::string::npos) break;
      start = comma + 1;
    }
    if (specs_.empty()) throw std::invalid_argument("object_classes is empty");
    const auto drinks = std::count_if(specs_.begin(), specs_.end(),
      [](const ObjSpec& s) { return s.drink; });
    if (drinks > 4 || specs_.size() - static_cast<std::size_t>(drinks) > 4)
      throw std::invalid_argument("each tray supports at most 4 objects");
  }

  moveit_msgs::msg::CollisionObject toCollision(const ObjSpec& s)
  {
    moveit_msgs::msg::CollisionObject o;
    o.id = s.id;
    o.header.frame_id = "world";
    shape_msgs::msg::SolidPrimitive p;
    p.type = static_cast<uint8_t>(s.type);
    p.dimensions.assign(s.dims.begin(), s.dims.end());
    o.primitives.push_back(p);
    o.pose = s.pose;
    geometry_msgs::msg::Pose local;
    local.orientation.w = 1.0;
    o.primitive_poses.push_back(local);
    o.operation = moveit_msgs::msg::CollisionObject::ADD;   // ADD acts as add-or-move
    return o;
  }

  // -------------------------------------------------- Stage B/C: detections in
  void onDetections(const vision_msgs::msg::Detection3DArray::SharedPtr& m)
  {
    if (m->header.frame_id != "world") {
      RCLCPP_WARN(node_->get_logger(), "rejecting detections: frame must be world");
      return;
    }
    std::lock_guard<std::mutex> lk(specs_mu_);
    for (const auto& d : m->detections) {
      if (d.results.empty() || d.id.empty()) continue;
      const auto& pos = d.bbox.center.position;
      if (!std::isfinite(pos.x) || !std::isfinite(pos.y) || !std::isfinite(pos.z) ||
          pos.x < 0.33 || pos.x > 0.52 || pos.y < -0.30 || pos.y > 0.30) continue;
      const std::string cls = lower(d.results.front().hypothesis.class_id);
      static const std::vector<std::string> supported =
        {"bottle", "cup", "wine glass", "vase", "banana", "apple", "orange", "box"};
      if (std::find(supported.begin(), supported.end(), cls) == supported.end()) continue;
      const std::string id = "sorting_demo_obj_det_" + d.id + "_" + under(cls);
      auto it = index_.find(id);
      if (it != index_.end()) {
        ObjSpec& s = specs_[it->second];
        if (s.placed || s.id == active_id_) continue;   // don't disturb active/placed
        s.pose = d.bbox.center;
        s.pose.position.z = s.cz;                        // stand the primitive on the table
        s.pose.orientation = geometry_msgs::msg::Quaternion();
        s.pose.orientation.w = 1.0;
        ++s.seen;
      } else {
        ObjSpec s;
        s.cls = cls;
        classify(s);
        s.id = id;
        s.pose = d.bbox.center;
        s.pose.position.z = s.cz;
        s.pose.orientation = geometry_msgs::msg::Quaternion();
        s.pose.orientation.w = 1.0;
        s.seen = 1;
        if (specs_.size() >= 256) continue;
        index_[id] = specs_.size();
        specs_.push_back(s);
      }
    }
  }

  // Put every stable, unprocessed detection into the PlanningScene (ADD = add-or-move)
  // so the arm plans around the neighbouring objects too. The mutex is held only
  // while reading/writing specs_, never while talking to MoveIt.
  void ensureInScene()
  {
    std::vector<moveit_msgs::msg::CollisionObject> objs;
    std::vector<moveit_msgs::msg::ObjectColor> cols;
    std::vector<std::pair<std::size_t, geometry_msgs::msg::Pose>> applied;
    {
      std::lock_guard<std::mutex> lk(specs_mu_);
      for (std::size_t i = 0; i < specs_.size(); ++i) {
        const auto& s = specs_[i];
        if (s.placed || s.seen < stable_need_) continue;
        if (s.in_scene &&
            std::hypot(s.pose.position.x - s.scene_pose.position.x,
                       s.pose.position.y - s.scene_pose.position.y) < 0.005)
          continue;                                     // already there, not moved
        objs.push_back(toCollision(s));
        moveit_msgs::msg::ObjectColor oc; oc.id = s.id; oc.color = s.color;
        cols.push_back(oc);
        applied.emplace_back(i, s.pose);
      }
    }
    if (objs.empty()) return;
    if (!psi_->applyCollisionObjects(objs, cols))
      throw std::runtime_error("applyCollisionObjects failed for detected objects");
    std::lock_guard<std::mutex> lk(specs_mu_);
    for (const auto& a : applied) {
      specs_[a.first].in_scene = true;
      specs_[a.first].scene_pose = a.second;
    }
  }

  // -------------------------------------------------------------- scene setup
  void waitForBackend()
  {
    const auto timeout = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(startup_timeout_));
    auto get_scene = node_->create_client<moveit_msgs::srv::GetPlanningScene>("/get_planning_scene");
    auto apply_scene = node_->create_client<moveit_msgs::srv::ApplyPlanningScene>("/apply_planning_scene");
    auto execute = rclcpp_action::create_client<
      moveit_task_constructor_msgs::action::ExecuteTaskSolution>(node_, "/execute_task_solution");
    if (!get_scene->wait_for_service(timeout) || !apply_scene->wait_for_service(timeout) ||
        !execute->wait_for_action_server(timeout))
      throw std::runtime_error("MoveIt backend unavailable: start MTC demo backend with execution capability");
  }

  void setupScene()
  {
    psi_ = std::make_unique<moveit::planning_interface::PlanningSceneInterface>();
    const auto attached = psi_->getAttachedObjects();
    if (!attached.empty())
      throw std::runtime_error("attached object exists; inspect/recover the simulation before restarting");
    std::vector<std::string> stale;
    for (const auto& id : psi_->getKnownObjectNames()) {
      if (id.rfind("sorting_demo_obj", 0) == 0 || id == "sorting_demo_table") stale.push_back(id);
    }
    if (!stale.empty() && !reset_scene_)
      throw std::runtime_error("previous demo objects exist; inspect then use reset_demo_scene:=true");

    std::vector<moveit_msgs::msg::CollisionObject> objs;
    std::vector<moveit_msgs::msg::ObjectColor> cols;
    for (const auto& id : stale) {
      moveit_msgs::msg::CollisionObject o;
      o.id = id; o.operation = moveit_msgs::msg::CollisionObject::REMOVE;
      objs.push_back(o);
    }
    // A support surface is collision geometry, not a Gazebo rigid body.
    ObjSpec table;
    table.id = "sorting_demo_table";
    table.type = Prim::BOX;
    table.dims = {0.78, 1.20, 0.05};
    table.pose.orientation.w = 1.0;
    table.pose.position.x = 0.50;
    table.pose.position.z = -0.025;
    objs.push_back(toCollision(table));

    // fixed mode installs the configured objects up-front
    std::vector<std::string> ids;
    if (source_mode_ == "fixed") {
      std::lock_guard<std::mutex> lk(specs_mu_);
      for (auto& s : specs_) {
        objs.push_back(toCollision(s));
        moveit_msgs::msg::ObjectColor oc; oc.id = s.id; oc.color = s.color;
        cols.push_back(oc);
        s.in_scene = true;
        ids.push_back(s.id);
      }
    }
    if (!psi_->applyCollisionObjects(objs, cols))
      throw std::runtime_error("applyCollisionObjects failed; no motion attempted");
    if (!ids.empty() && psi_->getObjects(ids).size() != ids.size())
      throw std::runtime_error("planning scene verification failed");
    publishMarkers();
  }

  // --------------------------------------------------------------- run modes
  bool runFixedPass()
  {
    int ok_count = 0;
    for (std::size_t i = 0; i < specs_.size() && rclcpp::ok(); ++i) {
      active_ = static_cast<int>(i);
      active_id_ = specs_[i].id;
      pubI(idx_pub_, static_cast<int>(i));
      pubI(cyc_pub_, static_cast<int>(i) + 1);
      { std_msgs::msg::String m; m.data = specs_[i].cls; cls_pub_->publish(m); }
      publishMarkers();
      const auto place = slotPose(specs_[i].drink, specs_[i].cz);
      RCLCPP_INFO(node_->get_logger(), "cycle %zu/%zu: %s (%s) -> (%.2f, %.2f)",
                  i + 1, specs_.size(), specs_[i].cls.c_str(),
                  specs_[i].drink ? "LEFT" : "RIGHT", place.position.x, place.position.y);
      if (planAndExecute(specs_[i], place)) {
        (specs_[i].drink ? left_n_ : right_n_)++;
        specs_[i].placed = true;
        specs_[i].pose = place;
        ++ok_count;
        publishMarkers();
        RCLCPP_INFO(node_->get_logger(), "  placed %s", specs_[i].id.c_str());
      } else {
        RCLCPP_ERROR(node_->get_logger(),
          "FAILED %s: stopping; scene/attachment retained for inspection. No automatic detach.",
          specs_[i].id.c_str());
        return finish(ok_count);
      }
      std::this_thread::sleep_for(std::chrono::duration<double>(pause_));
    }
    return finish(ok_count);
  }

  bool runDetections()
  {
    // wait for the first stable detection
    const auto t0 = std::chrono::steady_clock::now();
    while (rclcpp::ok()) {
      { std::lock_guard<std::mutex> lk(specs_mu_);
        if (hasUnprocessedLocked()) break; }
      if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() > detect_wait_)
        break;
      std::this_thread::sleep_for(100ms);
    }
    { std::lock_guard<std::mutex> lk(specs_mu_);
      if (specs_.empty())
        RCLCPP_WARN(node_->get_logger(),
          "no detections on %s within %.1f s (is object_spawner / perception live3d running?)",
          det_topic_.c_str(), detect_wait_); }

    int ok_count = 0, idx_cycle = 0;
    double idle = 0.0;
    while (rclcpp::ok()) {
      int target = -1;
      { std::lock_guard<std::mutex> lk(specs_mu_);
        target = pickNearestLocked(); }
      if (target < 0) {
        if (!autonomous_) break;
        std::this_thread::sleep_for(200ms);
        idle += 0.2;
        if (idle > 3.0) break;           // nothing new for 3 s -> done
        continue;
      }
      idle = 0.0;
      ObjSpec snap;
      { std::lock_guard<std::mutex> lk(specs_mu_);
        snap = specs_[target];
        active_ = target; active_id_ = snap.id; }

      ensureInScene();
      // Tray capacity guard (4 slots per tray): skip/stop instead of throwing.
      if ((snap.drink ? left_n_ : right_n_) >= 4) {
        RCLCPP_ERROR(node_->get_logger(), "  %s tray full (4/4); %s cannot be placed",
                     snap.drink ? "LEFT" : "RIGHT", snap.id.c_str());
        { std::lock_guard<std::mutex> lk(specs_mu_);
          specs_[target].skipped = true;
          active_ = -1;                  // v0.4.2: drop the highlight of the skipped object
          active_id_.clear(); }
        if (!autonomous_) return finish(ok_count);
        publishMarkers();
        continue;
      }

      pubI(idx_pub_, ++idx_cycle - 1);
      pubI(cyc_pub_, idx_cycle);
      { std_msgs::msg::String m; m.data = snap.cls; cls_pub_->publish(m); }
      publishMarkers();

      const auto place = slotPose(snap.drink, snap.cz);
      RCLCPP_INFO(node_->get_logger(), "target %s (%s) -> (%.2f, %.2f)",
                  snap.id.c_str(), snap.drink ? "LEFT" : "RIGHT", place.position.x, place.position.y);
      bool okc = false;
      for (int k = 0; k < retry_max_ && rclcpp::ok() && !okc; ++k) {
        if (k > 0) RCLCPP_WARN(node_->get_logger(), "retry %d/%d for %s", k, retry_max_ - 1, snap.id.c_str());
        okc = planAndExecute(snap, place);
        if (execution_started_ && !okc) {
          RCLCPP_ERROR(node_->get_logger(), "execution attempted but not verified; stop for manual recovery");
          return finish(ok_count);
        }
      }
      bool stop = false;
      {
        // Scoped lock: publishMarkers()/finish() lock specs_mu_ themselves,
        // so they must be called AFTER this block (std::mutex is not recursive).
        std::lock_guard<std::mutex> lk(specs_mu_);
        ObjSpec& s = specs_[target];
        if (okc) {
          (s.drink ? left_n_ : right_n_)++;
          s.placed = true;
          s.pose = place;
          ++ok_count;
          RCLCPP_INFO(node_->get_logger(), "  placed %s", s.id.c_str());
        } else if (autonomous_) {
          s.skipped = true;              // leave in scene, no auto-detach
          RCLCPP_ERROR(node_->get_logger(), "  SKIP %s after %d tries (no automatic detach)",
                       s.id.c_str(), retry_max_);
        } else {
          RCLCPP_ERROR(node_->get_logger(),
            "FAILED %s: stopping; scene/attachment retained. No automatic detach.", s.id.c_str());
          stop = true;
        }
        active_ = -1;
        active_id_.clear();
      }
      if (stop) return finish(ok_count);
      publishMarkers();
      std::this_thread::sleep_for(std::chrono::duration<double>(pause_));
    }
    return finish(ok_count);
  }

  bool hasUnprocessedLocked() const
  {
    for (const auto& s : specs_)
      if (!s.placed && !s.skipped && s.seen >= stable_need_) return true;
    return false;
  }

  int pickNearestLocked() const
  {
    int best = -1; double bd = 0.0;
    const double ox = 0.42, oy = 0.0;
    for (std::size_t i = 0; i < specs_.size(); ++i) {
      const auto& s = specs_[i];
      if (s.placed || s.skipped || s.seen < stable_need_) continue;
      const double d = std::hypot(s.pose.position.x - ox, s.pose.position.y - oy);
      if (best < 0 || d < bd) { best = static_cast<int>(i); bd = d; }
    }
    return best;
  }

  bool finish(int ok_count)
  {
    std::size_t total;
    { // active_id_ is read by onDetections() on the executor thread -> lock
      std::lock_guard<std::mutex> lk(specs_mu_);
      active_ = -1;
      active_id_.clear();
      total = specs_.size(); }
    pubI(idx_pub_, -1);
    std_msgs::msg::String state;
    // v0.4.2: "DONE" only when at least one object existed and all were placed
    // (0/0 = nothing detected -> INTERRUPTED, consistent with the return value).
    const bool all_done = total > 0 && ok_count == static_cast<int>(total);
    state.data = all_done ? "DONE" : "INTERRUPTED";
    cls_pub_->publish(state);
    publishMarkers();
    RCLCPP_INFO(node_->get_logger(), "Sorting demo finished. placed %d/%zu", ok_count, total);
    return all_done;
  }

  // --------------------------------------------------------------- markers
  visualization_msgs::msg::Marker textMarker(int id, const std::string& ns, double x, double y,
                                             double z, const std::string& text)
  {
    visualization_msgs::msg::Marker m;
    m.header.frame_id = "world";
    m.header.stamp = node_->now();
    m.ns = ns; m.id = id;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = z;
    m.pose.orientation.w = 1.0;
    m.scale.z = 0.05;
    m.color = rgba(1, 1, 1, 1);
    m.text = text;
    return m;
  }

  void publishMarkers()
  {
    visualization_msgs::msg::MarkerArray arr;
    int id = 0;
    arr.markers.push_back(textMarker(id++, "trays", 0.38, kTrayY, 0.30, "LEFT: drinks"));
    arr.markers.push_back(textMarker(id++, "trays", 0.38, -kTrayY, 0.30, "RIGHT: food"));
    std::lock_guard<std::mutex> lk(specs_mu_);
    for (std::size_t i = 0; i < specs_.size(); ++i) {
      const auto& s = specs_[i];
      std::string txt = s.cls;
      if (s.placed) txt = s.cls + " #" + std::to_string(i + 1);
      else if (s.skipped) txt = s.cls + " (skip)";
      arr.markers.push_back(textMarker(id++, "labels", s.pose.position.x, s.pose.position.y,
                                       s.pose.position.z + s.cz + 0.08, txt));
    }
    visualization_msgs::msg::Marker h;
    h.header.frame_id = "world"; h.header.stamp = node_->now();
    h.ns = "active"; h.id = 999;
    h.type = visualization_msgs::msg::Marker::SPHERE;
    if (active_ >= 0 && active_ < static_cast<int>(specs_.size())) {
      const auto& s = specs_[static_cast<std::size_t>(active_)];
      h.action = visualization_msgs::msg::Marker::ADD;
      h.pose = s.pose;
      h.pose.orientation.x = 0.0; h.pose.orientation.y = 0.0;
      h.pose.orientation.z = 0.0; h.pose.orientation.w = 1.0;
      h.scale.x = h.scale.y = h.scale.z = 0.10;
      h.color = rgba(1, 1, 0, 0.5);
    } else {
      h.action = visualization_msgs::msg::Marker::DELETE;
    }
    arr.markers.push_back(h);
    marker_pub_->publish(arr);
  }

  // --------------------------------------------------------------- MTC plan
  bool planAndExecute(const ObjSpec& s, const geometry_msgs::msg::Pose& place)
  {
    execution_started_ = false;
    if (!psi_->getAttachedObjects().empty()) {
      execution_started_ = true;
      return false;
    }
    try {
      last_task_ = std::make_unique<mtc::Task>(createTask(s, place));
      auto& task = *last_task_;
      task.init();
      if (!task.plan(static_cast<std::size_t>(max_sol_)) || task.solutions().empty()) {
        RCLCPP_ERROR(node_->get_logger(), "plan failed: %s", s.id.c_str());
        return false;
      }
      task.introspection().publishSolution(*task.solutions().front());
      execution_started_ = true;
      const auto result = task.execute(*task.solutions().front());
      if (result.val != moveit_msgs::msg::MoveItErrorCodes::SUCCESS) {
        RCLCPP_ERROR(node_->get_logger(), "execute failed (code %d): %s",
                     static_cast<int>(result.val), s.id.c_str());
        return false;
      }
      const auto poses = psi_->getObjectPoses({s.id});
      const auto found = poses.find(s.id);
      if (!psi_->getAttachedObjects({s.id}).empty() || found == poses.end())
        throw std::runtime_error("post-execution object missing or still attached");
      const auto& p = found->second.position;
      const auto& q = place.position;
      if (std::hypot(std::hypot(p.x - q.x, p.y - q.y), p.z - q.z) > 0.015)
        throw std::runtime_error("post-execution position differs from requested place pose");
      return true;
    } catch (const mtc::InitStageException& e) {
      RCLCPP_ERROR_STREAM(node_->get_logger(), e);
    } catch (const std::exception& e) {
      RCLCPP_ERROR(node_->get_logger(), "task exception: %s", e.what());
    }
    return false;
  }

  mtc::Task createTask(const ObjSpec& s, const geometry_msgs::msg::Pose& place)
  {
    const std::string arm = kArm, hand = kHand, hframe = kHandFrame;
    mtc::Task task;
    task.stages()->setName("sort " + s.id);
    task.loadRobotModel(node_);
    if (!task.getRobotModel()->getJointModelGroup(arm) ||
        !task.getRobotModel()->getJointModelGroup(hand) ||
        !task.getRobotModel()->getLinkModel(hframe) ||
        (task.getRobotModel()->getModelFrame() != "world"))
      throw std::runtime_error("expected Panda groups/frames missing; use matching Panda MoveIt config");
    task.setProperty("group", arm);
    task.setProperty("eef", hand);
    task.setProperty("ik_frame", hframe);

    mtc::Stage* cur = nullptr;
    {
      auto st = std::make_unique<mtc::stages::CurrentState>("current");
      cur = st.get();
      task.add(std::move(st));
    }
    auto sampling = std::make_shared<mtc::solvers::PipelinePlanner>(node_);
    auto interp = std::make_shared<mtc::solvers::JointInterpolationPlanner>();
    auto cart = std::make_shared<mtc::solvers::CartesianPath>();
    sampling->setMaxVelocityScalingFactor(vel_);
    sampling->setMaxAccelerationScalingFactor(acc_);
    cart->setMaxVelocityScalingFactor(vel_);
    cart->setMaxAccelerationScalingFactor(acc_);
    cart->setStepSize(0.005);
    interp->setMaxVelocityScalingFactor(vel_);
    interp->setMaxAccelerationScalingFactor(acc_);

    {
      auto st = std::make_unique<mtc::stages::MoveTo>("open hand", interp);
      st->setGroup(hand); st->setGoal("open"); task.add(std::move(st));
    }
    {
      auto st = std::make_unique<mtc::stages::Connect>(
        "move to pick", mtc::stages::Connect::GroupPlannerVector{{arm, sampling}});
      st->setTimeout(8.0);
      st->properties().configureInitFrom(mtc::Stage::PARENT);
      task.add(std::move(st));
    }

    mtc::Stage* attach = nullptr;
    {
      auto pick = std::make_unique<mtc::SerialContainer>("pick");
      task.properties().exposeTo(pick->properties(), {"eef", "group", "ik_frame"});
      pick->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group", "ik_frame"});
      {
        auto st = std::make_unique<mtc::stages::MoveRelative>("approach", cart);
        st->properties().set("marker_ns", "approach");
        st->properties().set("link", hframe);
        st->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        st->setMinMaxDistance(0.08, 0.14);
        geometry_msgs::msg::Vector3Stamped d;
        d.header.frame_id = hframe; d.vector.z = 1.0;
        st->setDirection(d);
        pick->insert(std::move(st));
      }
      {
        auto gen = std::make_unique<mtc::stages::GenerateGraspPose>("grasp pose");
        gen->properties().configureInitFrom(mtc::Stage::PARENT);
        gen->properties().set("marker_ns", "grasp");
        gen->setPreGraspPose("open");
        gen->setObject(s.id);
        gen->setAngleDelta(M_PI / 12.0);
        gen->setMonitoredStage(cur);
        Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
        t.linear() = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY()).toRotationMatrix();
        // Height is derived from the actual primitive, not an ID substring table.
        // This is a candidate calibration, NOT a proof of collision-free grasping.
        t.translation().z() = grasp_offset_mode_ == "baseline" ? 0.17 :
          std::clamp(2.0 * s.cz - 0.01, 0.05, 0.20);
        auto ik = std::make_unique<mtc::stages::ComputeIK>("grasp IK", std::move(gen));
        ik->setMaxIKSolutions(8);
        ik->setMinSolutionDistance(0.5);
        ik->setIKFrame(t, hframe);
        ik->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group"});
        ik->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose"});
        pick->insert(std::move(ik));
      }
      {
        auto st = std::make_unique<mtc::stages::ModifyPlanningScene>("allow hand-obj");
        st->allowCollisions(
          s.id,
          task.getRobotModel()->getJointModelGroup(hand)->getLinkModelNamesWithCollisionGeometry(),
          true);
        st->allowCollisions(s.id, "sorting_demo_table", true);
        pick->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::MoveTo>("close hand", interp);
        st->setGroup(hand); st->setGoal("close");
        pick->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::ModifyPlanningScene>("attach");
        st->attachObject(s.id, hframe);
        attach = st.get();
        pick->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::MoveRelative>("lift", cart);
        st->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        st->setMinMaxDistance(0.10, 0.20);
        st->setIKFrame(hframe);
        st->properties().set("marker_ns", "lift");
        geometry_msgs::msg::Vector3Stamped d;
        d.header.frame_id = "world"; d.vector.z = 1.0;
        st->setDirection(d);
        pick->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::ModifyPlanningScene>("clear table after lift");
        st->allowCollisions(s.id, "sorting_demo_table", false);
        pick->insert(std::move(st));
      }
      task.add(std::move(pick));
    }

    {
      auto st = std::make_unique<mtc::stages::Connect>(
        "move to place",
        mtc::stages::Connect::GroupPlannerVector{{arm, sampling}, {hand, interp}});
      st->setTimeout(8.0);
      st->properties().configureInitFrom(mtc::Stage::PARENT);
      task.add(std::move(st));
    }

    {
      auto place_c = std::make_unique<mtc::SerialContainer>("place");
      task.properties().exposeTo(place_c->properties(), {"eef", "group", "ik_frame"});
      place_c->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group", "ik_frame"});
      {
        auto st = std::make_unique<mtc::stages::ModifyPlanningScene>("allow placement contact");
        st->allowCollisions(s.id, "sorting_demo_table", true);
        place_c->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::MoveRelative>("lower to place", cart);
        st->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        st->setIKFrame(hframe);
        st->setMinMaxDistance(0.08, 0.12);
        geometry_msgs::msg::Vector3Stamped d;
        d.header.frame_id = "world"; d.vector.z = -1.0;
        st->setDirection(d);
        place_c->insert(std::move(st));
      }
      {
        auto gen = std::make_unique<mtc::stages::GeneratePlacePose>("place pose");
        gen->properties().configureInitFrom(mtc::Stage::PARENT);
        gen->properties().set("marker_ns", "place");
        gen->setObject(s.id);
        geometry_msgs::msg::PoseStamped pp;
        pp.header.frame_id = "world";
        pp.pose = place;
        pp.pose.orientation.w = 1.0;
        gen->setPose(pp);
        gen->setMonitoredStage(attach);
        auto ik = std::make_unique<mtc::stages::ComputeIK>("place IK", std::move(gen));
        ik->setMaxIKSolutions(4);
        ik->setMinSolutionDistance(0.5);
        ik->setIKFrame(s.id);
        ik->properties().configureInitFrom(mtc::Stage::PARENT, {"eef", "group"});
        ik->properties().configureInitFrom(mtc::Stage::INTERFACE, {"target_pose"});
        place_c->insert(std::move(ik));
      }
      {
        auto st = std::make_unique<mtc::stages::MoveTo>("open hand", interp);
        st->setGroup(hand); st->setGoal("open");
        place_c->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::ModifyPlanningScene>("detach");
        st->detachObject(s.id, hframe);
        place_c->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::ModifyPlanningScene>("forbid hand-obj");
        st->allowCollisions(s.id, "sorting_demo_table", false);
        st->allowCollisions(
          s.id,
          task.getRobotModel()->getJointModelGroup(hand)->getLinkModelNamesWithCollisionGeometry(),
          false);
        place_c->insert(std::move(st));
      }
      {
        auto st = std::make_unique<mtc::stages::MoveRelative>("retreat", cart);
        st->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
        st->setMinMaxDistance(0.08, 0.16);
        st->setIKFrame(hframe);
        st->properties().set("marker_ns", "retreat");
        geometry_msgs::msg::Vector3Stamped d;
        d.header.frame_id = "world"; d.vector.z = 1.0;
        st->setDirection(d);
        place_c->insert(std::move(st));
      }
      task.add(std::move(place_c));
    }

    {
      auto st = std::make_unique<mtc::stages::MoveTo>("return ready", sampling);
      st->properties().configureInitFrom(mtc::Stage::PARENT, {"group"});
      st->setGroup(arm); st->setGoal("ready");
      task.add(std::move(st));
    }
    return task;
  }

  // --------------------------------------------------------------- members
  rclcpp::Node::SharedPtr node_;
  std::unique_ptr<moveit::planning_interface::PlanningSceneInterface> psi_;
  std::unique_ptr<mtc::Task> last_task_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr idx_pub_, cyc_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr cls_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Subscription<vision_msgs::msg::Detection3DArray>::SharedPtr det_sub_;
  std::mutex specs_mu_;
  std::vector<ObjSpec> specs_;
  std::map<std::string, std::size_t> index_;
  std::string active_id_;
  std::string grasp_offset_mode_{"baseline"};
  std::string source_mode_{"fixed"}, det_topic_{"/perception/detections_3d"};
  bool reset_scene_{false}, autonomous_{false};
  bool execution_started_{false};
  double startup_timeout_{30.0}, detect_wait_{60.0};
  int left_n_{0}, right_n_{0}, active_{-1}, max_sol_{5}, retry_max_{3}, stable_need_{1};
  double vel_{0.2}, acc_{0.2}, pause_{1.0};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  int result = 1;
  rclcpp::executors::MultiThreadedExecutor executor;
  std::thread spin_thread;
  std::shared_ptr<MTCTaskNode> task_node;
  try {
    rclcpp::NodeOptions options;
    options.automatically_declare_parameters_from_overrides(true);
    task_node = std::make_shared<MTCTaskNode>(options);
    executor.add_node(task_node->getNodeBaseInterface());
    spin_thread = std::thread([&executor]() {
      try { executor.spin(); }
      catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("mtc_node"), "executor exception: %s", e.what());
        rclcpp::shutdown();
      }
    });
    result = task_node->run() ? 0 : 1;
    if (result == 0) {
      RCLCPP_INFO(rclcpp::get_logger("mtc_node"),
        "DONE. Keeping final markers/state available; Ctrl-C to exit.");
      while (rclcpp::ok()) std::this_thread::sleep_for(100ms);
    }
    executor.cancel();
    if (spin_thread.joinable()) spin_thread.join();
    executor.remove_node(task_node->getNodeBaseInterface());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(rclcpp::get_logger("mtc_node"), "%s", e.what());
  } catch (...) {
    RCLCPP_ERROR(rclcpp::get_logger("mtc_node"), "unknown exception");
  }
  executor.cancel();
  if (spin_thread.joinable()) spin_thread.join();
  if (rclcpp::ok()) rclcpp::shutdown();
  return result;
}

