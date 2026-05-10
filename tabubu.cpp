#include <bits/stdc++.h>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <algorithm>
#include <random>
#include <optional>

#define ll long long
#define pb push_back
#define mp make_pair
#define pii pair<int,int>
#define vi vector<int>
#define vd vector<double>
#define vvi vector<vector<int>>
#define vvd vector<vector<double>>
#define vpi vector<pair<int,int>>
#define all(v) v.begin(),v.end()
#define FOR(i,a,b) for(int i=a;i<=b;i++)
#define RFOR(i,a,b) for(int i=a-1;i>=b;i--)

using namespace std;

// Data structures and global variables
struct Point {
    double x = 0.0, y = 0.0;
    int id = -1;
    Point() = default;
    Point(double x_, double y_, int id_ = -1) : x(x_), y(y_), id(id_) {}
};

int n, h, d; //number of customers, number of trucks, number of drones
vector<Point> loc; // loc[i]: location (x, y) of customer i, if i = 0, it is depot
vd serve_truck, serve_drone; // time taken by truck and drone to serve each customer (seconds)
vi served_by_drone; //whether each customer can be served by drone or not, 1 if yes, 0 if no
vd deadline; //customer deadlines
vd demand; // demand[i]: demand of customer i
double Dh = 500.0; // truck capacity (all trucks) (kg)
double vmax = 15.6464; // truck base speed (m/s)
int L = 24; //number of time segments in a day
//vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
//vd time_segments_sigma = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; //sigma (truck velocity coefficient) for each time segments
vd time_segment = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}; // time segment boundaries in hours
vd time_segments_sigma = {0.9, 0.8, 0.4, 0.6,0.9, 0.8, 0.6, 0.8, 0.8, 0.7, 0.5, 0.8}; //sigma (truck velocity coefficient) for each time segments
double Dd = 2.27, E = 7200000.0; //drone's weight and energy capacities (for all drones)
double v_fly_drone = 31.3, v_take_off = 15.6, v_landing = 7.8; // speed of the drone
double height = 50; // height of the drone
//double height = 0; // height of the drone
//double power_beta = 0, power_gamma = 1.0; //coefficients for drone energy consumption per second
double power_beta = 24.2, power_gamma = 1329.0; //coefficients for drone energy consumption per second
vvd distance_matrix; //distance matrices for truck and drone

// Candidate lists (k-nearest neighbors) to filter neighborhood evaluations
static int CFG_KNN_K = 1000;           // number of nearest neighbors per customer
static int CFG_KNN_WINDOW = 1;       // insertion window around candidate anchors
static vvi KNN_LIST;                 // KNN_LIST[i] = up to K nearest neighbor customer ids for i (exclude depot 0)
static vector<vector<char>> KNN_ADJ; // KNN_ADJ[i][j] = 1 if j in KNN_LIST[i]

// Simple tabu structure for relocate moves: tabu_list_switch[cust][target_vehicle] stores iteration until which move is tabu
// target_vehicle is 0..h-1 for trucks, h..h+d-1 for drones
static vector<vector<int>> tabu_list_switch; // sized (n+1) x (h + d), initialized on first use
static int TABU_TENURE_BASE = 0; // default tenure; actual update done in tabu loop (not here)
// Separate tabu structure for swap moves: store until-iteration for swapping a pair (min_id,max_id)
static vector<vector<int>> tabu_list_10; // sized (n+1) x (n+1)
static int TABU_TENURE_10 = 0; // default tenure for swap moves
static vector<vector<int>> tabu_list_11; // sized (n+1) x (h + d)
static int TABU_TENURE_11 = 0; // default tenure for relocate moves
// Separate tabu list for intra-route reinsert (Or-opt-1) moves
static map<vector<int>, int> tabu_list_20; // keyed by (cust_id1, cust_id2, vehicle_id)
static int TABU_TENURE_20 = 0; // default tenure for reinsert moves
// Separate tabu list for 2-opt moves: keyed by segment endpoints (min_id,max_id)
static vector<vector<int>> tabu_list_2opt; // sized (n+1) x (n+1) 
static int TABU_TENURE_2OPT = 0; // default tenure for 2-opt moves
static vector<vector<int>> tabu_list_2opt_star; // sized (n+1) x (n+1)
static int TABU_TENURE_2OPT_STAR = 0; // default tenure for 2-opt-star moves
static map<vector<int>, int> tabu_list_21; // keyed by (a,b,c,d) for (2,1) moves
static int TABU_TENURE_21 = 0; // default tenure for (2,1) moves
static map<vector<int>, int> tabu_list_22; // keyed by (a,b,c,d) for (2,2) moves
static int TABU_TENURE_22 = 0; // default tenure for (2,2) moves
static map<vector<int>, int> tabu_list_ejection; // keyed by sorted customer sequence
static int TABU_TENURE_EJECTION = 0; // default tenure for ejection chain moves
const int NUM_NEIGHBORHOODS = 9;
const int NUM_OF_INITIAL_SOLUTIONS = 200;
const int MAX_SEGMENT = 200;
const int MAX_NO_IMPROVE = 1000;
const int MAX_ITER_PER_SEGMENT = 1000;
const double gamma1 = 0.5;
const double gamma2 = 0.3;
const double gamma3 = 0.1;
const double gamma4 = 0.3;

// Runtime-configurable search knobs (initialized from compile-time defaults)
static int CFG_NUM_INITIAL = NUM_OF_INITIAL_SOLUTIONS;
static int CFG_MAX_SEGMENT = MAX_SEGMENT;
static int CFG_MAX_NO_IMPROVE = MAX_NO_IMPROVE;
static int CFG_MAX_ITER_PER_SEGMENT = MAX_ITER_PER_SEGMENT;
static double CFG_TIME_LIMIT_SEC = 0.0; // 0 = unlimited

// Adaptive penalty coefficients for constraint violations
static double PENALTY_LAMBDA_CAPACITY = 1.0;      // λ for capacity violations
static double PENALTY_LAMBDA_ENERGY = 1.0;        // λ for energy violations  
static double PENALTY_LAMBDA_DEADLINE = 1.0;      // λ for deadline violations
static double PENALTY_EXPONENT = 0.5;             // exponent for penalty term *

static const double PENALTY_INCREASE = 1.2;       // multiply when violated *
static const double PENALTY_DECREASE = 1.2;       // divide when satisfied *
static const double PENALTY_MIN = 0.5;            // minimum λ value
static const double PENALTY_MAX = 1000.0;

static const double T0 = 150.0; // initial temperature for simulated annealing acceptance
double alpha = 0.998; // cooling rate for simulated annealing

// Destroy and repair helper
vvd edge_records; // edge_records[i][j]: stores working times for edge (i,j)
const double DESTROY_RATE = 0.3; // fraction of customers to remove during destroy phase

struct Solution {
    vvi truck_routes; //truck_routes[i]: sequence of customers served by truck i
    vvi drone_routes; //drone_routes[i]: sequence of customers served by drone i
    vd truck_route_times; //truck_route_times[i]: total time of truck i
    vd drone_route_times; //drone_route_times[i]: total time of drone i
    double total_makespan; //total makespan of the solution
    double capacity_violation = 0.0;    // sum of excess capacity / total capacity
    double energy_violation = 0.0;      // sum of excess energy / total battery
    double deadline_violation = 0.0;    // sum of deadline breaches / total deadlines
};

vector<Solution> elite_set; //store most promising solutions
const int ELITE_SET_SIZE = 10;

// Helper to parse key=value flags from argv
static bool parse_kv_flag(const std::string& s, const std::string& key, std::string& out) {
    if (s.rfind(key + "=", 0) == 0) { out = s.substr(key.size() + 1); return true; }
    return false;
}

// Build k-nearest neighbor lists based on Euclidean distance_matrix.
// Excludes depot (0) and self; sizes to n+1. Also builds adjacency for O(1) membership checks.
static void compute_knn_lists(int k) {
    int N = n;
    if (N <= 1) {
        KNN_LIST.assign(N + 1, {});
        KNN_ADJ.assign(N + 1, vector<char>(N + 1, 0));
        return;
    }
    KNN_LIST.assign(N + 1, {});
    KNN_ADJ.assign(N + 1, vector<char>(N + 1, 0));
    vector<pair<double,int>> cand;
    cand.reserve(max(0, N - 1));
    for (int i = 1; i <= N; ++i) {
        cand.clear();
        for (int j = 1; j <= N; ++j) {
            if (j == i) continue;
            cand.emplace_back(distance_matrix[i][j], j);
        }
        int kk = min(k, (int)cand.size());
        if ((int)cand.size() > kk) {
            nth_element(cand.begin(), cand.begin() + kk, cand.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            cand.resize(kk);
        } else {
            sort(cand.begin(), cand.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
        }
        KNN_LIST[i].reserve(kk);
        for (int t = 0; t < kk; ++t) {
            int j = cand[t].second;
            KNN_LIST[i].push_back(j);
            KNN_ADJ[i][j] = 1;
        }
    }
}


// Separate tabu list for 2-opt-star (inter-route exchange) moves: keyed by unordered edge endpoints (min(u,v), max(u,v))


void input(string filepath){
        // Open the file
        ifstream fin(filepath);
        if (!fin) {
            cerr << "Error: Cannot open " << filepath << endl;
            exit(1);
        }
        string line;
        n = h = d = -1;
        // Read trucks_count, drones_count, customers
        while (getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            stringstream ss(line);
            string key;
            ss >> key;
            if (key == "trucks_count") ss >> h;
            else if (key == "drones_count") ss >> d;
            else if (key == "customers") ss >> n;
            else if (key == "depot") break;
        }
        // Read depot location
        double depot_x = 0, depot_y = 0;
        stringstream ss_depot(line);
        string depot_key;
        ss_depot >> depot_key >> depot_x >> depot_y;
    // Prepare storage (use assign to ensure inner dimensions reset, avoiding stale sizes across batch runs)
    served_by_drone.assign(n+1, 0);
    serve_truck.assign(n+1, 0.0);
    serve_drone.assign(n+1, 0.0);
    deadline.assign(n+1, 0.0);
    demand.assign(n+1, 0.0);
    loc.assign(n+1, Point());
    distance_matrix.assign(n+1, vd(n+1, 0.0));
    loc[0] = {depot_x, depot_y, 0};
    // Skip headers until data lines
    int header_skips = 0;
    while (header_skips < 2 && getline(fin, line)) {
        if (!line.empty() && line[0] != '#') ++header_skips;
    }
    // Read customer data
    int cust = 1;
    while (cust <= n && getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        stringstream ss(line);
        double x, y, dronable, demand_val, drone_service, truck_service, deadline_val;
        ss >> x >> y >> dronable >> demand_val >> drone_service >> truck_service >> deadline_val;
        loc[cust] = {x, y, cust};
        served_by_drone[cust] = (int)dronable;
        demand[cust] = demand_val;
        serve_drone[cust] = drone_service;
        serve_truck[cust] = truck_service;
        deadline[cust] = deadline_val;
        ++cust;
    }
}

void update_tabu_tenures() {
    // Base heuristic: roughly proportional to sqrt(n) or n/5
    // sqrt(n) scales better for very large instances
    int base = max(20, (int)(2.0 * sqrt(n))); 
    
    TABU_TENURE_BASE = base;
    TABU_TENURE_10 = base;          // Swap
    TABU_TENURE_11 = base;          // Relocate
    TABU_TENURE_20 = base;          // Or-opt
    TABU_TENURE_2OPT = (int)(base * 1.2); // 2-opt usually benefits from slightly longer memory
    TABU_TENURE_2OPT_STAR = base; 
    TABU_TENURE_21 = base;
    TABU_TENURE_22 = base;
    TABU_TENURE_EJECTION = max(30, (int)(base * 2.0)); // Ejection chains are disruptive, keep longer
    
    /* cout << "Dynamic Tabu Tenures set to: " << base 
         << " (2-opt: " << TABU_TENURE_2OPT 
         << ", Ejection: " << TABU_TENURE_EJECTION << ")" << endl; */
}

// Returns pair of distance matrices
void compute_distance_matrices(const vector<Point>& loc) {
    int n = loc.size() - 1; // assuming loc[0] is depot
    for (int i = 0; i <= n; ++i) {
        for (int j = 0; j <= n; ++j) {
            distance_matrix[i][j] = sqrt((loc[i].x - loc[j].x) * (loc[i].x - loc[j].x)
                                         + (loc[i].y - loc[j].y) * (loc[i].y - loc[j].y)); // Euclidean
        }
    }
}

// Helper: get time segment index for a given time t (in hours)
int get_time_segment(double t) {
    // t is in hours. Use custom time_segment boundaries (in hours):
    // time_segment: [b0, b1, ..., bk] defines k segments [b0,b1), [b1,b2), ... [b{k-1}, b{k}]
    // Return 0-based segment index in [0, k-1].
    // If outside boundaries, loop back to the start segment.
    t = fmod(t, 12.0);
    if (time_segment.size() < 2) return 0;
    // Find first boundary strictly greater than t
    auto it = upper_bound(time_segment.begin(), time_segment.end(), t);
    int idx = static_cast<int>(it - time_segment.begin()) - 1; // index of segment start
    if (idx < 0) idx = 0;
    int max_idx = static_cast<int>(time_segment.size()) - 2; // last valid segment index
    if (idx > max_idx) idx = max_idx;
    return idx;
}

pair<double, double> compute_truck_route_time(const vi& route, double start=0) {
    double time = start; // seconds
    double deadline_feasible = 0.0;
    // Index by customer id (0..n), not by position in route, to avoid out-of-bounds writes
    vector<double> visit_times(n+1, 0.0); // visit_times[id]: time when node id is last visited
    vector<int> customers_since_last_depot;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        double dist_left = distance_matrix[from][to]; // meters
        // Defensive: cap number of segment steps to avoid infinite loops due to numeric edge cases
        int guard_steps = 0;
        while (dist_left > 1e-8) {
            if (++guard_steps > 1000000) {
                // Fallback: assume constant speed and finish remaining distance
                double v_safe = vmax > 1e-6 ? vmax : 1.0;
                time += dist_left / v_safe;
                dist_left = 0.0;
                break;
            }
            // Convert time to hours for segment lookup
            double t_hr = time / 3600.0;
            int seg = get_time_segment(t_hr); // 0-based index into time_segments_sigma
            double v = vmax * (seg < (int)time_segments_sigma.size() ? time_segments_sigma[seg] : 1.0); // m/s
            if (v <= 1e-8) v = vmax;
            // Time left in this custom segment (seconds to next boundary)
            double next_boundary_hr = (seg + 1 < (int)time_segment.size()) ? time_segment[seg + 1] : std::numeric_limits<double>::infinity();
            double segment_end_time_sec;
            if (std::isinf(next_boundary_hr)) segment_end_time_sec = time + 1e18; // effectively no boundary ahead
            else segment_end_time_sec = next_boundary_hr * 3600.0;
            double t_seg_end = segment_end_time_sec - time; // seconds remaining in this segment
            if (t_seg_end < 1e-8) t_seg_end = 1e-6; // minimal progress to avoid stalling
            double max_dist_this_seg = v * t_seg_end;
            if (max_dist_this_seg <= 1e-12) {
                // Make minimal forward progress to avoid stalling due to underflow
                max_dist_this_seg = std::max(1e-6, v * 1e-6);
            }
            if (dist_left <= max_dist_this_seg) {
                double t_needed = dist_left / v;
                time += t_needed;
                dist_left = 0;
            } else {
                time += t_seg_end;
                dist_left -= max_dist_this_seg;
            }
        }
        if (to != 0) {
            time += serve_truck[to]; // in seconds
            customers_since_last_depot.push_back(to);
        }
        visit_times[to] = time; // record departure from node 'to'
        // If we reach depot (except at start), check duration from leaving each customer to depot
        if (to == 0 && k != 1) {
            for (int cust : customers_since_last_depot) {
                // Duration from leaving customer to returning to depot
                double duration = time - visit_times[cust];
                if (duration > deadline[cust] + 1e-8) {
                    double deadline_norm = (deadline[cust] > 1e-6) ? deadline[cust] : 1.0;
                    deadline_feasible += (duration - deadline[cust]) / deadline_norm;
                }
            }
            // After returning to depot, reset visit times for customers
            for (int cust : customers_since_last_depot) {
                visit_times[cust] = time;
            }
            customers_since_last_depot.clear();
        }
    }
    return {time - start, deadline_feasible};
}

pair<double, double> compute_drone_route_energy(const vi& route) {
    double total_energy = 0, current_weight = 0;
    double energy_used = 0;
    double feasible = 0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        if (from == to) continue;
        double dist = distance_matrix[from][to]; // meters
        double v = v_fly_drone; // assume constant speed for simplicity
        double time = dist / v; // seconds
        time += height / v_take_off; // take-off time
        time += height / v_landing; // landing time
        // Energy consumption model: power = beta * weight + gamma
        double power = power_beta * (current_weight) + power_gamma; // watts
        energy_used += power * time; // energy in joules
        total_energy += power * time;
        if (energy_used > E + 1e-8) feasible += energy_used - E; // exceeded energy
        if (to != 0) current_weight += demand[to]; // add payload when delivering
        else {
            current_weight = 0; // reset weight when returning to depot
            energy_used = 0; // reset energy (charged at depot)
        }
    }
    return make_pair(total_energy, feasible);
}

pair<double, double> compute_drone_route_time(const vi& route) {
    double time = 0; // seconds
    double deadline_feasible = 0.0;
    // Index by customer id (0..n), not by position in route
    vector<double> visit_times(n+1, 0.0); // visit_times[id]: time when node id is last visited
    vector<int> customers_since_last_depot;
    for (int k = 1; k < (int)route.size(); ++k) {
        int from = route[k-1], to = route[k];
        if (from == to) continue;
        double dist = distance_matrix[from][to]; // meters
        double v = v_fly_drone; // assume constant speed for simplicity
        if (v <= 1e-8) v = v_fly_drone;
        double t = dist / v; // seconds
        t += height / v_take_off; // take-off time
        t += height / v_landing; // landing time
        time += t;
        if (to != 0) {
            time += serve_drone[to]; // in seconds
            customers_since_last_depot.push_back(to);
        }
        visit_times[to] = time;
        // If we reach depot (except at start), check duration from leaving each customer to depot
        if (to == 0 && k != 1) {
            for (int cust : customers_since_last_depot) {
                double duration = time - visit_times[cust];
                if (duration > deadline[cust] + 1e-8) {
                    double deadline_norm = (deadline[cust] > 1e-6) ? deadline[cust] : 1.0;
                    deadline_feasible += (duration - deadline[cust]) / deadline_norm;
                }
            }
            for (int cust : customers_since_last_depot) {
                visit_times[cust] = time;
            }
            customers_since_last_depot.clear();
        }
    }
    return {time, deadline_feasible};
}



void update_served_by_drone() {
    int depot = 0;
    for (int customer = 1; customer <= n; ++customer) {
        if (served_by_drone[customer] == 0) continue;
        // Capacity: demand[customer] <= Dd
        if (demand[customer] > Dd) {
            served_by_drone[customer] = 0;
            continue;
        }
        // Energy: use compute_drone_route_energy for depot->customer->depot
        vi route = {depot, customer, depot};
        auto [total_energy, energy_violation] = compute_drone_route_energy(route);
        if (energy_violation > 1e-8) {
            served_by_drone[customer] = 0;
            continue;
        }
        // Deadline: use compute_drone_route_time for depot->customer->depot
        auto [total_time, feasible_deadline] = compute_drone_route_time(route);
        if (feasible_deadline > 1e-8) {
            served_by_drone[customer] = 0;
            continue;
        }
        served_by_drone[customer] = 1;
    }
}

// Returns: [route_time, deadline_violation, energy_violation, capacity_violation]
vector<double> check_truck_route_feasibility(const vi& route, double start=0) {
    // Check deadlines
    auto [time, deadline_violation] = compute_truck_route_time(route, start);
    
    // Check capacity (reset at depot)
    double capacity_violation = 0.0;
    double total_demand = 0.0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            total_demand += demand[customer];
            if (total_demand > Dh + 1e-9) {
                capacity_violation += (total_demand - Dh) / Dh; // normalized excess
            }
        }
    }
    
    // Trucks don't use energy (set to 0)
    double energy_violation = 0.0;
    
    return {time, deadline_violation, energy_violation, capacity_violation};
}

// Returns: [route_time, deadline_violation, energy_violation, capacity_violation]
vector<double> check_drone_route_feasibility(const vi& route) {
    // Check deadlines
    auto [time, deadline_violation] = compute_drone_route_time(route);
    
    // Check capacity (reset at depot)
    double capacity_violation = 0.0;
    double total_demand = 0.0;
    for (int k = 1; k < (int)route.size(); ++k) {
        int customer = route[k];
        if (customer == 0) {
            total_demand = 0.0;
        } else {
            total_demand += demand[customer];
            if (total_demand > Dd + 1e-9) {
                capacity_violation += (total_demand - Dd) / Dd; // normalized excess
            }
        }
    }
    
    // Check energy
    auto energy_metrics = compute_drone_route_energy(route);
    double energy_violation = max(0.0, energy_metrics.second / E); // normalized excess beyond battery per sortie
    
    return {time, deadline_violation, energy_violation, capacity_violation};
}

// Unified wrapper
vector<double> check_route_feasibility(const vi& route, double start=0, bool is_truck = true) {
    if (is_truck) {
        return check_truck_route_feasibility(route, start);
    } else {
        return check_drone_route_feasibility(route);
    }
}

//score calculator
double solution_score_l2_norm(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    double sum_sq = 0.0;
    for (double t : sol.truck_route_times) sum_sq += t * t;
    for (double t : sol.drone_route_times) sum_sq += t * t;
    double l2_norm = std::sqrt(sum_sq);

    // 1e-3 ensures it acts as a tie-breaker without overriding the primary Makespan objective
    return (sol.total_makespan + l2_norm * 1e-3) * pow(penalty_multiplier, PENALTY_EXPONENT);
}

double solution_score_makespan(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    return (sol.total_makespan) * pow(penalty_multiplier, PENALTY_EXPONENT);
}

double solution_score_total_time(const Solution& sol) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * sol.capacity_violation
                                + PENALTY_LAMBDA_ENERGY * sol.energy_violation
                                + PENALTY_LAMBDA_DEADLINE * sol.deadline_violation;
    double sum_sq = 0.0;
    for (double t : sol.truck_route_times) sum_sq += t * t;
    for (double t : sol.drone_route_times) sum_sq += t * t;
    double l2_norm = std::sqrt(sum_sq);

    return (l2_norm) * pow(penalty_multiplier, PENALTY_EXPONENT);
}

double calculate_score_with_penalties(const double makespan, const double sum_sq, const double capacity_violation, const double energy_violation, const double deadline_violation) {
    double penalty_multiplier = 1.0 + PENALTY_LAMBDA_CAPACITY * capacity_violation
                                + PENALTY_LAMBDA_ENERGY * energy_violation
                                + PENALTY_LAMBDA_DEADLINE * deadline_violation;
    return (makespan + 1e-4 * std::sqrt(sum_sq / (h + d))) * pow(penalty_multiplier, PENALTY_EXPONENT);
}

void update_penalties(const Solution& sol) {
    // Capacity penalty
    if (sol.capacity_violation > 1e-9) {
        PENALTY_LAMBDA_CAPACITY = min(PENALTY_MAX, PENALTY_LAMBDA_CAPACITY * PENALTY_INCREASE);
    } else {
        PENALTY_LAMBDA_CAPACITY = max(PENALTY_MIN, PENALTY_LAMBDA_CAPACITY / PENALTY_DECREASE);
    }
    
    // Energy penalty
    if (sol.energy_violation > 1e-9) {
        PENALTY_LAMBDA_ENERGY = min(PENALTY_MAX, PENALTY_LAMBDA_ENERGY * PENALTY_INCREASE);
    } else {
        PENALTY_LAMBDA_ENERGY = max(PENALTY_MIN, PENALTY_LAMBDA_ENERGY / PENALTY_DECREASE);
    }
    
    // Deadline penalty
    if (sol.deadline_violation > 1e-9) {
        PENALTY_LAMBDA_DEADLINE = min(PENALTY_MAX, PENALTY_LAMBDA_DEADLINE * PENALTY_INCREASE);
    } else {
        PENALTY_LAMBDA_DEADLINE = max(PENALTY_MIN, PENALTY_LAMBDA_DEADLINE / PENALTY_DECREASE);
    }
}

vvi kmeans_clustering(int k, int max_iters=1000, uint64_t seed=UINT64_MAX) {
    if (n <= 0) return {};
    // Bound k to [1, n]
    if (k <= 0) k = 1;
    if (k > n) k = n;

    vvi clusters(k);
    vector<Point> centroids;
    centroids.reserve(k);

    // Random engine
    std::mt19937 gen(seed == UINT64_MAX ? std::random_device{}() : (uint32_t)seed);
    std::uniform_int_distribution<int> dis(1, n);

    // K-means++-like seeding: first random, next farthest from existing
    // First centroid
    centroids.push_back(loc[dis(gen)]);
    while ((int)centroids.size() < k) {
        double max_min_dist = -1.0;
        Point next_centroid = loc[1];
        for (int i = 1; i <= n; ++i) {
            const Point& p = loc[i];
            double min_dist = 1e18;
            for (const auto& c : centroids) {
                double dx = p.x - c.x;
                double dy = p.y - c.y;
                double dist = std::sqrt(dx*dx + dy*dy);
                min_dist = std::min(min_dist, dist);
            }
            if (min_dist > max_min_dist) {
                max_min_dist = min_dist;
                next_centroid = p;
            }
        }
        centroids.push_back(next_centroid);
    }

    // Iterations
    vector<int> assignment(n+1, -1); // assignment for customers 1..n
    for (int it = 0; it < max_iters; ++it) {
        bool changed = false;
        for (auto& cl : clusters) cl.clear();

        // Assign step
        for (int i = 1; i <= n; ++i) {
            const Point& p = loc[i];
            double bestDist2 = 1e30;
            int bestC = 0;
            for (int c = 0; c < k; ++c) {
                double dx = p.x - centroids[c].x;
                double dy = p.y - centroids[c].y;
                double d2 = dx*dx + dy*dy;
                if (d2 < bestDist2) {
                    bestDist2 = d2;
                    bestC = c;
                }
            }
            if (assignment[i] != bestC) {
                assignment[i] = bestC;
                changed = true;
            }
            clusters[bestC].push_back(i);
        }

        // Update step
        for (int c = 0; c < k; ++c) {
            if (clusters[c].empty()) {
                // Reinitialize empty cluster to a random customer to avoid dead clusters
                int pick = dis(gen);
                centroids[c].x = loc[pick].x;
                centroids[c].y = loc[pick].y;
                continue;
            }
            double sumx = 0.0, sumy = 0.0;
            for (int idx : clusters[c]) {
                sumx += loc[idx].x;
                sumy += loc[idx].y;
            }
            centroids[c].x = sumx / clusters[c].size();
            centroids[c].y = sumy / clusters[c].size();
        }

        if (!changed) break; // converged
    }

    return clusters;
}

Solution greedy_insert_customer(Solution sol, int customer, bool minimize_delta) {
    Solution best_sol = sol;
    double best_score = 1e18;
    auto try_insert = [&](vi base_route, bool is_truck, int route_idx) {
        vd base_metrics = check_route_feasibility(base_route, 0.0, is_truck);
        for (size_t pos = 1; pos < base_route.size(); ++pos) {
            vi new_route = base_route;
            new_route.insert(new_route.begin() + pos, customer);
            vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck);
            double new_makespan = 0.0;
            for (int t = 0; t < h; ++t){
                new_makespan = max(new_makespan, (t == route_idx && is_truck) ? new_metrics[0] : sol.truck_route_times[t]);
            }
            for (int t = 0; t < d; ++t){
                new_makespan = max(new_makespan, (t == route_idx && !is_truck) ? new_metrics[0] : sol.drone_route_times[t]);
            }
            double new_deadline_violation = sol.deadline_violation + new_metrics[1] - base_metrics[1];
            double new_energy_violation = sol.energy_violation + new_metrics[2] - base_metrics[2];
            double new_capacity_violation = sol.capacity_violation + new_metrics[3] - base_metrics[3];
            double violation = new_deadline_violation * 1e3 +
                               new_energy_violation * 1e3 +
                               new_capacity_violation * 1e3;
           double objective_val = 0.0;
            if (minimize_delta) {
                // Minimize added time (Cheapest Insertion) -> Creates tight clusters
                double delta = new_metrics[0] - base_metrics[0];
                
                // Get the current total time of the vehicle we are considering
                double current_load = is_truck ? sol.truck_route_times[route_idx] : sol.drone_route_times[route_idx];
                
                // Add a tiny penalty based on current load.
                // If Deltas are equal (common for drones), this forces the algorithm to pick the emptier vehicle.
                // 1e-4 is small enough not to override true geographic efficiency.
                objective_val = delta + (current_load * 1e-3);
            } else {
                // Minimize global makespan -> Balances load (but can cause crossing)
                objective_val = new_makespan;
            }

            double new_score = objective_val * pow((1.0 + violation), PENALTY_EXPONENT);
            if (new_score + 1e-8 < best_score) {
                best_score = new_score;
                best_sol = sol;
                best_sol.deadline_violation = new_deadline_violation;
                best_sol.energy_violation = new_energy_violation;
                best_sol.capacity_violation = new_capacity_violation; 
                best_sol.total_makespan = new_makespan;
                if (is_truck) {
                    best_sol.truck_routes[route_idx] = new_route;
                    best_sol.truck_route_times[route_idx] = new_metrics[0];
                } else {
                    best_sol.drone_routes[route_idx] = new_route;
                    best_sol.drone_route_times[route_idx] = new_metrics[0];
                }
            }
        }
        // Also attempt to insert at the end of the route
        {
            vi new_route = base_route;
            if (new_route.back() != 0) new_route.push_back(0);
            new_route.push_back(customer);
            new_route.push_back(0);
            vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck);
            double new_makespan = 0.0;
            for (int t = 0; t < h; ++t){
                new_makespan = max(new_makespan, (t == route_idx && is_truck) ? new_metrics[0] : sol.truck_route_times[t]);
            }
            for (int t = 0; t < d; ++t){
                new_makespan = max(new_makespan, (t == route_idx && !is_truck) ? new_metrics[0] : sol.drone_route_times[t]);
            }
            double new_deadline_violation = max(sol.deadline_violation + new_metrics[1] - base_metrics[1], 0.0);
            double new_energy_violation = max(sol.energy_violation + new_metrics[2] - base_metrics[2], 0.0);
            double new_capacity_violation = max(sol.capacity_violation + new_metrics[3] - base_metrics[3], 0.0);
            double violation = new_deadline_violation * 1e3 +
                               new_energy_violation * 1e3 +
                               new_capacity_violation * 1e3;
            double objective_val = 0.0;
            if (minimize_delta) {
                // Minimize added time (Cheapest Insertion) -> Creates tight clusters
                double delta = new_metrics[0] - base_metrics[0];
                
                // Get the current total time of the vehicle we are considering
                double current_load = is_truck ? sol.truck_route_times[route_idx] : sol.drone_route_times[route_idx];
                
                // Add a tiny penalty based on current load.
                // If Deltas are equal (common for drones), this forces the algorithm to pick the emptier vehicle.
                // 1e-4 is small enough not to override true geographic efficiency.
                objective_val = delta + (current_load * 1e-3);
            } else {
                // Minimize global makespan -> Balances load (but can cause crossing)
                objective_val = new_makespan;
            }

            double new_score = objective_val * pow((1.0 + violation), PENALTY_EXPONENT);
            if (new_score + 1e-8 < best_score) {
                best_score = new_score;
                best_sol = sol;
                best_sol.deadline_violation = new_deadline_violation;
                best_sol.energy_violation = new_energy_violation;
                best_sol.capacity_violation = new_capacity_violation;
                best_sol.total_makespan = new_makespan;
                if (is_truck) {
                    best_sol.truck_routes[route_idx] = new_route;
                    best_sol.truck_route_times[route_idx] = new_metrics[0];
                } else {
                    best_sol.drone_routes[route_idx] = new_route;
                    best_sol.drone_route_times[route_idx] = new_metrics[0];
                }
            }
        }
    };
    for (int i = 0; i < h; ++i) {
        try_insert(sol.truck_routes[i], true, i);
    }
    for (int i = 0; i < d; ++i) {
        try_insert(sol.drone_routes[i], false, i);
    }
    return best_sol;
}

Solution generate_initial_solution(uint64_t seed = UINT64_MAX){
    Solution sol;
    sol.truck_routes.resize(h);
    sol.drone_routes.resize(h); // as many drone routes as truck routes
    // Cluster customers into up to h groups (if h==0, nothing to do)
    vector<bool> visited(n+1, false);
    int num_of_visited_customers = 0;
    vvi clusters = (h > 0) ? kmeans_clustering(h, 1000, seed) : vvi{};
    // Optional: shuffle each cluster to randomize the intra-cluster selection order
    mt19937 rng(seed == UINT64_MAX ? std::random_device{}() : (uint32_t)seed);
    for (auto& vec : clusters) {
        shuffle(vec.begin(), vec.end(), rng);
    }
    vi cluster_assignment(n+1, -1);
    for (int i = 0; i < (int)clusters.size(); ++i) {
        for (int cust : clusters[i]) {
            cluster_assignment[cust] = i;
        }
    }
    vd service_times_truck(h, 0.0); // service times for each truck (and drone)
    vd service_times_drone(h, 0.0); // service times for each drone
    vd capacity_used_truck(h, 0); // capacity used by each truck
    vd capacity_used_drone(h, 0); // capacity used by each drone
    vd timebomb_truck(h, 1e18); // timebomb for each truck
    vd timebomb_drone(h, 1e18); // timebomb for each drone
    vd energy_used_drone(h, 0); // energy used by each drone
    // Simple initializer: for each index i in [0..h-1], pick first unvisited feasible customer
    // for truck, then pick first unvisited feasible customer for drone. Routes are {0, cust, 0}.
    // If none available or infeasible, assign {0}.
    for (int i = 0; i < h; ++i) {
        const vector<int>* cluster_ptr = (i < (int)clusters.size()) ? &clusters[i] : nullptr;
        bool assigned_truck = false;
        if (cluster_ptr) {
            for (int cust : *cluster_ptr) {
                if (visited[cust]) continue;
                vi r = {0, cust, 0};
                vd t_route = check_truck_route_feasibility(r, 0.0);
                double t = t_route[0];
                bool feas = (t_route[1] < 1e-8) && (t_route[3] < 1e-9); // deadline and capacity
                if (feas) {
                    r = {0, cust};
                    auto [t, feas] = compute_truck_route_time(r, 0.0);
                    sol.truck_routes[i] = r;
                    visited[cust] = true;
                    assigned_truck = true;
                    service_times_truck[i] += t;
                    num_of_visited_customers++;
                    capacity_used_truck[i] += demand[cust];
                    // Timebomb should be tied to the selected customer's deadline, not the truck index
                    timebomb_truck[i] = deadline[cust];
                    break;
                }
            }
        }

        if (!assigned_truck) {
            sol.truck_routes[i] = {0};
        }
        bool assigned_drone = false;
        if (cluster_ptr) {
            for (int cust : *cluster_ptr) {
                if (visited[cust]) continue;
                if (!served_by_drone[cust]) continue;
                vi r = {0, cust, 0};
                vd d_route = check_drone_route_feasibility(r);
                double t = d_route[0];
                bool feas = (d_route[1] < 1e-8) && (d_route[2] < 1e-8) && (d_route[3] < 1e-9); // deadline, energy, capacity
                if (feas) {
                    r = {0, cust};
                    auto [t, feas] = compute_drone_route_time(r);
                    service_times_drone[i] += t;
                    energy_used_drone[i] += compute_drone_route_energy(r).first;
                    // Timebomb should be tied to the selected customer's deadline, not the drone index
                    timebomb_drone[i] = deadline[cust];
                    capacity_used_drone[i] += demand[cust];
                    sol.drone_routes[i] = r;
                    visited[cust] = true;
                    assigned_drone = true;
                    num_of_visited_customers++;
                    break;
                }
            }
        }
        if (!assigned_drone) {
            sol.drone_routes[i] = {0};
        }
    }

    //loop until all customers are visited:
    int iter = 10000; // max iterations
    int stall_count = 0; // number of consecutive iterations without meaningful progress
    // Track whether each vehicle (truck/drone) is still active (eligible for selection)
    vector<char> active_truck(h, 1), active_drone(h, 1);
    while (num_of_visited_customers < n && iter > 0) {
        iter--;
        bool made_progress = false;
        // Select the vehicle (truck or drone) with the smallest current service time
        if (h <= 0) break; // no vehicles available
        int best_vehicle = -1; // 0..h-1 trucks, h..2h-1 drones
        double best_time_val = 1e100;
        int active_count = 0;
        for (int i = 0; i < h; ++i) {
            if (!active_truck[i]) continue;
            ++active_count;
            if (service_times_truck[i] < best_time_val) {
                best_time_val = service_times_truck[i];
                best_vehicle = i; // truck i
            }
        }
        for (int i = 0; i < h; ++i) { // consider paired drone index i
            if (!active_drone[i]) continue;
            ++active_count;
            if (service_times_drone[i] < best_time_val) {
                best_time_val = service_times_drone[i];
                best_vehicle = h + i; // drone i
            }
        }
        if (active_count == 0 || best_vehicle == -1) break; // no active vehicles remaining
        bool is_truck = (best_vehicle < h);
        // if it's a truck:
        if (is_truck) {
            int truck_idx = best_vehicle;
            bool assigned = false;
            double best_score = 1e18;
            vi current_route = sol.truck_routes[truck_idx];
            int current_node = current_route.empty() ? 0 : current_route.back();
            // Try to assign a new customer to this truck
            // Loop all customers in cluster[truck_idx] first
            // Find the best candidate customer based on a scoring function
            struct Candidate {
                int cust;
                double urgency_score;
                double capacity_ratio;
                double change_in_return_time;
                bool same_cluster;
                Candidate(int c, double u, double cr, double crt, bool sc)
                    : cust(c), urgency_score(u), capacity_ratio(cr), change_in_return_time(crt), same_cluster(sc) {}
            };
            Candidate* best_candidate = nullptr;
            vector<Candidate> candidates;
            double max_change_in_return_time = -1e18;
            double min_change_in_return_time = 1e18;
            double direct_return_time = compute_truck_route_time({current_node, 0}, service_times_truck[truck_idx]).first;  
            for (int cust = 1; cust <= n; ++cust) {
                if (visited[cust]) continue;
                if (capacity_used_truck[truck_idx] + demand[cust] > (double)Dh + 1e-9) continue; // capacity prune
                vi r = sol.truck_routes[truck_idx];
                // calculate a score for inserting cust into r
                // Note: compute_truck_route_time adds serve_truck[cust] for non-depot arrivals; subtract it to get pure travel
                double to_with_service = compute_truck_route_time({current_node, cust}, service_times_truck[truck_idx]).first;
                double travel_to_cust = max(0.0, to_with_service);
                double depart_time = service_times_truck[truck_idx] + travel_to_cust;
                double time_back_to_depot = compute_truck_route_time({cust, 0}, depart_time).first;
                double time_bomb_at_cust = min(timebomb_truck[truck_idx] - to_with_service, deadline[cust]);
                if (time_bomb_at_cust <= 0) continue; // cannot reach customer before its deadline
                double urgency_score = time_back_to_depot / time_bomb_at_cust;
                // Urgency check: must be able to serve customer and come back to depot before their deadline
                if (urgency_score > 1.0 + 1e-8) continue;
                if (urgency_score < 0) continue;
                double change_in_return_time = to_with_service + time_back_to_depot - direct_return_time;
                double capacity_ratio = (capacity_used_truck[truck_idx] + demand[cust]) / (double)Dh;
                if (capacity_ratio > 1.0 + 1e-8) continue; // capacity prune
                max_change_in_return_time = max(max_change_in_return_time, change_in_return_time);
                min_change_in_return_time = min(min_change_in_return_time, change_in_return_time);
                // check if same cluster with truck position
                bool same_cluster = (cluster_assignment[cust] == cluster_assignment[current_node]);
                candidates.emplace_back(cust, urgency_score, capacity_ratio, change_in_return_time, same_cluster);
            }
            // Select the best candidate based on a weighted scoring function
            //First calculate weights based on MAD:
            double mean_urgency = 0.0, mean_capacity = 0.0;
            for (auto& cand : candidates) {
                mean_urgency += cand.urgency_score;
                mean_capacity += cand.capacity_ratio;
            }
            mean_urgency /= candidates.size();
            mean_capacity /= candidates.size();
            double mad_urgency = 0.0, mad_capacity = 0.0;
            for (auto& cand : candidates) {
                mad_urgency += fabs(cand.urgency_score - mean_urgency);
                mad_capacity += fabs(cand.capacity_ratio - mean_capacity);
            }
            mad_urgency /= candidates.size();
            mad_capacity /= candidates.size();
            // Avoid zero MAD
            if (mad_urgency < 1e-8) mad_urgency = 1.0;
            if (mad_capacity < 1e-8) mad_capacity = 1.0;
            // Now score candidates and pick the best
            for (auto& cand : candidates) {
                double w1 = 1.0 / mad_urgency, w2 = 1.0 / mad_capacity; // weights for urgency, capacity, change in return time, same cluster
                // Normalize weights
                double w_sum = w1 + w2;
                w1 /= w_sum; w2 /= w_sum;
                // Normalize change_in_return_time to [0,1] based on min/max in candidates
                double norm_change = (max_change_in_return_time - min_change_in_return_time < 1e-8)
                                     ? 0.0
                                     : (cand.change_in_return_time - min_change_in_return_time) / (max_change_in_return_time - min_change_in_return_time);
                // test different scoring formulas here
                //double score = norm_change + (cand.same_cluster ? 0.0 : 1.0);
                double score = w1 * cand.urgency_score * cand.urgency_score + w2 * cand.capacity_ratio * cand.capacity_ratio + norm_change + (cand.same_cluster ? 0.0 : w1 + w2 + 1.0);
                if (score < best_score) {
                    best_score = score;
                    best_candidate = &cand;
                }
            }
            if (best_candidate) {
                int cust = best_candidate->cust;
                vi r = sol.truck_routes[truck_idx];
                r.push_back(cust);
                double time_to_cust = compute_truck_route_time({current_node, cust}, service_times_truck[truck_idx]).first;
                service_times_truck[truck_idx] += time_to_cust;
                capacity_used_truck[truck_idx] += demand[cust];
                timebomb_truck[truck_idx] = min(timebomb_truck[truck_idx] - time_to_cust, deadline[cust]);
                sol.truck_routes[truck_idx] = r;
                visited[cust] = true;
                assigned = true;
                num_of_visited_customers++;
                made_progress = true;
            }
            if (!assigned) {
                // No feasible customer found.
                int current_node2 = current_route.empty() ? 0 : current_route.back();
                if (current_node2 != 0) {
                    // Force return to depot
                    vi r = sol.truck_routes[truck_idx];
                    double time_to_depot = compute_truck_route_time({current_node2, 0}, service_times_truck[truck_idx]).first;
                    service_times_truck[truck_idx] += time_to_depot;
                    r.push_back(0);
                    sol.truck_routes[truck_idx] = r;
                    timebomb_truck[truck_idx] = 1e18; // reset timebomb after returning to depot
                    // Reset capacity after completing a tour at the depot
                    capacity_used_truck[truck_idx] = 0.0;
                    made_progress = true;
                }
                else {
                    active_truck[truck_idx] = 0;
                    made_progress = true;
                }
            }
        } else {
            int drone_idx = best_vehicle - h;
            bool assigned = false;
            vi current_route = sol.drone_routes[drone_idx];
            int current_node = current_route.empty() ? 0 : current_route.back();
            double best_score = 1e18;
            // Try to assign a new customer to this drone
            struct Candidate {
                int cust;
                double urgency_score;
                double capacity_ratio;
                double energy_ratio;
                double change_in_return_time;
                bool same_cluster;
                Candidate(int c, double u, double cr, double er, double crt, bool sc)
                    : cust(c), urgency_score(u), capacity_ratio(cr), energy_ratio(er), change_in_return_time(crt), same_cluster(sc) {}
            };
            Candidate* best_candidate = nullptr;
            vector<Candidate> candidates;
            double max_change_in_return_time = -1e18;
            double min_change_in_return_time = 1e18;
            double direct_return_time = compute_drone_route_time({current_node, 0}).first;  

            for (int cust = 1; cust <= n; ++cust) {
                if (visited[cust]) continue;
                if (!served_by_drone[cust]) continue;
                if (capacity_used_drone[drone_idx] + demand[cust] > Dd + 1e-9) continue; // capacity prune
                double to_with_service = compute_drone_route_time({current_node, cust}).first;
                double time_back_to_depot = compute_drone_route_time({cust, 0}).first;
                double time_bomb_at_cust = min(timebomb_drone[drone_idx] - to_with_service, deadline[cust]);
                if (time_bomb_at_cust <= 0) continue; // cannot reach customer before its deadline
                double urgency_score = time_back_to_depot / time_bomb_at_cust;
                if (urgency_score > 1.0 + 1e-8) continue;
                if (urgency_score < -1e-12) continue;
                double change_in_return_time = to_with_service + time_back_to_depot - direct_return_time;
                double capacity_ratio = (capacity_used_drone[drone_idx] + demand[cust]) / Dd;
                if (capacity_ratio > 1.0 + 1e-8) continue; // capacity prune
                // Estimate energy for sortie current_node -> cust -> depot at current payload
                double total_energy = (to_with_service - serve_drone[cust]) * (power_beta * capacity_used_drone[drone_idx] + power_gamma)
                                      + (time_back_to_depot) * (power_beta * (capacity_used_drone[drone_idx] + demand[cust]) + power_gamma);
                double energy_ratio = (energy_used_drone[drone_idx] + total_energy) / E;
                if (energy_ratio > 1.0 + 1e-8) continue; // energy prune
                max_change_in_return_time = max(max_change_in_return_time, change_in_return_time);
                min_change_in_return_time = min(min_change_in_return_time, change_in_return_time);
                bool same_cluster = (cluster_assignment[cust] == cluster_assignment[current_node]);
                candidates.emplace_back(cust, urgency_score, capacity_ratio, energy_ratio, change_in_return_time, same_cluster);
            }
            // Select the best candidate based on a weighted scoring function
            //First calculate weights based on MAD:
            double mean_urgency = 0.0, mean_capacity = 0.0, mean_energy = 0.0;
            for (auto& cand : candidates) {
                mean_urgency += cand.urgency_score;
                mean_capacity += cand.capacity_ratio;
                mean_energy += cand.energy_ratio;
            }
            mean_urgency /= candidates.size();
            mean_capacity /= candidates.size();
            mean_energy /= candidates.size();
            double mad_urgency = 0.0, mad_capacity = 0.0, mad_energy = 0.0;
            for (auto& cand : candidates) {
                mad_urgency += fabs(cand.urgency_score - mean_urgency);
                mad_capacity += fabs(cand.capacity_ratio - mean_capacity);
                mad_energy += fabs(cand.energy_ratio - mean_energy);
            }
            mad_urgency /= candidates.size();
            mad_capacity /= candidates.size();
            mad_energy /= candidates.size();
            // Avoid zero MAD
            if (mad_urgency < 1e-8) mad_urgency = 1.0;
            if (mad_capacity < 1e-8) mad_capacity = 1.0;
            if (mad_energy < 1e-8) mad_energy = 1.0;
            // Now score candidates and pick the best   

            for (auto& cand : candidates) {
                double w1 = 1.0 / mad_urgency, w2 = 1.0 / mad_capacity, w3 = 1.0 / mad_energy; // weights for urgency, capacity, energy, change in return time, same cluster
                // Normalize weights
                double w_sum = w1 + w2 + w3;
                w1 /= w_sum; w2 /= w_sum; w3 /= w_sum;
                // Normalize change_in_return_time to [0,1] based on min/max in candidates
                double norm_change = (max_change_in_return_time - min_change_in_return_time < 1e-8)
                                     ? 0.0
                                     : (cand.change_in_return_time - min_change_in_return_time) / (max_change_in_return_time - min_change_in_return_time);
                // Score: lower is better
                // Formula 1: remove capacity_ratio and energy_ratio
                //double score = norm_change + (cand.same_cluster ? 0.0 : 1.0);
                double score = w1 * cand.urgency_score * cand.urgency_score + w2 * cand.capacity_ratio * cand.capacity_ratio + w3 * cand.energy_ratio * cand.energy_ratio + norm_change + (cand.same_cluster ? 0.0 : w1 + w2 + w3 + 1.0);
                if (score < best_score) {
                    best_score = score;
                    best_candidate = &cand;
                }
            }
            if (best_candidate) {
                int cust = best_candidate->cust;
                vi r = sol.drone_routes[drone_idx];
                r.push_back(cust);
                double time_to_cust = compute_drone_route_time({current_node, cust}).first;
                service_times_drone[drone_idx] += time_to_cust;
                // Reserve energy for forward leg and the eventual return to depot
                double total_energy = (time_to_cust - serve_drone[cust]) * (power_beta * capacity_used_drone[drone_idx] + power_gamma);
                energy_used_drone[drone_idx] += total_energy;
                capacity_used_drone[drone_idx] += demand[cust];
                timebomb_drone[drone_idx] = min(timebomb_drone[drone_idx] - time_to_cust, deadline[cust]);
                sol.drone_routes[drone_idx] = r;
                visited[cust] = true;
                assigned = true;
                num_of_visited_customers++;
                made_progress = true;
            }
            if (!assigned) {
                int current_node2 = current_route.empty() ? 0 : current_route.back();
                if (current_node2 != 0) {
                    vi r = sol.drone_routes[drone_idx];
                    double time_to_depot = compute_drone_route_time({current_node2, 0}).first;
                    service_times_drone[drone_idx] += time_to_depot;
                    r.push_back(0);
                    sol.drone_routes[drone_idx] = r;
                    timebomb_drone[drone_idx] = 1e18; // reset timebomb after returning to depot
                    capacity_used_drone[drone_idx] = 0.0;
                    energy_used_drone[drone_idx] = 0.0;
                    made_progress = true;
                }
                else {
                    
                    active_drone[drone_idx] = 0; // deactivate drone
                    made_progress = true;
                }
            }
        }
        // Stall detection: if no progress this iteration, count and break after a threshold
        if (!made_progress) {
            if (++stall_count > max(100, 2*h + 2*d)) {
                cerr << "Warning: construction stalled; breaking after repeated no-progress iterations.\n";
                break;
            }
        } else {
            stall_count = 0;
        }
    }
    // Reallocate drone sorties (depot-to-depot) across d drones using LPT to minimize makespan
    {
        // Extract sorties from current drone routes
        struct Sortie { vi route; double duration; };
        vector<Sortie> sorties;
        sorties.reserve(n); // rough upper bound
        for (const auto &r : sol.drone_routes) {
            if (r.size() <= 1) continue;
            vi cur; cur.push_back(0);
            for (size_t i = 1; i < r.size(); ++i) {
                int node = r[i];
                if (node == 0) {
                    // close current sortie
                    if (cur.back() != 0) cur.push_back(0);
                    double dur = compute_drone_route_time(cur).first;
                    if (cur.size() > 2) sorties.push_back({cur, dur}); // non-empty sortie
                    cur.clear(); cur.push_back(0);
                } else {
                    cur.push_back(node);
                }
            }
            // If route ended without a closing depot, close it
            if (cur.size() > 1) {
                if (cur.back() != 0) cur.push_back(0);
                double dur = compute_drone_route_time(cur).first;
                if (cur.size() > 2) sorties.push_back({cur, dur});
            }
        }

        // Prepare target number of drones
        if (d <= 0) {
            sol.drone_routes.clear();
        } else {
            // LPT: sort sorties by duration descending
            vector<int> idx(sorties.size());
            iota(idx.begin(), idx.end(), 0);
            sort(idx.begin(), idx.end(), [&](int a, int b){ return sorties[a].duration > sorties[b].duration; });

            // Min-heap of (load, drone_id)
            struct Load { double t; int id; };
            struct Cmp { bool operator()(const Load &a, const Load &b) const { return a.t > b.t; } };
            priority_queue<Load, vector<Load>, Cmp> pq;
            for (int k = 0; k < d; ++k) pq.push({0.0, k});

            vector<vector<int>> assigned(d); // store concatenated routes per drone as sequences of nodes
            vector<double> loads(d, 0.0);
            // Assign each sortie to the least-loaded drone
            for (int id : idx) {
                Load cur = pq.top(); pq.pop();
                int k = cur.id;
                // Append sortie to drone k's sequence (avoid duplicating initial depot)
                const vi &s = sorties[id].route; // s is [0, ..., 0]
                if (assigned[k].empty()) assigned[k].push_back(0);
                // remove trailing 0 if present to avoid double depot before appending
                if (!assigned[k].empty() && assigned[k].back() == 0) {
                    // keep it; we'll append from s.begin()+1
                }
                assigned[k].insert(assigned[k].end(), s.begin() + 1, s.end());
                loads[k] = cur.t + sorties[id].duration;
                pq.push({loads[k], k});
            }

            // Build sol.drone_routes from assigned sequences
            sol.drone_routes.clear();
            sol.drone_routes.resize(d);
            for (int k = 0; k < d; ++k) {
                if (assigned[k].empty()) {
                    sol.drone_routes[k] = {0};
                } else {
                    sol.drone_routes[k] = std::move(assigned[k]);
                    // Ensure ends at depot
                    if (sol.drone_routes[k].back() != 0) sol.drone_routes[k].push_back(0);
                }
            }
        }
    }

    // Finally, ensure all routes end at depot
    for (int i = 0; i < h; ++i) {
        if (sol.truck_routes[i].empty() || sol.truck_routes[i].back() != 0) {
            int current_node = sol.truck_routes[i].empty() ? 0 : sol.truck_routes[i].back();
            if (current_node != 0) sol.truck_routes[i].push_back(0);
        }
    }
    // Drone routes may have size d (could be < h). Safely finalize only existing routes.
    for (int i = 0; i < (int)sol.drone_routes.size(); ++i) {
        if (sol.drone_routes[i].empty() || sol.drone_routes[i].back() != 0) {
            int current_node = sol.drone_routes[i].empty() ? 0 : sol.drone_routes[i].back();
            if (current_node != 0) sol.drone_routes[i].push_back(0);
        }
    }
    // Compute makespan and per-vehicle times (indexed vectors)
    double makespan = 0.0;
    sol.truck_route_times.assign(h, 0.0);
    for (int i = 0; i < h; ++i) {
        double t = 0.0;
        if (sol.truck_routes[i].size() > 1) {
            t = compute_truck_route_time(sol.truck_routes[i], 0.0).first;
        }
        sol.truck_route_times[i] = t;
        makespan = max(makespan, t);
    }
    sol.drone_route_times.assign((int)sol.drone_routes.size(), 0.0);
    for (int i = 0; i < (int)sol.drone_routes.size(); ++i) {
        double t = 0.0;
        if (sol.drone_routes[i].size() > 1) {
            t = compute_drone_route_time(sol.drone_routes[i]).first;
        }
        sol.drone_route_times[i] = t;
        makespan = max(makespan, t);
    }
    sol.total_makespan = makespan;
    // forced feasibility for now
    sol.capacity_violation = 0.0;
    sol.energy_violation = 0.0;
    sol.deadline_violation = 0.0;
    for (int i = 0; i < h; ++i) {
        vd metrics = check_truck_route_feasibility(sol.truck_routes[i], 0.0);
        sol.deadline_violation += metrics[1];
        sol.capacity_violation += metrics[3];
    }
    for (int i = 0; i < (int)sol.drone_routes.size(); ++i) {
        vd metrics = check_drone_route_feasibility(sol.drone_routes[i]);
        sol.deadline_violation += metrics[1];
        sol.energy_violation += metrics[2];
        sol.capacity_violation += metrics[3];
    }
        // if there are still unvisited customers, assign them greedily with smallest delta increase
    for (int cust = 1; cust <= n; ++cust) {
        if (visited[cust]) continue;
        cout << "Warning: customer " << cust << " unvisited after initial solution construction; assigning greedily.\n";
        sol = greedy_insert_customer(sol, cust, true);
    }
    return sol;
}

void print_solution(const Solution& sol) {
    cout << "Truck Routes:\n";
    for (int i = 0; i < h; ++i) {
        cout << "Truck " << i+1 << ": ";
        for (int node : sol.truck_routes[i]) {
            cout << node << " ";
        }
        cout << "\n";
    }
    cout << "Drone Routes:\n";
    for (int i = 0; i < d; ++i) {
        cout << "Drone " << i+1 << ": ";
        for (int node : sol.drone_routes[i]) {
            cout << node << " ";
        }
        cout << "\n";
    }
}

// Stream-based printer to avoid duplicating formatting on stdout/file
static void print_solution_stream(const Solution& sol, std::ostream& os) {
    os << "Truck Routes:\n";
    for (int i = 0; i < h; ++i) {
        os << "Truck " << i+1 << ": ";
        for (int node : sol.truck_routes[i]) {
            os << node << " ";
        }
        vd truck_metric = check_route_feasibility(sol.truck_routes[i], 0.0, true);
        os << "|Truck Time: " << sol.truck_route_times[i] << "|" << truck_metric[0] << "," << truck_metric[1] << "," << truck_metric[2] << "," << truck_metric[3];
        os << "\n";
    }
    os << "Drone Routes:\n";
    for (int i = 0; i < d; ++i) {
        os << "Drone " << i+1 << ": ";
        for (int node : sol.drone_routes[i]) {
            os << node << " ";
        }
        vd drone_metric = check_route_feasibility(sol.drone_routes[i], 0.0, false);
        os << "|Drone Time: " << sol.drone_route_times[i] << "|" << drone_metric[0] << "," << drone_metric[1] << "," << drone_metric[2] << "," << drone_metric[3];
        os << "\n";
    }
    os << "Total validation:" 
       << " Makespan=" << sol.total_makespan
       << ", Deadline violation=" << sol.deadline_violation
       << ", Energy violation=" << sol.energy_violation
       << ", Capacity violation=" << sol.capacity_violation
       << "\n";
}

pair<int, bool> critical_solution_index(const Solution& sol) {
    // Identify the vehicle (truck or drone) that contributes most to the penalized objective.
    // Drone 3 is indexed as h + 2. => returns (2, false)
    double best_violation_weight = -1.0;
    double best_score = -1.0;
    bool is_truck = true;
    int best_idx = -1; // unified index: trucks [0,h), drones [h,h+d)

    auto evaluate_route = [&](const vi& route, double cached_time, bool is_truck, int unified_idx) {
        vector<double> metrics = check_route_feasibility(route, 0.0, is_truck);
        double base_time = (route.size() > 1)
            ? (cached_time > 0.0 ? cached_time : metrics[0])
            : 0.0;
        double violation_weight =
            PENALTY_LAMBDA_DEADLINE * metrics[1] +
            PENALTY_LAMBDA_ENERGY   * metrics[2] +
            PENALTY_LAMBDA_CAPACITY * metrics[3];
        double penalty_multiplier = 1.0 + violation_weight;
        double score = base_time * pow(penalty_multiplier, PENALTY_EXPONENT);

        if (violation_weight > best_violation_weight + 1e-12 ||
            (std::fabs(violation_weight - best_violation_weight) <= 1e-12 && score > best_score + 1e-9)) {
            best_violation_weight = violation_weight;
            best_score = score;
            best_idx = unified_idx;
        }
    };

    for (int i = 0; i < h; ++i) {
        double cached_time = (i < (int)sol.truck_route_times.size()) ? sol.truck_route_times[i] : 0.0;
        evaluate_route(sol.truck_routes[i], cached_time, true, i);
    }
    for (int i = 0; i < (int)sol.drone_route_times.size(); ++i) {
        double cached_time = sol.drone_route_times[i];
        evaluate_route(sol.drone_routes[i], cached_time, false, h + i);
    }

    if (best_idx == -1) {
        // Fallback: pick the route with the largest cached time to keep progress moving.
        double max_time = -1.0;
        int fallback_idx = 0;
        for (int i = 0; i < h; ++i) {
            double t = (i < (int)sol.truck_route_times.size()) ? sol.truck_route_times[i] : 0.0;
            if (t > max_time) { max_time = t; fallback_idx = i; }
        }
        for (int i = 0; i < (int)sol.drone_route_times.size(); ++i) {
            double t = sol.drone_route_times[i];
            if (t > max_time) { max_time = t; fallback_idx = h + i; }
        }
        best_idx = fallback_idx;
    }
    if (best_idx < h) {
        is_truck = true;
    } else {
        is_truck = false;
        best_idx -= h;
    }
    return {best_idx, is_truck};
}

Solution local_search(const Solution& initial_solution, int neighbor_id, int current_iter, double best_cost, double (*solution_cost)(const Solution&)) {
    Solution best_neighbor = initial_solution;
    double best_neighbor_cost = 1e10;
    // Depending on neighbor_id, implement different neighborhood structures
    if (neighbor_id == 0) {
        // Relocate 1 customer from the critical (longest-time) vehicle route to another route of the same mode
        // 1) Identify critical vehicle (truck or drone) using precomputed times
        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

         // Ensure tabu list is sized to (n+1) x (h+d)
        int veh_count = h + d;
        if ((int)tabu_list_10.size() != n + 1 || (veh_count > 0 && (int)tabu_list_10[0].size() != veh_count)) {
            tabu_list_10.assign(n + 1, vector<int>(max(0, veh_count), 0));
        }

        // Prepare neighborhood best tracking
        int best_target = -1; // vehicle index in unified space (0..h-1 trucks, h..h+d-1 drones)
        int best_cust = -1;   // moved customer id
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;

        // Precompute metrics for delta evaluation
        double current_total_time_sq = 0.0;
        double second_max_makespan = 0.0;
        
        // Calculate current total time squared and find the second highest makespan (excluding the critical vehicle)
        for (int i = 0; i < h; ++i) {
            double t = initial_solution.truck_route_times[i];
            current_total_time_sq += t * t;
            if (!crit_is_truck || i != critical_idx) {
                second_max_makespan = max(second_max_makespan, t);
            }
        }
        for (int i = 0; i < d; ++i) {
            double t = initial_solution.drone_route_times[i];
            current_total_time_sq += t * t;
            if (crit_is_truck || i != critical_idx) {
                second_max_makespan = max(second_max_makespan, t);
            }
        }

        auto consider_relocate = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            // Lambda to normalize routes for comparison (detect no-ops)
            auto normalize_route = []( vi& route) -> vi {
                if (!route.empty() && route.front() != 0) 
                    route.insert(route.begin(), 0);
                if (!route.empty() && route.back() != 0) 
                    route.push_back(0);
                return route;
            };

            vd crit_route_time_feas = is_truck_mode
                ? check_route_feasibility(base_route, 0.0, true)
                : check_route_feasibility(base_route, 0.0, false);

            double l2_weight = 0.0;
                if (solution_cost == solution_score_l2_norm) l2_weight = 1e-3;

            // Collect positions of customers (exclude depots)
            vector<int> pos;
            for (int i = 0; i < (int)base_route.size(); ++i) if (base_route[i] != 0) pos.push_back(i);
            
            for (int idx = 0; idx < (int)pos.size(); ++idx) {
                int p = pos[idx];
                int cust = base_route[p];

                // Pre-calculate the base route with customer removed (for inter-route moves)
                vi base_route_removed = base_route;
                base_route_removed.erase(base_route_removed.begin() + p);
                
                vd base_route_removed_feas = is_truck_mode
                    ? check_route_feasibility(base_route_removed, 0.0, true)
                    : check_route_feasibility(base_route_removed, 0.0, false);

                // Try relocating cust to other vehicles
                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    //if (served_by_drone[cust] == 0 && target_veh >= h) continue; // cannot assign to drone
                    bool is_tabu = (tabu_list_10[cust][target_veh] > current_iter);
                    
                    if (target_veh == critical_vehicle_id) {
                        // --- INTRA-ROUTE RELOCATION (Same Vehicle) ---
                        auto evaluate_intra = [&](int p2) {
                            vi new_route = base_route;
                            new_route.erase(new_route.begin() + p);
                            int insert_idx = p2 - (p2 > p ? 1 : 0);
                            new_route.insert(new_route.begin() + insert_idx, cust);
                            vi new_norm = normalize_route(new_route);
                            
                            vd new_feas = is_truck_mode
                                ? check_route_feasibility(new_norm, 0.0, true)
                                : check_route_feasibility(new_norm, 0.0, false);

                            // Delta Eval
                            double new_deadline = initial_solution.deadline_violation + new_feas[1] - crit_route_time_feas[1];
                            double new_capacity = initial_solution.capacity_violation + new_feas[3] - crit_route_time_feas[3];
                            double new_energy = initial_solution.energy_violation + new_feas[2] - crit_route_time_feas[2];
                            
                            double new_makespan = max(second_max_makespan, new_feas[0]);
                            double new_total_sq = current_total_time_sq - (crit_route_time_feas[0]*crit_route_time_feas[0]) + (new_feas[0]*new_feas[0]);
                            
                            double pen = 1.0 + PENALTY_LAMBDA_CAPACITY * new_capacity + PENALTY_LAMBDA_ENERGY * new_energy + PENALTY_LAMBDA_DEADLINE * new_deadline;
                            // check score cost function: if it's solution_score_makespan, * 0, else * 1e-3
                            double score = (new_makespan + std::sqrt(new_total_sq) / (h + d) * l2_weight) * pow(pen, PENALTY_EXPONENT);
                            
                            bool feasible = new_deadline <= 1e-8 && new_capacity <= 1e-8 && new_energy <= 1e-8;
                            
                            if (is_tabu && !(score + 1e-8 < best_cost && feasible)) return;
                            
                            if (score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = score;
                                best_target = target_veh;
                                best_cust = cust;
                                best_candidate_neighbor = initial_solution;
                                if (is_truck_mode) {
                                    best_candidate_neighbor.truck_routes[critical_vehicle_id] = new_norm;
                                    best_candidate_neighbor.truck_route_times[critical_vehicle_id] = new_feas[0];
                                } else {
                                    best_candidate_neighbor.drone_routes[critical_vehicle_id - h] = new_norm;
                                    best_candidate_neighbor.drone_route_times[critical_vehicle_id - h] = new_feas[0];
                                }
                                best_candidate_neighbor.deadline_violation = new_deadline;
                                best_candidate_neighbor.capacity_violation = new_capacity;
                                best_candidate_neighbor.energy_violation = new_energy;
                                best_candidate_neighbor.total_makespan = new_makespan;
                            }
                        };

                        for (int p2 = 1; p2 < (int)base_route.size(); ++p2) {
                            if (p2 == p) continue;
                            evaluate_intra(p2);
                        }
                        // End of route
                        evaluate_intra(base_route.size());
                    } else {
                        // --- INTER-ROUTE RELOCATION (Different Vehicle) ---
                        const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh] : initial_solution.drone_routes[target_veh - h];
                        
                        vd target_route_feas = (target_veh < h)
                            ? check_route_feasibility(target_route, 0.0, true)
                            : check_route_feasibility(target_route, 0.0, false);

                        auto evaluate_inter = [&](int insert_pos) {
                            vi new_target = target_route;
                            if (insert_pos >= (int)new_target.size()) {
                                new_target.push_back(cust);
                                new_target.push_back(0);
                            } else {
                                new_target.insert(new_target.begin() + insert_pos, cust);
                            }
                            
                            vd new_target_feas = (target_veh < h)
                                ? check_route_feasibility(new_target, 0.0, true)
                                : check_route_feasibility(new_target, 0.0, false);
                            
                            double new_deadline = initial_solution.deadline_violation 
                                + (base_route_removed_feas[1] - crit_route_time_feas[1]) 
                                + (new_target_feas[1] - target_route_feas[1]);
                            double new_capacity = initial_solution.capacity_violation 
                                + (base_route_removed_feas[3] - crit_route_time_feas[3]) 
                                + (new_target_feas[3] - target_route_feas[3]);
                            double new_energy = initial_solution.energy_violation 
                                + (base_route_removed_feas[2] - crit_route_time_feas[2]) 
                                + (new_target_feas[2] - target_route_feas[2]);
                            
                            double new_makespan = max({second_max_makespan, base_route_removed_feas[0], new_target_feas[0]});
                            double new_total_sq = current_total_time_sq 
                                - (crit_route_time_feas[0]*crit_route_time_feas[0]) + (base_route_removed_feas[0]*base_route_removed_feas[0])
                                - (target_route_feas[0]*target_route_feas[0]) + (new_target_feas[0]*new_target_feas[0]);
                            
                            double pen = 1.0 + PENALTY_LAMBDA_CAPACITY * new_capacity + PENALTY_LAMBDA_ENERGY * new_energy + PENALTY_LAMBDA_DEADLINE * new_deadline;
                            double score = (new_makespan + l2_weight * (std::sqrt(new_total_sq) / (h + d))) * pow(pen, PENALTY_EXPONENT);
                            
                            bool feasible = new_deadline <= 1e-8 && new_capacity <= 1e-8 && new_energy <= 1e-8;
                            
                            if (is_tabu && !(score + 1e-8 < best_cost && feasible)){
                                //cout << "Tabu move skipped: cust " << cust << " to vehicle " << target_veh << " until iter " << tabu_list_10[cust][target_veh] << "\n";
                                return;
                            }
                            
                            if (score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = score;
                                best_target = target_veh;
                                best_cust = cust;
                                best_candidate_neighbor = initial_solution;
                                if (is_truck_mode) {
                                    best_candidate_neighbor.truck_routes[critical_vehicle_id] = base_route_removed;
                                    best_candidate_neighbor.truck_route_times[critical_vehicle_id] = base_route_removed_feas[0];
                                } else {
                                    best_candidate_neighbor.drone_routes[critical_vehicle_id - h] = base_route_removed;
                                    best_candidate_neighbor.drone_route_times[critical_vehicle_id - h] = base_route_removed_feas[0];
                                }
                                if (target_veh < h) {
                                    best_candidate_neighbor.truck_routes[target_veh] = new_target;
                                    best_candidate_neighbor.truck_route_times[target_veh] = new_target_feas[0];
                                } else {
                                    best_candidate_neighbor.drone_routes[target_veh - h] = new_target;
                                    best_candidate_neighbor.drone_route_times[target_veh - h] = new_target_feas[0];
                                }
                                best_candidate_neighbor.deadline_violation = new_deadline;
                                best_candidate_neighbor.capacity_violation = new_capacity;
                                best_candidate_neighbor.energy_violation = new_energy;
                                best_candidate_neighbor.total_makespan = new_makespan;
                            }
                        };

                        for (int insert_pos = 1; insert_pos < (int)target_route.size(); ++insert_pos) {
                            evaluate_inter(insert_pos);
                        }
                        evaluate_inter(target_route.size());
                    }
                }
            }
        };
        if (crit_is_truck) {
            consider_relocate(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_relocate(initial_solution.drone_routes[critical_idx], false, critical_idx + h);
        }

        // After evaluating all candidates, update tabu list if we found an improving move
        if (best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            // Update tabu list
            tabu_list_10[best_cust][best_target] = current_iter + TABU_TENURE_10;
        }
        return best_neighbor;
    } else if (neighbor_id == 1) {
        // Neighborhood 1: swap two customers, allowing cross-mode exchanges
        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        if ((int)tabu_list_11.size() != n + 1 || (n + 1 > 0 && (int)tabu_list_11[0].size() != n + 1)) {
            tabu_list_11.assign(n + 1, vector<int>(n + 1, 0));
        }

        int best_cust_a = -1, best_cust_b = -1;
        int best_pos_a = -1, best_pos_b = -1;
        int best_veh_a = -1, best_veh_b = -1;
        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;

        //calculate sum of squares of current route times
        double current_sum_squares = 0.0;
        for (int veh = 0; veh < h + d; ++veh) {
            const double& route_times = (veh < h) ? initial_solution.truck_route_times[veh] : initial_solution.drone_route_times[veh - h];
            current_sum_squares += route_times * route_times;
        }

        auto consider_swap = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 2) return; // nothing to swap
            double l2_weight = 0.0;
                if (solution_cost == solution_score_l2_norm) l2_weight = 1e-3;
            vd crit_metrics = is_truck_mode
                ? check_route_feasibility(base_route, 0.0, true)
                : check_route_feasibility(base_route, 0.0, false);

            vector<int> crit_positions;
            for (int i = 0; i < (int)base_route.size(); ++i) {
                if (base_route[i] != 0) crit_positions.push_back(i);
            }
            if (crit_positions.empty()) return;

            for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                if (target_veh == critical_vehicle_id) {
                    // Calculate second makespan excluding critical vehicle
                    double second_max_makespan = 0.0;
                    for (int v = 0; v < h + d; ++v) {
                        if (v != critical_vehicle_id) {
                            double t = (v < h) ? initial_solution.truck_route_times[v] : initial_solution.drone_route_times[v - h];
                            second_max_makespan = max(second_max_makespan, t);
                        }
                    }
                    for (int idx_a = 0; idx_a < (int)crit_positions.size(); ++idx_a) {
                        int pos_a = crit_positions[idx_a];
                        int cust_a = base_route[pos_a];

                        for (int idx_b = idx_a + 1; idx_b < (int)crit_positions.size(); ++idx_b) {
                            int pos_b = crit_positions[idx_b];
                            int cust_b = base_route[pos_b];

                            if (cust_a == cust_b) continue;

                            // Check tabu status
                            int u = min(cust_a, cust_b);
                            int v = max(cust_a, cust_b);
                            bool is_tabu = (tabu_list_11[u][v] > current_iter);

                            // Generate new route with swapped customers
                            vi new_crit_route = base_route;
                            new_crit_route[pos_a] = cust_b;
                            new_crit_route[pos_b] = cust_a;

                            vd new_crit_metrics = is_truck_mode
                                ? check_route_feasibility(new_crit_route, 0.0, true)
                                : check_route_feasibility(new_crit_route, 0.0, false);

                            // Delta evaluation
                            double new_deadline = initial_solution.deadline_violation
                                + (new_crit_metrics[1] - crit_metrics[1]);
                            double new_capacity = initial_solution.capacity_violation
                                + (new_crit_metrics[3] - crit_metrics[3]);
                            double new_energy = initial_solution.energy_violation
                                + (new_crit_metrics[2] - crit_metrics[2]);
                            double new_makespan = max(second_max_makespan, new_crit_metrics[0]);
                            double new_sum_squares = current_sum_squares
                                - (crit_metrics[0] * crit_metrics[0]) + (new_crit_metrics[0] * new_crit_metrics[0]);
                            double pen = 1.0 + PENALTY_LAMBDA_CAPACITY * new_capacity + PENALTY_LAMBDA_ENERGY * new_energy + PENALTY_LAMBDA_DEADLINE * new_deadline;
                            double score = (new_makespan + std::sqrt(new_sum_squares) / (h + d) * l2_weight) * pow(pen, PENALTY_EXPONENT);
                            bool feasible = new_deadline <= 1e-8 && new_capacity <= 1e-8 && new_energy <= 1e-8;
                            if (is_tabu && !(score + 1e-8 < best_cost && feasible)) continue;
                            if (score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = score;
                                best_cust_a = cust_a;
                                best_cust_b = cust_b;
                                best_pos_a = pos_a;
                                best_pos_b = pos_b;
                                best_veh_a = critical_vehicle_id;
                                best_veh_b = critical_vehicle_id;
                            }
                        }
                    }
                }

                if (target_veh == critical_vehicle_id) continue;

                const vi& target_route = (target_veh < h) ? initial_solution.truck_routes[target_veh] : initial_solution.drone_routes[target_veh - h];
                if (target_route.size() <= 2) continue;

                // Calculate third makespan excluding critical and target vehicles
                double third_max_makespan = 0.0;
                for (int v = 0; v < h + d; ++v) {
                    if (v != critical_vehicle_id && v != target_veh) {
                        double t = (v < h) ? initial_solution.truck_route_times[v] : initial_solution.drone_route_times[v - h];
                        third_max_makespan = max(third_max_makespan, t);
                    }
                }

                vd target_metrics = (target_veh < h)
                    ? check_route_feasibility(target_route, 0.0, true)
                    : check_route_feasibility(target_route, 0.0, false);

                vector<int> target_positions;
                for (int i = 0; i < (int)target_route.size(); ++i) {
                    if (target_route[i] != 0) target_positions.push_back(i);
                }
                if (target_positions.empty()) continue;

                for (int idx_a = 0; idx_a < (int)crit_positions.size(); ++idx_a) {
                    int pos_a = crit_positions[idx_a];
                    int cust_a = base_route[pos_a];

                    for (int idx_b = 0; idx_b < (int)target_positions.size(); ++idx_b) {
                        int pos_b = target_positions[idx_b];
                        int cust_b = target_route[pos_b];

                        if (cust_a == cust_b) continue;

                        if (served_by_drone[cust_a] == 0 && target_veh >= h) continue; // cannot assign cust_a to drone
                        if (served_by_drone[cust_b] == 0 && critical_vehicle_id >= h) continue; // cannot assign cust_b to drone

                        // Check tabu status
                        int u = min(cust_a, cust_b);
                        int v = max(cust_a, cust_b);
                        bool is_tabu = (tabu_list_11[u][v] > current_iter);

                        // Generate new routes with swapped customers
                        vi new_crit_route = base_route;
                        vi new_target_route = target_route;
                        new_crit_route[pos_a] = cust_b;
                        new_target_route[pos_b] = cust_a;

                        vd new_crit_metrics = is_truck_mode
                            ? check_route_feasibility(new_crit_route, 0.0, true)
                            : check_route_feasibility(new_crit_route, 0.0, false);
                        vd new_target_metrics = (target_veh < h)
                            ? check_route_feasibility(new_target_route, 0.0, true)
                            : check_route_feasibility(new_target_route, 0.0, false);

                        // Delta evaluation
                        double new_deadline = initial_solution.deadline_violation
                            + (new_crit_metrics[1] - crit_metrics[1])
                            + (new_target_metrics[1] - target_metrics[1]);
                        double new_capacity = initial_solution.capacity_violation
                            + (new_crit_metrics[3] - crit_metrics[3])
                            + (new_target_metrics[3] - target_metrics[3]);
                        double new_energy = initial_solution.energy_violation
                            + (new_crit_metrics[2] - crit_metrics[2])
                            + (new_target_metrics[2] - target_metrics[2]);
                        double new_makespan = max({third_max_makespan, new_crit_metrics[0], new_target_metrics[0]});
                        double new_sum_squares = current_sum_squares
                            - (crit_metrics[0] * crit_metrics[0]) + (new_crit_metrics[0] * new_crit_metrics[0])
                            - (target_metrics[0] * target_metrics[0]) + (new_target_metrics[0] * new_target_metrics[0]);
                        double pen = 1.0 + PENALTY_LAMBDA_CAPACITY * new_capacity + PENALTY_LAMBDA_ENERGY * new_energy + PENALTY_LAMBDA_DEADLINE * new_deadline;
                        double score = (new_makespan + std::sqrt(new_sum_squares) / (h + d) * l2_weight) * pow(pen, PENALTY_EXPONENT);
                        bool feasible = new_deadline <= 1e-8 && new_capacity <= 1e-8 && new_energy <= 1e-8;
                        if (is_tabu && !(score + 1e-8 < best_cost && feasible)) continue;
                        if (score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = score;
                            best_cust_a = cust_a;
                            best_cust_b = cust_b;
                            best_pos_a = pos_a;
                            best_pos_b = pos_b;
                            best_veh_a = critical_vehicle_id;
                            best_veh_b = target_veh;
                        }
                    }
                }
            }
        };

        for (int veh = 0; veh < h + d; ++veh) {
            bool is_truck = veh < h;
            const vi& route = is_truck ? initial_solution.truck_routes[veh]
                                       : initial_solution.drone_routes[veh - h];
            // only consider the critical vehicle for swaps
            if (veh == (crit_is_truck ? critical_idx : critical_idx + h)) {
                consider_swap(route, is_truck, veh);
            }
        }

        // Construct best neighbor if found
        if (best_cust_a != -1 && best_cust_b != -1) {
            Solution candidate = initial_solution;
            vd old_metric_a = (best_veh_a < h)
                ? check_route_feasibility(initial_solution.truck_routes[best_veh_a], 0.0, true)
                : check_route_feasibility(initial_solution.drone_routes[best_veh_a - h], 0.0, false);
            vd old_metric_b = (best_veh_b < h)
                ? check_route_feasibility(initial_solution.truck_routes[best_veh_b], 0.0, true)
                : check_route_feasibility(initial_solution.drone_routes[best_veh_b - h], 0.0, false);
            if (best_veh_a == best_veh_b) {
                // Same vehicle swap
                bool is_truck = best_veh_a < h;
                vi new_route = is_truck ? candidate.truck_routes[best_veh_a]
                                        : candidate.drone_routes[best_veh_a - h];
                new_route[best_pos_a] = best_cust_b;
                new_route[best_pos_b] = best_cust_a;
                vd new_metrics = is_truck
                    ? check_route_feasibility(new_route, 0.0, true)
                    : check_route_feasibility(new_route, 0.0, false);
                if (is_truck) {
                    candidate.truck_routes[best_veh_a] = new_route;
                    candidate.truck_route_times[best_veh_a] = new_metrics[0];
                } else {
                    candidate.drone_routes[best_veh_a - h] = new_route;
                    candidate.drone_route_times[best_veh_a - h] = new_metrics[0];
                }
                candidate.deadline_violation += (new_metrics[1] - old_metric_a[1]);
                candidate.capacity_violation += (new_metrics[3] - old_metric_a[3]);
                candidate.energy_violation += (new_metrics[2] - old_metric_a[2]);
                candidate.total_makespan = 0.0;
                for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
            } else {
                // Different vehicle swap
                bool is_truck_a = best_veh_a < h;
                bool is_truck_b = best_veh_b < h;

                vi route_a = is_truck_a ? candidate.truck_routes[best_veh_a]
                                        : candidate.drone_routes[best_veh_a - h];
                vi route_b = is_truck_b ? candidate.truck_routes[best_veh_b]
                                        : candidate.drone_routes[best_veh_b - h];
                route_a[best_pos_a] = best_cust_b;
                route_b[best_pos_b] = best_cust_a;
                vd new_metrics_a = is_truck_a
                    ? check_route_feasibility(route_a, 0.0, true)
                    : check_route_feasibility(route_a, 0.0, false);
                vd new_metrics_b = is_truck_b
                    ? check_route_feasibility(route_b, 0.0, true)
                    : check_route_feasibility(route_b, 0.0, false);
                if (is_truck_a) {
                    candidate.truck_routes[best_veh_a] = route_a;
                    candidate.truck_route_times[best_veh_a] = new_metrics_a[0];
                } else {
                    candidate.drone_routes[best_veh_a - h] = route_a;
                    candidate.drone_route_times[best_veh_a - h] = new_metrics_a[0];
                }
                if (is_truck_b) {
                    candidate.truck_routes[best_veh_b] = route_b;
                    candidate.truck_route_times[best_veh_b] = new_metrics_b[0];
                } else {
                    candidate.drone_routes[best_veh_b - h] = route_b;
                    candidate.drone_route_times[best_veh_b - h] = new_metrics_b[0];
                }
                candidate.deadline_violation += (new_metrics_a[1] - old_metric_a[1]) + (new_metrics_b[1] - old_metric_b[1]);
                candidate.capacity_violation += (new_metrics_a[3] - old_metric_a[3]) + (new_metrics_b[3] - old_metric_b[3]);
                candidate.energy_violation += (new_metrics_a[2] - old_metric_a[2]) + (new_metrics_b[2] - old_metric_b[2]);
                candidate.total_makespan = 0.0;
                for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
            }
            best_candidate_neighbor = candidate;
        }

        if (best_cust_a != -1 && best_cust_b != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            int u = min(best_cust_a, best_cust_b);
            int v = max(best_cust_a, best_cust_b);
            tabu_list_11[u][v] = current_iter + TABU_TENURE_11;

            // Debug: print swap info
            /*  cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N1] swap " << best_cust_a << " and " << best_cust_b
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << best_neighbor_cost_local
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 2) {
        // Neighborhood 2: relocate a consecutive pair (2,0)-move from the critical vehicle to another vehicle
        // Structure mirrors neighborhood 0: identify critical vehicle, enumerate candidate relocations,
        // respect tabu_list_20 keyed by (min(c1,c2), max(c1,c2), target_vehicle).
        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        // Prepare best tracking
        Solution best_candidate_neighbor = initial_solution;
        double best_neighbor_cost_local = 1e10;
        int best_c1 = -1, best_c2 = -1;
        int best_target_vehicle = -1;
        int best_src_pos = -1, best_target_pos = -1;

        auto consider_relocate_pair = [&](const vi& base_route, bool is_truck_mode, int critical_vehicle_id) {
            if (base_route.size() <= 3) return; // nothing to do if fewer than two customers

            auto normalize_route = [](const vi& route) -> vi {
                vi normalized;
                for (int node : route) {
                    if (normalized.empty() || node != 0 || normalized.back() != 0) normalized.push_back(node);
                }
                if (normalized.empty()) return vi{0};
                if (normalized.front() != 0) normalized.insert(normalized.begin(), 0);
                if (normalized.back() != 0) normalized.push_back(0);
                return normalized;
            };

            vi orig = normalize_route(base_route);
            if (orig.size() <= 3) return;
            vd orig_metrics = is_truck_mode
                ? check_route_feasibility(orig, 0.0, true)
                : check_route_feasibility(orig, 0.0, false);

            vector<int> pos;
            for (int i = 0; i + 1 < (int)orig.size(); ++i) {
                if (orig[i] != 0 && orig[i + 1] != 0) pos.push_back(i);
            }
            if (pos.empty()) return;

            auto apply_candidate = [&](const vi& crit_route, const vd& crit_metrics_local,
                                      const optional<pair<int, vi>>& target_change) {
                Solution candidate = initial_solution;
                candidate.deadline_violation += crit_metrics_local[1] - orig_metrics[1];
                candidate.capacity_violation += crit_metrics_local[3] - orig_metrics[3];
                candidate.energy_violation += crit_metrics_local[2] - orig_metrics[2];
                if (is_truck_mode) {
                    candidate.truck_routes[critical_vehicle_id] = crit_route;
                    candidate.truck_route_times[critical_vehicle_id] = (crit_route.size() > 1) ? crit_metrics_local[0] : 0.0;
                } else {
                    candidate.drone_routes[critical_vehicle_id - h] = crit_route;
                    candidate.drone_route_times[critical_vehicle_id - h] = (crit_route.size() > 1) ? crit_metrics_local[0] : 0.0;
                }

                if (target_change.has_value()) {
                    int target_vehicle = target_change->first;
                    const vi& new_target_route = target_change->second;
                    bool target_is_truck = target_vehicle < h;
                    vd target_metrics_before = target_is_truck
                        ? check_route_feasibility(target_is_truck ? initial_solution.truck_routes[target_vehicle]
                                                                 : initial_solution.drone_routes[target_vehicle - h], 0.0, target_is_truck)
                        : check_route_feasibility(initial_solution.drone_routes[target_vehicle - h], 0.0, false);
                    vd target_metrics_after = target_is_truck
                        ? check_route_feasibility(new_target_route, 0.0, true)
                        : check_route_feasibility(new_target_route, 0.0, false);
                    candidate.deadline_violation += target_metrics_after[1] - target_metrics_before[1];
                    candidate.capacity_violation += target_metrics_after[3] - target_metrics_before[3];
                    candidate.energy_violation += target_metrics_after[2] - target_metrics_before[2];
                    if (target_is_truck) {
                        candidate.truck_routes[target_vehicle] = new_target_route;
                        candidate.truck_route_times[target_vehicle] = (new_target_route.size() > 1) ? target_metrics_after[0] : 0.0;
                    } else {
                        candidate.drone_routes[target_vehicle - h] = new_target_route;
                        candidate.drone_route_times[target_vehicle - h] = (new_target_route.size() > 1) ? target_metrics_after[0] : 0.0;
                    }
                }

                candidate.total_makespan = 0.0;
                for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                for (double tt : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, tt);
                return candidate;
            };

            for (int p : pos) {
                int c1 = orig[p];
                int c2 = orig[p + 1];

                vi reduced = orig;
                reduced.erase(reduced.begin() + p, reduced.begin() + p + 2);
                reduced = normalize_route(reduced);
                vd reduced_metrics = is_truck_mode
                    ? check_route_feasibility(reduced, 0.0, true)
                    : check_route_feasibility(reduced, 0.0, false);

                for (int ip = 1; ip <= (int)reduced.size(); ++ip) {
                    vi r = reduced;
                    r.insert(r.begin() + ip, c1);
                    r.insert(r.begin() + ip + 1, c2);
                    vi r_norm = normalize_route(r);
                    if (r_norm == orig) continue;

                    vd new_metrics = is_truck_mode
                        ? check_route_feasibility(r_norm, 0.0, true)
                        : check_route_feasibility(r_norm, 0.0, false);
                    Solution candidate = apply_candidate(r_norm, new_metrics, nullopt);

                    vector<int> key = { min(c1, c2), max(c1, c2), critical_vehicle_id };
                    auto it = tabu_list_20.find(key);
                    bool is_tabu = (it != tabu_list_20.end() && it->second > current_iter);
                    double candidate_score = solution_cost(candidate);
                    bool candidate_feasible = candidate.deadline_violation <= 1e-8 &&
                                               candidate.capacity_violation <= 1e-8 &&
                                               candidate.energy_violation <= 1e-8;
                    if (is_tabu && !(candidate_score + 1e-8 < best_cost && candidate_feasible)) continue;

                    if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                        best_neighbor_cost_local = candidate_score;
                        best_candidate_neighbor = candidate;
                        best_c1 = c1; best_c2 = c2;
                        best_target_vehicle = critical_vehicle_id;
                        best_src_pos = p;
                        best_target_pos = ip;
                    }
                }

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == critical_vehicle_id) continue;
                    if ((served_by_drone[c1] == 0 || served_by_drone[c2] == 0) && target_veh >= h) continue;
                    bool target_is_truck = target_veh < h;
                    vi target_route = target_is_truck
                        ? initial_solution.truck_routes[target_veh]
                        : initial_solution.drone_routes[target_veh - h];
                    target_route = normalize_route(target_route);

                    for (int insert_pos = 1; insert_pos <= (int)target_route.size(); ++insert_pos) {
                        vi new_target = target_route;
                        new_target.insert(new_target.begin() + insert_pos, c1);
                        new_target.insert(new_target.begin() + insert_pos + 1, c2);
                        new_target = normalize_route(new_target);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation = initial_solution.deadline_violation;
                        candidate.capacity_violation = initial_solution.capacity_violation;
                        candidate.energy_violation = initial_solution.energy_violation;

                        // build candidate with reduced critical route + modified target route
                        vd crit_metrics_ready = reduced_metrics;
                        vd target_metrics_new = target_is_truck
                            ? check_route_feasibility(new_target, 0.0, true)
                            : check_route_feasibility(new_target, 0.0, false);
                        vd target_metrics_old = target_is_truck
                            ? check_route_feasibility(target_route, 0.0, true)
                            : check_route_feasibility(target_route, 0.0, false);

                        candidate.deadline_violation += crit_metrics_ready[1] - orig_metrics[1];
                        candidate.deadline_violation += target_metrics_new[1] - target_metrics_old[1];
                        candidate.capacity_violation += crit_metrics_ready[3] - orig_metrics[3];
                        candidate.capacity_violation += target_metrics_new[3] - target_metrics_old[3];
                        candidate.energy_violation += crit_metrics_ready[2] - orig_metrics[2];
                        candidate.energy_violation += target_metrics_new[2] - target_metrics_old[2];

                        if (is_truck_mode) {
                            candidate.truck_routes[critical_vehicle_id] = reduced;
                            candidate.truck_route_times[critical_vehicle_id] = (reduced.size() > 1) ? crit_metrics_ready[0] : 0.0;
                        } else {
                            candidate.drone_routes[critical_vehicle_id - h] = reduced;
                            candidate.drone_route_times[critical_vehicle_id - h] = (reduced.size() > 1) ? crit_metrics_ready[0] : 0.0;
                        }
                        if (target_is_truck) {
                            candidate.truck_routes[target_veh] = new_target;
                            candidate.truck_route_times[target_veh] = (new_target.size() > 1) ? target_metrics_new[0] : 0.0;
                        } else {
                            candidate.drone_routes[target_veh - h] = new_target;
                            candidate.drone_route_times[target_veh - h] = (new_target.size() > 1) ? target_metrics_new[0] : 0.0;
                        }

                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double tt : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, tt);

                        vector<int> key = { min(c1, c2), max(c1, c2), target_veh };
                        auto it = tabu_list_20.find(key);
                        bool is_tabu = (it != tabu_list_20.end() && it->second > current_iter);
                        double candidate_score = solution_cost(candidate);
                        bool feasible = candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8;
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost && feasible)) continue;

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_c1 = c1; best_c2 = c2;
                            best_target_vehicle = target_veh;
                            best_src_pos = p;
                            best_target_pos = insert_pos;
                        }
                    }
                }
            }
        };

        if (crit_is_truck) {
            consider_relocate_pair(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_relocate_pair(initial_solution.drone_routes[critical_idx], false, h + critical_idx);
        }

        // apply best move if found
        if (best_c1 != -1 && best_c2 != -1 && best_target_vehicle != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            // update tabu
            vector<int> key = { min(best_c1, best_c2), max(best_c1, best_c2), best_target_vehicle };
            tabu_list_20[key] = current_iter + TABU_TENURE_20;
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            // Debug:
            /*  cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N2] relocate pair (" << best_c1 << "," << best_c2 << ") to vehicle " << best_target_vehicle
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */
            // return the chosen neighbor (already fully assembled in best_candidate_neighbor)*/
            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 3) {
        // Neighborhood 3: 2-opt within each subroute (between depot nodes) for trucks or drones.
        // Finds the best 2-opt move across all routes that yields the largest local time drop.

        if ((int)tabu_list_2opt.size() != n + 1 || ((int)tabu_list_2opt.size() > 0 && (int)tabu_list_2opt[0].size() != n + 1)) {
            tabu_list_2opt.assign(n + 1, vector<int>(n + 1, 0));
        }

        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e10;
        int best_edge_u = -1, best_edge_v = -1;
        int best_i = -1, best_j = -1;
        bool best_is_truck = crit_is_truck;

        auto normalize_route = [](vi r) -> vi {
            if (r.empty()) return r;
            if (r.front() != 0) r.insert(r.begin(), 0);
            if (r.back() != 0) r.push_back(0);
            vi cleaned; cleaned.reserve(r.size());
            for (int node : r) {
                if (!cleaned.empty() && cleaned.back() == 0 && node == 0) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto consider_2opt = [&](const vi& base_route, bool is_truck_mode, int route_idx) {
            if (base_route.size() <= 3) return;

            vd route_metrics = check_route_feasibility(base_route, 0.0, is_truck_mode);
            int m = (int)base_route.size();
            int start = 0;
            while (start < m) {
                while (start < m && base_route[start] == 0) ++start;
                if (start >= m) break;
                int seg_end = start;
                while (seg_end + 1 < m && base_route[seg_end + 1] != 0) ++seg_end;

                for (int i = start; i < seg_end; ++i) {
                    for (int j = i + 1; j <= seg_end; ++j) {
                        vi new_route = base_route;
                        reverse(new_route.begin() + i, new_route.begin() + j + 1);
                        if (new_route == base_route) continue;

                        new_route = normalize_route(new_route);
                        vd new_metrics = check_route_feasibility(new_route, 0.0, is_truck_mode);
                        int u = min(base_route[i], base_route[j]);
                        int v = max(base_route[i], base_route[j]);
                        if (u < 0 || v < 0) continue;
                        bool is_tabu = (tabu_list_2opt.size() > (size_t)u &&
                                        tabu_list_2opt[u].size() > (size_t)v &&
                                        tabu_list_2opt[u][v] > current_iter);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += new_metrics[1] - route_metrics[1];
                        candidate.capacity_violation += new_metrics[3] - route_metrics[3];
                        candidate.energy_violation += new_metrics[2] - route_metrics[2];
                        if (is_truck_mode) {
                            candidate.truck_routes[route_idx] = new_route;
                            candidate.truck_route_times[route_idx] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                        } else {
                            int drone_route_idx = route_idx - h;
                            if (drone_route_idx >= 0 && drone_route_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[drone_route_idx] = new_route;
                                candidate.drone_route_times[drone_route_idx] = (new_route.size() > 1) ? new_metrics[0] : 0.0;
                            }
                        }
                        candidate.total_makespan = 0.0;
                        for (int t = 0; t < h; ++t) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[t]);
                        for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                        double candidate_score = solution_cost(candidate);
                        if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                         candidate.deadline_violation <= 1e-8 &&
                                         candidate.capacity_violation <= 1e-8 &&
                                         candidate.energy_violation <= 1e-8)) continue;

                        if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                            best_neighbor_cost_local = candidate_score;
                            best_candidate_neighbor = candidate;
                            best_edge_u = u;
                            best_edge_v = v;
                            best_i = i;
                            best_j = j;
                            best_is_truck = is_truck_mode;
                        }
                    }
                }
                start = seg_end + 1;
            }
        };

        if (crit_is_truck) {
            consider_2opt(initial_solution.truck_routes[critical_idx], true, critical_idx);
        } else {
            consider_2opt(initial_solution.drone_routes[critical_idx], false, critical_idx + h);
        }

        if (best_edge_u != -1 && best_edge_v != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_2opt[best_edge_u][best_edge_v] = current_iter + TABU_TENURE_2OPT;

            // Debug N3
            /* cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N3] 2-opt on " << (best_is_truck ? "truck" : "drone") << " #"
                 << (crit_is_truck ? critical_idx + 1 : critical_idx + 1)
                 << " between positions " << best_i << " and " << best_j
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;

    } else if (neighbor_id == 4) {
        if ((int)tabu_list_2opt_star.size() != n + 1 || ((int)tabu_list_2opt_star.size() > 0 && (int)tabu_list_2opt_star[0].size() != n + 1)) {
            tabu_list_2opt_star.assign(n + 1, vector<int>(n + 1, 0));
        }

        auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        int best_ua = -1, best_va = -1, best_ub = -1, best_vb = -1;

        auto enumerate_segments = [](const vi& route) {
            vector<pair<int,int>> segs;
            int m = (int)route.size();
            int start = 0;
            while (start < m) {
                while (start < m && route[start] == 0) ++start;
                if (start >= m) break;
                int end = start;
                while (end + 1 < m && route[end + 1] != 0) ++end;
                segs.emplace_back(start, end);
                start = end + 1;
            }
            return segs;
        };

        vi crit_route_raw = crit_is_truck
            ? initial_solution.truck_routes[critical_idx]
            : initial_solution.drone_routes[critical_idx];
        vi crit_route = normalize_route(crit_route_raw);
        vd crit_metrics = check_route_feasibility(crit_route_raw, 0.0, crit_is_truck);
        auto crit_segs = enumerate_segments(crit_route);

        auto evaluate_two_opt_star = [&](const vi& other_route_raw, bool other_is_truck, int other_idx) {
            vi other_route = normalize_route(other_route_raw);
            if (other_route.size() <= 3) return;
            vd other_metrics = check_route_feasibility(other_route_raw, 0.0, other_is_truck);
            auto other_segs = enumerate_segments(other_route);

            for (const auto& segA : crit_segs) {
                for (int i = segA.first; i < segA.second; ++i) {
                    int a1 = crit_route[i], a2 = crit_route[i + 1];
                    int ua = min(a1, a2), va = max(a1, a2);
                    if (a1 == 0 || a2 == 0) continue;

                    for (const auto& segB : other_segs) {
                        for (int j = segB.first; j < segB.second; ++j) {
                            int b1 = other_route[j], b2 = other_route[j + 1];
                            int ub = min(b1, b2), vb = max(b1, b2);
                            if (b1 == 0 || b2 == 0) continue;

                            bool is_tabu = (tabu_list_2opt_star[ua][va] > current_iter) ||
                                           (tabu_list_2opt_star[ub][vb] > current_iter);

                            vi crit_new = crit_route;
                            vi other_new = other_route;

                            vi tailA(crit_new.begin() + i + 1, crit_new.begin() + segA.second + 1);
                            vi tailB(other_new.begin() + j + 1, other_new.begin() + segB.second + 1);

                            crit_new.erase(crit_new.begin() + i + 1, crit_new.begin() + segA.second + 1);
                            other_new.erase(other_new.begin() + j + 1, other_new.begin() + segB.second + 1);

                            crit_new.insert(crit_new.begin() + i + 1, tailB.begin(), tailB.end());
                            other_new.insert(other_new.begin() + j + 1, tailA.begin(), tailA.end());

                            crit_new = normalize_route(crit_new);
                            other_new = normalize_route(other_new);
                            if (crit_new == crit_route && other_new == other_route) continue;

                            vd crit_metrics_new = check_route_feasibility(crit_new, 0.0, crit_is_truck);
                            vd other_metrics_new = check_route_feasibility(other_new, 0.0, other_is_truck);

                            Solution candidate = initial_solution;
                            candidate.deadline_violation += crit_metrics_new[1] - crit_metrics[1];
                            candidate.capacity_violation += crit_metrics_new[3] - crit_metrics[3];
                            candidate.energy_violation += crit_metrics_new[2] - crit_metrics[2];
                            candidate.deadline_violation += other_metrics_new[1] - other_metrics[1];
                            candidate.capacity_violation += other_metrics_new[3] - other_metrics[3];
                            candidate.energy_violation += other_metrics_new[2] - other_metrics[2];

                            if (crit_is_truck) {
                                candidate.truck_routes[critical_idx] = crit_new;
                                candidate.truck_route_times[critical_idx] = (crit_new.size() > 1) ? crit_metrics_new[0] : 0.0;
                            } else {
                                int crit_drone_idx = critical_idx;
                                if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[crit_drone_idx] = crit_new;
                                    candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_metrics_new[0] : 0.0;
                                }
                            }

                            if (other_is_truck) {
                                candidate.truck_routes[other_idx] = other_new;
                                candidate.truck_route_times[other_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                            } else {
                                int other_drone_idx = other_idx;
                                if (other_drone_idx >= 0 && other_drone_idx < (int)candidate.drone_routes.size()) {
                                    candidate.drone_routes[other_drone_idx] = other_new;
                                    candidate.drone_route_times[other_drone_idx] = (other_new.size() > 1) ? other_metrics_new[0] : 0.0;
                                }
                            }

                            candidate.total_makespan = 0.0;
                            for (int i = 0; i < h; ++i) candidate.total_makespan = max(candidate.total_makespan, candidate.truck_route_times[i]);
                            for (double t : candidate.drone_route_times) candidate.total_makespan = max(candidate.total_makespan, t);
                            double candidate_score = solution_cost(candidate);
                            if (is_tabu && !(candidate_score + 1e-8 < best_cost &&
                                             candidate.deadline_violation <= 1e-8 &&
                                             candidate.capacity_violation <= 1e-8 &&
                                             candidate.energy_violation <= 1e-8)) {
                                continue;
                            }

                            if (candidate_score + 1e-8 < best_neighbor_cost_local) {
                                best_neighbor_cost_local = candidate_score;
                                best_candidate_neighbor = candidate;
                                best_ua = ua; best_va = va;
                                best_ub = ub; best_vb = vb;
                            }
                        }
                    }
                }
            }
        };

        if (crit_is_truck) {
            for (int j = 0; j < h; ++j) {
                if (j == critical_idx) continue;
                evaluate_two_opt_star(initial_solution.truck_routes[j], true, j);
            }
        } else {
            for (int j = 0; j < (int)initial_solution.drone_routes.size(); ++j) {
                if (j == critical_idx) continue;
                evaluate_two_opt_star(initial_solution.drone_routes[j], false, j);
            }
        }

        if (best_ua != -1 && best_neighbor_cost_local + 1e-8 < best_neighbor_cost) {
            best_neighbor = best_candidate_neighbor;
            best_neighbor_cost = best_neighbor_cost_local;
            tabu_list_2opt_star[best_ua][best_va] = current_iter + TABU_TENURE_2OPT_STAR;
            tabu_list_2opt_star[best_ub][best_vb] = current_iter + TABU_TENURE_2OPT_STAR;

            //Debug N4
/*              cout.setf(std::ios::fixed);
            cout << setprecision(6);
            cout << "[N4] 2-opt* cuts (" << best_ua << "," << best_va << ") & (" << best_ub << "," << best_vb << ")"
                 << ", score: " << solution_score(initial_solution)
                 << " -> " << solution_score(best_candidate_neighbor)
                 << ", iter " << current_iter << "\n"; */

            return best_neighbor;
        }
        return initial_solution;
    } else if (neighbor_id == 5) {
       auto [critical_idx, crit_is_truck] = critical_solution_index(initial_solution);

        Solution best_candidate_neighbor = best_neighbor;
        double best_neighbor_cost_local = 1e18;
        vector<int> best_tabu_triple;
        int best_pair_a = -1, best_pair_b = -1, best_single = -1;
        bool best_pair_from_critical = true;
        int best_other_vehicle = -1;
        bool best_other_is_truck = true;

        auto normalize_route = [](vi route) {
            if (route.empty()) return route;
            if (route.front() != 0) route.insert(route.begin(), 0);
            if (route.back() != 0) route.push_back(0);
            vi cleaned;
            cleaned.reserve(route.size());
            for (int node : route) {
                if (!cleaned.empty() && cleaned.back() == node) continue;
                cleaned.push_back(node);
            }
            return cleaned;
        };

        auto consider_pair_vs_single = [&](const vi& crit_route, bool crit_mode_truck, int crit_global_idx, int crit_route_idx) {
            if (crit_route.size() <= 3) return;

            vd crit_metrics = check_route_feasibility(crit_route, 0.0, crit_mode_truck);
            vector<int> pair_positions;
            for (int i = 0; i + 1 < (int)crit_route.size(); ++i)
                if (crit_route[i] != 0 && crit_route[i + 1] != 0) pair_positions.push_back(i);
            if (pair_positions.empty()) return;

            auto near_enough = [&](int u, int v) {
                return !KNN_ADJ.empty() && KNN_ADJ.size() > (size_t)u && KNN_ADJ[u].size() > (size_t)v && KNN_ADJ[u][v];
            };

            for (int pair_idx : pair_positions) {
                int c1 = crit_route[pair_idx];
                int c2 = crit_route[pair_idx + 1];

                for (int target_veh = 0; target_veh < h + d; ++target_veh) {
                    if (target_veh == (crit_mode_truck ? crit_route_idx : h + crit_route_idx)) continue;
                    bool target_is_truck = target_veh < h;
                    if (!target_is_truck && (!served_by_drone[c1] || !served_by_drone[c2])) continue;

                    int target_idx = target_is_truck ? target_veh : target_veh - h;
                    const vi& target_route = target_is_truck
                        ? initial_solution.truck_routes[target_idx]
                        : initial_solution.drone_routes[target_idx];
                    if (target_route.size() <= 2) continue;

                    vector<int> target_positions;
                    for (int j = 0; j < (int)target_route.size(); ++j)
                        if (target_route[j] != 0) target_positions.push_back(j);
                    if (target_positions.empty()) continue;

                    vd target_metrics = check_route_feasibility(target_route, 0.0, target_is_truck);

                    for (int pos_single : target_positions) {
                        int single = target_route[pos_single];
                        if (!crit_mode_truck && !served_by_drone[single]) continue;

                        if (!KNN_ADJ.empty()) {
                            bool ok = near_enough(c1, single) || near_enough(single, c1) ||
                                      near_enough(c2, single) || near_enough(single, c2);
                            if (!ok) continue;
                        }

                        vi crit_new = crit_route;
                        crit_new.erase(crit_new.begin() + pair_idx);
                        crit_new.erase(crit_new.begin() + pair_idx);
                        crit_new.insert(crit_new.begin() + pair_idx, single);
                        crit_new = normalize_route(crit_new);

                        vi target_new = target_route;
                        target_new.erase(target_new.begin() + pos_single);
                        target_new.insert(target_new.begin() + pos_single, c1);
                        target_new.insert(target_new.begin() + pos_single + 1, c2);
                        target_new = normalize_route(target_new);

                        vd crit_new_metrics = check_route_feasibility(crit_new, 0.0, crit_mode_truck);
                        vd target_new_metrics = check_route_feasibility(target_new, 0.0, target_is_truck);

                        Solution candidate = initial_solution;
                        candidate.deadline_violation += crit_new_metrics[1] - crit_metrics[1];
                        candidate.capacity_violation += crit_new_metrics[3] - crit_metrics[3];
                        candidate.energy_violation += crit_new_metrics[2] - crit_metrics[2];
                        candidate.deadline_violation += target_new_metrics[1] - target_metrics[1];
                        candidate.capacity_violation += target_new_metrics[3] - target_metrics[3];
                        candidate.energy_violation += target_new_metrics[2] - target_metrics[2];

                        if (crit_mode_truck) {
                            candidate.truck_routes[crit_route_idx] = crit_new;
                            candidate.truck_route_times[crit_route_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                        } else {
                            int crit_drone_idx = crit_route_idx;
                            if (crit_drone_idx >= 0 && crit_drone_idx < (int)candidate.drone_routes.size()) {
                                candidate.drone_routes[crit_drone_idx] = crit_new;
                                candidate.drone_route_times[crit_drone_idx] = (crit_new.size() > 1) ? crit_new_metrics[0] : 0.0;
                            }
                 