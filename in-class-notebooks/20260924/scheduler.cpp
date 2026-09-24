#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

using i64 = long long;

struct Job {
    std::string id;
    i64 p, r, d, w;
};

struct Metrics {
    i64 value[4] = {0, 0, std::numeric_limits<i64>::min(), 0};
};

Metrics evaluate(const std::vector<Job>& jobs, const std::vector<int>& order) {
    Metrics result;
    i64 clock = 0;
    for (int index : order) {
        const Job& job = jobs[index];
        clock = std::max(clock, job.r) + job.p;
        const i64 late = clock - job.d;
        result.value[0] += clock;
        result.value[1] += job.w * clock;
        result.value[2] = std::max(result.value[2], late);
        result.value[3] += job.w * std::max<i64>(0, late);
    }
    return result;
}

bool better(const Job& a, const Job& b, int rule, i64 clock) {
    __int128 left, right;
    if (rule == 0) {
        left = a.d;
        right = b.d;
    } else if (rule == 1) {
        left = a.p;
        right = b.p;
    } else if (rule == 2) {
        left = static_cast<__int128>(a.d - clock) * b.p;
        right = static_cast<__int128>(b.d - clock) * a.p;
    } else {
        left = static_cast<__int128>(a.p) * b.w;
        right = static_cast<__int128>(b.p) * a.w;
    }
    return left == right ? a.id < b.id : left < right;
}

std::vector<int> construct(const std::vector<Job>& jobs, int rule) {
    std::vector<int> order;
    std::vector<bool> used(jobs.size());
    i64 clock = 0;
    while (order.size() < jobs.size()) {
        int chosen = -1;
        for (size_t i = 0; i < jobs.size(); ++i) {
            if (!used[i] && jobs[i].r <= clock &&
                (chosen < 0 || better(jobs[i], jobs[chosen], rule, clock))) {
                chosen = static_cast<int>(i);
            }
        }
        if (chosen < 0) {
            i64 next = std::numeric_limits<i64>::max();
            for (size_t i = 0; i < jobs.size(); ++i) {
                if (!used[i]) next = std::min(next, jobs[i].r);
            }
            clock = next;
            continue;
        }
        used[chosen] = true;
        order.push_back(chosen);
        clock += jobs[chosen].p;
    }
    return order;
}

void build_prefix(const std::vector<Job>& jobs, const std::vector<int>& order,
                  int objective, std::vector<i64>& clocks, std::vector<i64>& values) {
    clocks.assign(order.size() + 1, 0);
    values.assign(order.size() + 1, 0);
    if (objective == 2) values[0] = std::numeric_limits<i64>::min();
    for (size_t i = 0; i < order.size(); ++i) {
        const Job& job = jobs[order[i]];
        const i64 finish = std::max(clocks[i], job.r) + job.p;
        const i64 late = finish - job.d;
        clocks[i + 1] = finish;
        if (objective == 0) values[i + 1] = values[i] + finish;
        else if (objective == 1) values[i + 1] = values[i] + job.w * finish;
        else if (objective == 2) values[i + 1] = std::max(values[i], late);
        else values[i + 1] = values[i] + job.w * std::max<i64>(0, late);
    }
}

i64 candidate_value(const std::vector<Job>& jobs, const std::vector<int>& order,
                    int objective, size_t first, const std::vector<i64>& clocks,
                    const std::vector<i64>& values) {
    i64 clock = clocks[first], value = values[first];
    for (size_t i = first; i < order.size(); ++i) {
        const Job& job = jobs[order[i]];
        clock = std::max(clock, job.r) + job.p;
        const i64 late = clock - job.d;
        if (objective == 0) value += clock;
        else if (objective == 1) value += job.w * clock;
        else if (objective == 2) value = std::max(value, late);
        else value += job.w * std::max<i64>(0, late);
    }
    return value;
}

void apply_move(std::vector<int>& order, size_t first, size_t last, int neighborhood) {
    if (neighborhood == 2) std::reverse(order.begin() + first, order.begin() + last + 1);
    else std::swap(order[first], order[last]);
}

struct SearchResult {
    std::vector<int> order;
    Metrics metrics;
    i64 evaluations = 0, moves = 0;
    double seconds = 0;
};

SearchResult search(const std::vector<Job>& jobs, std::vector<int> order,
                    int objective, int neighborhood) {
    SearchResult result;
    const auto started = std::chrono::steady_clock::now();
    std::vector<i64> clocks, values;
    build_prefix(jobs, order, objective, clocks, values);
    while (true) {
        bool improved = false;
        for (size_t first = 0; first + 1 < order.size() && !improved; ++first) {
            const size_t end = neighborhood == 0 ? first + 2 : order.size();
            for (size_t last = first + 1; last < end; ++last) {
                apply_move(order, first, last, neighborhood);
                ++result.evaluations;
                if (candidate_value(jobs, order, objective, first, clocks, values) < values.back()) {
                    build_prefix(jobs, order, objective, clocks, values);
                    ++result.moves;
                    improved = true;
                    break;
                }
                apply_move(order, first, last, neighborhood);
            }
        }
        if (!improved) break;
    }
    result.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    result.order = std::move(order);
    result.metrics = evaluate(jobs, result.order);
    if (result.metrics.value[objective] != values.back()) {
        std::cerr << "Incremental evaluation mismatch\n";
        std::exit(2);
    }
    return result;
}

int main() {
    size_t n;
    std::cin >> n;
    std::vector<Job> jobs(n);
    for (Job& job : jobs) std::cin >> job.id >> job.p >> job.r >> job.d >> job.w;
    for (int rule = 0; rule < 4; ++rule) {
        const std::vector<int> initial = construct(jobs, rule);
        const Metrics baseline = evaluate(jobs, initial);
        for (int objective = 0; objective < 4; ++objective) {
            for (int neighborhood = 0; neighborhood < 3; ++neighborhood) {
                const SearchResult result = search(jobs, initial, objective, neighborhood);
                std::cout << rule << '\t' << objective << '\t' << neighborhood;
                for (i64 value : baseline.value) std::cout << '\t' << value;
                for (i64 value : result.metrics.value) std::cout << '\t' << value;
                std::cout << '\t' << result.evaluations << '\t' << result.moves
                          << '\t' << result.seconds << '\t';
                for (size_t i = 0; i < result.order.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << jobs[result.order[i]].id;
                }
                std::cout << '\n';
            }
        }
    }
}
