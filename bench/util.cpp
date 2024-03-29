#include <random>
#include "util.h"

#include <benchmark/benchmark.h>
#include <fmt/format.h>
#include <defs.h>


#include <picosha2.h>

#include <random>
#include <unordered_map>

namespace vxsort_bench {
using namespace vxsort::types;

std::random_device global_bench_random_device;

std::random_device::result_type global_bench_random_seed = global_bench_random_device();

Counter make_time_per_n_counter(i64 n) {
    return {(double)n, Counter::kAvgThreadsRate | Counter::kIsIterationInvariantRate | Counter::kInvert, Counter::kIs1000};
}

Counter make_cycle_per_n_counter(double n) {
    return {n, Counter::kDefaults, Counter::kIs1000};
}

std::string get_crypto_hash(void *start, void *end) {
    auto *begin = (char *) start;
    auto *finish = (char *) end;
    //auto length = finish - begin;
    std::string hash(picosha2::k_digest_size * 2, '0');
    std::string hash_hex_str = picosha2::hash256_hex_string(begin, finish);
    return hash_hex_str;
}

std::unordered_map<std::string, std::string> perf_counter_names = {
    { "branches", "branches" },
    { "branch-instructions", "branch-insn" },
    { "branch-misses", "branch-misses" },
    { "cache-misses", "cache-misses" },
    { "cache-references", "cache-refs" },
    { "cpu-cycles", "cpu-cycles" },
    { "cycles", "cycles" },
    { "instructions", "insn" },
    { "L1-dcache-load-misses", "L1d$-load-misses" },
    { "L1-dcache-loads", "L1d$-loads" },
    { "L1-dcache-stores", "L1d$-stores" },
    { "L1-icache-load-misses", "L1i$-load-misses" },
    { "LLC-load-misses", "LLC-load-misses" },
    { "LLC-loads", "LLC-loads" },
    { "LLC-store-misses", "LLC-store-misses" },
    { "LLC-stores", "LLC-stores" },
    { "branch-load-misses", "branch-load-misses" },
    { "branch-loads", "branch-loads" },
    { "dTLB-load-misses", "dTLB-load-misses" },
    { "dTLB-loads", "dTLB-loads" },
    { "dTLB-store-misses", "dTLB-store-misses" },
    { "dTLB-stores", "dTLB-stores" },
    { "iTLB-load-misses", "iTLB-load-misses" },
    { "node-load-misses", "node-load-misses" },
    { "node-loads", "node-loads" },
    { "node-store-misses", "node-store-misses" },
    { "node-stores", "node-stores" },
    { "uops_dispatched.port_0", "uops p0" },
    { "uops_dispatched.port_1", "uops p1" },
    { "uops_dispatched.port_2_3", "uops p2+3" },
    { "uops_dispatched.port_4_9", "uops p4+9" },
    { "uops_dispatched.port_5", "uops p5" },
    { "uops_dispatched.port_6", "uops p6" },
    { "uops_dispatched.port_7_8", "uops p7+8" },
    { "uops_executed.core", "uops_executed.core" },
    { "uops_executed.core_cycles_ge_1", "uops_exe core>=1c" },
    { "uops_executed.core_cycles_ge_2", "uops_exe core>=2c" },
    { "uops_executed.core_cycles_ge_3", "uops_exe core>=3c" },
    { "uops_executed.core_cycles_ge_4", "uops_exe core>=4c" },
    { "uops_executed.cycles_ge_1", "uops_exe.cycles_ge_1" },
    { "uops_executed.cycles_ge_2", "uops_exe.cycles_ge_2" },
    { "uops_executed.cycles_ge_3", "uops_exe.cycles_ge_3" },
    { "uops_executed.cycles_ge_4", "uops_exe.cycles_ge_4" },
    { "uops_executed.stall_cycles", "uops_exe.stall_cycles" },
    { "uops_executed.thread", "uops_executed.thread" },
    { "uops_executed.x87", "uops_executed.x87" },
    { "uops_issued.any", "uops_issued.any" },
    { "uops_issued.stall_cycles", "uops_issued.stall_cycles" },
    { "uops_issued.vector_width_mismatch", "uops_issued.vector_width_mismatch" },
    { "uops_retired.slots", "uops_retired.slots" },
    { "uops_retired.stall_cycles", "uops_retired.stall_cycles" },
    { "uops_retired.total_cycles", "uops_retired.total_cycles" },
    { "cycle_activity.stalls_l3_miss", "cycle_activity.stalls_l3_miss" },
    { "cycle_activity.cycles_l1d_miss", "cycle_activity.cycles_l1d_miss" },
    { "cycle_activity.cycles_l2_miss", "cycle_activity.cycles_l2_miss" },
    { "cycle_activity.cycles_mem_any", "cycle_activity.cycles_mem_any" },
    { "cycle_activity.stalls_l1d_miss", "cycle_activity.stalls_l1d_miss" },
    { "cycle_activity.stalls_l2_miss", "cycle_activity.stalls_l2_miss" },
    { "cycle_activity.stalls_mem_any", "cycle_activity.stalls_mem_any" },
    { "cycle_activity.stalls_total", "cycle_activity.stalls_total" },
    { "exe_activity.1_ports_util", "exe_act 1p" },
    { "exe_activity.2_ports_util", "exe_act 2p" },
    { "exe_activity.3_ports_util", "exe_act 3p" },
    { "exe_activity.4_ports_util", "exe_act 4p" },
    { "exe_activity.bound_on_loads", "exe_act load_bound" },
    { "exe_activity.bound_on_stores", "exe_act store_bound" },
    { "exe_activity.exe_bound_0_ports", "exe_act exe_bound" },
    { "frontend_retired.dsb_miss", "frontend_retired.dsb_miss" },
    { "frontend_retired.itlb_miss", "frontend_retired.itlb_miss" },
    { "frontend_retired.l1i_miss", "frontend_retired.l1i_miss" },
    { "frontend_retired.l2_miss", "frontend_retired.l2_miss" },
    { "frontend_retired.latency_ge_1", "frontend_retired.latency_ge_1" },
    { "frontend_retired.latency_ge_128", "frontend_retired.latency_ge_128" },
    { "frontend_retired.latency_ge_16", "frontend_retired.latency_ge_16" },
    { "frontend_retired.latency_ge_2", "frontend_retired.latency_ge_2" },
    { "frontend_retired.latency_ge_256", "frontend_retired.latency_ge_256" },
    { "frontend_retired.latency_ge_2_bubbles_ge_1", "frontend_retired.latency_ge_2_bubbles_ge_1" },
    { "frontend_retired.latency_ge_32", "frontend_retired.latency_ge_32" },
    { "frontend_retired.latency_ge_4", "frontend_retired.latency_ge_4" },
    { "frontend_retired.latency_ge_512", "frontend_retired.latency_ge_512" },
    { "frontend_retired.latency_ge_64", "frontend_retired.latency_ge_64" },
    { "frontend_retired.latency_ge_8", "frontend_retired.latency_ge_8" },
    { "frontend_retired.stlb_miss", "frontend_retired.stlb_miss" },
    { "dtlb_load_misses.stlb_hit", "dtlb_load_misses.stlb_hit" },
    { "dtlb_load_misses.walk_active", "dtlb_load_misses.walk_active" },
    { "dtlb_load_misses.walk_completed", "dtlb_load_misses.walk_completed" },
    { "dtlb_load_misses.walk_completed_2m_4m", "dtlb_load_misses.walk_completed_2m_4m" },
    { "dtlb_load_misses.walk_completed_4k", "dtlb_load_misses.walk_completed_4k" },
    { "dtlb_load_misses.walk_pending", "dtlb_load_misses.walk_pending" },
    { "dtlb_store_misses.stlb_hit", "dtlb_store_misses.stlb_hit" },
    { "dtlb_store_misses.walk_active", "dtlb_store_misses.walk_active" },
    { "dtlb_store_misses.walk_completed", "dtlb_store_misses.walk_completed" },
    { "dtlb_store_misses.walk_completed_2m_4m", "dtlb_store_misses.walk_completed_2m_4m" },
    { "dtlb_store_misses.walk_completed_4k", "dtlb_store_misses.walk_completed_4k" },
    { "dtlb_store_misses.walk_pending", "dtlb_store_misses.walk_pending" },
    { "itlb_misses.stlb_hit", "itlb_misses.stlb_hit" },
    { "itlb_misses.walk_active", "itlb_misses.walk_active" },
    { "itlb_misses.walk_completed", "itlb_misses.walk_completed" },
    { "itlb_misses.walk_completed_2m_4m", "itlb_misses.walk_completed_2m_4m" },
    { "itlb_misses.walk_completed_4k", "itlb_misses.walk_completed_4k" },
    { "itlb_misses.walk_pending", "itlb_misses.walk_pending" },
    { "rs_events.empty_cycles", "rs_events.empty_cycles" },
    { "rs_events.empty_end", "rs_events.empty_end" },
    { "l1d_pend_miss.fb_full", "l1d_pend_miss.fb_full" },
    { "l1d_pend_miss.fb_full_periods", "l1d_pend_miss.fb_full_periods" },
    { "l1d_pend_miss.l2_stall", "l1d_pend_miss.l2_stall" },
    { "l1d_pend_miss.pending", "l1d_pend_miss.pending" },
    { "l1d_pend_miss.pending_cycles", "l1d_pend_miss.pending_cycles" },
    { "l2_lines_in.all", "l2_lines_in.all" },
    { "l2_lines_out.non_silent", "l2_lines_out.non_silent" },
    { "l2_lines_out.silent", "l2_lines_out.silent" },
    { "l2_rqsts.all_code_rd", "l2_rqsts.all_code_rd" },
    { "l2_rqsts.all_rfo", "l2_rqsts.all_rfo" },
    { "l2_rqsts.code_rd_hit", "l2_rqsts.code_rd_hit" },
    { "l2_rqsts.code_rd_miss", "l2_rqsts.code_rd_miss" },
    { "l2_rqsts.miss", "l2_rqsts.miss" },
    { "l2_rqsts.rfo_hit", "l2_rqsts.rfo_hit" },
    { "l2_rqsts.rfo_miss", "l2_rqsts.rfo_miss" },
    { "l2_rqsts.swpf_hit", "l2_rqsts.swpf_hit" },
    { "l2_rqsts.swpf_miss", "l2_rqsts.swpf_miss" },
    { "l2_trans.l2_wb", "l2_trans.l2_wb" },
    { "mem_load_l3_hit_retired.xsnp_fwd", "mem_load_l3_hit_retired.xsnp_fwd" },
    { "mem_load_l3_hit_retired.xsnp_miss", "mem_load_l3_hit_retired.xsnp_miss" },
    { "mem_load_l3_hit_retired.xsnp_no_fwd", "mem_load_l3_hit_retired.xsnp_no_fwd" },
    { "mem_load_l3_hit_retired.xsnp_none", "mem_load_l3_hit_retired.xsnp_none" },
    { "mem_load_retired.fb_hit", "mem_load_retired.fb_hit" },
    { "mem_load_retired.l1_hit", "mem_load_retired.l1_hit" },
    { "mem_load_retired.l1_miss", "mem_load_retired.l1_miss" },
    { "mem_load_retired.l2_hit", "mem_load_retired.l2_hit" },
    { "mem_load_retired.l2_miss", "mem_load_retired.l2_miss" },
    { "mem_load_retired.l3_hit", "mem_load_retired.l3_hit" },
    { "mem_load_retired.l3_miss", "mem_load_retired.l3_miss" },
    { "mem_trans_retired.load_latency_gt_128", "mem_trans_retired.load_latency_gt_128" },
    { "mem_trans_retired.load_latency_gt_16", "mem_trans_retired.load_latency_gt_16" },
    { "mem_trans_retired.load_latency_gt_256", "mem_trans_retired.load_latency_gt_256" },
    { "mem_trans_retired.load_latency_gt_32", "mem_trans_retired.load_latency_gt_32" },
    { "mem_trans_retired.load_latency_gt_4", "mem_trans_retired.load_latency_gt_4" },
    { "mem_trans_retired.load_latency_gt_512", "mem_trans_retired.load_latency_gt_512" },
    { "mem_trans_retired.load_latency_gt_64", "mem_trans_retired.load_latency_gt_64" },
    { "mem_trans_retired.load_latency_gt_8", "mem_trans_retired.load_latency_gt_8" },
    { "core_power.lvl0_turbo_license", "core_power.lvl0_turbo_license" },
    { "core_power.lvl1_turbo_license", "core_power.lvl1_turbo_license" },
    { "core_power.lvl2_turbo_license", "core_power.lvl2_turbo_license" },
    { "opdown-bad-spec", "opdown-bad-spec" },
    { "topdown-be-bound", "topdown-be-bound" },
    { "topdown-fe-bound", "topdown-fe-bound" },
    { "topdown-retiring", "topdown-retiring" },
    { "topdown.backend_bound_slots", "topdown.backend_bound_slots" },
    { "topdown.br_mispredict_slots", "topdown.br_mispredict_slots" },
    { "topdown.slots", "topdown.slots" },
    { "topdown.slots_p", "topdown.slots_p" },
};

void process_perf_counters(UserCounters &counters, i64 num_elements) {
    for (auto const &[k, v] : perf_counter_names) {
        if (!counters.contains(k))
            continue;

        auto counter = counters[k];
        auto per_n_name = fmt::format("{}/N", v);
        counters[per_n_name] = make_cycle_per_n_counter(counter.value / (f64) num_elements);
        counters.erase(k);
    }
}

}
