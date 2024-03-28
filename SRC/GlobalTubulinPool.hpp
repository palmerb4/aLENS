#ifndef GLOBALTUBULINPOOL_HPP_
#define GLOBALTUBULINPOOL_HPP_

#include "Util/TRngPool.hpp"
#include <memory>
#include <mpi.h>
#include <omp.h>
#include <vector>

/// @brief A class for controlling a global pool of tubulin that can bind to microtubules.
///
/// Each of the M unbound tubulin in the global pool binds to a single microtubule out of N total microtubules with a constant binding rate of k_on.
///
/// The probability of a tubulin binding within a single timestep is p = 1 - exp(-N*K_on*dt). Because multiple tubulin may bind to the same microtubule,
/// binding is an independent process. We can draw M U01 random numbers and check if each is less than p. If so, we know a tubulin binds. We then need to
/// draw a random number between 1 to N (inclusive) to decide which microtubule to bind to.
///
/// The above is simple in a single-rank system but becomes complex when microtubules are distributed across processors. Break N into R ranks such that
/// rank r owns n_r microtubules and the sum of n_r for all ranks is N.
/// |--------------------------------N----------------------------|
/// |--n_1--|-----n_2-----|-------------...-------------|---n_R---|
///
/// We need to be careful to avoid race conditions while also evenly distributing the workload. We will create a class that evenly distributes M across all
/// processes such that each rank owns M/R + ((M - M/R) < r) tubulin. It's important to note that the users may locally increment (but never decrement) the
/// free tubulin on each process. This may cause load imbalance and may require recalculation and redistribution of M as it evolves in time. Binding can be
/// performed in a distributed manner by having each process create a vector  bind_count_per_microtubule of size N. Each rank loops over each of its locally
/// owned unbound tubulin and draws a U01 random number. If that number is less than p, the tubulin binds to one of the N total microtubules (not necessarily
/// one owned by the current process) and the process draws a random integer between 0 to N-1 and increments bind_count_per_microtubule[i] by one. Once each
/// process has finished this process, we perform an all-to-all sum of bind_count_per_microtubule. Each process can then loop over its locally owned microtubules,
/// fetch the bind counts, and increment the length of the microtubule accordingly. (Optional) Every so many timesteps, the user should perform an all-to-all
/// sum of the locally owned microtubule counts to get M and then redistribute it among the processes.
class GlobalTubulinPool {
  public:
    GlobalTubulinPool() = delete;

    GlobalTubulinPool(const int &initial_global_microtubule_count, const int &initial_global_tubulin_count,
                      const int &num_procs, const int rank, TRngPool *const rng_pool_ptr)
        : global_microtubule_count_(initial_global_microtubule_count),
          global_tubulin_count_(initial_global_tubulin_count), local_tubulin_count_(0), num_procs_(num_procs),
          rank_(rank), rng_pool_ptr_(rng_pool_ptr) {
        assert(rng_pool_ptr_ != nullptr);
        bind_count_per_microtubule_.resize(global_microtubule_count_);
        std::fill(bind_count_per_microtubule_.begin(), bind_count_per_microtubule_.end(), 0);
        distribute();
    }

    void distribute() {
        // Evenly distribute the global tubulin count among the processes
        int global_tubulin_count_per_proc = global_tubulin_count_ / num_procs_;
        int global_tubulin_count_remainder = global_tubulin_count_ % num_procs_;
        int global_tubulin_count_local = global_tubulin_count_per_proc + (rank_ < global_tubulin_count_remainder);
        local_tubulin_count_ = global_tubulin_count_local;
    }

    void synchronize() {
        // The global count may differ from the sum of the local counts due to the user incrementing the local count.
        // We need to sum the local counts to update the global count.
        int global_tubulin_count_sum = 0;
        MPI_Allreduce(&local_tubulin_count_, &global_tubulin_count_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        global_tubulin_count_ = global_tubulin_count_sum;
    }

    void count_tubulin_binding(const double &binding_rate, const double &dt) {
        // Loop over each locally owned tubulin and draw a U01 random number. If it's less than the binding probability, bind to a microtubule.
        // We need to draw a random number between 0 and N-1 to decide which microtubule to bind to.
        // We then increment the bind count for that microtubule.
        const double binding_probability = 1 - std::exp(-global_microtubule_count_ * binding_rate * dt);
        std::fill(bind_count_per_microtubule_.begin(), bind_count_per_microtubule_.end(), 0);
        std::vector local_bind_count_per_microtubule(global_microtubule_count_, 0);
        const int initial_local_tubulin_count = local_tubulin_count_;
#pragma omp parallel
        {
            const int thread_id = omp_get_thread_num();
#pragma omp for
            for (int i = 0; i < initial_local_tubulin_count; i++) {
                const double rand_u01 = rng_pool_ptr_->getU01(thread_id);
                if (rand_u01 < binding_probability) {
                    const int global_microtubule_index =
                        static_cast<int>(rng_pool_ptr_->getU01(thread_id) * global_microtubule_count_);
                    local_bind_count_per_microtubule[global_microtubule_index]++;

                    // Decrement the local unbound tubulin count. This may cause load imbalance and must be done in a thread-safe manner.
#pragma omp critical
                    local_tubulin_count_--;
                }
            }
        }

        // Perform an all-to-all sum of the bind counts.
        MPI_Allreduce(local_bind_count_per_microtubule.data(), bind_count_per_microtubule_.data(),
                      global_microtubule_count_, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }

    int get_bind_count(const int &global_microtubule_index) const {
        return bind_count_per_microtubule_[global_microtubule_index];
    }

    int get_global_tubulin_count() {
        synchronize();
        return global_tubulin_count_;
    }

    int get_global_microtubule_count() {
        return global_microtubule_count_;
    }

    int get_local_tubulin_count() {
        return local_tubulin_count_;
    }

    void set_global_microtubule_count(const int &global_microtubule_count) {
        global_microtubule_count_ = global_microtubule_count;
        bind_count_per_microtubule_.resize(global_microtubule_count_);
        std::fill(bind_count_per_microtubule_.begin(), bind_count_per_microtubule_.end(), 0);
    }

    void set_global_tubulin_count(const int &global_tubulin_count) {
        global_tubulin_count_ = global_tubulin_count;
        distribute();
    }

    void increment() {
        local_tubulin_count_++;
    }

  private:
    int global_microtubule_count_;
    int global_tubulin_count_;
    int local_tubulin_count_;
    int num_procs_;
    int rank_;
    TRngPool *const rng_pool_ptr_;
    std::vector<int> bind_count_per_microtubule_;
}; // GlobalTubulinPool

#endif
