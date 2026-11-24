// GPU bin-size heuristics, derived from device attributes (L2, bus width, shared
// memory) rather than per-architecture tables. Owns the shared-memory accounting
// used to validate that a chosen bin/np combination fits.

#include <algorithm>
#include <array>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include <cuda.h>

#include <cufinufft/contrib/helper_cuda.h>
#include <cufinufft/heuristics.hpp>
#include <cufinufft/spreadinterp.hpp>
#include <cufinufft/utils.hpp>
#include <finufft_common/kernel.h>

// GpuCapabilities member declared in types.hpp; defined here to keep <cstdio>
// out of the header.
void GpuCapabilities::print_classification(int debug_level) const {
  if (debug_level < 2) return;

  // Only for the name; every other field comes from cudaDeviceGetAttribute.
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, device_id);
  printf("[cufinufft] GPU Classification:\n");
  printf("  Name: %s\n", prop.name);
  printf("  Compute Capability: %d.%d\n", cc_major, cc_minor);
  printf("  Multiprocessor Count: %d SMs\n", multiprocessor_count);
  printf("  Shared Memory:\n");
  printf("    Per-block (opt-in): %.1f KB\n", max_smem_per_block_optin / 1024.0);
  printf("    Per-SM: %.1f KB\n", max_smem_per_sm / 1024.0);
  printf("  Occupancy:\n");
  printf("    Max warps/SM: %d\n", max_warps_per_sm());
  printf("    Max threads/SM: %d\n", max_threads_per_sm);
  printf("  L2: %.1f MB\n", l2_cache_size / 1048576.0);
  printf("  Memory bus: %d-bit (%s)\n", memory_bus_width,
         wide_memory_bus() ? "wide" : "narrow");
}

namespace cufinufft {
namespace common {
using std::max;

template<typename T> std::size_t shared_memory_per_point(int dim, int ns) {
  return ns * sizeof(T) * dim       // kernel evaluations
         + sizeof(int) * dim        // indexes
         + sizeof(cuda_complex<T>); // strength
}
template std::size_t shared_memory_per_point<float>(int dim, int ns);
template std::size_t shared_memory_per_point<double>(int dim, int ns);

// Marco: 4/18/25 not 100% happy of having np here, but the alternatives seem worse to me
template<typename T>
std::size_t shared_memory_required(int dim, int ns, int bin_size_x, int bin_size_y,
                                   int bin_size_z, int np) {
  const auto shmem_per_point = shared_memory_per_point<T>(dim, ns);
  const int ns_2             = (ns + 1) / 2;
  std::size_t grid_size      = bin_size_x + 2 * ns_2;
  if (dim > 1) grid_size *= bin_size_y + 2 * ns_2;
  if (dim > 2) grid_size *= bin_size_z + 2 * ns_2;
  return grid_size * sizeof(cuda_complex<T>) + shmem_per_point * np;
}
template std::size_t shared_memory_required<float>(
    int dim, int ns, int bin_size_x, int bin_size_y, int bin_size_z, int np);
template std::size_t shared_memory_required<double>(
    int dim, int ns, int bin_size_x, int bin_size_y, int bin_size_z, int np);

// Largest r with r^dim <= value. std::pow alone is short by one ulp on exact powers
// (pow(64, 1/3.) < 4 in IEEE 754), so retry the next integer.
static int integer_root(double value, int dim) {
  // std::pow of a negative value is NaN, and NaN-to-int is UB
  if (value < 1) return 0;
  auto r = int(std::floor(std::pow(value, 1.0 / dim)));
  // successor in int64: r + 1 overflows int at value = INT_MAX
  if (std::pow(double(std::int64_t(r) + 1), dim) <= value) ++r;
  return r;
}

// Points the derived pick stages per batch: one warp. Larger np costs shared
// memory and occupancy without adding work.
static constexpr int method3_default_np = 32;

static bool has_user_bin(const cufinufft_opts &opts) {
  return (opts.gpu_binsizex | opts.gpu_binsizey | opts.gpu_binsizez) != 0;
}

static int method3_target_bin(int dim) {
  if (dim == 1) return 256;
  if (dim == 2) return 16;
  return 6;
}

// Function to find bin_size_x == bin_size_y
// where bin_size_x * bin_size_y * bin_size_z < mem_size
template<typename T> int find_bin_size(std::size_t mem_size, int dim, int ns) {
  const auto elements        = mem_size / sizeof(cuda_complex<T>);
  const auto padded_bin_size = integer_root(double(elements), dim);
  const auto bin_size        = padded_bin_size - 2 * ((ns + 1) / 2);
  return bin_size;
}

template<typename T>
void cufinufft_setup_binsize(const GpuCapabilities &gpu, int type, int ns, int dim,
                             const CUFINUFFT_BIGINT *mstu, cufinufft_opts *opts) {
  const int shmem_limit = gpu.max_smem_per_block_optin;

  gpu.print_classification(opts->debug);

  auto set_bins = [&](int bin) {
    if (opts->gpu_binsizex == 0) opts->gpu_binsizex = bin;
    if (opts->gpu_binsizey == 0) opts->gpu_binsizey = (dim >= 2) ? bin : 1;
    if (opts->gpu_binsizez == 0) opts->gpu_binsizez = (dim >= 3) ? bin : 1;
  };

  auto validate_fit = [&](int np) {
    size_t need = shared_memory_required<T>(dim, ns, opts->gpu_binsizex,
                                            opts->gpu_binsizey, opts->gpu_binsizez, np);
    if (need > static_cast<size_t>(shmem_limit)) {
      throw std::runtime_error("[cufinufft] Config exceeds " +
                               std::to_string(shmem_limit) + " bytes available (needs " +
                               std::to_string(need) + " bytes)");
    }
  };

  auto debug_print = [&](int method, int np, const char *note) {
    if (opts->debug < 1) return;
    printf("[cufinufft] Method %d: dim=%d, ns=%d, bin=%dx%dx%d", method, dim, ns,
           opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez);
    if (np > 0) printf(", np=%d", np);
    if (note[0]) printf(" %s", note);
    printf("\n");
    if (opts->debug >= 2) {
      size_t use = shared_memory_required<T>(dim, ns, opts->gpu_binsizex,
                                             opts->gpu_binsizey, opts->gpu_binsizez, np);
      printf("  Shmem: %zu/%d bytes (%.1f%%)\n", use, shmem_limit,
             100.0 * use / shmem_limit);
    }
  };

  switch (opts->gpu_method) {
  case 1: {
    // The GM kernel stages no tile in shared memory; the bin width only sets setpts'
    // sort granularity, which trades locality of the scattered grid atomics (small
    // bins) against their spread over L2 sectors (large bins). bin = nf makes one bin
    // per dim and skips the sort entirely.

    // Fine grid per dim; type 3's outer plan and unset modes leave it unknown.
    std::array<CUFINUFFT_BIGINT, 3> nf{1, 1, 1};
    bool known = type != 3;
    for (int d = 0; known && d < dim; ++d) {
      known = mstu[d] > 0;
      // fine_grid_len is the plan's next235 rule: nf[d] is the exact fine-grid length.
      if (opts->gpu_spreadinterponly)
        nf[d] = mstu[d];
      else
        nf[d] = CUFINUFFT_BIGINT(
            finufft::common::fine_grid_len(opts->upsampfac, mstu[d], ns));
    }

    // Halvings per dim, bin = ceil(nf / 2^shift); none selects the pre-#807 fixed bins.
    std::optional<int> shift;
    const auto cells  = std::int64_t(nf[0]) * nf[1] * nf[2];
    const auto budget = gpu.l2_complex_budget<T>();
    const bool wide   = gpu.wide_memory_bus();
    if (known) {
      if (type == 2) {
        // t2 gather reads the grid through the texture path: 1D never wants a sort, 2D
        // only pays once the grid leaves ~2 budgets, 3D always sorts.
        if (dim == 1 || (dim == 2 && cells <= 2 * budget)) shift = 0;
      } else if (cells <= ((wide || dim == 1) ? 2 * budget : budget)) {
        // t1 scatter keeps one bin while the grid stays L2-resident: 2 budgets on
        // wide-bus parts and in 1D, 1 budget on 384-bit GDDR.
        shift = 0;
      } else if (dim > 1) {
        // Then halve per dim while each 2^(k*dim) tile still fits K, inside a per-dim
        // spill band. K: 1 budget on Hopper/Blackwell, 1/2 on split-L2 sm_80 and GDDR.
        const std::int64_t tile_cap = (wide && gpu.cc_major != 8) ? budget : budget / 2;
        // Band edges: where the halved bins stopped beating the fixed bins in the sweep.
        const std::int64_t band     = [&] {
          if (dim == 2 && wide) return std::numeric_limits<std::int64_t>::max();
          if (dim == 2) return 4 * budget;
          if (wide) return 8 * budget;
          return 2 * budget;
        }();
        if (cells <= band)
          for (int k = 1; k <= 2 && !shift; ++k)
            if ((cells >> (k * dim)) <= tile_cap) shift = k;
      }
    }

    if (shift) {
      const auto halved = [&](int d) {
        const auto v = (nf[d] + (CUFINUFFT_BIGINT(1) << *shift) - 1) >> *shift;
        return int(std::clamp<CUFINUFFT_BIGINT>(v, 1, std::numeric_limits<int>::max()));
      };
      if (opts->gpu_binsizex == 0) opts->gpu_binsizex = halved(0);
      if (opts->gpu_binsizey == 0) opts->gpu_binsizey = dim >= 2 ? halved(1) : 1;
      if (opts->gpu_binsizez == 0) opts->gpu_binsizez = dim >= 3 ? halved(2) : 1;
    } else
      set_bins(dim == 1 ? 1024 : dim == 2 ? 40 : 8);
    debug_print(1, 0, "");
    break;
  }

  case 2: {
    // One subproblem is one block's batch of <= maxsubprobsize points, so size the
    // bin to fill a subproblem at the ~1 point/cell design density: bin^dim = msub,
    // capped by the largest padded tile that fits shared memory. Dense f64 3D on
    // large-smem parts wants a smaller bin, but its optimum is device-jagged and
    // method 3 serves those cells faster, so no table is worth reintroducing.
    if (opts->gpu_maxsubprobsize < 1)
      throw std::runtime_error("[cufinufft] Invalid gpu_maxsubprobsize (" +
                               std::to_string(opts->gpu_maxsubprobsize) +
                               ") for Method 2.");
    const int bin = std::min(integer_root(double(opts->gpu_maxsubprobsize), dim),
                             find_bin_size<T>(shmem_limit, dim, ns));
    if (bin < 1)
      throw std::runtime_error("[cufinufft] Insufficient shmem for Method 2 (ns=" +
                               std::to_string(ns) + "). Try Method 1.");
    set_bins(bin);
    validate_fit(0);
    debug_print(2, 0, "");
    break;
  }

  case 3: {
    const auto config = method3_shared_config<T>(gpu, dim, ns, *opts);
    if (!config)
      throw std::runtime_error("[cufinufft] Insufficient shmem for Method 3 (ns=" +
                               std::to_string(ns) + ", gpu_np=" +
                               std::to_string(opts->gpu_np) + ")");
    const bool user_np  = (opts->gpu_np != 0);
    const bool user_bin = has_user_bin(*opts);
    const char *note    = user_bin ? (user_np ? "(user)" : "(user bin)")
                                   : (user_np ? "(user np)" : "");
    opts->gpu_binsizex  = config->binsize[0];
    opts->gpu_binsizey  = config->binsize[1];
    opts->gpu_binsizez  = config->binsize[2];
    opts->gpu_np        = config->np;
    debug_print(3, opts->gpu_np, note);
    break;
  }

  case 4:
    if (opts->gpu_obinsizex == 0) opts->gpu_obinsizex = 8;
    if (opts->gpu_obinsizey == 0) opts->gpu_obinsizey = 8;
    if (opts->gpu_obinsizez == 0) opts->gpu_obinsizez = 8;
    if (opts->gpu_binsizex == 0) opts->gpu_binsizex = 4;
    if (opts->gpu_binsizey == 0) opts->gpu_binsizey = 4;
    if (opts->gpu_binsizez == 0) opts->gpu_binsizez = 4;
    break;

  default:
    throw std::runtime_error(
        "[cufinufft] Invalid gpu_method=" + std::to_string(opts->gpu_method));
  }

  if (opts->gpu_binsizex < 1 || opts->gpu_binsizey < 1 || opts->gpu_binsizez < 1)
    throw std::runtime_error(
        "[cufinufft] BUG: Invalid bin sizes (method=" + std::to_string(opts->gpu_method) +
        ", ns=" + std::to_string(ns) + ")");
}

template<typename T>
std::optional<method3_config> method3_shared_config(const GpuCapabilities &gpu, int dim,
                                                    int ns, const cufinufft_opts &opts) {
  const int shmem_limit  = gpu.max_smem_per_block_optin;
  const int shmem_per_pt = static_cast<int>(shared_memory_per_point<T>(dim, ns));
  const bool user_np     = (opts.gpu_np != 0);
  const bool user_bin    = has_user_bin(opts);

  // The bin response is a broad valley: its lower edge (halo amplification,
  // (1+2*ceil(ns/2)/bin)^dim work per point) is device-independent, its upper edge
  // (resident blocks/SM = smem/tile) only rises with more shared memory. A
  // low-valley pick therefore transfers across devices: bin 256/16/6, capped by the
  // largest tile that fits.
  const auto valley_bin  = [&] {
    // The tile is capped against the memory left after the points.
    const int tile_mem = std::max(0, shmem_limit - method3_default_np * shmem_per_pt);
    return std::min(method3_target_bin(dim), find_bin_size<T>(tile_mem, dim, ns));
  };

  method3_config c{{opts.gpu_binsizex, opts.gpu_binsizey, opts.gpu_binsizez},
                   opts.gpu_np};
  const auto set_bins = [&](int bin) {
    if (c.binsize[0] == 0) c.binsize[0] = bin;
    if (c.binsize[1] == 0) c.binsize[1] = (dim >= 2) ? bin : 1;
    if (c.binsize[2] == 0) c.binsize[2] = (dim >= 3) ? bin : 1;
  };
  const auto shmem_needed = [&](int np) {
    return shared_memory_required<T>(dim, ns, c.binsize[0], c.binsize[1], c.binsize[2],
                                     np);
  };

  if (user_np && !user_bin) {
    const int avail = shmem_limit - c.np * shmem_per_pt;
    if (avail <= 0) return std::nullopt;
    set_bins(find_bin_size<T>(avail, dim, ns));
  } else if (user_bin) {
    // The bin-size guard rejects a 0 dim: fill the unset dims with the derived pick.
    set_bins(std::max(1, valley_bin()));
    if (!user_np) { // fill the shared memory left after the tile
      const std::size_t tile = shmem_needed(0);
      if (tile + 16 * std::size_t(shmem_per_pt) > std::size_t(shmem_limit))
        return std::nullopt;
      const int rem = shmem_limit - int(tile);
      c.np          = (rem / shmem_per_pt) & ~15;
    }
  } else {
    set_bins(valley_bin());
    c.np = method3_default_np;
  }
  if (c.binsize[0] < 1 || c.binsize[1] < 1 || c.binsize[2] < 1) return std::nullopt;
  if (shmem_needed(c.np) > std::size_t(shmem_limit)) return std::nullopt;
  return c;
}

template<typename T>
bool method3_uses_global(const GpuCapabilities &gpu, int dim, int ns,
                         const cufinufft_opts &opts) {
  const auto shared = method3_shared_config<T>(gpu, dim, ns, opts);
  if (!shared) return true;
  return !has_user_bin(opts) && 4 * shared->binsize[0] <= 3 * method3_target_bin(dim);
}

std::int64_t method3_tile_cells(int dim, int ns, const cufinufft_opts &opts) {
  const int pad      = 2 * ((ns + 1) / 2);
  const std::array<int, 3> bins = {opts.gpu_binsizex, opts.gpu_binsizey,
                                   opts.gpu_binsizez};
  std::int64_t cells = 1;
  for (int d = 0; d < dim; ++d) {
    cells *= std::int64_t(bins[d]) + pad;
    // already over the budget; stop before overflow
    if (cells > std::numeric_limits<int>::max()) break;
  }
  return cells;
}

template<typename T>
void cufinufft_setup_method3_global(const GpuCapabilities &gpu, int dim, int ns,
                                    cufinufft_opts *opts) {
  const int bin = method3_target_bin(dim);
  if (opts->gpu_binsizex == 0) opts->gpu_binsizex = bin;
  if (opts->gpu_binsizey == 0) opts->gpu_binsizey = (dim >= 2) ? bin : 1;
  if (opts->gpu_binsizez == 0) opts->gpu_binsizez = (dim >= 3) ? bin : 1;
  if (opts->gpu_np == 0) opts->gpu_np = method3_default_np;
  if (opts->gpu_binsizex < 1 || opts->gpu_binsizey < 1 || opts->gpu_binsizez < 1 ||
      opts->gpu_np < 1) {
    fprintf(stderr, "[cufinufft] Invalid bin %dx%dx%d or gpu_np (%d) for Method 3.\n",
            opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, opts->gpu_np);
    throw finufft::exception(FINUFFT_ERR_INVALID_ARGUMENT);
  }
  const std::size_t points = shared_memory_per_point<T>(dim, ns) * opts->gpu_np;
  if (points > std::size_t(gpu.max_smem_per_block_optin)) {
    fprintf(stderr,
            "[cufinufft] Method 3 with gpu_np=%d needs %zu bytes of shared memory, but "
            "the device supports only %d bytes.\n",
            opts->gpu_np, points, gpu.max_smem_per_block_optin);
    throw finufft::exception(FINUFFT_ERR_INSUFFICIENT_SHMEM);
  }
  const std::int64_t cells = method3_tile_cells(dim, ns, *opts);
  if (cells > std::int64_t(method3_global_budget / sizeof(cuda_complex<T>))) {
    fprintf(stderr,
            "[cufinufft] Method 3 tile %dx%dx%d is larger than the %zu-byte limit of "
            "global memory.\n",
            opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez,
            method3_global_budget);
    throw finufft::exception(FINUFFT_ERR_MAXNALLOC);
  }
  if (opts->debug >= 1)
    printf("[cufinufft] Method 3: dim=%d, ns=%d, bin=%dx%dx%d, np=%d (global)\n", dim, ns,
           opts->gpu_binsizex, opts->gpu_binsizey, opts->gpu_binsizez, opts->gpu_np);
  if (opts->debug >= 2)
    printf("  Shmem: %zu/%d bytes, global memory: %zu bytes per block\n", points,
           gpu.max_smem_per_block_optin, std::size_t(cells) * sizeof(cuda_complex<T>));
}

template<typename T>
int choose_batchsize(const GpuCapabilities &gpu, const cufinufft_opts &opts, int ntransf,
                     std::int64_t nf) {
  // Cap at ntransf: a larger batch would make cuFFT transform grids that are then
  // discarded.
  if (opts.gpu_maxbatchsize) return std::min(opts.gpu_maxbatchsize, ntransf);

  // No FFT to amortize a batch against, so it would only widen the working set. For a
  // type 3's outer plan it would also evict the L2 that the inner type-2's FFT needs.
  // Timings: https://github.com/flatironinstitute/finufft/pull/873
  if (opts.gpu_spreadinterponly) return 1;

  // Keep nf*batchsize inside the L2 budget, up to 32 to fill the SMs at small nf. Past
  // the budget a batch only adds FFT work the grid cannot hold. Multi-GPU timings:
  // https://github.com/flatironinstitute/finufft/pull/873
  const std::int64_t l2_elems = gpu.l2_complex_budget<T>();
  const int cap = int(std::clamp<std::int64_t>(l2_elems / nf, 1, std::min(ntransf, 32)));

  // Spread out the batches evenly: the cuFFT plan is fixed at batchsize, and cufft_ex has
  // no per-call count, so the last batch always transforms batchsize grids even when only
  // blksize of them hold data. ntransf=9 with cap 8 would do 2x8 grid FFTs for 9
  // transforms; balancing gives 2x5 instead. Never above cap, so the L2 bound still
  // holds.
  const int nbatch = 1 + (ntransf - 1) / cap;
  return 1 + (ntransf - 1) / nbatch;
}

template void cufinufft_setup_binsize<float>(const GpuCapabilities &, int type, int ns,
                                             int dim, const CUFINUFFT_BIGINT *mstu,
                                             cufinufft_opts *opts);
template void cufinufft_setup_binsize<double>(const GpuCapabilities &, int type, int ns,
                                              int dim, const CUFINUFFT_BIGINT *mstu,
                                              cufinufft_opts *opts);
template std::optional<method3_config> method3_shared_config<float>(
    const GpuCapabilities &, int, int, const cufinufft_opts &);
template std::optional<method3_config> method3_shared_config<double>(
    const GpuCapabilities &, int, int, const cufinufft_opts &);
template bool method3_uses_global<float>(const GpuCapabilities &, int, int,
                                         const cufinufft_opts &);
template bool method3_uses_global<double>(const GpuCapabilities &, int, int,
                                          const cufinufft_opts &);
template void cufinufft_setup_method3_global<float>(const GpuCapabilities &, int, int,
                                                    cufinufft_opts *);
template void cufinufft_setup_method3_global<double>(const GpuCapabilities &, int, int,
                                                     cufinufft_opts *);
template int choose_batchsize<float>(const GpuCapabilities &, const cufinufft_opts &, int,
                                     std::int64_t);
template int choose_batchsize<double>(const GpuCapabilities &, const cufinufft_opts &,
                                      int, std::int64_t);
} // namespace common
} // namespace cufinufft

template<typename T> std::size_t cufinufft_plan_t<T>::shared_memory_required() const {
  if (method3_global)
    return cufinufft::common::shared_memory_per_point<T>(dim, spopts.nspread) *
           opts.gpu_np;
  return cufinufft::common::shared_memory_required<T>(
      dim, spopts.nspread, opts.gpu_binsizex, opts.gpu_binsizey, opts.gpu_binsizez,
      opts.gpu_np);
}
template std::size_t cufinufft_plan_t<float>::shared_memory_required() const;
template std::size_t cufinufft_plan_t<double>::shared_memory_required() const;
