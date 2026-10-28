// Per-dim instantiation TU: output-driven spread (gpu_method = 3).
// Compiled three times via CMake foreach with -DCUFINUFFT_DIM={1,2,3}.
// Prep is shared with gpu_method=2 and instantiated in spread_subprob_inst.cu.

#include "spreadinterp_common.cuh"
#include <cstdint>
#include <cufinufft/spreadinterp.hpp>
#include <poet/poet.hpp>

#ifndef CUFINUFFT_DIM
#error "CUFINUFFT_DIM must be defined to 1, 2, or 3 (set by CMake)"
#endif

namespace cufinufft {
namespace spreadinterp {

// Spread one subproblem's points into a tile in scratch, then add the tile to fw.
// scratch is shared memory or the block's buffer in global memory.
template<typename T, int KEREVALMETH, int ndim, int ns>
__device__ FINUFFT_FLATTEN void spread_output_driven_subprob(
    const cufinufft_gpu_data<T> &p, const cuda_complex<T> *c, cuda_complex<T> *fw, int np,
    int subpidx, char *scratch) {
  T sigma   = p.sigma;
  T es_c    = p.es_c;
  T es_beta = p.es_beta;

  // assume that bin_size > ns/2;
  auto info         = compute_subprob_block_info<T, ndim>(p, subpidx);
  auto &binsizes    = info.binsizes;
  auto &offset      = info.offset;
  const int ptstart = info.ptstart;
  const int nupts   = info.nupts;

  static constexpr auto ns_2  = (ns + 1) / 2;
  int total                   = 1;

  for (int idim = 0; idim < ndim; ++idim) total *= ns;

  auto [padded_size, local_subgrid_size] = get_padded_subgrid_info<ndim, ns>(binsizes);

  using kernel_data = cuda::std::array<cuda::std::array<T, ns>, ndim>;
  auto *kerevals    = reinterpret_cast<kernel_data *>(scratch);
  // Offset pointer into scratch after kerevals
  auto *nupts_sm =
      reinterpret_cast<cuda_complex<T> *>(scratch + np * sizeof(kernel_data));
  auto *shift = reinterpret_cast<cuda::std::array<int, ndim> *>(
      scratch + np * sizeof(kernel_data) + np * sizeof(cuda_complex<T>));

  auto *local_subgrid = reinterpret_cast<cuda_complex<T> *>(
      scratch + np * sizeof(kernel_data) + np * sizeof(cuda_complex<T>) +
      np * sizeof(cuda::std::array<int, ndim>));

  // set local_subgrid to zero
  for (int i = threadIdx.x; i < local_subgrid_size; i += blockDim.x) {
    local_subgrid[i] = {0, 0};
  }

  __syncthreads();

  for (int batch_begin = 0; batch_begin < nupts; batch_begin += np) {
    const auto batch_size = min(np, nupts - batch_begin);
    for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
      const int nuptsidx      = loadReadOnly(p.idxnupts + ptstart + i + batch_begin);
      nupts_sm[i]             = loadReadOnly(c + nuptsidx);
      auto [ker, local_shift] = get_kerval_and_local_start<T, KEREVALMETH, ndim, ns>(
          nuptsidx, p.xyz, p.nf123, offset, sigma, es_c, es_beta);
      kerevals[i] = ker;
      shift[i]    = local_shift;
    }
    __syncthreads();

    for (auto i = 0; i < batch_size; i++) {
      const auto cnow  = nupts_sm[i];
      const auto start = shift[i];

      for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        // strength from scratch
        int tmp       = idx;
        int idxout    = 0;
        T kervalue    = 1;
        int strideout = 1;
        for (int idim = 0; idim + 1 < ndim; ++idim) {
          int s = tmp % ns;
          kervalue *= kerevals[i][idim][s];
          idxout += strideout * (s + start[idim] + ns_2);
          strideout *= padded_size[idim];
          tmp /= ns;
        }
        // last dimension can be done more cheaply
        kervalue *= kerevals[i][ndim - 1][tmp];
        idxout += strideout * (tmp + start[ndim - 1] + ns_2);

        local_subgrid[idxout] += cnow * kervalue;
      }
      __syncthreads();
    }
  }
  /* write to global memory */
  // The tile can be wider than the fine grid, e.g. a 1D bin of 256 on a small grid.
  // Cells past nf + ns/2 get no points and would be written out of bounds, so skip
  // them as method 2 does.
  shared_mem_copy_helper<T, ndim, ns>(
      binsizes, offset, p.nf123, [fw, local_subgrid](int idx_local, int idx_global) {
        atomicAddComplexGlobal<T>(fw + idx_global, local_subgrid[idx_local]);
      });
}

// Output-driven spreading kernel. Shared memory: one block per subproblem. Global
// memory: fewer blocks, each doing subproblems blockIdx.x, blockIdx.x + gridDim.x, ...
// in its own buffer.
template<typename T, int KEREVALMETH, int ndim, int ns, bool GLOBAL>
__global__ FINUFFT_FLATTEN void spread_output_driven(
    cufinufft_gpu_data<T> p, const cuda_complex<T> *c, cuda_complex<T> *fw, int np,
    char *scratch, std::size_t block_bytes) {
  extern __shared__ char sharedbuf[];
  if constexpr (!GLOBAL) {
    spread_output_driven_subprob<T, KEREVALMETH, ndim, ns>(p, c, fw, np, blockIdx.x,
                                                           sharedbuf);
  } else {
    char *buffer = scratch + blockIdx.x * block_bytes;
    for (int subpidx = blockIdx.x; subpidx < p.totalnumsubprob; subpidx += gridDim.x) {
      spread_output_driven_subprob<T, KEREVALMETH, ndim, ns>(p, c, fw, np, subpidx,
                                                             buffer);
      // Wait until all threads are done with this tile before the next one clears it.
      __syncthreads();
    }
  }
}

// Output-driven spreading CPU driver
template<typename T, int ndim, int ns>
void spread_output_driven_launch(const cufinufft_plan_t<T> &d_plan,
                                 const cuda_complex<T> *c, cuda_complex<T> *fw,
                                 int blksize) {
  int bufsz = 1;
  for (int idim = 0; idim < ndim; ++idim) bufsz *= ns;

  const bool global            = d_plan.opts.gpu_m3_storage == 2;
  // Global memory needs no dynamic shared memory.
  const auto sharedplanorysize = global ? 0 : d_plan.shared_memory_required();
  const int nthreads           = std::min(256, std::max(bufsz, d_plan.opts.gpu_np));
  const int nblocks = global ? std::min(d_plan.method3_blocks, d_plan.totalnumsubprob)
                             : d_plan.totalnumsubprob;
  char *scratch = global ? const_cast<char *>(dethrust(d_plan.method3_scratch)) : nullptr;
  const std::size_t block_bytes =
      global ? cufinufft::common::method3_global_block_bytes<T>(ndim, ns, d_plan.opts)
             : 0;

  const auto launch = [&](auto kernel) {
    if (!global) {
      cufinufft_set_shared_memory(kernel, d_plan);
      cudaFuncSetSharedMemConfig(kernel, cudaSharedMemBankSizeEightByte);
      THROW_IF_CUDA_ERROR();
    }
    for (std::int64_t t = 0; t < blksize; t++) {
      kernel<<<nblocks, nthreads, sharedplanorysize, d_plan.stream>>>(
          d_plan, c + t * d_plan.M, fw + t * d_plan.nf, d_plan.opts.gpu_np, scratch,
          block_bytes);
      THROW_IF_CUDA_ERROR();
    }
  };
  const bool horner = d_plan.opts.gpu_kerevalmeth == 1;
  if (global)
    horner ? launch(spread_output_driven<T, 1, ndim, ns, true>)
           : launch(spread_output_driven<T, 0, ndim, ns, true>);
  else
    horner ? launch(spread_output_driven<T, 1, ndim, ns, false>)
           : launch(spread_output_driven<T, 0, ndim, ns, false>);
}

template<typename T, int Ndim> struct SpreadOutputDrivenCaller {
  const cufinufft_plan_t<T> &p;
  const cuda_complex<T> *c;
  cuda_complex<T> *fw;
  int blksize;
  template<int Ns> void operator()() const {
    spread_output_driven_launch<T, Ndim, Ns>(p, c, fw, blksize);
  }
};

template<typename T, int Ndim>
void do_spread_output_driven(const cufinufft_plan_t<T> &p, const cuda_complex<T> *c,
                             cuda_complex<T> *fw, int blksize) {
  using namespace finufft::common;
  SpreadOutputDrivenCaller<T, Ndim> caller{p, c, fw, blksize};
  using NsSeq = poet::inclusive_range<MIN_NSPREAD, MAX_NSPREAD<T>>;
  poet::dispatch(caller, std::make_tuple(poet::dispatch_param<NsSeq>{p.spopts.nspread}));
}

template void do_spread_output_driven<float, CUFINUFFT_DIM>(
    const cufinufft_plan_t<float> &, const cuda_complex<float> *, cuda_complex<float> *,
    int);
template void do_spread_output_driven<double, CUFINUFFT_DIM>(
    const cufinufft_plan_t<double> &, const cuda_complex<double> *,
    cuda_complex<double> *, int);

} // namespace spreadinterp
} // namespace cufinufft
