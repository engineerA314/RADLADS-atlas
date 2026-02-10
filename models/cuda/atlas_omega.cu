// atlas_omega.cu - CUDA kernels for Atlas Omega RNN
// Exact omega_window=16 implementation with checkpoint-based backward
// Based on expert's provided code

#include <assert.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#if defined(__HIP_PLATFORM_HCC__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#define bf __hip_bfloat16
__device__ __forceinline__ float to_float(bf x) { return __bfloat162float(x); }
__device__ __forceinline__ bf to_bf(float x) { return __float2bfloat16(x); }
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#define bf __nv_bfloat16
__device__ __forceinline__ float to_float(bf x) { return __bfloat162float(x); }
__device__ __forceinline__ bf to_bf(float x) { return __float2bfloat16(x); }
#endif

#ifndef _C_
#define _C_ 64
#endif

#ifndef _E_
#define _E_ 16
#endif

static constexpr int C = _C_;
static constexpr int E = _E_;
// Fixed checkpoint interval for backward re-anchoring (numerical stability).
// Must be a power of two for the (t & (K-1)) fast-path below.
static constexpr int K = 4;

__device__ __forceinline__ float clamp_eps(float x, float eps=1e-4f) {
    return x < eps ? eps : x;
}

// Forward kernel
// phi_k, phi_q, delta: [BH, T, C] bf16
// lr, decay, beta, gate: [BH, T] bf16
// S0, Z0: [BH, C, C] bf16 (row-major)
// y: [BH, T, C] bf16
// S_T, Z_T: [BH, C, C] bf16
__global__ void forward_exact_kernel(
    int T,
    int n_ckpt,
    const bf* __restrict__ phi_k,
    const bf* __restrict__ phi_q,
    const bf* __restrict__ delta,
    const bf* __restrict__ lr,
    const bf* __restrict__ decay,
    const bf* __restrict__ beta,
    const bf* __restrict__ gate,
    const bf* __restrict__ S0,
    const bf* __restrict__ Z0,
    bf* __restrict__ y,
    bf* __restrict__ S_T,
    bf* __restrict__ Z_T,
    float* __restrict__ S_ckpt,
    float* __restrict__ Z_ckpt
) {
    int bh = (int)blockIdx.x;
    int i = (int)threadIdx.x;  // row index 0..63

    // shared ring for phi (vector), delta (vector), and gate (scalar)
    __shared__ bf phi_ring[E][C];
    __shared__ bf delta_ring[E][C];  // FIXED: store full delta vector, not per-thread scalar
    __shared__ float gate_ring[E];
    __shared__ float qvec[C];

    // init shared ring
    #pragma unroll
    for (int s=0; s<E; s++) {
        phi_ring[s][i] = to_bf(0.f);
        delta_ring[s][i] = to_bf(0.f);  // FIXED: init full vector
    }
    if (i < E) gate_ring[i] = 0.f;
    __syncthreads();

    // load initial states
    float Srow[C];
    float Zrow[C];
    float gwin[C];
    #pragma unroll
    for (int j=0; j<C; j++) {
        int idx = bh*C*C + i*C + j;
        Srow[j] = to_float(S0[idx]);
        Zrow[j] = to_float(Z0[idx]);
        gwin[j] = 0.f;
    }

    // main loop
    for (int t=0; t<T; t++) {
        // load q vector into shared
        qvec[i] = to_float(phi_q[(bh*T + t)*C + i]);
        __syncthreads();

        // y_t = S_{t-1} @ q_t (row i)
        float yi = 0.f;
        #pragma unroll
        for (int j=0; j<C; j++) yi += Srow[j] * qvec[j];
        y[(bh*T + t)*C + i] = to_bf(yi);
        __syncthreads();

        // update gwin by ring: gwin += g_raw[t] - g_raw[t-E]
        // g[i,j] = delta[i] * phi[j] (outer product)
        int slot = t & (E-1);
        if (t >= E) {
            // Remove old contribution: gwin[j] -= gate * delta[i] * phi[j]
            float delta_i_old = to_float(delta_ring[slot][i]);  // FIXED: use row i's delta
            float coeff_old = gate_ring[slot] * delta_i_old;
            #pragma unroll
            for (int j=0; j<C; j++) {
                gwin[j] -= coeff_old * to_float(phi_ring[slot][j]);
            }
        }

        // overwrite slot with current token
        phi_ring[slot][i] = phi_k[(bh*T + t)*C + i];
        delta_ring[slot][i] = delta[(bh*T + t)*C + i];  // FIXED: store full vector
        if (i == 0) gate_ring[slot] = to_float(gate[bh*T + t]);
        __syncthreads();

        // Add new contribution: gwin[j] += gate * delta[i] * phi[j]
        float delta_i_cur = to_float(delta_ring[slot][i]);  // FIXED: use row i's delta
        float coeff_cur = gate_ring[slot] * delta_i_cur;
        #pragma unroll
        for (int j=0; j<C; j++) {
            gwin[j] += coeff_cur * to_float(phi_ring[slot][j]);
        }

        // load scalars
        float lr_t = to_float(lr[bh*T + t]);
        float decay_t = clamp_eps(to_float(decay[bh*T + t]));
        float beta_t = clamp_eps(to_float(beta[bh*T + t]));

        // momentum + state update
        #pragma unroll
        for (int j=0; j<C; j++) {
            Zrow[j] = beta_t * Zrow[j] + gwin[j];
            Srow[j] = decay_t * Srow[j] - lr_t * Zrow[j];
        }
        __syncthreads();

        // Save checkpoints at fixed interval (and always at the end).
        if (((t & (K - 1)) == (K - 1)) || (t == T - 1)) {
            int ckpt_idx = t / K;
            if (ckpt_idx < n_ckpt) {
                #pragma unroll
                for (int j=0; j<C; j++) {
                    int off = ((bh * n_ckpt + ckpt_idx) * C * C) + i * C + j;
                    S_ckpt[off] = Srow[j];
                    Z_ckpt[off] = Zrow[j];
                }
            }
        }
    }

    // save final states
    #pragma unroll
    for (int j=0; j<C; j++) {
        int idx = bh*C*C + i*C + j;
        S_T[idx] = to_bf(Srow[j]);
        Z_T[idx] = to_bf(Zrow[j]);
    }
}

// Backward kernel
// dy: [BH,T,C] bf16
// dS_T,dZ_T: [BH,C,C] bf16 (can be zeros)
// outputs: dphi_k,dphi_q,ddelta: [BH,T,C] bf16
// dlr,ddecay,dbeta,dgate: [BH,T] bf16
// dS0,dZ0: [BH,C,C] bf16
// dg_ring_global: [BH, E, C, C] fp32 (global memory for "future 16 sum")
__global__ void backward_exact_kernel(
    int T,
    int n_ckpt,
    const bf* __restrict__ phi_k,
    const bf* __restrict__ phi_q,
    const bf* __restrict__ delta,
    const bf* __restrict__ lr,
    const bf* __restrict__ decay,
    const bf* __restrict__ beta,
    const bf* __restrict__ gate,
    const bf* __restrict__ S_T,
    const bf* __restrict__ Z_T,
    const float* __restrict__ S_ckpt,
    const float* __restrict__ Z_ckpt,
    const bf* __restrict__ S0,
    const bf* __restrict__ Z0,
    const bf* __restrict__ dy,
    const bf* __restrict__ dS_T_in,
    const bf* __restrict__ dZ_T_in,
    bf* __restrict__ dphi_k,
    bf* __restrict__ dphi_q,
    bf* __restrict__ ddelta,
    bf* __restrict__ dlr,
    bf* __restrict__ ddecay,
    bf* __restrict__ dbeta,
    bf* __restrict__ dgate,
    bf* __restrict__ dS0,
    bf* __restrict__ dZ0,
    float* __restrict__ dg_ring_global  // global memory ring buffer (fp32)
) {
    int bh = (int)blockIdx.x;
    int i = (int)threadIdx.x;  // row 0..63

    // ---- shared for mat-vec ops (no dg_ring, now in global memory)
    extern __shared__ float smem[];  // Dynamic shared memory for mat
    float* mat = smem;              // [C * C] float
    __shared__ float v1[C];          // vector buffer (dy or delta)
    __shared__ float v2[C];          // vector buffer (q or phi)
    __shared__ float red[C];         // reduction buffer

    // dg_ring now in global memory: [BH, E, C, C] (fp32)
    // Access pattern: dg_ring_global[bh * E * C * C + slot * C * C + i * C + j]
    float* dg_ring = &dg_ring_global[bh * E * C * C];  // This block's ring buffer

    // init dg_ring to 0 (each thread writes its own row)
    #pragma unroll
    for (int s=0; s<E; s++) {
        #pragma unroll
        for (int j=0; j<C; j++) {
            dg_ring[s * C * C + i * C + j] = 0.f;
        }
    }

    // load end states and end grads
    float Srow[C], Zrow[C], dSrow[C], dZrow[C];
    #pragma unroll
    for (int j=0; j<C; j++) {
        int idx = bh*C*C + i*C + j;
        Srow[j] = to_float(S_T[idx]);
        Zrow[j] = to_float(Z_T[idx]);
        dSrow[j] = to_float(dS_T_in[idx]);
        dZrow[j] = to_float(dZ_T_in[idx]);
    }

    // init gwin for t=T-1 by summing last E tokens (exact)
    // FIXED: Load full delta vector to shared memory
    __shared__ bf delta_vec[C];
    __shared__ bf phi_vec[C];
    
    float gwin[C];
    #pragma unroll
    for (int j=0; j<C; j++) gwin[j] = 0.f;
    int start = (T > E) ? (T - E) : 0;
    for (int p=start; p<T; p++) {
        // Load delta and phi vectors to shared memory
        delta_vec[i] = delta[(bh*T + p)*C + i];
        phi_vec[i] = phi_k[(bh*T + p)*C + i];
        __syncthreads();
        
        float gp = to_float(gate[bh*T + p]);
        float delta_i = to_float(delta_vec[i]);  // FIXED: use row i's delta
        float coeff = gp * delta_i;
        
        #pragma unroll
        for (int j=0; j<C; j++) {
            gwin[j] += coeff * to_float(phi_vec[j]);
        }
        __syncthreads();
    }

    // dg_sum row = sum_{u=t}^{t+E-1} dZ_u (maintained as we go backward)
    float dg_sum[C];
    #pragma unroll
    for (int j=0; j<C; j++) dg_sum[j] = 0.f;

    for (int t=T-1; t>=0; t--) {
        // scalars
        float lr_t = to_float(lr[bh*T + t]);
        float decay_t = clamp_eps(to_float(decay[bh*T + t]));
        float beta_t = clamp_eps(to_float(beta[bh*T + t]));
        float gate_t = to_float(gate[bh*T + t]);

        // Periodically re-anchor S_t and Z_t from forward checkpoints to reduce drift
        if (((t & (K - 1)) == (K - 1)) || (t == T - 1)) {
            int ckpt_idx = t / K;
            if (ckpt_idx < n_ckpt) {
                #pragma unroll
                for (int j=0; j<C; j++) {
                    int off = ((bh * n_ckpt + ckpt_idx) * C * C) + i * C + j;
                    Srow[j] = S_ckpt[off];
                    Zrow[j] = Z_ckpt[off];
                }
            }
        }

        // Recompute S_prev and Z_prev via forward within K-segment to reduce drift
        float Sprev[C];
        float Zprev[C];
        {
            int t_end = (t | (K - 1));
            if (t_end >= T) t_end = T - 1;
            int prev = t_end - K;
            if ((t_end == (T - 1)) && ((t_end & (K - 1)) != (K - 1))) {
                prev = (t_end / K) * K - 1;
            }

            if (prev >= 0) {
                int ckpt_idx = prev / K;
                if (ckpt_idx < n_ckpt) {
                    #pragma unroll
                    for (int j=0; j<C; j++) {
                        int off = ((bh * n_ckpt + ckpt_idx) * C * C) + i * C + j;
                        Sprev[j] = S_ckpt[off];
                        Zprev[j] = Z_ckpt[off];
                    }
                } else {
                    #pragma unroll
                    for (int j=0; j<C; j++) {
                        Sprev[j] = Srow[j];
                        Zprev[j] = Zrow[j];
                    }
                }
            } else {
                #pragma unroll
                for (int j=0; j<C; j++) {
                    int idx = bh*C*C + i*C + j;
                    Sprev[j] = to_float(S0[idx]);
                    Zprev[j] = to_float(Z0[idx]);
                }
            }

            // forward recompute from prev+1 to t-1
            for (int u = prev + 1; u <= t - 1; u++) {
                float gtmp[C];
                #pragma unroll
                for (int j=0; j<C; j++) gtmp[j] = 0.f;
                int start_u = (u > (E - 1)) ? (u - (E - 1)) : 0;
                for (int p = start_u; p <= u; p++) {
                    delta_vec[i] = delta[(bh*T + p)*C + i];
                    phi_vec[i] = phi_k[(bh*T + p)*C + i];
                    __syncthreads();
                    float gp = to_float(gate[bh*T + p]);
                    float delta_i = to_float(delta_vec[i]);
                    float coeff = gp * delta_i;
                    #pragma unroll
                    for (int j=0; j<C; j++) {
                        gtmp[j] += coeff * to_float(phi_vec[j]);
                    }
                    __syncthreads();
                }

                float lr_u = to_float(lr[bh*T + u]);
                float decay_u = clamp_eps(to_float(decay[bh*T + u]));
                float beta_u = clamp_eps(to_float(beta[bh*T + u]));
                #pragma unroll
                for (int j=0; j<C; j++) {
                    Zprev[j] = beta_u * Zprev[j] + gtmp[j];
                    Sprev[j] = decay_u * Sprev[j] - lr_u * Zprev[j];
                }
            }
        }

        // ---- y path: dphi_q[t] = Sprev^T @ dy[t], dSprev += dy ⊗ q^T
        // write Sprev row to mat
        #pragma unroll
        for (int j=0; j<C; j++) mat[i * C + j] = Sprev[j];
        // load dy and q vectors
        v1[i] = to_float(dy[(bh*T + t)*C + i]);  // dy
        v2[i] = to_float(phi_q[(bh*T + t)*C + i]);  // q
        __syncthreads();

        // compute dphi_q element i = column-i dot dy
        float dqi = 0.f;
        #pragma unroll
        for (int r=0; r<C; r++) dqi += mat[r * C + i] * v1[r];
        dphi_q[(bh*T + t)*C + i] = to_bf(dqi);

        // dSprev from y: outer(dy, q)
        float dSprev[C];
        #pragma unroll
        for (int j=0; j<C; j++) dSprev[j] = v1[i] * v2[j];
        __syncthreads();

        // ---- S update backward: S = decay*Sprev - lr*Z
        float part_decay = 0.f;
        float part_lr = 0.f;
        #pragma unroll
        for (int j=0; j<C; j++) {
            part_decay += dSrow[j] * Sprev[j];
            part_lr += dSrow[j] * Zrow[j];
            dSprev[j] += decay_t * dSrow[j];
            dZrow[j] += -lr_t * dSrow[j];
        }

        // reduce and write ddecay, dlr
        red[i] = part_decay;
        __syncthreads();
        if (i == 0) {
            float s = 0.f;
            #pragma unroll
            for (int r=0; r<C; r++) s += red[r];
            ddecay[bh*T + t] = to_bf(s);
        }
        __syncthreads();

        red[i] = part_lr;
        __syncthreads();
        if (i == 0) {
            float s = 0.f;
            #pragma unroll
            for (int r=0; r<C; r++) s += red[r];
            dlr[bh*T + t] = to_bf(-s);
        }
        __syncthreads();

        // ---- Z update backward: Z = beta*Zprev + g
        float part_beta = 0.f;
        #pragma unroll
        for (int j=0; j<C; j++) part_beta += dZrow[j] * Zprev[j];
        red[i] = part_beta;
        __syncthreads();
        if (i == 0) {
            float s = 0.f;
            #pragma unroll
            for (int r=0; r<C; r++) s += red[r];
            dbeta[bh*T + t] = to_bf(s);
        }
        __syncthreads();

        // dZprev for next step
        float dZprev[C];
        #pragma unroll
        for (int j=0; j<C; j++) dZprev[j] = beta_t * dZrow[j];

        // ---- maintain dg_sum = sum_{u=t}^{t+E-1} dZ_u (future window)
        // subtract leaving dZ_{t+E}
        if (t + E < T) {
            int slot_leave = (t + E) & (E - 1);
            int offset_leave = slot_leave * C * C + i * C;
            #pragma unroll
            for (int j=0; j<C; j++) {
            dg_sum[j] -= dg_ring[offset_leave + j];
            }
        }

        // add current dZ_t
        #pragma unroll
        for (int j=0; j<C; j++) dg_sum[j] += dZrow[j];

        // store current dZ_t into ring slot (per-thread row)
        int slot_store = t & (E - 1);
        int offset_store = slot_store * C * C + i * C;
        #pragma unroll
        for (int j=0; j<C; j++) {
            dg_ring[offset_store + j] = dZrow[j];
        }

        // ---- g_raw[t] grads using dGraw = dg_sum
        // FIXED: load full phi and delta vectors for token t to shared memory
        phi_vec[i] = phi_k[(bh*T + t)*C + i];
        delta_vec[i] = delta[(bh*T + t)*C + i];
        __syncthreads();

        // u_i = (dGraw @ phi)_i = dot(dg_sum_row, phi)
        float u = 0.f;
        #pragma unroll
        for (int j=0; j<C; j++) u += dg_sum[j] * to_float(phi_vec[j]);

        // ddelta_i = gate * u_i
        {
            float ddi = gate_t * u;
            ddelta[(bh*T + t)*C + i] = to_bf(ddi);
        }

        // dgate = sum_i delta_i * u_i (FIXED: use full delta vector)
        red[i] = to_float(delta_vec[i]) * u;
        __syncthreads();
        if (i == 0) {
            float s = 0.f;
            #pragma unroll
            for (int r=0; r<C; r++) s += red[r];
            dgate[bh*T + t] = to_bf(s);
        }
        __syncthreads();

        // dphi = gate * (dGraw^T @ delta)
        // reuse mat = dg_sum
        #pragma unroll
        for (int j=0; j<C; j++) mat[i * C + j] = dg_sum[j];
        __syncthreads();

        float vcol = 0.f;
        #pragma unroll
        // vcol = (dG^T @ delta)_i, where dG rows are dg_sum per row.
        // FIXED: v1[] no longer holds delta; use shared delta_vec instead.
        for (int r=0; r<C; r++) vcol += mat[r * C + i] * to_float(delta_vec[r]);
        {
            float dki = gate_t * vcol;
            dphi_k[(bh*T + t)*C + i] = to_bf(dki);
        }
        __syncthreads();

        // ---- advance to next (t-1): update gwin, then overwrite states+grads
        // gwin_{t-1} = gwin_t - g_raw[t] + g_raw[t-E]
        // FIXED: use delta_vec[i] which already loaded in shared memory
        float coeff_cur = gate_t * to_float(delta_vec[i]);
        #pragma unroll
        for (int j=0; j<C; j++) gwin[j] -= coeff_cur * to_float(phi_vec[j]);

        if (t - E >= 0) {
            // FIXED: load old delta and phi vectors to shared memory
            delta_vec[i] = delta[(bh*T + (t - E))*C + i];
            phi_vec[i] = phi_k[(bh*T + (t - E))*C + i];
            __syncthreads();
            
            float gate_old = to_float(gate[bh*T + (t - E)]);
            float del_old_i = to_float(delta_vec[i]);
            float coeff_old = gate_old * del_old_i;
            #pragma unroll
            for (int j=0; j<C; j++) {
                gwin[j] += coeff_old * to_float(phi_vec[j]);
            }
            __syncthreads();
        }

        // overwrite state/grads for next iteration
        #pragma unroll
        for (int j=0; j<C; j++) {
            Srow[j] = Sprev[j];
            Zrow[j] = Zprev[j];
            dSrow[j] = dSprev[j];
            dZrow[j] = dZprev[j];
        }
        __syncthreads();
    }

    // output dS0,dZ0 (grad wrt initial states)
    #pragma unroll
    for (int j=0; j<C; j++) {
        int idx = bh*C*C + i*C + j;
        dS0[idx] = to_bf(dSrow[j]);
        dZ0[idx] = to_bf(dZrow[j]);
    }

}

void cuda_forward_exact(
    int BH, int T,
    int n_ckpt,
    torch::Tensor phi_k,
    torch::Tensor phi_q,
    torch::Tensor delta,
    torch::Tensor lr,
    torch::Tensor decay,
    torch::Tensor beta,
    torch::Tensor gate,
    torch::Tensor S0,
    torch::Tensor Z0,
    torch::Tensor y,
    torch::Tensor S_T,
    torch::Tensor Z_T,
    torch::Tensor S_ckpt,
    torch::Tensor Z_ckpt
) {
    auto stream = at::cuda::getCurrentCUDAStream();
    forward_exact_kernel<<<dim3(BH), dim3(C), 0, stream>>>(
        T,
        n_ckpt,
        (bf*)phi_k.data_ptr<at::BFloat16>(),
        (bf*)phi_q.data_ptr<at::BFloat16>(),
        (bf*)delta.data_ptr<at::BFloat16>(),
        (bf*)lr.data_ptr<at::BFloat16>(),
        (bf*)decay.data_ptr<at::BFloat16>(),
        (bf*)beta.data_ptr<at::BFloat16>(),
        (bf*)gate.data_ptr<at::BFloat16>(),
        (bf*)S0.data_ptr<at::BFloat16>(),
        (bf*)Z0.data_ptr<at::BFloat16>(),
        (bf*)y.data_ptr<at::BFloat16>(),
        (bf*)S_T.data_ptr<at::BFloat16>(),
        (bf*)Z_T.data_ptr<at::BFloat16>(),
        (float*)S_ckpt.data_ptr<float>(),
        (float*)Z_ckpt.data_ptr<float>()
    );
}

void cuda_backward_exact(
    int BH, int T,
    int n_ckpt,
    torch::Tensor phi_k,
    torch::Tensor phi_q,
    torch::Tensor delta,
    torch::Tensor lr,
    torch::Tensor decay,
    torch::Tensor beta,
    torch::Tensor gate,
    torch::Tensor S_T,
    torch::Tensor Z_T,
    torch::Tensor S_ckpt,
    torch::Tensor Z_ckpt,
    torch::Tensor S0,
    torch::Tensor Z0,
    torch::Tensor dy,
    torch::Tensor dS_T,
    torch::Tensor dZ_T,
    torch::Tensor dphi_k,
    torch::Tensor dphi_q,
    torch::Tensor ddelta,
    torch::Tensor dlr,
    torch::Tensor ddecay,
    torch::Tensor dbeta,
    torch::Tensor dgate,
    torch::Tensor dS0,
    torch::Tensor dZ0
) {
    // Allocate global memory for dg_ring: [BH, E, C, C] fp32
    // Size: BH * 16 * 64 * 64 * 4 bytes = BH * 262,144 bytes
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(phi_k.device());
    auto dg_ring_global = torch::zeros({BH, E, C, C}, options);
    
    auto stream = at::cuda::getCurrentCUDAStream();
    size_t shmem_bytes = (size_t)C * (size_t)C * sizeof(float);
#if !defined(__HIP_PLATFORM_HCC__) && !defined(__HIPCC__)
    cudaFuncSetAttribute(
        backward_exact_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)shmem_bytes
    );
#endif
    backward_exact_kernel<<<dim3(BH), dim3(C), shmem_bytes, stream>>>(
        T,
        n_ckpt,
        (bf*)phi_k.data_ptr<at::BFloat16>(),
        (bf*)phi_q.data_ptr<at::BFloat16>(),
        (bf*)delta.data_ptr<at::BFloat16>(),
        (bf*)lr.data_ptr<at::BFloat16>(),
        (bf*)decay.data_ptr<at::BFloat16>(),
        (bf*)beta.data_ptr<at::BFloat16>(),
        (bf*)gate.data_ptr<at::BFloat16>(),
        (bf*)S_T.data_ptr<at::BFloat16>(),
        (bf*)Z_T.data_ptr<at::BFloat16>(),
        (float*)S_ckpt.data_ptr<float>(),
        (float*)Z_ckpt.data_ptr<float>(),
        (bf*)S0.data_ptr<at::BFloat16>(),
        (bf*)Z0.data_ptr<at::BFloat16>(),
        (bf*)dy.data_ptr<at::BFloat16>(),
        (bf*)dS_T.data_ptr<at::BFloat16>(),
        (bf*)dZ_T.data_ptr<at::BFloat16>(),
        (bf*)dphi_k.data_ptr<at::BFloat16>(),
        (bf*)dphi_q.data_ptr<at::BFloat16>(),
        (bf*)ddelta.data_ptr<at::BFloat16>(),
        (bf*)dlr.data_ptr<at::BFloat16>(),
        (bf*)ddecay.data_ptr<at::BFloat16>(),
        (bf*)dbeta.data_ptr<at::BFloat16>(),
        (bf*)dgate.data_ptr<at::BFloat16>(),
        (bf*)dS0.data_ptr<at::BFloat16>(),
        (bf*)dZ0.data_ptr<at::BFloat16>(),
        (float*)dg_ring_global.data_ptr<float>()
    );
    // dg_ring_global will be automatically freed when it goes out of scope
}
