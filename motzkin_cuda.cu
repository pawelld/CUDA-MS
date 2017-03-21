/*******************************************************************************
 * motzkin_cuda.cu
 *
 * Copyright 2017 Pawel Daniluk
 *
 *
 * This file is part of CUDA-MS.
 *
 * CUDA-MS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CUDA-MS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with CUDA-MS.  If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************/

#include<stdio.h>
#include"bitops.h"

#include<unistd.h>

#include"simple_macros.h"
#include"arrays.h"
#include<cuda.h>
#include<curand_kernel.h>

#include<omp.h>

#include"motzkin_cuda.h"
#include"random.h"

#include"common_cuda.cuh"

#define N_ITER 10000

#define ROW_SLICE_L 128
#define ROW_SLICE_S 64

#define VEC_SKIP 32

#define UNSOLVED(s, nmap) ((float)(nmap-s->nzeroes)-s->csize)

#define DIVROUNDUP(val, mod) (((val)+(mod)-1)/(mod))

#define LARGE_N 8192
#define SMALL_N 1024

#define MAX_DOUBLE_SIZE_MAT 40000

curandState_t *d_randstates=0;
int cuda_device=-1;

#define MAX_CUDA_DEV 8

#define MAX_SIM_CUDA_RUNS 16

omp_lock_t dev_read_wait_lock[MAX_CUDA_DEV];

int dev_readcount[MAX_CUDA_DEV];

#pragma omp threadprivate (d_randstates, cuda_device)

/* #define DEBUG */

/* #define NO_CONCURRENCY */

int threaddev[32]={-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
int rand_init[32]={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

static void lib_constructor() __attribute__((constructor));
static void lib_destructor() __attribute__((destructor));

void lib_constructor(void)
{
    for(int i=0; i<MAX_CUDA_DEV; i++) {
        omp_init_lock(&dev_read_wait_lock[i]);
    }
    memset(dev_readcount, 0, sizeof(dev_readcount));
}

void lib_destructor(void)
{
    for(int i=0; i<MAX_CUDA_DEV; i++) {
        omp_destroy_lock(&dev_read_wait_lock[i]);
    }
}

static void checkCUDAError(const char *msg, int line)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
#pragma omp critical (checkCUDAError)
        {
            printf("Cuda error: line: %d:  %s: %s.\n", line, msg, cudaGetErrorString( err) );
            P_INT(omp_get_thread_num()) P_INT(cuda_device) P_NL;
            P_INT_ARR(threaddev, 32) P_NL;
        }
        abort();
    }
}


static void cuda_start(void) 
{

    omp_set_lock(&dev_read_wait_lock[cuda_device]);
#pragma omp critical (read_mutex)
    {
        dev_readcount[cuda_device]++;
        if(dev_readcount[cuda_device]<MAX_SIM_CUDA_RUNS) {
            omp_unset_lock(&dev_read_wait_lock[cuda_device]);
        }
    }
}

static void cuda_end(void)
{
#pragma omp critical (read_mutex)
    {
        dev_readcount[cuda_device]--;
        if(dev_readcount[cuda_device]<MAX_SIM_CUDA_RUNS) {
            omp_unset_lock(&dev_read_wait_lock[cuda_device]);
        }
    }
}

cudaError_t cudaCalloc(void **devPtr, size_t size)
{
    cudaMalloc(devPtr, size);
    return cudaMemset(*devPtr, 0, size);
}

void set_csize(struct cuda_clique_status *d_status, float val, cudaStream_t stream)
{
    cudaMemcpyAsync(&d_status->csize, &val, sizeof(float), cudaMemcpyHostToDevice, stream);
}


template <unsigned int blockSize> __device__ __forceinline__  void maxdiffGPU_dev(float *d_x0, float *d_x1, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, int *d_map)
{
	extern __shared__ float sdata[];

	int n=d_map[-1];

    unsigned int tid = threadIdx.x;
    unsigned int i=tid;

    float norm0=d_x0_status->norm;
    float norm1=d_x1_status->norm;

    sdata[tid]=0;
    while (i < n) {
        sdata[tid] =MAX(sdata[tid], d_x0[i]/norm0 -  d_x1[i]/norm1);
        i += blockSize;
    }

    __syncthreads();

    reduce_max<blockSize, float>(sdata, tid);

    if(tid == 0) {
        d_x1_status->maxdiff=sdata[0];
    }
}

template <unsigned int blockSize> __global__ void maxdiffGPU(float *d_x0, float *d_x1, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, int *d_map)
{
    maxdiffGPU_dev<blockSize>(d_x0, d_x1, d_x0_status, d_x1_status, d_map);
}


template <unsigned int blockSize, int biased> __device__ __forceinline__  void normiterGPU_dev(struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, float *d_x1,  int n, float alpha=0)
{
	extern __shared__ float sdata[];

    unsigned int i=threadIdx.x;

    sdata[threadIdx.x]=0;

    if(biased) sdata[blockSize+threadIdx.x]=0;
    while (i < n) {
        sdata[threadIdx.x] += d_x1[i];

        if(biased) sdata[blockSize+threadIdx.x] += d_x1[i]*d_x1[i];
        i += blockSize;
    }

    __syncthreads();
    reduce_sum<blockSize, float>(sdata, threadIdx.x);
    if(biased) reduce_sum<blockSize, float>(&(sdata[blockSize]), threadIdx.x);

    if(threadIdx.x == 0) {
        float norm;
        float normsq;

        if(sdata[0]>0) {
            norm=sdata[0]/d_x0_status->norm2;
            if(biased) normsq=sdata[blockSize]/(d_x0_status->norm2 * d_x0_status->norm2);
        } else {
            norm=0.0;
        }
        d_x1_status->norm=norm;
        d_x1_status->norm2=norm*norm;
        if(biased) {
            d_x1_status->csize=1/(1-norm+alpha*normsq/(norm*norm));     // *****  BIASED
        } else {
            d_x1_status->csize=1/(1-norm);      // *****  not BIASED
        }
    }
}

template <unsigned int blockSize, int biased> __global__ void normiterGPU(float *d_x1, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, int *d_map, float alpha=0)
{
    normiterGPU_dev<blockSize, biased>(d_x0_status, d_x1_status, d_x1, d_map[-1], alpha);
}

template <unsigned int blockSize> __global__ void normGPU(float *d_x1, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, int *d_map)
{
	extern __shared__ float sdata[];

	int n=d_map[-1];

    unsigned int tid = threadIdx.x;
    unsigned int i=tid;

    sdata[tid]=0;
    while (i < n) {
        sdata[tid] += d_x1[i];
        i += blockSize;
    }

    __syncthreads();

    reduce_sum<blockSize, float>(sdata, tid);

    __syncthreads();
    if(sdata[0]>0) for(i=tid; i< n; i+= blockSize){
        d_x1[i]/=sdata[0];
    }

    if(tid==0) {
        d_x0_status->norm=1;
        d_x0_status->norm2=1;
        d_x1_status->norm=1;
        d_x1_status->norm2=1;
    }
}

__global__ void randomizeGPU(float *d_x2, curandState_t *d_randstates, float c, int *d_map)
{
    unsigned int tid = threadIdx.x;
    unsigned int i=tid;

	int n=d_map[-1];

    while (i < n) {
        float r=curand_uniform(&(d_randstates[tid]));
        d_x2[i]*=(1-2*r)*c+1;
        i += blockDim.x;
    }
}

struct isZero_data {
    float zero;
    float *x;
};

struct isOne_data {
    int *full_row;
};

// struct isZeroOrOne_data: isZero_data, isOne_data {}; // This definition cannot be used, because brace-enclosed
// initialization of this structure would not be possible.

struct isZeroOrOne_data {
    float zero;
    float *x;
    int *one_revmap;
};


__device__ __forceinline__  int isZero(int i, isZero_data data)
{
    // return 0;
    return data.x[i]<data.zero;
}

__device__ __forceinline__  int isNotFullRow(int i, isOne_data data)
{
    return data.full_row[i]==0;
}

__device__ __forceinline__  int isZeroOrOne(int i, isZeroOrOne_data data)
{
    // return /*data.x[i]<data.zero ||*/ data.one_revmap[i]>=0;
    return data.x[i]<data.zero || data.one_revmap[i]>=0;
}


template <unsigned int blockSize, typename T, int test(int i, T)> __global__ void elim1GPU(int *d_map, int *d_new_revmap, T data)
/*
 *  Fills d_new_revmap with consecutive numbers with increments at positions (i) for which test(d_x1, d_fullrow, i, zero) is true.
 *  Each block is filled separately.
 */
{
	extern __shared__ int idata[];

    unsigned int tid = threadIdx.x;
	int n=d_map[-1];

    int i=tid+blockIdx.x*blockSize*2;

    idata[tid]=0;
    if (i < n) idata[tid]=test(i, data) ? 0 : 1;
    if (i+blockSize < n) idata[tid+blockSize]=test(i+blockSize, data) ? 0 : 1;

    __syncthreads();
    prefixscan<blockSize, int, true>(idata, tid);
    __syncthreads();

    if (i < n) d_new_revmap[i]=idata[tid+1];
    if (i+blockSize < n) d_new_revmap[i+blockSize]=idata[tid+1+blockSize];
}


template <unsigned int blockSize> __global__ void elim15GPU(int *tmp, int *d_new_revmap, int n)
/*
 *  Prescans last elements of blocks. A simple unparalellized iteration is good enough.
 */
{
    tmp[0]=-1;
    for(int i=1; i<n; i++) {
        tmp[i]=tmp[i-1]+d_new_revmap[i*blockSize*2-1];
    }
}

template <typename T, int test(int i, T), int n_skip, int update_stats, int remap_ones> __device__ __forceinline__  void elim2_saveres(int pos, int n, int *d_new_revmap, T data, int acc, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, struct cuda_clique_status *d_x2_status, int *d_ones)
{
    if(pos<n) {
        if(n_skip) d_new_revmap[pos]=MAX(d_new_revmap[pos]+acc-n_skip, -1);
        else d_new_revmap[pos]+=acc;
        if(pos==n-1) {
            d_new_revmap[-1]=d_new_revmap[pos]+1;
            if(update_stats) {
                d_x1_status->nzeroes=n-d_new_revmap[-1]-d_x1_status->nones;

                if(d_x1_status->nzeroes<0) {
                    printf("n: %d d_new_revmap[-1]: %d d_x1_status->nones: %d\n", n, d_new_revmap[-1], d_x1_status->nones);
                }

            }
            if(remap_ones) {
                d_ones[-2]=d_new_revmap[-1];
                d_x0_status->nones=d_ones[-2];
                d_x1_status->nones=d_ones[-2];
                d_x2_status->nones=d_ones[-2];
            }
        }
        if(test(pos, data)) d_new_revmap[pos]=-1;

        if(remap_ones) if(d_new_revmap[pos]>=0) d_ones[d_ones[-1]+d_new_revmap[pos]]=pos;
    }

}


template <unsigned int blockSize, typename T, int test(int i, T), int n_skip, int update_stats, int remap_ones> __global__ void elim2GPU(int *d_map, int *d_new_revmap, T data, int *tmp, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, struct cuda_clique_status *d_x2_status, int *d_ones)
/*
 *  Makes numbers in d_new_revmap concecutive between blocks. Positions for which test fails equal -1.
 */
{
    unsigned int tid = threadIdx.x;
	int acc=tmp[blockIdx.x];

    int n=d_map[-1];
    int i=tid+blockIdx.x*blockSize*2;


    elim2_saveres<T, test, n_skip, update_stats, remap_ones>(i, n, d_new_revmap, data, acc, d_x0_status, d_x1_status, d_x2_status, d_ones);
    elim2_saveres<T, test, n_skip, update_stats, remap_ones>(i+blockSize, n, d_new_revmap, data, acc, d_x0_status, d_x1_status, d_x2_status, d_ones);
}

template <unsigned int blockSize, typename T, int test(int i, T), int n_skip, int update_stats, int remap_ones> __global__ void elim_oneblockGPU(int *d_map, int *d_new_revmap, T data, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, struct cuda_clique_status *d_x2_status, int *d_ones)
/*
 *  Fills d_new_revmap with consecutive numbers with increments at positions (i) for which test(d_x1, d_fullrow, i, zero) is true.
 *  Each block is filled separately.
 */
{
	extern __shared__ int idata[];

    unsigned int tid = threadIdx.x;
	int n=d_map[-1];

    int start=0;

    while(start<n) {
        int i=tid+start;

        idata[tid]=0;
        idata[tid+blockSize]=0;
        if (i < n) idata[tid]=test(i, data) ? 0 : 1;
        if (i+blockSize < n) idata[tid+blockSize]=test(i+blockSize, data) ? 0 : 1;

        __syncthreads();
        prefixscan<blockSize, int, true>(idata, tid);
        __syncthreads();

        if (i < n) d_new_revmap[i]=idata[tid+1];
        if (i+blockSize < n) d_new_revmap[i+blockSize]=idata[tid+1+blockSize];

        __syncthreads();  // This sync could be omitted if assignments above referred to idata[tid]
        start+=blockSize*2;
    }


    if(tid==0) {
        idata[0]=-1;
        for(int i=1; i*blockSize*2<=n; i++) {
            idata[i]=idata[i-1]+d_new_revmap[i*blockSize*2-1];
        }
    }

    __syncthreads();

    start=0;
    int blk=0;

    while(start<n) {
        int acc=idata[blk];

        int i=tid+start;

        elim2_saveres<T, test, n_skip, update_stats, remap_ones>(i, n, d_new_revmap, data, acc, d_x0_status, d_x1_status, d_x2_status, d_ones);
        elim2_saveres<T, test, n_skip, update_stats, remap_ones>(i+blockSize, n, d_new_revmap, data, acc, d_x0_status, d_x1_status, d_x2_status, d_ones);
        start+=blockSize*2;
        blk++;
    }
}

template <unsigned int blockSizeX, unsigned int blockSizeY, int biased, int final_mult> __device__ __forceinline__  void iter1GPU_dev(float *d_x1, float *d_x2, unsigned char *d_mat, size_t mat_pitch, size_t x_pitch, int *d_map, float alpha, float omega)
{
    extern __shared__ float sdata[];

    float *d_x=sdata+blockSizeY;

    float *pows=d_x+blockSizeY;

    unsigned int tid = threadIdx.x;
    int n;
    int n1;

    n=MIN(d_map[-1], (blockIdx.x+1)*blockSizeX);
    n1=MIN(d_map[-1], (blockIdx.y+1)*blockSizeY);

    int i=tid+blockIdx.x*blockSizeX;

    if(i<n) {
        d_x[tid]=d_x1[i];
    }

    if(biased==3) {
        for(int i=tid; i<255; i+=blockDim.x) {
            pows[i] = powf(omega, (i-1)*(i-1));
        }
    }

    __syncthreads();

    n=MIN((int)(d_map[-1]- blockIdx.x*blockSizeX), blockSizeX);
    int row = blockIdx.y*blockSizeY + tid;

    if(row < n1) {
        int j=blockIdx.x*blockSizeX;

        sdata[tid]=0;

        for(i=0; i<n; i++) {
            int pos = row + d_map[j]*mat_pitch;

            if(biased<=1 && d_mat[pos]) {
                sdata[tid] += d_x[i];
            } else if(biased==2) {
                if(d_mat[pos]>=2) {
                    sdata[tid] += omega * d_x[i];
                } else if(d_mat[pos]) {
                    sdata[tid] += d_x[i];
                }
            } else if(biased==3 && d_mat[pos]) {
                sdata[tid] += pows[d_mat[pos]] * d_x[i];
            }
            j++;
        }

        if(final_mult) {
            d_x2[row+x_pitch*blockIdx.x] = d_x1[row] * sdata[tid];
        } else {
            d_x2[row+x_pitch*blockIdx.x] = sdata[tid];
        }
    }
}

template <int biased, int final_mult> __device__ __forceinline__  void iter2GPU_dev(struct cuda_clique_status *d_x0_status, float *d_x1, float *d_x2, size_t x_pitch, int *d_map, int rows, float alpha)
{
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    int i=tid+blockIdx.x*blockDim.x;

    float *d_x;

    float norm=d_x0_status->norm2;

    if(final_mult) {
        norm=norm*norm;
    }

    d_x=d_x2;


    if(i<d_map[-1] && norm>0) {
        sdata[tid]=0;
        for(int r=0; r<rows; r++) {
            sdata[tid]+=d_x[i+r*x_pitch];
        }

        if(biased) {
            if(final_mult) {
                sdata[tid] += alpha * d_x1[i] * d_x1[i];
            } else {
                sdata[tid] += alpha * d_x1[i];
            }
        }

        d_x[i]=sdata[tid]>=0 ? sdata[tid]/norm : 0;
    }
}

template <unsigned int blockSizeX, unsigned int blockSizeY, int biased, int final_mult> __global__ void __launch_bounds__ (blockSizeY, 2048/blockSizeY) iter1GPU(float *d_x1, float *d_x2, unsigned char *d_mat, size_t mat_pitch, size_t x_pitch, int *d_map, float alpha, float omega)
/*
 *  Constraints: blockDim.x == blockSizeY
 *  Shared memory required: biased != 3 ? 2 * blockDim.x * sizeof(float) : (2 * blockDim.x + 256) * sizeof(float)
 */

{
    iter1GPU_dev<blockSizeX, blockSizeY, biased, final_mult>(d_x1, d_x2, d_mat, mat_pitch, x_pitch, d_map, alpha, omega);
}


template <int biased, int final_mult> __global__ void iter2GPU(struct cuda_clique_status *d_x0_status, float *d_x1, float *d_x2, size_t x_pitch, int *d_map, int rows, float alpha)
{
    iter2GPU_dev<biased, final_mult>(d_x0_status, d_x1, d_x2, x_pitch, d_map, rows, alpha);
}


template <int blockSize, int biased, int usedBlocks, int final_mult> __device__ __forceinline__ void iterGPU_dev(struct cuda_clique_status *d_x0_status, float *d_x1, float *d_x2, unsigned char *d_mat, size_t mat_pitch, size_t x_pitch, int *d_map, float alpha, float omega)
{
    extern __shared__ float sdata[];

    float *pows=sdata+blockDim.x;

    unsigned int tid = threadIdx.x;
    int row = blockIdx.x;

    float norm=d_x0_status->norm2;
    if(final_mult) {
        norm=norm*norm;
    }

    if(biased==3) {
        for(int i=tid; i<256; i+=blockDim.x) {
            pows[i] = powf(omega, (i-1)*(i-1));
        }
    }

    __syncthreads();

    while(row < d_map[-1]) {
        sdata[tid] = 0;

        int rr=d_map[row]*mat_pitch;

        for(int i=tid; i<d_map[-1]; i+=blockDim.x) {
            if(biased==2 && d_mat[i + rr]>=2) {
                sdata[tid] += omega * d_x1[i];
            } else if(biased==3 && d_mat[i + rr]>=2) {
                sdata[tid] += pows[d_mat[i + rr]] * d_x1[i];
            } else if(d_mat[i + rr]) {
                sdata[tid] += d_x1[i];
            }

        }
        __syncthreads();

        reduce_sum<blockSize, float>(sdata, tid);

        if(tid==0) {
            if(biased) sdata[0] += d_x1[row] * alpha;
            if(final_mult) {
                d_x2[row] = sdata[0] >= 0 ? d_x1[row] * sdata[0] / norm : 0;
            } else {
                d_x2[row] = sdata[0] >= 0 ? sdata[0] / norm : 0;
            }
        }

        row += gridDim.x-usedBlocks; // Some blocks may be used for something else (like normiterGPU_dev)
    }

}

template <int blockSize, int biased, int maxdiff, int final_mult> __global__ void iterGPU(struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, float *d_x0, float *d_x1, float *d_x2, unsigned char *d_mat, size_t mat_pitch, size_t x_pitch, int *d_map, float alpha, float omega)
{
    if(blockIdx.x < gridDim.x-1) {
        iterGPU_dev<blockSize, biased, 1, final_mult>(d_x0_status, d_x1, d_x2, d_mat, mat_pitch, x_pitch, d_map, alpha, omega);
    } else { 
        normiterGPU_dev<blockSize, biased>(d_x0_status, d_x1_status, d_x1, d_map[-1], alpha);
        if(maxdiff) {
            maxdiffGPU_dev<blockSize>(d_x0, d_x1, d_x0_status, d_x1_status, d_map);
        }
    }
}

__global__ void fillMapGPU(int *d_map, int n, int *d_ones)
{
	int step=blockDim.x*gridDim.x;

	for(int i= threadIdx.x+blockIdx.x*blockDim.x; i< n; i+=step){
		d_map[i]=i;
	}

	if(threadIdx.x==0) {
		d_map[-1]=n;
        d_ones[-2]=0;
        d_ones[-1]=0;
	}
}

__global__ void rewriteMatrixGPU(unsigned char *d_mat, unsigned char *d_start_mat, int *d_map, int mat_pitch)
{
	int nmap=d_map[-1];

    for(int i = blockIdx.x; i < nmap; i += gridDim.x) if(d_map[i]>=0) {
        unsigned char *row = d_mat + mat_pitch* d_map[i];
		unsigned char *row_start = d_start_mat + mat_pitch*d_map[i];

        for(int j= threadIdx.x; j< nmap; j+= blockDim.x){
            if(d_map[j]>=0) {
                row[j]=row_start[d_map[j]];
            } else {
                row[j]=0;
            }
        }
    }
}

__global__ void rewriteMatrixGPU1(unsigned char *d_mat, int *d_map, int *d_new_map, int mat_pitch)
{
	int nnew_map=d_new_map[-1];
	int nmap=d_map[-1];

    int j,jj;
    for(int i = threadIdx.x+blockIdx.x*blockDim.x; i < nnew_map; i += blockDim.x*gridDim.x) {

        unsigned char *row = d_mat + mat_pitch* d_new_map[i];

        for(j = 0, jj = 0; jj< nmap; jj++) {
            if(d_new_map[j]==d_map[jj]) {
                row[j++]=row[jj];
            }
        }
    }
}

__global__ void updateMapGPU(int *d_new_rev_map, float *d_x1, int *d_map, float *d_x2, int *d_new_map)
{
    int n=d_map[-1];

    int i=threadIdx.x+blockIdx.x*blockDim.x;

    if(i<n) {
        if(d_new_rev_map[i]>=0) {
            d_new_map[d_new_rev_map[i]]=d_map[i];
            d_x2[d_new_rev_map[i]]=d_x1[i];
        }
    }

    if(i==0) {
        d_new_map[-1]=d_new_rev_map[-1];
    }
}


__global__ void fullrowsGPU(int *d_fullrow, unsigned char *d_mat, int *d_map, unsigned int mat_pitch, float *d_x1)
{
    unsigned int tid = threadIdx.x;
	int n=d_map[-1];

    int row=blockIdx.x+tid*gridDim.x;
    if(row<n) {
        d_fullrow[row]=1;
    }

    __syncthreads();

	for(int row = blockIdx.x;  row < n; row+=gridDim.x){
		int rowStart = mat_pitch*d_map[row];

        int i=tid;
        while (i < n) {
            if(__any(!d_mat[rowStart+i] && i!=row && d_x1[i]>0)) { // This makes all threads in a warp break out of the loop together. Other threads in the block will scan until they find other zeroes.
                d_fullrow[row]=0;
                break;
            }
            i += blockDim.x;
        }
    }
}

__global__ void unmapOnesGPU(int *d_ones, int *d_map, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, struct cuda_clique_status *d_x2_status)
{
    unsigned int tid = threadIdx.x;

    int i=tid+blockIdx.x*blockDim.x;

    int n=d_ones[-1];
    int n1=d_ones[-1]+d_ones[-2];

    i+=n;


    for(;i<n1;i+=blockDim.x*gridDim.x) {
        d_ones[i]=d_map[d_ones[i]];
    }

    if(tid==0 && blockIdx.x==0) {
        d_ones[-1]+=d_ones[-2];
        d_ones[-2]=0;
        d_x0_status->nones=d_ones[-2];
        d_x1_status->nones=d_ones[-2];
        d_x2_status->nones=d_ones[-2];
    }
}


template <unsigned int blockSize> __global__ void unmapResGPU(float *d_x0, float *d_x1, struct cuda_clique_status *d_x0_status, int *d_ones, int *d_map, int n)
{
	extern __shared__ int idata[];

    unsigned int tid = threadIdx.x;

    float one=1.0/MAX(1.0,d_x0_status->csize);

	for(int i= threadIdx.x; i < n; i+= blockDim.x){
		d_x1[i]=0;
	}

    int cnt=0;

	for(int i= threadIdx.x; i < d_map[-1]; i+= blockDim.x){
		if(d_map[i]>=0) {
			d_x1[d_map[i]]=d_x0[i];
            cnt++;
		}
	}
    idata[tid] = cnt;


    __syncthreads();

    reduce_sum<blockSize, int>(idata, tid);

    __syncthreads();

    if(idata[0]+d_ones[-1]>0) {
        for(int i= threadIdx.x; i < d_ones[-1]; i+= blockDim.x) {
            d_x1[d_ones[i]]=one;
        }
    } else {
        one=1/(float)n;
        for(int i= threadIdx.x; i < n; i+= blockDim.x) {
            d_x1[i]=one;
        }
    }

    if(threadIdx.x==0 && blockIdx.x==0) d_map[-1]=n;
}

int isDone(struct cuda_clique_status *h_x1_status, int n, int nmap, int max_unsolved, int *abortcheck_cb(void))
{
    float csize=h_x1_status->csize;
    int nzeroes=h_x1_status->nzeroes;
    int nones=h_x1_status->nones;
    float maxdiff=h_x1_status->maxdiff;

    float unsolved=UNSOLVED(h_x1_status, nmap);
    float comp_diff = 0.000001f;

    int done=0;

    if(unsolved<.5+max_unsolved) {
        done=DONE_SOLVED;
        if(nzeroes>0 || nones>0) {
            done=DONE_CLEANUP;
        }
    } else if((nzeroes>nmap/10 || nzeroes>=32) || nzeroes>0 || nones>0) {
        done=DONE_CLEANUP;
    } else if(maxdiff<comp_diff/100 || ( maxdiff<comp_diff && unsolved < 0.5 * n )) {
        done=DONE_CONVERGED;
    } else if(abortcheck_cb && abortcheck_cb()) {
        done=DONE_ABORTED;
    }


#ifdef DEBUG
    P_T("is_done:: ") P_FLOAT(csize) P_FLOAT(unsolved) P_INT(max_unsolved) P_FLOATE(maxdiff) P_FLOATE(comp_diff) P_INT(nzeroes) P_INT(nones) P_INT(nmap) P_INT(done) P_NL;
#endif

    if(h_x1_status->nzeroes<0) {
        P_T("is_done:: ") P_FLOAT(csize) P_FLOAT(unsolved) P_INT(max_unsolved) P_FLOATE(maxdiff) P_FLOATE(comp_diff) P_INT(nzeroes) P_INT(nones) P_INT(nmap) P_INT(done) P_NL; 
        P_INT(h_x1_status->nzeroes) P_NL;
        abort();
    }

    return done;
}

template<unsigned int blockSize> inline void run_normGPU(cudaStream_t stream,  float *d_x1, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, int *d_map)
{
    normGPU<blockSize><<<1, blockSize, blockSize*4, stream>>>(d_x1, d_x0_status, d_x1_status, d_map);
}


template<unsigned int blockSize> inline void run_normiterGPU(cudaStream_t stream,  float *d_x1, struct cuda_clique_status *d_x0_status, struct cuda_clique_status *d_x1_status, int *d_map, float alpha)
{
    if(alpha==0) {
        normiterGPU<blockSize, 0><<<1, blockSize, blockSize*4, stream>>>(d_x1, d_x0_status, d_x1_status, d_map);
    } else {
        normiterGPU<blockSize, 1><<<1, blockSize, blockSize*8, stream>>>(d_x1, d_x0_status, d_x1_status, d_map, alpha);
    }
}

static void normalize_cuda(struct cuda_clique_data *data)
/*
   Uses: [1]d_x, [0]d_x_status
   Computes: [1]d_x_status
*/
{
    cudaStream_t stream=data->norm_stream;

    cudaStreamWaitEvent(stream, data->instances[1].quadratic_done, 0);

    run_normiterGPU<128>(stream, data->instances[1].d_x, data->instances[0].d_x_status, data->instances[1].d_x_status, data->d_map, data->alpha);
    cudaEventRecord(data->instances[1].norm_done, stream);
}

template<int part, int maxdiff, int final_mult> static void quadratic_cuda(struct cuda_clique_data *data)
/*
   Uses: [1]d_x, [0]d_x_status
   Computes: [2]d_x, ([1]d_x_status if data->n<SMALL_N)
*/

{
    cudaStream_t stream=data->iter_stream;

    if(data->nmap>=SMALL_N) {
        int rows;
        int rows1;

        if(data->nmap>=LARGE_N) {
            rows=DIVROUNDUP(data->nmap, ROW_SLICE_L);
            rows1=DIVROUNDUP(data->nmap, 128);
        } else {
            rows=DIVROUNDUP(data->nmap, ROW_SLICE_S);
            rows1=DIVROUNDUP(data->nmap, 64);
        }

        if(part==1) {
            normalize_cuda(data);
            cudaStreamWaitEvent(stream, data->instances[0].norm_done, 0);
            cudaStreamWaitEvent(stream, data->instances[0].zero_done, 0);
            cudaStreamWaitEvent(stream, data->instances[2].zero_done, 0);

            struct iter1GPU_pars {
                float *d_x1;
                float *d_x2;
                unsigned char *d_mat;
                size_t mat_pitch;
                size_t x_pitch;
                int *d_map;
                float alpha;
                float omega;
            };

            struct iter1GPU_pars pars={data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega};


            if(data->nmap>=LARGE_N) {
                cudaConfigureCall(dim3(rows, rows1), 128, 2048, stream);
                cudaSetupArgument(pars, 0);
                if(data->mode==MODE_SIMPLE) {
                    cudaLaunch(iter1GPU<ROW_SLICE_L, 128, 0, final_mult>);
                    /* iter1GPU<ROW_SLICE_L, 128, 0, final_mult><<<dim3(rows, rows1), 128, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                } else if(data->mode==MODE_REGULAR) {
                    cudaLaunch(iter1GPU<ROW_SLICE_L, 128, 1, final_mult>);
                    /* iter1GPU<ROW_SLICE_L, 128, 1, final_mult><<<dim3(rows, rows1), 128, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                } else if(data->mode==MODE_ATTEN) {
                    cudaLaunch(iter1GPU<ROW_SLICE_L, 128, 2, final_mult>);
                    /* iter1GPU<ROW_SLICE_L, 128, 2, final_mult><<<dim3(rows, rows1), 128, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                } else { // MODE_ATTEN
                    cudaLaunch(iter1GPU<ROW_SLICE_L, 128, 3, final_mult>);
                    /* iter1GPU<ROW_SLICE_L, 128, 3, final_mult><<<dim3(rows, rows1), 128, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                }
            } else {
                cudaConfigureCall(dim3(rows, rows1), 64, 2048, stream);
                cudaSetupArgument(pars, 0);
                if(data->mode==MODE_SIMPLE) {
                    cudaLaunch(iter1GPU<ROW_SLICE_S, 64, 0, final_mult>);
                    /* iter1GPU<ROW_SLICE_S, 64, 0, final_mult><<<dim3(rows, rows1), 64, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                } else if(data->mode==MODE_REGULAR) {
                    cudaLaunch(iter1GPU<ROW_SLICE_S, 64, 1, final_mult>);
                    /* iter1GPU<ROW_SLICE_S, 64, 1, final_mult><<<dim3(rows, rows1), 64, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                } else if(data->mode==MODE_ATTEN) {
                    cudaLaunch(iter1GPU<ROW_SLICE_S, 64, 2, final_mult>);
                    /* iter1GPU<ROW_SLICE_S, 64, 2, final_mult><<<dim3(rows, rows1), 64, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                } else { // MODE_ATTEN
                    cudaLaunch(iter1GPU<ROW_SLICE_S, 64, 3, final_mult>);
                    /* iter1GPU<ROW_SLICE_S, 64, 3, final_mult><<<dim3(rows, rows1), 64, 2048, stream>>>(data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
                }
            }
        } else if(part==2) {
            struct iter2GPU_pars {
                struct cuda_clique_status *d_x0_status;
                float *d_x1;
                float *d_x2;
                size_t x_pitch;
                int *d_map;
                int rows;
                float alpha;
            };

            struct iter2GPU_pars pars={data->instances[0].d_x_status, data->instances[1].d_x, data->instances[2].d_x, data->x_pitch, data->d_map, rows, data->alpha};

            cudaConfigureCall(rows1, 128, 512, stream);
            cudaSetupArgument(pars, 0);

            if(data->mode==MODE_SIMPLE) {
                cudaLaunch(iter2GPU<0, final_mult>);
                /* iter2GPU<0, final_mult><<<rows1, 128, 512, stream>>>(data->instances[0].d_x_status, data->instances[1].d_x, data->instances[2].d_x, data->x_pitch, data->d_map, rows, data->alpha); */
            } else {
                cudaLaunch(iter2GPU<1, final_mult>);
                /* iter2GPU<1, final_mult><<<rows1, 128, 512, stream>>>(data->instances[0].d_x_status, data->instances[1].d_x, data->instances[2].d_x, data->x_pitch, data->d_map, rows, data->alpha); */
            }
            cudaEventRecord(data->instances[2].quadratic_done, stream);
        }
    } else {
        if(part==1) {
            cudaStreamWaitEvent(stream, data->instances[0].norm_done, 0);
            cudaStreamWaitEvent(stream, data->instances[0].zero_done, 0);
            cudaStreamWaitEvent(stream, data->instances[2].zero_done, 0);

            struct iterGPU_pars {
                struct cuda_clique_status *d_x0_status;
                struct cuda_clique_status *d_x1_status;
                float *d_x0;
                float *d_x1;
                float *d_x2;
                unsigned char *d_mat;
                size_t mat_pitch;
                size_t x_pitch;
                int *d_map;
                float alpha;
                float omega;
            };

            struct iterGPU_pars pars={data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[0].d_x, data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega};


            cudaConfigureCall(MIN(1024, data->nmap), 128, (128+256)*sizeof(float), stream);
            cudaSetupArgument(pars, 0);

            if(data->mode==MODE_SIMPLE) {
                cudaLaunch(iterGPU<128, 0, maxdiff, final_mult>);
                /* iterGPU<128, 0, maxdiff, final_mult><<<MIN(1024, data->nmap), 128, (128+256)*sizeof(float), stream>>>(data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[0].d_x, data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
            } else if(data->mode==MODE_REGULAR) {
                cudaLaunch(iterGPU<128, 1, maxdiff, final_mult>);
                /* iterGPU<128, 1, maxdiff, final_mult><<<MIN(1024, data->nmap), 128, (128+256)*sizeof(float), stream>>>(data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[0].d_x, data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
            } else if(data->mode==MODE_ATTEN) {
                cudaLaunch(iterGPU<128, 2, maxdiff, final_mult>);
                /* iterGPU<128, 2, maxdiff, final_mult><<<MIN(1024, data->nmap), 128, (128+256)*sizeof(float), stream>>>(data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[0].d_x, data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
            } else { // MODE_ATTEN
                cudaLaunch(iterGPU<128, 3, maxdiff, final_mult>);
                /* iterGPU<128, 3, maxdiff, final_mult><<<MIN(1024, data->nmap), 128, (128+256)*sizeof(float), stream>>>(data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[0].d_x, data->instances[1].d_x, data->instances[2].d_x, data->d_mat, data->mat_pitch, data->x_pitch, data->d_map, data->alpha, data->omega); */
            }

            cudaEventRecord(data->instances[2].quadratic_done, stream);
        }
    }
}


template<typename T, int test(int i, T), int n_skip, int update_stats, int remap_ones> static void elim_cuda(struct cuda_clique_data *data, int *revmap, T test_data, cudaStream_t stream)
{
    if(data->nmap>=LARGE_N) {
        int n_blocks=DIVROUNDUP(data->nmap, 128 * 2);

        elim1GPU<128, T, test><<<n_blocks, 128, (128*2+1)*sizeof(int), stream>>>(data->d_map, revmap, test_data);
        elim15GPU<128><<<1, 1, 0, stream>>>(data->d_tmp, revmap, n_blocks);
        elim2GPU<128, T, test, n_skip, update_stats, remap_ones><<<n_blocks, 128, (128*2+1)*sizeof(int), stream>>>(data->d_map, revmap, test_data, data->d_tmp, data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[2].d_x_status, data->d_ones);
    } else {
        elim_oneblockGPU<128, T, test, n_skip, update_stats, remap_ones><<<1, 128, (128*2+1)*sizeof(int), stream>>>(data->d_map, revmap, test_data, data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[2].d_x_status, data->d_ones);
    }
}

template<int part> static void find_full_rows_cuda(struct cuda_clique_data *data)
// Find full rows in current matrix. Done at the beginning of the 10 iter cycle, iff matrix was changed.
{
    cudaStream_t stream=data->zero_stream;

    if(part==1) {
        fullrowsGPU<<<128, 128, 0, stream>>>(data->d_incident, data->d_mat, data->d_map, data->mat_pitch, data->instances[1].d_x);
    } else if(part==2) {
        elim_cuda<isOne_data, isNotFullRow, 1/*n_skip*/, 0, 1 /*remap_ones*/>(data, data->d_one_revmap, {data->d_incident}, stream);
    }
}

static void compute_stats_cuda(struct cuda_clique_data *data, int mat_changed)
{
    cudaStream_t stream=data->zero_stream;

    cudaStreamWaitEvent(stream, data->instances[1].quadratic_done, 0);

    if(mat_changed && data->alpha>=0 && data->omega==1) {
        elim_cuda<isZeroOrOne_data, isZeroOrOne, 0, 1 /*update_stats*/, 0>(data, data->d_new_revmap, {data->zero, data->instances[1].d_x, data->d_one_revmap}, stream);
    } else {
        elim_cuda<isZero_data, isZero, 0, 1 /*update_stats*/, 0>(data, data->d_new_revmap, {data->zero, data->instances[1].d_x}, stream);
    }

    if(data->nmap>=SMALL_N) {
        maxdiffGPU<128><<<1, 128, 128*4, stream>>>(data->instances[0].d_x, data->instances[1].d_x, data->instances[0].d_x_status, data->instances[1].d_x_status, data->d_map);
    } else {
        cudaStreamWaitEvent(stream, data->instances[2].quadratic_done, 0);
    }

    cudaMemcpyAsync(data->instances[1].h_x_status, data->instances[1].d_x_status, sizeof(struct cuda_clique_status), cudaMemcpyDeviceToHost, stream);

    cudaEventRecord(data->instances[1].zero_done, stream);
}

static void cleanup_matrix_cuda(struct cuda_clique_data *data, int elim_zeroes)
{
    int n = data->n;


	fillMapGPU<<<1, 256, 0, data->iter_stream>>>(data->d_map, data->n, data->d_ones);
	fillMapGPU<<<1, 256, 0, data->iter_stream>>>(data->d_new_map, data->n, data->d_ones);
    data->nmap=data->n;
    run_normGPU<256>(data->iter_stream, data->instances[1].d_x, data->instances[0].d_x_status, data->instances[1].d_x_status, data->d_map);

    if(elim_zeroes) {
        elim_cuda<isZero_data, isZero, 0, 1 /*update_stats*/, 0>(data, data->d_new_revmap, {data->zero, data->instances[1].d_x}, data->iter_stream);
        updateMapGPU<<<DIVROUNDUP(data->n, 256), 256, 0, data->iter_stream>>>(data->d_new_revmap, data->instances[1].d_x, data->d_map, data->instances[2].d_x, data->d_new_map);
    }

    if(data->d_start_mat) {
        rewriteMatrixGPU<<<512,512, 0, data->iter_stream>>>(data->d_mat, data->d_start_mat, data->d_new_map, data->mat_pitch);
    } else {
        cudaMemcpy2D(data->d_mat, data->mat_pitch, data->h_start_mat, n*sizeof(char), n*sizeof(char), n, cudaMemcpyHostToDevice);
        rewriteMatrixGPU1<<<128,256, 0, data->iter_stream>>>(data->d_mat, data->d_map, data->d_new_map, data->mat_pitch);
    }
    cudaEventRecord(data->instances[0].norm_done, data->iter_stream);
    cudaStreamWaitEvent(data->zero_stream, data->instances[0].norm_done, 0);
    cudaMemcpyAsync(&(data->nmap), data->d_map-1, sizeof(int), cudaMemcpyDeviceToHost, data->zero_stream);
}

static void randomize_cuda(struct cuda_clique_data *data, float fact)
{
    if(fact>0) {
        randomizeGPU<<<1, 256, 0,data->iter_stream>>>(data->instances[2].d_x, d_randstates, fact, data->d_map);
    }
    run_normGPU<256>(data->iter_stream, data->instances[2].d_x, data->instances[1].d_x_status, data->instances[2].d_x_status, data->d_map);
}

static void remove_unnecessary_nodes_cuda(struct cuda_clique_data *data)
{
    updateMapGPU<<<DIVROUNDUP(data->nmap, 256), 256, 0, data->iter_stream>>>(data->d_new_revmap, data->instances[2].d_x, data->d_map, data->instances[0].d_x, data->d_new_map);
    unmapOnesGPU<<<DIVROUNDUP(data->nmap, 256), 256, 0, data->zero_stream>>>(data->d_ones, data->d_map, data->instances[0].d_x_status, data->instances[1].d_x_status, data->instances[2].d_x_status);
    if(data->d_start_mat) {
        rewriteMatrixGPU<<<512, 512, 0, data->iter_stream>>>(data->d_mat, data->d_start_mat, data->d_new_map, data->mat_pitch);
    } else {
        rewriteMatrixGPU1<<<1,512, 0, data->iter_stream>>>(data->d_mat, data->d_map, data->d_new_map, data->mat_pitch);
    }
    run_normGPU<256>(data->iter_stream, data->instances[0].d_x, data->instances[1].d_x_status, data->instances[0].d_x_status, data->d_map);
    cudaMemcpyAsync(&(data->nmap), data->d_new_map-1, sizeof(int), cudaMemcpyDeviceToHost, data->iter_stream);
}


static int d_iterate(struct cuda_clique_data *data, int *abortcheck_cb(void))
{

    int iter_cnt=1;


    SWAP(data->instances[0], data->instances[1]);
    cleanup_matrix_cuda(data, 1 /*elim_zeroes*/);
    SWAP(data->instances[1], data->instances[2]);

    SWAP(data->d_map, data->d_new_map);

    int done=0;
    int mat_changed=1;
    int randomized=0;

    checkCUDAError("memcpy", __LINE__);

#ifdef DEBUG
    P_INT(data->nmap) P_NL;
#endif

    int done3_cnt=0;

#define CHECK_PERIOD 10

	while(!done) {
        if(iter_cnt%CHECK_PERIOD==CHECK_PERIOD-2) {
            quadratic_cuda<1, 1, 1 /* final_mult */>(data);
        } else {
            quadratic_cuda<1, 0, 1 /* final_mult */>(data);
        }

        if(data->alpha>=0 && data->omega==1) {
            if((iter_cnt%CHECK_PERIOD==1) && mat_changed) find_full_rows_cuda<1>(data);
            if((iter_cnt%CHECK_PERIOD==2) && mat_changed) find_full_rows_cuda<2>(data);
        }

        if(iter_cnt%CHECK_PERIOD==CHECK_PERIOD-2) {
            compute_stats_cuda(data, mat_changed);
        }

        quadratic_cuda<2, 0, 1 /* final_mult */>(data);

        if(iter_cnt%CHECK_PERIOD==CHECK_PERIOD-1) {
            cudaEventSynchronize(data->instances[0].zero_done);
            cudaEventSynchronize(data->instances[0].norm_done);

            done=isDone(data->instances[0].h_x_status, data->n, data->nmap, data->max_unsolved, abortcheck_cb);
#ifdef DEBUG
            P_INT(done) P_INT(done3_cnt) P_FLOAT(UNSOLVED(data->instances[0].h_x_status, data->nmap)) P_INT(data->max_unsolved) P_NL;
#endif
            mat_changed=0;
            randomized=0;

            if(done==DONE_CLEANUP) {
                remove_unnecessary_nodes_cuda(data);

                SWAP(data->instances[0], data->instances[2]);
                SWAP(data->d_map, data->d_new_map);

                done=0;
                mat_changed=1;

                cudaStreamSynchronize(data->iter_stream);
                cudaStreamSynchronize(data->zero_stream);

#ifdef DEBUG
                P_INT(data->nmap) P_NL;
#endif

                if(data->nmap<=1) {
                    done=DONE_SOLVED;
                    set_csize(data->instances[2].d_x_status, data->nmap, data->zero_stream);
                    SWAP(data->instances[1], data->instances[2]); // Special case:  In this case only the last iteration is valid. One before last has a wrong size.
                }
            }

            if(done==DONE_CONVERGED && done3_cnt<3 && UNSOLVED(data->instances[0].h_x_status, data->nmap) > data->max_unsolved) {
                randomize_cuda(data, 0.10);
                done=0;
                randomized=1;
                done3_cnt++;
            }

        }

        checkCUDAError("isDone", __LINE__);

        SWAP(data->instances[0], data->instances[1]); SWAP(data->instances[1], data->instances[2]);

		if(iter_cnt>=N_ITER && ! mat_changed && !randomized) break;
		iter_cnt++;


	}

#ifdef DEBUG
    P_INT(iter_cnt) P_INT(N_ITER) P_NL;
#endif

// x0 contains an acceptable solution. x1 holds the next iteration which will be ignored.

    run_normGPU<256>(data->iter_stream, data->instances[0].d_x, data->instances[0].d_x_status, data->instances[1].d_x_status, data->d_map);
	unmapResGPU<256><<<1,256, 1024, data->iter_stream>>>(data->instances[0].d_x, data->instances[1].d_x, data->instances[0].d_x_status, data->d_ones, data->d_map, data->n);
    run_normGPU<256>(data->iter_stream, data->instances[1].d_x, data->instances[0].d_x_status, data->instances[1].d_x_status, data->d_map);

    return 0;
}

static float iter_one(struct cuda_clique_data *data)
{

    SWAP(data->instances[0], data->instances[1]);
    cleanup_matrix_cuda(data, 0 /* elim_zeroes */);

    quadratic_cuda<1, 0, 1 /* final_mult */>(data);
    quadratic_cuda<2, 0, 1 /* final_mult */>(data);


    SWAP(data->instances[0], data->instances[1]); SWAP(data->instances[1], data->instances[2]);
    normalize_cuda(data);

    cudaMemcpy(&(data->instances[1].h_x_status->csize), &(data->instances[1].d_x_status->csize), sizeof(float), cudaMemcpyDeviceToHost);
    float res=data->instances[1].h_x_status->csize;

    SWAP(data->instances[1], data->instances[2]); SWAP(data->instances[0], data->instances[1]);

    quadratic_cuda<1, 0, 0 /* final_mult */>(data);
    quadratic_cuda<2, 0, 0 /* final_mult */>(data);

    cudaStreamSynchronize(data->norm_stream);
    cudaStreamSynchronize(data->iter_stream);

    cudaMemcpy(&(data->instances[1].h_x_status->csize), &(data->instances[1].d_x_status->csize), sizeof(float), cudaMemcpyDeviceToHost);

    checkCUDAError("iter_one", __LINE__);

    return res;
}

template <typename t_data, int offset> void alloc_with_status(t_data **d_data, void *d_raw, int pos, size_t pitch)
{
	*d_data=(t_data *)((char *)d_raw + pos * pitch)+offset;
}

void alloc_instance(struct cuda_clique_instance *res, void *h_instances, void *d_instances, size_t h_size_st, size_t h_size_x, size_t d_size_st, size_t d_size_x, int n_inst, int pos)
{

    res->h_x_status=(struct cuda_clique_status *)((char *)h_instances + pos*h_size_st);
    res->d_x_status=(struct cuda_clique_status *)((char *)d_instances + pos*d_size_st);

    res->h_x=(float *)((char *)h_instances + n_inst*h_size_st + pos*h_size_x);
    res->d_x=(float *)((char *)d_instances + n_inst*d_size_st + pos*d_size_x);

    cudaEventCreateWithFlags(&res->quadratic_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&res->norm_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&res->zero_done, cudaEventDisableTiming);
}

void alloc_instances(struct cuda_clique_data *res, int n_inst)
{
    int vec_size = res->vec_size;

    cudaMallocHost((void **)&(res->h_instances), 2*vec_size*n_inst);
    cudaMallocPitch((void **)&(res->d_instances), &(res->x_pitch), vec_size, (DIVROUNDUP(res->n, ROW_SLICE_S)+1)*n_inst);

    for(int i=0; i<n_inst; i++) alloc_instance(&(res->instances[i]), res->h_instances, res->d_instances, vec_size, vec_size, res->x_pitch, res->x_pitch * DIVROUNDUP(res->n, ROW_SLICE_S), n_inst, i);
}

void free_instance(struct cuda_clique_instance res)
{
    cudaEventDestroy(res.quadratic_done);
    cudaEventDestroy(res.norm_done);
    cudaEventDestroy(res.zero_done);
}


extern "C" int count_untouched_cuda_clique(struct cuda_clique_data *data)
{
    int n=data->n;

    int res=0;

    for(int i=0; i<n; i++)
        for(int j=i+1; j<n; j++)
            if(data->h_start_mat[i*n+j]==1)
                res++;

    return res*2;
}


extern "C" void apply_mask_cuda_clique(struct cuda_clique_data *res, t_bitmask mask, int e)
{
    int n=res->n;

    for(int i=0; i<n; i++) if(BIT_TEST(mask, i))
        for(int j=i+1; j<n; j++) if(BIT_TEST(mask, j)) {
            if(res->h_start_mat[i*n+j]>0) {
                res->h_start_mat[i*n+j]=MIN((int)ceilf(sqrtf((res->h_start_mat[i*n+j] * res->h_start_mat[i*n+j])+e)), 255);
                res->h_start_mat[j*n+i]=res->h_start_mat[i*n+j];
            }
        }

    if(res->d_start_mat) {
        cudaMemcpy2D(res->d_start_mat, res->mat_pitch, res->h_start_mat, n*sizeof(char), n*sizeof(char), n, cudaMemcpyHostToDevice);
    }
}


__global__ void rand_setup_kernel(curandState_t *state) 
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(1234, id, 0, &state[id]);
}


extern "C" void init_cuda()
{
    if(cuda_device==-1) {
        int device_count;
        cudaGetDeviceCount(&device_count);

        cuda_device=int_ran(0, device_count-1);
        cudaSetDevice(cuda_device);

        threaddev[omp_get_thread_num()]=cuda_device;
        cudaFree(0);
    }
}


extern "C" void init_cuda_clique(struct cuda_clique_data *res, char **graph, int n)
{

    cuda_start();

    if(!rand_init[omp_get_thread_num()]) {
        cudaMalloc((void **)&d_randstates, 256 * sizeof(curandState_t));
        rand_setup_kernel<<<1, 256>>>(d_randstates);
        rand_init[omp_get_thread_num()]=1;
    }

	res->n=n;

	res->h_start_mat=(unsigned char *)calloc(n*n, sizeof(char));

    res->n_edges=0;

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			res->h_start_mat[i*n+j]=i!=j ? graph[i][j] : 0;

            res->n_edges+=res->h_start_mat[i*n+j];
		}
	}

    res->vec_size=MAX(n, VEC_SKIP) * sizeof(float);

    res->instances=(struct cuda_clique_instance *)malloc(3*sizeof(struct cuda_clique_instance));

    alloc_instances(res, 3);

    if(res->x_pitch % sizeof(float) != 0) {
        printf("internal error\n");
        abort();
    }

    res->x_pitch/=sizeof(float);

    size_t pitch;

    cudaMallocPitch((void **)&(res->d_raw), &pitch, (n+VEC_SKIP) * sizeof(int), 6);

    alloc_with_status<int, VEC_SKIP>(&res->d_map, res->d_raw, 0, pitch);
    alloc_with_status<int, VEC_SKIP>(&res->d_new_map, res->d_raw, 1, pitch);

    alloc_with_status<int, VEC_SKIP>(&res->d_one_revmap, res->d_raw, 2, pitch);
    alloc_with_status<int, VEC_SKIP>(&res->d_ones, res->d_raw, 3, pitch);
    alloc_with_status<int, VEC_SKIP>(&res->d_new_revmap, res->d_raw, 4, pitch);
    alloc_with_status<int, VEC_SKIP>(&res->d_incident, res->d_raw, 5, pitch);

	cudaCalloc( (void **) &res->d_tmp, (1024)*sizeof(int)); // Used in elim_cuda. 128 is enough, some surplus for safety.


    if(n>MAX_DOUBLE_SIZE_MAT) {
        res->d_start_mat = 0;
        cudaMallocPitch((void **) &res->d_mat, &res->mat_pitch, n*sizeof(char), n);
    } else {
        cudaMallocPitch((void **) &res->d_start_mat, &res->mat_pitch, n*sizeof(char), 2*n);
        res->d_mat=res->d_start_mat + n*res->mat_pitch;
    }


    if(res->d_start_mat) {
        cudaMemcpy2D (res->d_start_mat, res->mat_pitch, res->h_start_mat, n*sizeof(char), n*sizeof(char), n, cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy2D (res->d_mat, res->mat_pitch, res->h_start_mat, n*sizeof(char), n*sizeof(char), n, cudaMemcpyHostToDevice);
    }

#ifdef NO_CONCURRENCY
    res->iter_stream=0;
    res->norm_stream=0;
    res->zero_stream=0;
#else
    cudaStreamCreate(&res->iter_stream);
    cudaStreamCreate(&res->norm_stream);
    cudaStreamCreate(&res->zero_stream);

#endif

    checkCUDAError("init_cuda_clique", __LINE__);
}

extern "C" void clear_cuda_clique(struct cuda_clique_data *res)
{

    for(int i=0; i<3; i++) free_instance(res->instances[i]);


    cudaFree(res->d_instances);
    checkCUDAError("cudaFree(res->d_instances);", __LINE__);

	cudaFree(res->d_raw);
    checkCUDAError("cudaFree(res->d_raw);", __LINE__);

    if(res->d_start_mat) {
        cudaFree(res->d_start_mat);
        checkCUDAError("cudaFree(res->d_start_mat);", __LINE__);
    } else {
        cudaFree(res->d_mat);
        checkCUDAError("cudaFree(res->d_mat);", __LINE__);
    }

	cudaFree(res->d_tmp);
    checkCUDAError("cudaFree(res->d_tmp);", __LINE__);

    cudaStreamDestroy(res->iter_stream);
    cudaStreamDestroy(res->norm_stream);
    cudaStreamDestroy(res->zero_stream);

	checkCUDAError("clear_cuda_clique", __LINE__);

    cudaFreeHost(res->h_instances);
	free(res->h_start_mat);
    free(res->instances);

    cuda_end();
}

static int mode(float alpha, float omega)
{
    if(alpha==0 && omega==1) {
        return MODE_SIMPLE;
    } else if(omega==1) {
        return MODE_REGULAR;
    } else {
        return MODE_EXP_ATTEN;
    }
}

extern "C" float iterate_cuda_clique(struct cuda_clique_data *data, float *x, int max_unsolved, float zero, float alpha, float omega, float *par_unsolved, int *abortcheck_cb(void))
{
	memset(data->instances[0].h_x, 0, sizeof(float) * data->n);

	for(int i=0; i<data->n; i++) {
        data->instances[0].h_x[i]=x[i];
    }

	data->max_unsolved=max_unsolved;
	data->max_unsolved=0;

    if(zero>0) data->zero=zero;
    else data->zero=0.0001f;

    data->zero /= (float)data->n;

    data->alpha=alpha;
    data->omega=omega;
    data->mode=mode(alpha, omega);


#ifdef DEBUG
    P_FLOAT(data->alpha) P_FLOAT(data->omega) P_NL;
    P_INT(data->n_edges) P_FLOAT(data->alpha) P_NL;
#endif

	{
        cudaMemcpy(data->instances[0].d_x, data->instances[0].h_x, data->n * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemset(data->instances[0].d_x_status, 0, sizeof(struct cuda_clique_status));
        cudaMemset(data->instances[1].d_x_status, 0, sizeof(struct cuda_clique_status));
        cudaMemset(data->instances[2].d_x_status, 0, sizeof(struct cuda_clique_status));

		d_iterate(data, abortcheck_cb);

        cudaMemcpy(data->instances[0].h_x_status, data->instances[0].d_x_status, sizeof(struct cuda_clique_status), cudaMemcpyDeviceToHost );
        cudaMemcpy(data->instances[1].h_x, data->instances[1].d_x, data->n * sizeof(float), cudaMemcpyDeviceToHost );
        cudaMemcpy(&(data->nmap), data->d_map-1, sizeof(int), cudaMemcpyDeviceToHost );
        cudaMemcpy(&(data->h_nones), data->d_ones-1, sizeof(int), cudaMemcpyDeviceToHost );

        data->instances[0].h_x_status->csize=MAX(data->instances[0].h_x_status->csize, 0); // ugly hack to prevent stupid things if iteration was externally aborted too early


        checkCUDAError("iterate_cuda_clique", __LINE__);
    }


    for(int i=0; i<data->n; i++) x[i]=data->instances[1].h_x[i];

    // P_FLOAT_ARR(x, data->n);

    if(par_unsolved) *par_unsolved=UNSOLVED(data->instances[0].h_x_status, data->n);

#ifdef DEBUG
    P_T("iterate_cuda_clique done") P_NL
    P_INT(data->n)
    P_FLOAT(data->instances[0].h_x_status->csize) P_INT(data->instances[0].h_x_status->nones) P_INT(data->instances[0].h_x_status->nzeroes)
    P_INT(data->h_nones)
    P_FLOAT(data->alpha) P_FLOAT(data->omega) P_NL;
    P_FLOAT(UNSOLVED(data->instances[0].h_x_status, data->nmap));
    P_NL;
#endif

    return data->instances[0].h_x_status->csize+data->h_nones;
}

extern "C" float cuda_clique_size(struct cuda_clique_data *data, float *x, float alpha, float omega, float *aux_x)
{
    memset(data->instances[0].h_x, 0, sizeof(float) * data->n);

    for(int i=0; i<data->n; i++) {
        data->instances[0].h_x[i]=x[i];
    }

#ifdef DEBUG
    P_FLOAT_ARR(x, data->n) P_NL;
#endif

    data->alpha=alpha;
    data->omega=omega;
    data->mode=mode(alpha, omega);

    float res;

    cudaMemcpy(data->instances[0].d_x, data->instances[0].h_x, data->n * sizeof(float), cudaMemcpyHostToDevice);

    res=iter_one(data);

    if(aux_x) {
        cudaMemcpy(data->instances[2].h_x, data->instances[2].d_x, data->n * sizeof(float), cudaMemcpyDeviceToHost);

        for(int i=0; i<data->n; i++) {
            aux_x[i]=data->instances[2].h_x[i];
        }
#ifdef DEBUG
        P_FLOAT_ARR(aux_x, data->n) P_NL;
#endif
    }
    checkCUDAError("iterate_cuda_clique", __LINE__);

    return res;
}
