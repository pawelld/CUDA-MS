/*******************************************************************************
 * common_cuda.cu
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

#ifndef __COMMON_CUDA_CUH__
#define __COMMON_CUDA_CUH__

template <unsigned int blockSize, typename t,  void (*op)(t *, int, int)> __device__ void reduce(t *sdata, unsigned int tid)
{

    if (blockSize >= 512) { if (tid < 256) { op(sdata, tid, tid + 256); } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { op(sdata, tid, tid + 128); } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { op(sdata, tid, tid + 64 ); } __syncthreads(); }

    if (blockSize >=  64) { if (tid < 32)  { op(sdata, tid, tid + 32); } }
    if (blockSize >=  32) { if (tid < 16)  { op(sdata, tid, tid + 16); } }
    if (blockSize >=  16) { if (tid <  8)  { op(sdata, tid, tid +  8); } }
    if (blockSize >=   8) { if (tid <  4)  { op(sdata, tid, tid +  4); } }
    if (blockSize >=   4) { if (tid <  2)  { op(sdata, tid, tid +  2); } }
    if (blockSize >=   2) { if (tid <  1)  { op(sdata, tid, tid +  1); } }
}

template <unsigned int blockSize, typename t> inline __device__ void min_reduce(t *data, int pos1, int pos2)
{
    data[pos1]=MIN(data[pos1], data[pos2]);
}

template <unsigned int blockSize, typename t> inline __device__ void max_reduce(t *data, int pos1, int pos2)
{
    data[pos1]=MAX(data[pos1], data[pos2]);
}

template <unsigned int blockSize, typename t> inline __device__ void sum_reduce(t *data, int pos1, int pos2)
{
    data[pos1]=data[pos1] + data[pos2];
}

template <unsigned int blockSize, typename t> __device__ void reduce_sum(t *sdata, unsigned int tid)
{

    if (blockSize >= 512) { if (tid < 256) { sdata[tid]+=sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid]+=sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { sdata[tid]+=sdata[tid + 64 ]; } __syncthreads(); }

    if (blockSize >=  64) { if (tid < 32)  { sdata[tid]+=sdata[tid + 32]; } }
    if (blockSize >=  32) { if (tid < 16)  { sdata[tid]+=sdata[tid + 16]; } }
    if (blockSize >=  16) { if (tid <  8)  { sdata[tid]+=sdata[tid +  8]; } }
    if (blockSize >=   8) { if (tid <  4)  { sdata[tid]+=sdata[tid +  4]; } }
    if (blockSize >=   4) { if (tid <  2)  { sdata[tid]+=sdata[tid +  2]; } }
    if (blockSize >=   2) { if (tid <  1)  { sdata[tid]+=sdata[tid +  1]; } }
}

template <unsigned int blockSize, typename t> __device__ void reduce_max(t *sdata, unsigned int tid)
{

    if (blockSize >= 512) { if (tid < 256) { if(sdata[tid + 256] > sdata[tid]) sdata[tid]=sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { if(sdata[tid + 128] > sdata[tid]) sdata[tid]=sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid < 64)  { if(sdata[tid + 64 ] > sdata[tid]) sdata[tid]=sdata[tid + 64 ]; } __syncthreads(); }

    if (blockSize >=  64) { if (tid < 32)  { if(sdata[tid + 32] > sdata[tid]) sdata[tid]=sdata[tid + 32]; } }
    if (blockSize >=  32) { if (tid < 16)  { if(sdata[tid + 16] > sdata[tid]) sdata[tid]=sdata[tid + 16]; } }
    if (blockSize >=  16) { if (tid <  8)  { if(sdata[tid +  8] > sdata[tid]) sdata[tid]=sdata[tid +  8]; } }
    if (blockSize >=   8) { if (tid <  4)  { if(sdata[tid +  4] > sdata[tid]) sdata[tid]=sdata[tid +  4]; } }
    if (blockSize >=   4) { if (tid <  2)  { if(sdata[tid +  2] > sdata[tid]) sdata[tid]=sdata[tid +  2]; } }
    if (blockSize >=   2) { if (tid <  1)  { if(sdata[tid +  1] > sdata[tid]) sdata[tid]=sdata[tid +  1]; } }
}

template <unsigned int blockSize, typename t, bool with_last> __device__ void prefixscan(t *data, int tid)
{
    if(with_last) {
        if(tid==blockSize-1) data[2*blockSize]=data[tid+blockSize];
    }

    __syncthreads();

    int offset=1;
    int n2=blockSize<<1;

    for(int d=n2>>1; d>0; d>>=1) {
        __syncthreads();

        if(tid<d) {
            int ai=offset*(2*tid+1)-1;
            int bi=offset*(2*tid+2)-1;

            data[bi]+=data[ai];
        }

        offset<<=1;
    }

    if(tid==0) data[n2-1]=0;

    for(int d=1; d<n2; d<<=1) {
        offset>>=1;
        __syncthreads();

        if(tid<d) {
            int ai=offset*(2*tid+1)-1;
            int bi=offset*(2*tid+2)-1;

            int tmp=data[ai];
            data[ai]=data[bi];
            data[bi]+=tmp;
        }
    }

    if(with_last) {
        if(tid==blockSize-1) data[2*blockSize]+=data[tid+blockSize];
    }
}

#endif
