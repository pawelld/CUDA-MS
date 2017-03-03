/*******************************************************************************
 * motzkin_cuda.h
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

#ifndef __MOTZKIN_CUDA_H__
#define __MOTZKIN_CUDA_H__


#include"cudams.h"

#include<cuda.h>
#include<curand.h>

#define DONE_SOLVED 1
#define DONE_CLEANUP 2
#define DONE_CONVERGED 3

#define MODE_SIMPLE 1
#define MODE_REGULAR    2
#define MODE_ATTEN 3
#define MODE_EXP_ATTEN 4

// #ifdef __cplusplus
// #include<curand_kernel.h>
// #else
// typedef int curandState_t;
// #endif

#ifdef __cplusplus
extern "C" {
#endif


struct cuda_clique_status {
    float norm;
    float norm2;
	float csize;
	float maxdiff;
	int nzeroes;
	int nones;
};

struct cuda_clique_instance {
	float *d_x;
	float *h_x;

    struct cuda_clique_status *d_x_status;
    struct cuda_clique_status *h_x_status;

    cudaEvent_t quadratic_done;
    cudaEvent_t norm_done;
    cudaEvent_t zero_done;
};

struct cuda_clique_data {
	int n;
    int n_edges;
	size_t mat_pitch;

    int mode;
    float alpha;
    float omega;

    int vec_size;

    struct cuda_clique_instance *instances;
    void *d_instances;
    void *h_instances;

	unsigned char *h_start_mat;
	unsigned char *d_start_mat;
	unsigned char *d_mat;


    void *d_raw;

	int *d_map;
	int *d_new_map;
	int *d_new_revmap;
	int *d_one_revmap;
	int *d_ones;

    int *d_tmp;


    int h_nones;

    int *d_incident;

    int nmap;

	// float *h_x_min;

    // float *d_random;

    // curandGenerator_t generator;

    // curandState_t *d_randstates;


	size_t x_pitch;

	int max_unsolved;

    float zero;

    // float *h_csize0;
    // float *h_csize1;
    // float *h_csize2;

    // int *h_nmap;

    cudaStream_t iter_stream;
    cudaStream_t norm_stream;
    cudaStream_t zero_stream;

    int failed;
};


void init_cuda(void);

void reset_cuda(void);

int init_cuda_clique(struct cuda_clique_data *res, char **graph, int n);
void clear_cuda_clique(struct cuda_clique_data *res);
float iterate_cuda_clique(struct cuda_clique_data *data, float *x, int max_unsolved, float zero, float alpha, float omega, float *par_unsolved, int *abortcheck_cb(void));

void apply_mask_cuda_clique(struct cuda_clique_data *res, t_bitmask mask, int e);
float cuda_clique_size(struct cuda_clique_data *data, float *x, float alpha, float omega, float *aux_x);
int count_untouched_cuda_clique(struct cuda_clique_data *data);


#ifdef __cplusplus
}
#endif

#endif
