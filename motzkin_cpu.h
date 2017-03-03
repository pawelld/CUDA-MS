/*******************************************************************************
 * motzkin_cpu.h
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

#ifndef __MOTZKIN_CPU_H__
#define __MOTZKIN_CPU_H__

struct cpu_clique_data {
	int n;

	int **lists_all;
    unsigned char **weights_all;
	int **lists;
    unsigned char **weights;

	int *list;
	int *el_mask;
	int *one_mask;

	float *x0;
	float *x1;

	float csize;

	float maxdiff;
	int nzeroes;
    int nones;

    float norm;

    int rem_cnt;
};

#define DONE_SOLVED 1
#define DONE_CLEANUP 2
#define DONE_CONVERGED 3

//void init_cpu_clique(struct cpu_clique_data *res, char **graph, int par_n_al);
int init_cpu_clique(struct cpu_clique_data *res, char **graph, int n);
void clear_cpu_clique(struct cpu_clique_data *res);
//float iterate_cpu_clique(struct cpu_clique_data *data, float *x, int max_unsolved, float zero, int biased, float *par_unsolved);
float iterate_cpu_clique(struct cpu_clique_data *data, float *x, int max_unsolved, float zero, float alpha, float omega, float *par_unsolved, int *abortcheck_cb(void));

void apply_mask_cpu_clique(struct cpu_clique_data *res, t_bitmask mask, int e);
float cpu_clique_size(struct cpu_clique_data *data, float *x, float alpha, float omega, float *aux_x);
int count_untouched_cpu_clique(struct cpu_clique_data *data);

#endif
