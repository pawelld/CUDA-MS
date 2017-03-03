/*******************************************************************************
 * cudams.h
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

#ifndef __CUDAMS_H__
#define __CUDAMS_H__

#define MODE_REPL_UNBIASED 1
#define MODE_REPL 2
#define MODE_REPL_ANNEAL 3
#define MODE_REPL_ATTEN 4
#define MODE_REPL_ANNEAL_AUTO 5
#define MODE_REPL_ATTEN_AUTO 6

#include"bitops.h"

float graph_clique(t_bitmask *res, t_bitmask *res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode);
float graph_clique_multi(t_bitmask *res, t_bitmask *res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode, int max_res, t_bitmask *res_all, int *n_res, int *abortcheck_cb(void));

float graph_clique_cpu(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode);
float graph_clique_multi_cpu(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode, int max_res, t_bitmask *res_all, int *n_res, int *abortcheck_cb(void));

#ifndef NO_CUDA
float graph_clique_cuda(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode);
float graph_clique_multi_cuda(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode, int max_res, t_bitmask *res_all, int *n_res, int *abortcheck_cb(void));
#endif

#endif
