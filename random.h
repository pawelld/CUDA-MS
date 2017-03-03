/*******************************************************************************
 * random.h
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

#ifndef __RANDOM_H__
#define __RANDOM_H__

#ifdef __cplusplus
extern "C" {
#endif

void init_ran();
int sign_ran();
int int_ran(int min, int max);
double uni_ran(double min,double  max);
int geom_ran(int max, double p);
double gauss_ran(double mean,double dev);
void permutation(int *perm, int n_points);
//void permutation(int n_points; int perm[n_points], int n_points);

#ifdef __cplusplus
}
#endif

#endif
