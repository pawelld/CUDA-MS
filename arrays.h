/*******************************************************************************
 * arrays.h
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

#ifndef __ARRAYS_H__
#define __ARRAYS_H__

#include<stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


int memory_available(size_t size, float frac);
size_t array_size(long int n, long int m, int size);
void **make_array(long int n, long int m, int size);
void copy_array(void **dst, void ** src, long int n, long int m, int size);
void destroy_array(void **array);

#ifdef __cplusplus
}
#endif

#endif
