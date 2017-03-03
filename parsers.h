/*******************************************************************************
 * parsers.h
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

#ifndef __PARSERS_H__
#define __PARSERS_H__

int read_graph_DIMACS_bin(char *file, char ***arr,  int *edges);
int read_graph_DIMACS_ascii(char *file, char ***arr,  int *edges);
int read_graph_adjmat(char *file, char ***arr,  int *edges);

#endif
