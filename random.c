/*******************************************************************************
 * random.c
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

#include<string.h>
#include<math.h>
#include<stdlib.h>
#include<time.h>
#include<unistd.h>


#include"random.h"

void init_ran()
{
	unsigned int seed;

	seed=time(NULL)%getpid()+20;

#pragma omp critical(rand)
	{
		srand(seed);
	}
}

int sign_ran()
{
	return(int_ran(0,1)*2-1);
}

int int_ran(int min, int max)
{
	int res;

#pragma omp critical(rand)
	{
		res=min+rand()%(max-min+1);
	}

	return res;
	//	return (min+(int)((double)(max-min+1)*rand()/(RAND_MAX+1.0)));
}

double uni_ran(double min,double  max)
{
	double res;

#pragma omp critical(rand)
	{
		res=(min+((max-min)*rand()/(RAND_MAX)));
	}

	return res;
}

int geom_ran(int max, double p)
{
	double r;
	do
	{
		r = uni_ran(0,1);
		if(max) r *= 1.0 - pow(1-p, (double)max);
		r = log(1.0 - r) / log(1-p);
	} while( r >= max);            // safety

	return (int)r+1;
}

double gauss_ran(double mean,double dev)
{
	static int iset=0;
	static double gset;
	float fac, rsq, v1, v2;

	if(iset==0) {
		do {
			v1=uni_ran(-1.0, 1.0);
			v2=uni_ran(-1.0, 1.0);
			rsq=v1*v1+v2*v2;
		} while (rsq>=1.0 || rsq==0.0);

		fac=sqrt(-2.0*log(rsq)/rsq);

		gset=v1*fac;
		iset=1;
		return (v2*fac*dev+mean);
	} else {
		iset=0;
		return (gset*dev+mean);
	}
}

/* void permutation(int n_points; int perm[n_points], int n_points) */
void permutation(int *perm, int n_points)
{
	int perm_vect[n_points];

	memset(perm_vect, 0, sizeof(perm_vect));

	for(int i=0; i<n_points; i++) {
		int j=int_ran(1,n_points-i);
		int k=-1;
		while(j) if(!perm_vect[++k]) j--;
		perm_vect[k]=1;
		perm[i]=k;
	}

}
