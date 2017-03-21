/*******************************************************************************
 * motzkin_cpu.c
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
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<limits.h>
#include<stdarg.h>
#include<unistd.h>
#include<sys/types.h>

#include"random.h"
#include"bitops.h"
#include"cudams.h"
#include"motzkin_cpu.h"
#include"simple_macros.h"
#include"arrays.h"

#define N_ITER 10000
#define ZERO 0.05f
#define MAX_DIFF_FACT 0.05

void init_cpu_clique(struct cpu_clique_data *res, char **graph, int n)
{
	res->n=n;

	res->lists_all=(int **)make_array(n+1, n, sizeof(int));
	res->weights_all=(unsigned char**)make_array(n+1, n, sizeof(char));
	res->lists=(int **)make_array(n+1, n, sizeof(int));
	res->weights=(unsigned char **)make_array(n+1, n, sizeof(char));

	res->list=calloc(n+1, sizeof(int));
	res->el_mask=calloc(n+1, sizeof(int));
	res->one_mask=calloc(n+1, sizeof(int));

	res->x0=calloc(n, sizeof(float));
	res->x1=calloc(n, sizeof(float));

	for(int i=0; i<n; i++) {
		for(int j=0; j<n; j++) {
			if(graph[i][j]==1 && i!=j) {
                res->lists_all[i][0]++;
                res->lists_all[i][res->lists_all[i][0]]=j;
                res->weights_all[i][res->lists_all[i][0]]=1;
            }
		}
	}
}


void clear_cpu_clique(struct cpu_clique_data *res)
{
	destroy_array((void **)res->lists_all);
	destroy_array((void **)res->lists);
	destroy_array((void **)res->weights_all);
	destroy_array((void **)res->weights);

	free(res->list);
	free(res->el_mask);
	free(res->one_mask);

	free(res->x0);
	free(res->x1);
}

int count_untouched_cpu_clique(struct cpu_clique_data *data)
{
    int n=data->n;

    int res=0;

    for(int i=0; i<n; i++)
        for(int j=1; j<=data->lists_all[i][0]; j++)
            if(data->weights_all[i][j]==1)
                res++;

    return res;
}

void apply_mask_cpu_clique(struct cpu_clique_data *res, t_bitmask mask, int e)
{
    int n=res->n;

    for(int i=0; i<n; i++) if(BIT_TEST(mask, i))
        for(int j=1; j<=res->lists_all[i][0]; j++)
            if(BIT_TEST(mask, res->lists_all[i][j])) {
                res->weights_all[i][j]=MIN((int)ceilf(sqrtf((res->weights_all[i][j] * res->weights_all[i][j])+e)), 255); // MIN(e+res->weights_all[i][j], 255);
            }
}


static void normalize(struct cpu_clique_data *data, float *x, float alpha)
{
	int n=data->n;
    float sum=0;
    float sumsq=0;

    int cnt=0;

    if(alpha==0) {
        for(int i=0; i<n; i++) {
            sum+=x[i];
            cnt+=data->el_mask[i];
        }
    } else {
        for(int i=0; i<n; i++) {
            sum+=x[i];
            sumsq+=x[i]*x[i];
            cnt+=data->el_mask[i];
        }
    }
    if(sum>0) {
        for(int i=0; i<n; i++) x[i]=x[i]/sum;
    } else if(cnt>0) {
        for(int i=0; i<n; i++) if(data->el_mask[i]) {
            x[i]=1/(float)cnt;
        }
    }

    data->norm=sum;

    /* P_INT(cnt) P_FLOAT(sum) P_FLOAT(sumsq) P_FLOAT(alpha); P_NL; */

    if(cnt==0) {
        data->csize=0;
    } else if(sum==0) {
        data->csize=1;
    } else {
        data->csize=1/(1-sum+alpha*sumsq/*/(sum*sum)*/);
    }

    data->csize+=data->nones;

    /* P_INT(data->nones) P_FLOAT(data->csize) P_NL; */
}

static void put_ones(struct cpu_clique_data *data, float *x)
{
    int n=data->n;


    float one = (data->csize-data->nones >0) ? 1.0/(data->csize-data->nones) : 1.0;

    float sum=0;

    /* P_FLOAT_ARR(x, n) P_NL; */
    /* P_INT_ARR(data->one_mask, n) P_NL; */
    /* P_FLOAT(one) P_NL; */

    for(int i=0; i<n; i++) {
        if(data->one_mask[i]==1) x[i]=one;

        sum+=x[i];
    }

    for(int i=0; i<n; i++) {
        x[i]/=sum;
    }

    /* P_FLOAT_ARR(x, n) P_NL; */
}

static void iter(struct cpu_clique_data *data, float *x1, float *x0, float zero, float alpha, float omega, float *aux_x)
{
    int n=data->n;
    int list_pos=0;

    /* P_FLOAT_ARR(x0, n) P_NL; */
    /* P_INT_ARR(data->el_mask, n) P_NL; */
    /* P_INT_ARR(data->one_mask, n) P_NL; */

    memset(x1, 0, sizeof(float) *n);

    float weights[256];
    weights[0]=0;
    for(int i=1; i<256; i++) {
        weights[i]=powf(omega, (i-1)*(i-1));
    }

    if(aux_x==0 && alpha>=0 && omega==1 /*data->rem_cnt>data->list[0]/10*/) {
        float sum=0;
        int skipped=0;
        for(int ii=1; ii<=data->list[0]; ii++) {int i=data->list[ii];
            if(data->one_mask[i]==-1) {
                if(! skipped) {
                    skipped=1;
                } else {
                    data->one_mask[i]=1;
                    data->nones++;
                    data->el_mask[i]=0;
                    x0[i]=0;
                }
            }
            sum+=x0[i];
        }
        /* P_INT(data->nones) P_NL; */
        if(sum>0) {
            for(int ii=1; ii<=data->list[0]; ii++) {int i=data->list[ii];
                x0[i]/=sum;
            }
        }
        data->rem_cnt=0;
    }

    for(int ii=1; ii<=data->list[0]; ii++) {int i=data->list[ii];
        if(x0[i]<=zero && aux_x==0 /*|| lists[i][0]<csize*/){
            data->el_mask[i]=0;
            /* data->rem_cnt++; */
            x1[i]=0;
        } else {
            if(data->el_mask[i]==0) {
                P_INT(data->el_mask[i]) P_NL;
                abort();
            }
            data->list[++list_pos]=i;
            float sum=0;

            int pos=0;

            for(int jj=1; jj<=data->lists[i][0]; jj++) {int j=data->lists[i][jj]; int w=data->weights[i][jj];
                /* P_INT(j) P_INT(uj) P_NL; */
                if(data->el_mask[j]) {
                    data->lists[i][++pos]=j;
                    data->weights[i][pos]=w;
                    if(w>1) {
                        /* sum+=powf(omega, w-1)*x0[j]; */
                        sum+=weights[w]*x0[j];
                    } else {
                        sum+=x0[j];
                    }
                }
            }

            data->lists[i][0]=pos;

            sum+=x0[i]*alpha;

            x1[i]=MAX(sum*x0[i],0);
            if(aux_x) aux_x[i]=MAX(sum, 0);

            if(pos==data->list[0]-1) {
                data->one_mask[i]=-1;
                data->rem_cnt++;
            }
        }

    }
    /* P_INT(data->rem_cnt) P_NL; */
    data->rem_cnt=0;

    data->list[0]=list_pos;

    normalize(data, x1, alpha);
}

void cleanup_matrix(struct cpu_clique_data *data)
{
    int n=data->n;

	copy_array((void **)data->lists, (void **)data->lists_all, n+1, n, sizeof(int));
	copy_array((void **)data->weights, (void **)data->weights_all, n+1, n, sizeof(char));
	for(int i=1; i<=n; i++) {
        data->list[i]=i-1;
        data->el_mask[i-1]=1;
    }
	data->list[0]=n;

    data->nones=0;
    data->nzeroes=0;
    data->rem_cnt=0;

	memset(data->one_mask, 0, sizeof(int) * n);
}

float cpu_clique_size(struct cpu_clique_data *data, float *x, float alpha, float omega, float *aux_x)
{
    float csize=0;

    float *tmp=calloc(data->n, sizeof(float));


    cleanup_matrix(data);

    iter(data, tmp, x, 0, alpha, omega, aux_x);

    free(tmp);

    csize=data->csize;

    return csize;
}



float iterate_cpu_clique(struct cpu_clique_data *data, float *x, int max_unsolved, float zero, float alpha, float omega, float *par_unsolved, int *abortcheck_cb(void))
{
    int n=data->n;


    float unsolved=n;

    if(zero==0) zero=0.0001f;

    zero /= (float)n;

    /* P_FLOAT(alpha) P_FLOAT(omega)  P_FLOAT(zero) P_NL; */

	data->rem_cnt=0;


	void randomize(float *x1, float c) {
        for(int ii=1; ii<=data->list[0]; ii++) {int i=data->list[ii];
            x1[i]*=uni_ran(-c,c)+1;
        }
		normalize(data, x1, alpha);
    }

	int is_done(float *x0, float *x1) {
		float maxdiff=0;

		/* data->csize=clique_size(x1); */

        /* float csize=data->csize; */

		data->nzeroes=0;

		for(int i=0; i<n; i++) {
			maxdiff=MAX(maxdiff, fabs(x1[i]-x0[i]));

			if(data->one_mask[i]==0 && x1[i]<=zero) data->nzeroes++;
		}

		unsolved=n-data->nzeroes-data->csize;

		float comp_diff;//=MAX_DIFF_FACT*0.04512900561816465f/powf(csize,1.772342376167655f);
#ifdef TIMER
		P_LONG_INT(get_timer()); P_FLOAT(csize); P_FLOAT(unsolved); P_FLOAT(maxdiff); P_FLOAT(comp_diff); P_NL;
#endif

        /* int nzeroes=data->nzeroes; */

		data->maxdiff=maxdiff;
        comp_diff = 0.000001f;
        /* comp_diff = 0.000000001f; */

        int done=0;

		if(/*maxdiff<comp_diff ||*/ unsolved<.5+max_unsolved) {
			done=DONE_SOLVED;
		} else if(maxdiff<comp_diff/100 || ( maxdiff<comp_diff && unsolved < 0.5 * n ) ) {
            done=DONE_CONVERGED;
        } else if(abortcheck_cb && abortcheck_cb()) {
            done=DONE_ABORTED;
        }

		/* P_FLOAT(csize) P_FLOAT(unsolved) P_FLOAT(maxdiff) P_FLOAT(comp_diff) P_INT(nzeroes) P_INT(data->list[0]) P_INT(done) P_INT(data->rem_cnt) P_NL; */
		/* P_T("is_done:: ") P_FLOAT(data->csize) P_FLOAT(unsolved) P_INT(max_unsolved) P_FLOATE(maxdiff) P_INT(n) P_FLOATE(comp_diff) P_INT(data->nzeroes) P_INT(data->nones) P_INT(done) P_INT(data->list[0]) P_NL; */

        return done;

/* 		if(maxdiff<comp_diff ||  unsolved<.5+max_unsolved) { */
/* 			return 1; */
/* 		} */
/*  */
/* 		return 0; */
	}


	int done=0;

	int iter_cnt=1;

    cleanup_matrix(data);

/* 	copy_array((void **)data->lists, (void **)data->lists_all, n+1, n, sizeof(int)); */
/* 	copy_array((void **)data->weights, (void **)data->weights_all, n+1, n, sizeof(char)); */
/* 	for(int i=1; i<=n; i++) data->list[i]=i-1; */
/* 	data->list[0]=n; */
/*  */
/* 	memset(data->el_mask, 1, sizeof(int) * n); */

	data->csize=0;
	memcpy(data->x0, x, sizeof(float) * n);
    memset(data->x1, 0, sizeof(float) * n);

    int done3_cnt=0;

	while(!done) {
        /* P_INT(iter_cnt) P_NL; */
/*         P_INT_ARR_L(data->list) P_NL; */
/*  */
/*         P_FLOAT_ARR(data->x0, n) P_NL; */
/*  */
/*         for(int i=0; i<1; i++) { */
/*             P_INT(i) P_FLOAT(data->x1[i]) P_NL; */
/*         } */

        /* P_FLOAT_ARR(data->x1, n) P_NL; */

		iter(data, data->x1, data->x0, zero, alpha, omega, 0);

        /* if(iter_cnt==1) { */
        /*     P_FLOAT_ARR(data->x1, n) P_NL; */
        /* } */


        /* P_FLOAT_ARR(data->x0, n) P_NL; */
        /* for(int i=0; i<1; i++) { */
        /*     P_INT(i) P_FLOAT(data->x1[i]) P_NL; */
        /* } */
        /* P_FLOAT_ARR(data->x1, n) P_NL; */

		if(! (iter_cnt % 10)) 
            done=is_done(data->x0, data->x1);

        if(done==DONE_CONVERGED && done3_cnt<3) {
            randomize(data->x1, 0.05);
            done=0;
            done3_cnt++;
        }

        /* P_INT(iter_cnt) P_FLOAT(alpha) P_NL; */

		SWAP(data->x1, data->x0);
		iter_cnt++;
		if(iter_cnt>=N_ITER) done=1;
	}


    put_ones(data, data->x0);

    /* P_T("CPU:  ") P_INT(iter_cnt) P_INT(N_ITER) P_NL; */

	memcpy(x, data->x0, sizeof(float) * n);

	if(par_unsolved) *par_unsolved=unsolved;

    /* P_T("interate_cpu_clique: ") P_FLOAT(data->csize) P_NL; */

	return MAX(data->csize, 0);
}
