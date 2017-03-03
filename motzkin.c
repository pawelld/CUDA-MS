/*******************************************************************************
 * motzkin.c
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
#include<string.h>
#include<sys/types.h>

#include<sys/time.h>

#include<math.h>

#include"bitops.h"
#include"cudams.h"
#include"simple_macros.h"
#include"random.h"
#include"arrays.h"

#ifndef NO_CUDA
#include<cuda.h>
#include"motzkin_cuda.h"
#endif
#include"motzkin_cpu.h"

#define MIN_SIZE 1


int failed=0;

long long int init_time, free_time;

//#define VERBOSE

#define BASE_WEIGHT 0.98388

static long int get_timestamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(long int)1000000+tv.tv_usec;
}
#ifndef NO_CUDA

#define declare_data() \
	struct cuda_clique_data cuda_data;\
	memset(&cuda_data, 0, sizeof(cuda_data));\
	struct cpu_clique_data cpu_data;\
	memset(&cpu_data, 0, sizeof(cpu_data));

#define init_clique(graph, graph_size) ({\
    int __res;\
    if(cuda) {\
        long int start = get_timestamp();\
        __res=init_cuda_clique(&cuda_data, graph, graph_size);\
        long int end = get_timestamp();\
        init_time=end-start;\
    } else {\
        __res=init_cpu_clique(&cpu_data, graph, graph_size);\
    }\
    __res;\
})

#define iterate_clique(x, max_unsolved, zero, alpha, omega, par_unsolved, abortcheck_cb) ({\
    float __res;\
    if(cuda) {\
        __res=iterate_cuda_clique(&cuda_data, x, max_unsolved, zero, alpha, omega, par_unsolved, abortcheck_cb);\
    } else {\
        __res=iterate_cpu_clique(&cpu_data, x, max_unsolved, zero, alpha, omega, par_unsolved, abortcheck_cb);\
    }\
    __res;\
})

#define clique_size(x, alpha, omega, aux_x) ({\
    float __res;\
    if(cuda) {\
        __res=cuda_clique_size(&cuda_data, x, alpha, omega, aux_x);\
    } else {\
        __res=cpu_clique_size(&cpu_data, x, alpha, omega, aux_x);\
    }\
    __res;\
})

#define reset() ({\
    printf("reset\n"); P_INT(cuda) P_NL;\
    if(cuda) {\
        reset_cuda();\
    }\
})

#define clear_clique() ({\
    if(cuda) {\
        long int start = get_timestamp();\
        clear_cuda_clique(&cuda_data);\
        long int end = get_timestamp();\
        free_time=end-start;\
    } else {\
        clear_cpu_clique(&cpu_data);\
    }\
})

#define count_untouched_clique() ({\
    int __res;\
    if(cuda) {\
        __res=count_untouched_cuda_clique(&cuda_data);\
    } else {\
        __res=count_untouched_cpu_clique(&cpu_data);\
    }\
    __res;\
})

#define apply_mask_clique(mask, n) ({\
    if(cuda) {\
        apply_mask_cuda_clique(&cuda_data, mask, n);\
    } else {\
        apply_mask_cpu_clique(&cpu_data, mask, n);\
    }\
})
#else

#define declare_data() \
	struct cpu_clique_data cpu_data;\
	memset(&cpu_data, 0, sizeof(cpu_data));

#define init_clique(graph, graph_size) ({init_cpu_clique(&cpu_data, graph, graph_size)})
#define iterate_clique(x, max_unsolved, zero, alpha, omega, par_unsolved, abortcheck_cb) ({iterate_cpu_clique(&cpu_data, x, max_unsolved, zero, alpha, omega, par_unsolved, abortcheck_cb);})
#define clique_size(x, alpha, omega, aux_x) ({cpu_clique_size(&cpu_data, x, alpha, omega, aux_x);})
#define clear_clique() ({clear_cpu_clique(&cpu_data);})

#define count_untouched_clique() ({count_untouched_cpu_clique(&cpu_data);})
#define apply_mask_clique(mask, n) ({apply_mask_cpu_clique(&cpu_data, mask, n);})

#endif


static int is_clique_extendable(char** graph, int graph_size, t_bitmask mask, int el) 
{
    for(int i=0; i<graph_size; i++) if(i!=el) if(BIT_TEST(mask, i) && !graph[i][el]) return 0;

    return 1;
}

static int build_clique(t_bitmask lower, t_bitmask upper, float *x, char **graph, int graph_size, int max_unsolved)
{
    mask_zeroall(lower, graph_size);
    mask_zeroall(upper, graph_size);

    int n_lower=0;
    int n_upper=0;


    int order[graph_size];

    for(int i=0; i<graph_size; i++) order[i]=i;

    int order_comp(const int *a, const int *b) {
        if (x[*a] < x[*b])
            return 1;
        else if (x[*a] > x[*b])
            return -1;
        else
            return 0;
    }

    qsort(order, graph_size, sizeof(int), (comparison_fn_t) order_comp);


    for(int i=0; i<graph_size; i++) if(x[order[i]]>0) {
        SET_BIT(upper, order[i]);
        n_upper++;
    }

    for(int i=0; i<graph_size; i++) if(BIT_TEST(upper, order[i])) {
        if((max_unsolved && is_clique_extendable(graph, graph_size, upper, order[i])) || (!max_unsolved && is_clique_extendable(graph, graph_size, lower, order[i]))) {
            SET_BIT(lower, order[i]);
            n_lower++;
        } else {
            if(!max_unsolved) break;
        }
    }

    int log=0;

    void add_to_res(void) {
        for(int i=0; i<graph_size; i++) if(BIT_TEST(upper, order[i]) && !BIT_TEST(lower, order[i])) {
            if(is_clique_extendable(graph, graph_size, upper, order[i])) {
                SET_BIT(lower, order[i]);
                n_lower++;
                log=0;
            } else break;
        }
    }

    void remove_from_upper(void) {
        for(int i=graph_size-1; i>0; i--) if(BIT_TEST(upper, order[i]) && !BIT_TEST(lower, order[i])) {
            if(!is_clique_extendable(graph, graph_size, upper, order[i])) {
                UNSET_BIT(upper, order[i]);
                n_upper--;
                log=0;
                break;
            }
        }
    }

    while(n_upper-n_lower>max_unsolved || n_lower < MIN_SIZE) {
        add_to_res();
        remove_from_upper();

        log++;
        if(log>=2) break;
    }
    add_to_res();

    for(int i=0; i<graph_size; i++) if(BIT_TEST(lower, i)) {
        for(int j=i+1; j<graph_size; j++) if(BIT_TEST(lower, j)) {
            if(! graph[i][j]) {

                for(int i=0; i<graph_size; i++) if(BIT_TEST(lower, i)) {
                    for(int j=0; j<graph_size; j++) if(BIT_TEST(lower, j)) {
                        printf("%d ", graph[i][j]);
                    }
                    P_NL;
                }

                abort();

            }
        }
    }



    return n_lower;
}


static float find_clique_motzkin(char **graph, int graph_size, float *x, int max_unsolved, float zero, float alpha, int mode, int cuda)
{

    declare_data();

    float csize=-1;

retry:

    if(init_clique(graph, graph_size)) goto skip;
    if(mode == MODE_REPL_UNBIASED) alpha=0;
    csize=iterate_clique(x, max_unsolved, zero, alpha, 1., 0, 0);

skip:
    clear_clique();

    if(csize<0) {
        printf("!!!! iterate_clique has failed. Retrying.\n");
        reset();
        goto retry;
    }


    return csize;
}


static void vec_to_mask(t_bitmask mask, float *x, int n)
{
    mask_zeroall(mask, n);

    float val=0.01 * 1/(float)n;

    for(int i=0; i<n; i++) if(x[i]>=val) SET_BIT(mask, i);
}

static void fix_zeroes(float *x, int n, float zero)
{
    float sum=0;
    float sum1=0;
    float eps=1./(float)n;
    for(int i=0; i<n; sum+=x[i++]);
    for(int i=0; i<n; i++) {
        x[i]/=sum;
        x[i]=MAX(x[i], eps);
        sum1+=x[i];
    }
    for(int i=0; i<n; x[i++]/=sum1);

}

static void vec_sim(float *s0in1, float *s1in0, float *x0, float *x1, int graph_size)
{
    int n0=0;
    int n1=0;
    int n01=0;

    for(int i=0; i<graph_size; i++) {
        if(x0[i]>0 && x1[i]>0) n01++;
        else if(x0[i]>0) n0++;
        else if(x1[i]>0) n1++;
    }

    *s0in1=(float)(n01)/(float)(n0+n01);
    *s1in0=(float)(n01)/(float)(n1+n01);
}

static int add_one_to_best(float **all_res, float *all_csize, int n_res, int max_res, float *max_sim, float *res, float csize, int graph_size)
{

    int pos=-1;

    int n_del=0;
    int to_del[max_res];

    int min_csize=-1;
    int max_csize=0;

    int first_empty=-1;

    for(int i=0; i<max_res; i++) {
        if(all_csize[i]>0) {
            max_csize=MAX(max_csize, all_csize[i]);
            if(min_csize>0) min_csize=MIN(min_csize, all_csize[i]);
            else min_csize=all_csize[i];
        } else if(first_empty<0) {
            first_empty=i;
        }
    }

    memset(to_del, 0, sizeof(to_del));

    for(int i=0; i<max_res; i++) if(all_csize[i]>0) {
        float snewinres;
        float sresinnew;

        vec_sim(&snewinres, &sresinnew, res, all_res[i], graph_size);

        if(snewinres>=*max_sim && all_csize[i]>=csize) {
            pos=-2;
            break;
        } else if(sresinnew>*max_sim && all_csize[i]<csize) {
            to_del[i]=1;
            if(n_del==0) pos=i;
            n_del++;
        }
    }

    if(n_res<max_res && pos!=-2) {
        pos=first_empty;
        n_res++;
    }

    if(pos==-1 && csize>MIN(min_csize, max_csize*.5) ) {
        for(int i=0; i<max_res; i++) if(all_csize[i]>0) {
            for(int j=i+1; j<max_res; j++) if(all_csize[j]>0) {
                float sij, sji;

                vec_sim(&sij, &sji, all_res[i], all_res[j], graph_size);

                if(*max_sim==MAX(sij, sji)) {
                    to_del[all_csize[i]>=all_csize[j] ? j : i]=1;
                    n_del++;
                }
            }
        }

        if(n_del>=1) {
            for(int i=0; i<max_res; i++) if(to_del[i]) {
                if(pos<0) pos=i;
                else {
                    if(all_csize[i]<all_csize[pos]) pos=i;
                }
            }
        }

        n_del=0;
    }

    n_res = n_res-(n_del ? n_del-1 : 0);

    if(pos>=0) {
        if(n_del) {
            for(int i=0; i<max_res; i++) if(to_del[i]) {
                all_csize[i]=0;
                memset(all_res[i], 0, graph_size * sizeof(float));
            }
        }
        memcpy(all_res[pos], res, sizeof(float)*graph_size);
        all_csize[pos]=csize;

        if(n_res < max_res) {
            *max_sim=1;
        } else {
            *max_sim=-1;

            for(int i=0; i<max_res; i++) if(all_csize[i]>0) {
                for(int j=i+1; j<max_res; j++) if(all_csize[j]>0) {
                    float sij, sji;

                    vec_sim(&sij, &sji, all_res[i], all_res[j], graph_size);
                    *max_sim=MAX3(*max_sim, sij, sji);
                }
            }
            if(*max_sim<0) *max_sim=1;
        }
    }

    return n_res;
}


static int auto_alphas(char **graph, int graph_size, float **par_alpha, float min_alpha)
{

    int edge_count=0;

    for(int i=0; i<graph_size; i++)
        for(int j=i+1; j<graph_size; j++) {
            edge_count += graph[i][j];
        }


    float q = 2 * (float)edge_count / (float) (graph_size * (graph_size-1));

    if(graph_size<=2 || q>=0.999) {
        float alpha[3];
        int n_alpha=0;
        alpha[n_alpha++]=-0.5;
        alpha[n_alpha++]=0;
        alpha[n_alpha++]=0.5;

        *par_alpha=calloc(n_alpha, sizeof(float));

        memcpy(*par_alpha, alpha, n_alpha*sizeof(float));

        return n_alpha;

    }

    float log1q = logf(1/q);

    int m = (int) ceilf(2*(logf(graph_size) - logf(logf(graph_size) / log1q) + (1+logf(0.5)))/log1q + 1);

    if(m<0) {
#ifdef VERBOSE
        P_INT(m) P_NL;
#endif
        m = (int) ceilf(graph_size/(2*logf(graph_size * q)) * logf(1/(1-q)));
    }

#ifdef VERBOSE
    P_INT(m) P_NL;
#endif

    float gamma[m+1];

    for(int i=1; i<=m; i++) {
        gamma[i] = 1 - (1-q)*i - sqrtf(i * q * (1-q)) * pow(0.01, -0.5/(graph_size - i));
    }


    int pos=m;

    float alpha[m+5];
    int n_alpha=0;
    while(pos>=2) {
        alpha[n_alpha] = (gamma[pos] + gamma[pos-1])*.5;
        pos--;
        if(alpha[n_alpha]>0) {
            break;
        }
        if(n_alpha==0 || (alpha[n_alpha]-alpha[n_alpha-1]) > min_alpha) n_alpha++;
    }

    alpha[n_alpha++]=0;
    alpha[n_alpha++]=0.5;

    *par_alpha=calloc(n_alpha, sizeof(float));

    memcpy(*par_alpha, alpha, n_alpha*sizeof(float));

    return n_alpha;
}

static int simple_alphas(float **par_alpha, float min_alpha)
{
    min_alpha=floor(min_alpha+.5)-.5;

    int n_alpha=(int)(-floor(min_alpha)+1);

    float alpha[n_alpha];
    for(int i=0; i<n_alpha; i++) alpha[i]=min_alpha+i;

    *par_alpha=calloc(n_alpha, sizeof(float));

    memcpy(*par_alpha, alpha, n_alpha*sizeof(float));

    return n_alpha;
}

static int fill_alphas(char **graph, int graph_size, float **par_alpha, float min_alpha, int auto_alpha)
{
    int n_alpha;

    if(auto_alpha) {
        n_alpha=auto_alphas(graph, graph_size, par_alpha, min_alpha);
    } else {
        n_alpha=simple_alphas(par_alpha, min_alpha);
    }

    printf("alpha[0 .. %d]: ", n_alpha);
    for(int i=0; i<n_alpha; i++) printf("%f ", (*par_alpha)[i]);
    printf("\n");

    return n_alpha;
}

static float find_clique_anneal(char **graph, int graph_size, float *res_x, int max_unsolved, float zero, float min_alpha, int cuda, int auto_alpha)
{
    declare_data();


    float *alpha;
    int n_alpha;

    n_alpha=fill_alphas(graph, graph_size, &alpha, min_alpha, auto_alpha);


    float csize;


retry:

    if(init_clique(graph, graph_size)) goto skip;
    for(int i=0; i<n_alpha; i++) {
        fix_zeroes(res_x, graph_size, zero);


        csize=iterate_clique(res_x, max_unsolved, zero, alpha[i], 1, 0, 0);

        if(csize<0) goto skip;
    }

    csize=clique_size(res_x, .0, 1, 0);

    clear_clique();

    free(alpha);

    return csize;

skip:
    clear_clique();
    printf("!!!! iterate_clique has failed. Retrying.\n");
    reset();
    goto retry;
}


static float find_clique_atten(char **graph, int graph_size, float *res_x, int max_unsolved, float zero, float min_alpha, int max_masks, int cuda, float **all_res, int max_res, int auto_alpha, int *abortcheck_cb(void))
{
    float *alpha;
    int n_alpha;

    n_alpha=fill_alphas(graph, graph_size, &alpha, min_alpha, auto_alpha);

    declare_data();

    t_bitmask *masks=calloc(max_masks, sizeof(void*));

    for(int i=0; i<max_masks; i++) masks[i]=mask_alloc(graph_size);


#ifdef VERBOSE
    P_INT(graph_size) P_INT(max_unsolved)  P_FLOAT(zero) P_FLOAT(min_alpha)  P_INT(max_masks) P_INT(cuda) P_INT(max_res) P_NL;
#endif

    int n_res=0;
    float max_sim=1;
    int n_masks=0;

    float *x=calloc(graph_size, sizeof(float));
    float csize;

    float *tmp_x=calloc(graph_size, sizeof(float));
    float tmp_csize;

    float *aux_x=calloc(graph_size, sizeof(float));

    float *ones=calloc(graph_size, sizeof(float));


    float *max_x=calloc(graph_size, sizeof(float));
    float max_csize=-1;

    float *all_csize=0;

    if(all_res) {
        all_csize=calloc(max_res, sizeof(float));
    }

    float first_csize=0;

retry:


    n_res=0;
    max_sim=1;
    n_masks=0;
    for(int i=0; i<graph_size; i++) ones[i]=1.;



    if(init_clique(graph, graph_size)) goto skip;
    while(n_masks<=max_masks) {
        if(abortcheck_cb && abortcheck_cb()) break;

#ifdef VERBOSE
        P_INT(n_masks) P_NL;
        for(int i=0; i<n_masks; i++) {
            P_MASK_BIT(masks[i], graph_size) P_NL;
        }
#endif

        if(count_untouched_clique()==0) break;

        memcpy(x, ones, graph_size*sizeof(float));

        for(int i=0; i<n_alpha; i++) {

            fix_zeroes(x, graph_size, zero);

            tmp_csize=iterate_clique(x, max_unsolved, zero, alpha[i], BASE_WEIGHT, 0, abortcheck_cb);

            if(tmp_csize<0) goto skip;
        }

        csize = clique_size(x, .0, BASE_WEIGHT, aux_x);

        if(n_masks==0) first_csize=csize;

        if(csize>max_csize) {
            memcpy(max_x, x, sizeof(float)*graph_size);

            max_csize=csize;
        }

        n_res=add_one_to_best(all_res, all_csize, n_res, max_res, &max_sim, x, csize, graph_size);

        printf("iteration: %d clique size: %f max clique size: %f\n", n_masks+1, csize, max_csize);


#ifdef VERBOSE
        P_INT(n_masks)  P_FLOAT(csize)  P_FLOAT(max_csize) P_NL;
#endif

        if(n_masks==max_masks || csize/first_csize<0.6) break;

        vec_to_mask(masks[n_masks++], x, graph_size);

        float xAx = 1 - 1/csize;

        for(int i=0; i<graph_size; i++) {
            if(! BIT_TEST(masks[n_masks-1], i)) aux_x[i]/=xAx;
            else aux_x[i]=0;
        }

        int float_comp(const float *a, const float *b) {
            if(*a < *b)
                return 1;
            else if (*a > *b)
                return -1;
            else
                return 0;
        }

        qsort(aux_x, graph_size, sizeof(int), (comparison_fn_t) float_comp);

        int pos=(graph_size - mask_size(masks[n_masks-1], graph_size)) / 2;

        while(aux_x[pos]<=0 && pos>=1) pos--;

        float val=ceilf(logf(MIN(aux_x[pos], BASE_WEIGHT))/logf(BASE_WEIGHT));

        apply_mask_clique(masks[n_masks-1], (int) val); 
    }

    clear_clique();

    memcpy(res_x, max_x, sizeof(float)*graph_size);

    free(ones);
    free(tmp_x);
    free(aux_x);
    free(max_x);

    if(all_csize) free(all_csize);

    free(x);

    for(int i=0; i<max_masks; i++) mask_free(masks[i]);

    free(masks);
    free(alpha);

    return max_csize;

skip:
    clear_clique();
    printf("!!!! iterate_clique has failed. Retrying.\n");
    reset();

    char fname[128];
    sprintf(fname, "graph%d", ++failed);
    FILE *f=fopen(fname, "w");

    for(int i=0; i<graph_size; i++) {
        for(int j=0; j<graph_size; j++) {
            fprintf(f, "%c ", graph[i][j]==0 ? '0' : '1');
        }
        fprintf(f, "\n");
    }
    fclose(f);

    goto retry;
}

static float find_clique_simplex_int(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int graph_size, t_bitmask allowed, int max_unsolved, 
        float zero, float alpha, int max_masks, int mode, int cuda, int max_res, t_bitmask *par_allres, int *par_n_res, int *abortcheck_cb(void))
{
    t_bitmask lower=mask_alloc(graph_size);
    t_bitmask upper=mask_alloc(graph_size);

    float *x0=calloc(graph_size, sizeof(float));

    if(allowed) {
        for(int i=0; i<graph_size; i++) if(BIT_TEST(allowed, i)) x0[i]=1.;
    } else {
        for(int i=0; i<graph_size; i++) x0[i]=1.;
    }

    float csize;

    float **x_all=0;

    if(par_allres) {
        if(mode!=MODE_REPL_ATTEN && mode!=MODE_REPL_ATTEN_AUTO) {
            printf("Invalid mode for computation of multiple cliques: %d\n", mode);
            abort();
        }
        x_all = (float **) make_array(max_res, graph_size, sizeof(float));
    }


    if(mode==MODE_REPL_UNBIASED || mode==MODE_REPL) {
        csize=find_clique_motzkin(graph, graph_size, x0, max_unsolved, zero, alpha, mode, cuda);
    } else if(mode==MODE_REPL_ANNEAL) {
        csize=find_clique_anneal(graph, graph_size, x0, max_unsolved, zero, alpha, cuda, 0 /* auto_alpha */);
    } else if(mode==MODE_REPL_ANNEAL_AUTO) {
        csize=find_clique_anneal(graph, graph_size, x0, max_unsolved, zero, alpha, cuda, 1 /* auto_alpha */);
    } else if(mode==MODE_REPL_ATTEN) {
        csize=find_clique_atten(graph, graph_size, x0, max_unsolved, zero, alpha, max_masks, cuda, x_all, max_res, 0 /* auto_alpha */, abortcheck_cb);
    } else if(mode==MODE_REPL_ATTEN_AUTO) {
        csize=find_clique_atten(graph, graph_size, x0, max_unsolved, zero, alpha, max_masks, cuda, x_all, max_res, 1 /* auto_alpha */, abortcheck_cb);
    } else {
        printf("Invalid mode: %d\n", mode);
        abort();
    }

    if(x_all) {
        int n_res=0;
        for(int i=0; i<max_res; i++) {
            par_allres[i]=mask_alloc(graph_size);

            if(build_clique(par_allres[n_res], upper, x_all[i], graph, graph_size, 0)) n_res++;
        }
        destroy_array((void **)x_all);

        for(int i=max_res-1; i>=n_res; i--) {
            mask_free(par_allres[i]);
        }

        *par_n_res=n_res;
    }

    build_clique(lower, upper, x0, graph, graph_size, 0 /*max_unsolved*/);

    free(x0);

	*par_res=lower;
    if(par_res_upper) *par_res_upper=upper;
    else mask_free(upper);

	return csize;
}

float graph_clique_cpu(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode)
{
#ifdef DEBUG
	printf("find_clique_simplex\n");
#endif
	return find_clique_simplex_int(par_res, par_res_upper, graph, n, allowed, max_unsolved, zero, alpha, max_masks, mode, 0, 0, 0, 0, 0);
}

#ifndef NO_CUDA
float graph_clique_cuda(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode)
{
#ifdef DEBUG
	printf("find_clique_simplex_cuda max_unsolved: %d\n", max_unsolved);
#endif

	return find_clique_simplex_int(par_res, par_res_upper, graph, n, allowed, max_unsolved, zero, alpha, max_masks, mode, 1, 0, 0, 0, 0);
}
#endif

float graph_clique(t_bitmask *res, t_bitmask *res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode)
{
	float csize=0;

#ifndef NO_CUDA
	csize=graph_clique_cuda(res, res_upper, graph, n, allowed, max_unsolved, zero, alpha, max_masks, mode);
#else
	csize=graph_clique_cpu(res, res_upper, graph, n, allowed, max_unsolved, zero, max_masks, alpha, mode);
#endif

	return csize;
}



float graph_clique_multi_cpu(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode, int max_res, t_bitmask *res_all, int *n_res, int *abortcheck_cb(void))
{
#ifdef DEBUG
	printf("find_clique_simplex\n");
#endif
	return find_clique_simplex_int(par_res, par_res_upper, graph, n, allowed, max_unsolved, zero, alpha, max_masks, mode, 0, max_res, res_all, n_res, abortcheck_cb);
}

#ifndef NO_CUDA
float graph_clique_multi_cuda(t_bitmask *par_res, t_bitmask *par_res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode, int max_res, t_bitmask *res_all, int *n_res, int *abortcheck_cb(void))
{
#ifdef DEBUG
	printf("find_clique_simplex_cuda max_unsolved: %d\n", max_unsolved);
#endif

    init_cuda();

	float res=find_clique_simplex_int(par_res, par_res_upper, graph, n, allowed, max_unsolved, zero, alpha, max_masks, mode, 1, max_res, res_all, n_res, abortcheck_cb);

    return res;
}
#endif

float graph_clique_multi(t_bitmask *res, t_bitmask *res_upper, char **graph, int n, t_bitmask allowed, int max_unsolved, float zero, float alpha, int max_masks, int mode, int max_res, t_bitmask *res_all, int *n_res, int *abortcheck_cb(void))
{
	float csize=0;

#ifndef NO_CUDA

    if(n<1024) {
        csize=graph_clique_multi_cpu(res, res_upper, graph, n, allowed, max_unsolved, zero, alpha, max_masks, mode, max_res, res_all, n_res, abortcheck_cb);
    } else {
        csize=graph_clique_multi_cuda(res, res_upper, graph, n, allowed, max_unsolved, zero, alpha, max_masks, mode, max_res, res_all, n_res, abortcheck_cb);
    }

#else
    csize=graph_clique_multi_cpu(res, res_upper, graph, n, allowed, max_unsolved, zero, max_masks, alpha, mode, max_res, res_all, n_res, abortcheck_cb);
#endif

    return csize;
}
