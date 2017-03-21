/*******************************************************************************
 * find_cliques.c
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
#include<time.h>
#include<unistd.h>

#include<string.h>

#include <argp.h>

#include"sys/time.h"

#include"simple_macros.h"
#include"arrays.h"
#include"parsers.h"

#include"cudams.h"

const char *argp_program_version = "find_cliques";
const char *argp_program_bug_address = "<pawel@bioexploratorium.pl>";

static char doc[] = "A program searching for the largest clique in a graph using a iterative approach based on the Motzkin-Straus theorem.";

static char args_doc[] = "[--DIMACSbin|--DIMACSascii] GRAPH\n--anneal [--DIMACSbin|--DIMACSascii] GRAPH\n--atten[=NUM] [--max-res=NUM] [--DIMACSbin|--DIMACSascii] GRAPH";

static struct argp_option options[] = {
    {"atten"           , 't', "NUM"     , OPTION_ARG_OPTIONAL, "Enable attenuation scheme. Optionally number of iterations can be given (default: 10)."},
    {"anneal"          , 'n', 0         , 0, "Enable annealing scheme." },
    {"alpha"           , 'a', "NUM|auto", 0, "Value of paramerer alpha (default: auto). When 'auto' is given in annealing and attenuation modes a size of maximal clique is estimated based on a graph size and density, for single iteration 0.5 is taken."},
    {"alpha-step"      , 's', "NUM"     , 0, "Minimal difference between consecutive values in automatic determination of alpha values (default: 0)."},
    {"cpu"             , 'c', 0         , 0, "Use CPU implementation." },
    {"gpu"             , 'g', 0         , 0, "Use GPU implementation." },
    {"max-res"         , 'm', "NUM"     , 0, "Maximal number of nodes of cliques to be returned. Applicable only in attenuation mode." },
    {"max-unsolved"    , 'u', "NUM"     , 0, "Maximal number of nodes for which iteration doesn't have to converge. Maximal cliques for these will be computed by exhaustive search." },
    {"zero"            , 'z', "NUM"     , 0, "Threshold below which values of elements will be squashed to 0." },
    {"DIMACSbin"       ,  1 , 0         , 0, "Load graph in DIMACS binary format." },
    {"DIMACSascii"     ,  2 , 0         , 0, "Load graph in DIMACS ASCII format." },
    { 0 }
};

struct arguments
{
    char *graph_file;
    int mode;
    int max_res;
    float alpha;
    float alpha_step;
    int alpha_auto;
    int max_masks;
    int gpu;
    int cpu;
    int max_unsolved;
    float zero;
    int file_format;
};

static error_t parse_opt (int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;

    switch (key)
    {
        case 1: // DIMACSbin
            if(arguments->file_format>0)
                argp_usage (state);
            arguments->file_format=1;
            break;

        case 2: // DIMACSascii
            if(arguments->file_format>0)
                argp_usage (state);
            arguments->file_format=2;
            break;

        case 't':  // atten
            if(arguments->mode>1)
                argp_usage (state);
            arguments->mode=3;
            arguments->max_masks = arg ? atoi(arg) : 10;
            break;

        case 'n':  // anneal
            if(arguments->mode>1)
                argp_usage (state);
            arguments->mode=2;
            break;

        case 'a':  // alpha
            if(strcmp(arg, "auto")) {
                arguments->alpha = atof(arg);
                arguments->alpha_auto=0;
            }
            break;

        case 'c':  // cpu
            arguments->cpu = 1;
            break;

        case 'g':  // gpu
            arguments->gpu = 1;
            break;

        case 'm':  // max-res
            arguments->max_res = atoi(arg);
            break;

        case 'u':  // max-unsolved
            arguments->max_unsolved = atoi(arg);
            break;

        case 'z':  // zero
            arguments->zero = atof(arg);
            break;

        case 's':  // alpha-step
            arguments->alpha_step = atof(arg);
            if(arguments->alpha_step<0)
                argp_usage(state);
            break;

        case ARGP_KEY_ARG:
            if(!arguments->mode)
                arguments->mode=1;
            if((arguments->max_res>1 && arguments->mode!=3) || state->arg_num >= 1)
                argp_usage (state);
            if(!arguments->alpha_auto && arguments->alpha_step!=0)
                argp_usage (state);
            arguments->graph_file = arg;
            break;

        case ARGP_KEY_END:
            if(state->arg_num < 1)
                argp_usage (state);
            break;

        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = { options, parse_opt, args_doc, doc };

long int get_timestamp() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return tv.tv_sec*(long int)1000000+tv.tv_usec;
}

void init_cuda(void);

int main(int argc, char **argv)
{
   struct arguments arguments;

    /* Default values. */
    arguments.mode = 0;
    arguments.max_res = 1;
    arguments.max_masks = 5;
    arguments.alpha=0.5;
    arguments.alpha_auto=1;
    arguments.alpha_step=0;
    arguments.gpu = 0;
    arguments.cpu = 0;
    arguments.max_unsolved = 5;
    arguments.zero = 0.001;
    arguments.file_format = 0;

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    if(!arguments.gpu && !arguments.cpu) arguments.gpu=1;

    int mode;

    if(arguments.mode==1) {
        if(arguments.alpha==0) mode=MODE_REPL_UNBIASED;
        else mode=MODE_REPL;
    } else if(arguments.mode==2) {
        if(arguments.alpha_auto) mode=MODE_REPL_ANNEAL_AUTO;
        else mode=MODE_REPL_ANNEAL;
    } else if(arguments.mode==3) {
        if(arguments.alpha_auto) mode=MODE_REPL_ATTEN_AUTO;
        else mode=MODE_REPL_ATTEN;
    }


	int N, n_edges;
	char **arr;

    switch(arguments.file_format) {
        case 1:
            N=read_graph_DIMACS_bin(arguments.graph_file, &arr, &n_edges);
            break;

        case 2:
            N=read_graph_DIMACS_ascii(arguments.graph_file, &arr, &n_edges);
            break;

        default:
            N=read_graph_adjmat(arguments.graph_file, &arr, &n_edges);
            break;
    }


    printf("Graph file: %s nodes: %d edges: %d\n", arguments.graph_file, N, n_edges);

    if(arguments.mode==1) {
        printf("Mode: single iteration\n");
        printf("Alpha: %f\n", arguments.alpha);
        if(arguments.alpha==0) mode=MODE_REPL_UNBIASED;
        else mode=MODE_REPL;
    } else if(arguments.mode==2) {
        printf("Mode: annealing\n");
        if(arguments.alpha_auto) {
            mode=MODE_REPL_ANNEAL_AUTO;
        } else {
            mode=MODE_REPL_ANNEAL;
        }
    } else if(arguments.mode==3) {
        printf("Mode: attenuation max iterations: %d\n", arguments.max_masks);
        arguments.max_masks--;
        if(arguments.alpha_auto) {
            mode=MODE_REPL_ATTEN_AUTO;
        } else {
            mode=MODE_REPL_ATTEN;
        }
    }

    if(arguments.mode>1) {
        if(arguments.alpha_auto) {
            printf("Starting alpha: auto min-step: %f\n", arguments.alpha_step);
            arguments.alpha=arguments.alpha_step;
        } else {
            printf("Starting alpha: %f\n", arguments.alpha);
        }
    }

    quiet=0;

    void run(int cuda)
    {
        t_bitmask res;
        t_bitmask res_upper;
        t_bitmask res_all[arguments.max_res];

        int n_res;

        printf("%s: \n", cuda ? "GPU" : "CPU");

        if(cuda) init_cuda();

        long int start = get_timestamp();

        if(arguments.max_res>1) {
            if(cuda) {
                graph_clique_multi_cuda(&res, &res_upper, arr, N, 0, arguments.max_unsolved, arguments.zero, arguments.alpha, arguments.max_masks, mode, arguments.max_res, res_all, &n_res, 0);
            } else {
                graph_clique_multi_cpu(&res, &res_upper, arr, N, 0, arguments.max_unsolved, arguments.zero, arguments.alpha, arguments.max_masks, mode, arguments.max_res, res_all, &n_res, 0);
            }
        } else {
            if(cuda) {
                graph_clique_cuda(&res, &res_upper, arr, N, 0, arguments.max_unsolved, arguments.zero, arguments.alpha, arguments.max_masks, mode);
            } else {
                graph_clique_cpu(&res, &res_upper, arr, N, 0, arguments.max_unsolved, arguments.zero, arguments.alpha, arguments.max_masks, mode);
            }
        }

        long int end = get_timestamp();

        void print_res(t_bitmask res) {
            for(int i=0; i<N; i++) if(BIT_TEST(res, i)) printf("%d ", i);
        }

        if(arguments.max_res>1) for(int i=0; i<n_res; i++) {
            int order[n_res];

            for(int i=0; i<n_res; i++) order[i]=i;


            int size_comp(int *a, int *b) {
                return mask_size(res_all[*b], N) - mask_size(res_all[*a], N);
            }

            qsort(order, n_res, sizeof(int), (comparison_fn_t) size_comp);

            printf("clique %d size: %d nodes:\n", i+1, mask_size(res_all[order[i]], N));
            print_res(res_all[order[i]]);
            printf("\n");
        } else {
            printf("clique size: %d nodes:\n", mask_size(res, N));
            print_res(res);
            printf("\n");
        }

        printf("%s time (us): %ld\n", cuda ? "GPU" : "CPU", end-start);
        printf("%s time (s): %.3f\n", cuda ? "GPU" : "CPU", (end-start)*0.000001);
    }

    if(arguments.cpu) run(0);
    if(arguments.gpu) run(1);

}


