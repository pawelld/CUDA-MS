/*******************************************************************************
 * parsers.c
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "simple_macros.h"
#include "parsers.h"
#include "arrays.h"

static int get_params(char *pp)
{
    char c;
    int stop = 0;

    int N=0;
    int n_edges = 0;

    while (!stop && (c = *pp++) != '\0'){
        switch (c)
        {
            case 'c':
                while ((c = *pp++) != '\n' && c != '\0');
                break;

            case 'p':
                sscanf(pp, "%*s %d %d\n", &N, &n_edges);
                stop = 1;
                break;

            default:
                break;
        }
    }


    if (N == 0 || n_edges == 0)
        return 0;  /* error */
    else
        printf ("vertices: %d, edges %d\n", N, n_edges);
    return N;
}


int read_graph_DIMACS_bin(char *file, char ***arr,  int *edges)
{

    int length = 0;
    FILE *fp;

    if ( (fp=fopen(file,"r"))==NULL ) {
        printf("ERROR: Cannot open infile\n");
        exit(10);
    }

    if (!fscanf(fp, "%d\n", &length)) {
        printf("ERROR: Corrupted preamble.\n");
        exit(10);
    }

    char *preamble=malloc(length+1);

    fread(preamble, 1, length, fp);
    preamble[length] = '\0';

    int N=get_params(preamble);
    free(preamble);

    if(!N) {
        printf("ERROR: Corrupted preamble.\n");
        exit(10);
    }

	*arr=(char **)make_array(N, N, sizeof(char));

    int n_edges=0;
    char *bitmask=malloc(N/8+1);

    char dimmasks[ 8 ] = { 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };

    for(int i = 0; i < N; i++ ) {
        if(! fread(bitmask, 1, (int)((i + 8)/8), fp)) {
            printf("ERROR: Corrupted file.\n");
            exit(10);
        }
        for(int j=0; j<i; j++) {
            int bit  = j & 0x00000007;
            int byte = j >> 3;

            if(bitmask[byte] & dimmasks[bit]) {
                (*arr)[i][j]=1;
                (*arr)[j][i]=1;
                n_edges++;
            }
        }
    }

    free(bitmask);

    fclose(fp);
    if(edges) *edges=n_edges;
    return N;
}

int read_graph_DIMACS_ascii(char *file, char ***arr,  int *edges)
{

    FILE *fp;


    if ( (fp=fopen(file,"r"))==NULL ) {
        printf("ERROR: Cannot open infile\n");
        exit(10);
    }

    char c;
    int N=-1;
    int n_edges=-1;
    int stop = 0;

    while (!stop && (c = fgetc(fp)) != EOF ){
        switch (c)
        {
            case 'c':
                while ((c = fgetc(fp)) != '\n' && c != '\0');
                break;

            case 'p':
                fscanf(fp, "%*s %d %d\n", &N, &n_edges);
                break;

            case 'e':
                ungetc(c, fp);
                stop=1;
                break;

            default:
                printf("ERROR: corrupted infile\n");
                exit(10);
                break;
        }
    }


    if (N<0 || n_edges<0) {
        printf("ERROR: corrupted infile\n");
        exit(10);
    }

	*arr=(char **)make_array(N, N, sizeof(char));

    while ((c = fgetc(fp)) != EOF){
        int i, j;
        switch (c) {
            case 'e':
                if (!fscanf(fp, "%d %d", &i, &j)) {
                    printf("ERROR: corrupted inputfile\n");
                    exit(10);
                }

                (*arr)[i-1][j-1]=1;
                (*arr)[j-1][i-1]=1;
                break;

            default:
                break;
        }
    }

    fclose(fp);

    *edges=n_edges;

    return N;
}


int read_graph_adjmat(char *file, char ***arr,  int *edges)
{
    FILE *fp;

    if ( (fp=fopen(file,"r"))==NULL ) {
        printf("ERROR: Cannot open infile\n");
        exit(10);
    }

    #define BUFSIZE 512

    char buf[BUFSIZE];

    int bufpos=BUFSIZE+1;
    int buflen=BUFSIZE;
    int N=0;
    int done=0;
    while(!done) {
        if(bufpos>=buflen) {
            if(buflen<BUFSIZE) {
                printf("ERROR: Corrupted file.\n");
                exit(10);
            }
            if(!(buflen=fread(buf, 1, BUFSIZE, fp))) {
                printf("ERROR: Corrupted file.\n");
                exit(10);
            }
            bufpos=0;
        }

        while(bufpos<buflen) {

            if(buf[bufpos]=='\n') {
                done=1;
                break;
            } else if(buf[bufpos]=='1' || buf[bufpos]=='0') {
                N++;
            }
            bufpos++;
        }
    }

	*arr=(char **)make_array(N, N, sizeof(char));
    int n_edges=0;

    rewind(fp);

    for(int i=0; i<N; i++) for(int j=0; j<N; j++) {
        int qq;

        fscanf(fp, "%d", &qq);

        if(qq!=0 && qq!=1) {
            printf("ERROR: Corrupted file.\n");
            exit(10);
        }

        (*arr)[i][j]=qq==1?1:0;

        if(qq && i>j) n_edges++;
    }

    fclose(fp);
    if(edges) *edges=n_edges;

    return N;
}
