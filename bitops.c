/*******************************************************************************
 * bitops.c
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

#include<stdlib.h>
#include<stdio.h>
#include<string.h>
#include"bitops.h"
#include"random.h"

/*
char bit_count[256];

void fill_bit_count(void)
{
	int k;
	unsigned char i, j;

	i=0;
	for(k=0; k<256; k++) {
		
		bit_count[i]=0;
		for(j=0; j<8; j++)
			bit_count[i]+=(i>>j) & 1;
		i++;
	}
}
*/

int bits_32(unsigned int v) 
{
	int c;

//	printf("32:\n");
	v = v - ((v >> 1) & 0x55555555);                    // reuse input as temporary
//		printf("%llx\n", v);
	v = (v & 0x33333333) + ((v >> 2) & 0x33333333);     // temp
//		printf("%llx\n", v);
	v =((v + (v >> 4)) & 0x0F0F0F0F); // count
//		printf("%llx\n", v);
	c = (int) (((v * 0x1010101) >> 24)); // count

	return c;
}

int bits_64(unsigned long long int v) 
{
	int c;

//	printf("64:\n");
	v = v - ((v >> 1) & 0x5555555555555555ULL);                    // reuse input as temporary
//		printf("%llx\n", v);
	v = (v & 0x3333333333333333ULL) + ((v >> 2) & 0x3333333333333333ULL);     // temp
//		printf("%llx\n", v);
	v = ((v + (v >> 4)) & 0x0F0F0F0F0F0F0F0FULL); // count
//		printf("%llx\n", v);
	c = (int) (((v * 0x0101010101010101ULL) >> 56)); // count

	return c;
}
/*
int bits_old(t_maskcell m)
{
	unsigned char *c;

	c=(unsigned char *)&m;

	int res=0;

	for(int i=0;i<sizeof(m); i++) res+=bit_count[c[i]];

	return(res);

	return(bit_count[c[0]]+bit_count[c[1]]+bit_count[c[2]]+bit_count[c[3]]);
}
*/
int bits(t_maskcell m)
{
#ifdef __x86_64__
	return bits_64(m);
#else
	return bits_32(m);
#endif
}

t_bitmask mask_alloc(int len)
{
	t_bitmask res=calloc(MASK_LENGTH(len), sizeof(t_maskcell));

//	mask_zeroall(res, len);

	return res;
}

void mask_free(t_bitmask mask)
{
	free(mask);
}

void mask_zeroall(t_bitmask mask, int len)
{
	memset(mask, 0, MASK_LENGTH(len)*sizeof(t_maskcell));
}

void mask_setall(t_bitmask mask, int len)
{
	memset(mask, 255, (MASK_LENGTH(len)-1)*sizeof(t_maskcell));
	for(int i=MASK_CELL_SIZE*(MASK_LENGTH(len)-1); i<len; i++) {
		SET_BIT(mask, i);
	}
}

int mask_size(t_bitmask mask, int len)
{
	int res=0;
	
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) res+=bits(mask[i]);

	return res;
}

int mask_empty(t_bitmask mask, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) if(mask[i]) return 0;

	return 1;
}

void mask_not(t_bitmask mask, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) mask[i]=~mask[i];
}

void mask_or(t_bitmask mask, t_bitmask mask1, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) mask[i]|=mask1[i];
}

void mask_and(t_bitmask mask, t_bitmask mask1, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) mask[i]&=mask1[i];
}

int mask_and_empty(t_bitmask mask, t_bitmask mask1, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) if(mask[i]&mask1[i]) return 0;

	return 1;
}

void mask_sub(t_bitmask mask, t_bitmask mask1, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) mask[i]&=~mask1[i];
}

int mask_sub_empty(t_bitmask mask, t_bitmask mask1, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) if(mask[i]&~mask1[i]) return 0;

	return 1;
}

void mask_cpy(t_bitmask mask, t_bitmask mask1, int len)
{
	memcpy(mask, mask1, MASK_LENGTH(len)*sizeof(t_maskcell));
}

int mask_eq(t_bitmask mask1, t_bitmask mask2, int len)
{
	for(int i=0; i<=len/MASK_CELL_SIZE; i++) {
		if(mask1[i] != mask2[i]) return 0;
	}

	return 1;
}

int mask_comp(t_bitmask mask1, t_bitmask mask2, int len)
{
	int res=MASK_SUBSET | MASK_SUPERSET | MASK_EQUAL;

	for(int i=0; i<=len/MASK_CELL_SIZE; i++) {
		if(mask1[i] & ~mask2[i]) res=res&~(MASK_EQUAL | MASK_SUBSET);
		if(~mask1[i] & mask2[i]) res=res&~(MASK_EQUAL | MASK_SUPERSET);
	}


	return res;
}

int mask_to_list(int *l, int n, t_bitmask mask, int len)
{


	int p=0;

	for(int i=0; i<len; i++) {
		if(BIT_TEST(mask, i)) {
			if(p>=n) return -1;
			l[p++]=i;
		}
	}

	return p;
}

int list_to_mask(t_bitmask mask, int len, int *l, int n)
{
	mask_zeroall(mask, len);

	int p=0;

	for(int i=0; i<n; i++) {
		if(l[i]>=0) {
			SET_BIT(mask, l[i]);
			p++;
		}
	}

	return p;
}


t_bitmask random_mask(int len)
{
	t_bitmask res=mask_alloc(len);

	for(int i=0; i<len; i++) if(int_ran(0,1)) SET_BIT(res, i);

	return res;
}
