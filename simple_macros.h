/*******************************************************************************
 * simple_macros.h
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

#ifndef __SIMPLE_MACROS_H__
#define __SIMPLE_MACROS_H__

#define MIN(X, Y) ({ typeof (X) __x = (X), __y = (Y); (__x < __y) ? __x : __y; })
#define MINI(X, Y) ({ int __x = (X), __y = (Y); (__x < __y) ? __x : __y; })
#define MIN3(X, Y, Z) ({ typeof (X) __x = (X), __y = (Y), __z = (Z); typeof (X) __i1 = ((__x < __y ) ? __x : __y);  (__i1 < __z) ? __i1 : __z; })
#define MAX(X, Y) ({ typeof (X) __x = (X), __y = (Y); (__x > __y) ? __x : __y; })
#define MAX3(X, Y, Z) ({ typeof (X) __x = (X), __y = (Y), __z = (Z); typeof (X) __i1 = ((__x > __y ) ? __x : __y);  (__i1 > __z) ? __i1 : __z; })
#define MAX4(X, Y, Z, V) ({ typeof (X) __x = (X), __y = (Y), __z = (Z), __v = (V); typeof (X) __i1 = ((__x > __y ) ? __x : __y); typeof (X) __i2 = ((__z > __v) ? __z : __v); (__i1 > __i2) ? __i1 : __i2; })

#define XOR(X, Y) ({ typeof (X) __x = (X), __y = (Y); (__x || __y) && ! (__x && __y); })

#define SWAP(X, Y) ({ typeof (X) __tmp; __tmp=(Y); 	(Y)=(X); (X)=__tmp; })
#define OVERLAP(start1, end1, start2, end2) ((start2<=start1 && start1<=end2) || (start1<=start2 && start2<=end1)) 

//#define INT2PTR(x) ({sizeof(void *)==8 ? (void *)((long int)(x)) : (void *)(x);})
//#define PTR2INT(x) ({sizeof(void *)==8 ? (int)((long int)(x)) : (int)(x);})

#define INT2PTR(x) ((void *)((size_t)(x)))
#define PTR2INT(x) ((int)((size_t)(x)))

#define P_MESSAGE(x) {printf("%s: %s", __func__, x);}

#define P_INT(x) {printf(#x ": %d ", x);}
#define P_LONG_INT(x) {printf(#x ": %lld ", x);}
#define P_POINT(x) {printf(#x ": %p ", x);}
#define __P_INT_ARR(x, f, t, arr) printf(#x "[%d .. %d]: ", f, t); for(int __i=f; __i<=t; printf("%d ", arr[__i++]))
#define P_INT_ARR(x,n) {typeof (*x) *__x=(x); __P_INT_ARR(x, 0, ((n)-1), __x);}
#define P_INT_ARR_L(x) {typeof (*x) *__x=(x); __P_INT_ARR(x, 1, __x[0], __x);}
#define P_SHORT_ARR(x,n) {short *__x=(x); __P_INT_ARR(x, 0, ((n)-1), __x);}
#define P_SHORT_ARR_L(x) {short *__x=(x); __P_INT_ARR(x, 0, __x[0], __x);}
#define P_CHAR_ARR(x,n) {char *__x=(x); __P_INT_ARR(x, 0, ((n)-1), __x);}
#define P_CHAR_ARR_L(x) {char *__x=(x); __P_INT_ARR(x, 0, __x[0], __x);}
//#define P_INT_ARR(x,n) {int *__x=x; printf(#x "[0 .. %d]: ", n); for(int i=0; i<n; printf("%d ", __x[i++]));}
//#define P_INT_ARR_L(x) {int *__x=x; printf(#x "[1 .. %d]: ", __x[0]); for(int i=1; i<=__x[0]; printf("%d ", __x[i++]));}
#define __P_FLOAT_ARR(x, f, t, arr) printf(#x "[%d .. %d]: ", f, t); for(int __i=f; __i<=t; printf("%f ", arr[__i++]))
#define __P_FLOATE_ARR(x, f, t, arr) printf(#x "[%d .. %d]: ", f, t); for(int __i=f; __i<=t; printf("%g ", arr[__i++]))
#define P_FLOAT_ARR(x,n) {float *__x=(x); __P_FLOAT_ARR(x, 0, ((n)-1), __x);}
#define P_FLOATE_ARR(x,n) {float *__x=(x); __P_FLOATE_ARR(x, 0, ((n)-1), __x);}
#define P_FLOAT(x) {printf(#x ": %f ", (x));}
#define P_FLOATE(x) {printf(#x ": %g ", x);}
#define P_S(x) {printf(#x ": %s ", x);}
#define P_VOID(x) {printf(#x); x;}
#define P_T(x) {printf(x);}
#define P_NL {printf("\n");}
#define P_RET {printf("\r");}
#define P_TAB {printf("\t");}
#define P_INT_MASK_INT(x, n) {printf(#x ": "); typeof(*x) *__x=x; for(int i=0; i<n; i++) if((__x[i])) {printf("%d ", i);}}
#define P_MASK_BIT(x, n) {printf(#x ": "); t_bitmask __x=x; for(int i=0; i<n; i++) printf(BIT_TEST(__x, i) ? "1" : "0");}
#define P_MASK_INT(x, n) {printf(#x ": "); t_bitmask __x=x; for(int i=0; i<n; i++) if(BIT_TEST(__x, i)) {printf("%d ", i);}}
#define P_MASK_RANGES(x, n) { \
	printf(#x ": "); \
	t_bitmask __x=x; \
	int start=-1; \
	if(BIT_TEST(__x, 0)) start=0; \
\
	for(int i=1; i<n; i++) { \
		if(BIT_TEST(__x, i) && !BIT_TEST(__x, i-1)) start=i; \
		else if(!BIT_TEST(__x, i) && BIT_TEST(__x, i-1)) { \
			if(i-1==start) printf("%d ", i-1); \
			else printf("%d-%d ", start, i-1); \
			start=-1; \
		} \
	} \
	if(start>=0) { \
		if(start==n) printf("%d ", n); \
		else printf("%d-%d ", start, n); \
	} \
}

#ifdef SOLARIS
#define NAN ({float a=0.0; a/a;})
typedef int (*comparison_fn_t) (__const void *, __const void *);
#endif

typedef int (*comparison_fn_t) (__const void *, __const void *);
#endif
