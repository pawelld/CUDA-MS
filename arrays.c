/*******************************************************************************
 * arrays.c
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
#include<unistd.h>

#ifdef  __APPLE__
#include<mach/mach.h>
#else
#include<sys/sysinfo.h>
#endif


#include"simple_macros.h"
#include"arrays.h"

#ifdef  __APPLE__
void do_host_statistics(host_name_port_t host, host_flavor_t flavor, host_info_t info, mach_msg_type_number_t *count)
{
    kern_return_t kr;

    kr = host_statistics(host, flavor, (host_info_t)info, count);
    if (kr != KERN_SUCCESS) {
        (void)mach_port_deallocate(mach_task_self(), host);
        mach_error("host_info:", kr);
        abort();
    }
}
#endif

int memory_available(size_t size, float frac)
{
	int page_size;
	long int avpages;

#ifdef  __APPLE__
    vm_size_t tmp_page_size;
    vm_statistics_data_t vm_stat;
    mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
    host_name_port_t host = mach_host_self();

    host_page_size(host, &tmp_page_size);
    do_host_statistics(host, HOST_VM_INFO, (host_info_t)&vm_stat, &count);

	page_size=tmp_page_size;

	avpages=vm_stat.free_count+vm_stat.inactive_count;

//	P_INT(page_size); P_INT((int)avpages); P_INT(page_size*avpages); P_NL;
#else
	page_size=getpagesize();
	avpages=get_avphys_pages();
#endif

	long int req_pages=size/page_size;

	if((float)req_pages/(float)avpages<=frac) return 1;
	
	return 0;
}

size_t array_size(long int n, long int m, int size)
{
	size_t res=sizeof(void *)*n+size*n*m;

	return res;
}

void **make_array(long int n, long int m, int size)
{
	void **rres=malloc(sizeof(void*)*n);

	if(!rres || !n) {
		printf("alloc error: %ld\n", (long int)sizeof(void*)*n);
//		abort();
//		print_trace();
		return 0;
	}

 
	rres[0]=malloc(size*n*m);

	if(!rres[0]) {
		printf("alloc error: %ld\n", (long int)size*n*m);
//		print_trace();
		free(rres);
//		abort();
		return 0;
	}

	memset(rres[0], 0, size*n*m);

	for(int i=1; i<n; i++) rres[i]=rres[i-1]+m*size;

	return rres;
}

void copy_array(void **dst, void ** src, long int n, long int m, int size)
{
	memcpy(dst[0], src[0], size*n*m);
}

void destroy_array(void **array)
{
	free(array[0]);
	free(array);
}


