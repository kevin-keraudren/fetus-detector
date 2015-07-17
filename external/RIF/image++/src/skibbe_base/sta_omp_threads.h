/*#############################################################################
*
*	Copyright 2011 by Henrik Skibbe and Marco Reisert
* 
*     
*	This file is part of the STA-ImageAnalysisToolbox
* 
*	STA-ImageAnalysisToolbox is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
* 
*	STA-ImageAnalysisToolbox is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
* 
*	You should have received a copy of the GNU General Public License
*	along with STA-ImageAnalysisToolbox. 
*	If not, see <http://www.gnu.org/licenses/>.
*
*
*#############################################################################*/

#ifndef STA_OMP_THREADS_H
#define STA_OMP_THREADS_H

#include <cstddef>
#include <complex>
#include <cmath>
#ifdef _STA_MULTI_THREAD   
#include <omp.h>
#endif
#include <sstream>
#include <cstddef>
#include <vector>
#ifndef WIN32
#include <unistd.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string>

#ifndef _STA_DEFAULT_NUM_CPUS
#define _STA_DEFAULT_NUM_CPUS 1
#endif


namespace hanalysis
{

	template<typename T>
	std::string toString(T value)
	{
		std::stringstream s;
		s<<value;
		return s.str();
	}

	template<typename T>
	T toType ( std::string s )
	{
		std::stringstream tmp;
		tmp<<s;
		T result;
		tmp>>result;
		return result;
	}

	inline int get_numCPUs_system()
	{
#ifdef _STA_MULTI_THREAD    
		char * sysenv=getenv ( "STA_NUMTHREADS" );
		if ( sysenv!=NULL )
		{
			std::string system_settings=std::string ( sysenv );
			//printf("%s %d %d\n",system_settings.c_str(),omp_get_num_procs(),omp_get_max_threads());
			if ( system_settings.length() >0 )
			{
				if ( system_settings.compare ( "ALL" ) != 0 )
					return toType<int> ( system_settings );
				else
					return omp_get_num_procs();
			}
		}

		return _STA_DEFAULT_NUM_CPUS;
#else
		return 1;
#endif
	}

	inline int get_numCPUs()
	{
		int numcpus=get_numCPUs_system();
		//printf("numthreads %d\n",numcpus);
		return numcpus;
	}

}

#endif
