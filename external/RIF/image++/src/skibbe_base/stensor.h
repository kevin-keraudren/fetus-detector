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

#ifndef STA_STENSOR_H
#define STA_STENSOR_H

#include "gsl/gsl_sf_coupling.h"
#include "gsl/gsl_sf_laguerre.h"
#include "gsl/gsl_sf_gamma.h"
#include "gsl/gsl_sf_legendre.h"
#include "gsl/gsl_sf_bessel.h"
#include <cstddef>
#include <complex>
#include <cmath>
#include <sstream>
#include <cstddef>
#include <vector>
#ifndef WIN32
#include <unistd.h>
#else
#define M_PI 3.14159265358979323846
#endif
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <limits>

#include "sta_omp_threads.h"

#ifdef _STA_LINK_FFTW
#include "fftw3.h"
#endif


//#define _STA_SINGLE_THREAD

/*! \mainpage The STA-ImageAnalysisToolbox
* \section intro_sec Introduction
*
* all relevant low-level functions can be found here: \ref hanalysis \n
* we also provide a high-level interface based on the \ref hanalysis::stafield class \n
*/







//!  The STA-ImageAnalysisToolkit namespace
namespace hanalysis
{

	static int verbose=0;



	/*
	computes Clebsch Gordan coupling coefficients
	\f$ \langle ja~ma, jb~mb \hspace{1.5pt}|\hspace{1.5pt} J M \rangle \f$. \n
	\returns  \f$
	\left\{
	\begin{array}{ll}
	\langle ja~ma, jb~mb \hspace{1.5pt}|\hspace{1.5pt} J M \rangle   & \mbox{if }
	M=ma+mb \text{ and } \lvert ja-jb  \rvert \leq J \text{ and } ja+jb \geq J \\
	0 & \mbox{else }
	\end{array}
	\right.
	\f$
	*/
	inline
		double clebschGordan ( const int ja,
		const int ma,
		const int jb,
		const int mb,
		const int J,
		const int M )
	{
		int J2=2*J;
		gsl_sf_result _result;
		gsl_sf_coupling_3j_e ( 2*ja, 2*jb,  J2,
			2*ma, 2*mb, -2*M, &_result );
		double norm = sqrt ( ( double ) ( J2+1.0 ) );
		int phase = ( ja-jb+M );
		double sign = ( phase & 1 ) ? -1.0 : 1.0;
		return _result.val*sign*norm;
	}


	/*
	orthonormal spherical harmonic basis functions
	*/
	inline
		std::complex<double> basis_SphericalHarmonics(int l, int m, double theta,double phi)
	{
		bool sym=false;
		if (m<0) {
			sym=true;
			m*=-1;
		}
		double legendre=gsl_sf_legendre_sphPlm (l, m, std::cos(theta)); //already normalized
		std::complex<double>  tmp;
		tmp=legendre*std::exp(std::complex<double>(0,1)*(double)m*phi);
		if (sym)
		{
			if (m%2==0) return std::conj(tmp);
			else return -std::conj(tmp);
		}
		return tmp;
	}



	/*
	orthogonal spherical harmonic basis functions (semi schmidt)
	*/
	inline
		std::complex<double> basis_SphericalHarmonicsSemiSchmidt(int l, int m, double theta,double phi)
	{

		double norm=std::sqrt(4.0*M_PI/(2.0*l+1.0));;
		return norm*basis_SphericalHarmonics(l,m,theta,phi);
	}













	template<typename T,typename S>
	int sta_laplace_1component (
		const S * stIn,std::complex<T> * stOut ,
		const std::size_t shape[],
		int type=1,
		std::complex<T> alpha= ( T ) 1.0,
		const T  v_size[]=NULL,
		bool clear_field=false)
	{


		if ( v_size!=NULL )
		{
			if (hanalysis::verbose>0)
				printf ( "WARNING! element size is not considered yet!\n" );
		}


		T voxel_size[3];
		voxel_size[0]=voxel_size[1]=voxel_size[2]=1;
		if ( v_size!=NULL )
		{
			voxel_size[0]/=v_size[0]; // Zdir
			voxel_size[1]/=v_size[1]; // Ydir
			voxel_size[2]/=v_size[2]; // Xdir
		}



		std::size_t jumpz=shape[1]*shape[2];
		std::size_t jumpy=shape[2];

		switch ( type )
		{

		case 0:
			{
				alpha*= ( T ) 0.2;
#pragma omp parallel for num_threads(get_numCPUs())
				for ( std::size_t z=0;z<shape[0];z++ )
				{
					std::size_t Z[3];
					Z[1]=z+shape[0];
					Z[0]=Z[1]-1;
					Z[2]=Z[1]+1;
					Z[0]%=shape[0];
					Z[1]%=shape[0];
					Z[2]%=shape[0];

					Z[0]*=jumpz;
					Z[1]*=jumpz;
					Z[2]*=jumpz;

					for ( std::size_t y=0;y<shape[1];y++ )
					{
						std::size_t Y[3];
						Y[1]=y+shape[1];
						Y[0]=Y[1]-1;
						Y[2]=Y[1]+1;
						Y[0]%=shape[1];
						Y[1]%=shape[1];
						Y[2]%=shape[1];

						Y[0]*=jumpy;
						Y[1]*=jumpy;
						Y[2]*=jumpy;

						for ( std::size_t x=0;x<shape[2];x++ )
						{
							std::size_t X[3];
							X[1]=x+shape[2];
							X[0]=X[1]-1;
							X[2]=X[1]+1;
							X[0]%=shape[2];
							X[1]%=shape[2];
							X[2]%=shape[2];

							std::size_t offset= ( Z[1]+Y[1]+X[1] );
							std::complex<T> & current=stOut[offset];
							if ( clear_field ) current=T ( 0 );

							current+=alpha* (
								stIn[ ( Z[0]+Y[0]+X[1] ) ] +
								stIn[ ( Z[0]+Y[1]+X[0] ) ] +
								stIn[ ( Z[0]+Y[1]+X[1] ) ] +
								stIn[ ( Z[0]+Y[1]+X[2] ) ] +
								stIn[ ( Z[0]+Y[2]+X[1] ) ] +
								stIn[ ( Z[1]+Y[0]+X[0] ) ] +
								stIn[ ( Z[1]+Y[0]+X[1] ) ] +
								stIn[ ( Z[1]+Y[0]+X[2] ) ] +
								stIn[ ( Z[1]+Y[1]+X[0] ) ]-
								T ( 18 ) *stIn[ ( Z[1]+Y[1]+X[1] ) ] +
								stIn[ ( Z[1]+Y[1]+X[2] ) ] +
								stIn[ ( Z[1]+Y[2]+X[0] ) ] +
								stIn[ ( Z[1]+Y[2]+X[1] ) ] +
								stIn[ ( Z[1]+Y[2]+X[2] ) ] +
								stIn[ ( Z[2]+Y[0]+X[1] ) ] +
								stIn[ ( Z[2]+Y[1]+X[0] ) ] +
								stIn[ ( Z[2]+Y[1]+X[1] ) ] +
								stIn[ ( Z[2]+Y[1]+X[2] ) ] +
								stIn[ ( Z[2]+Y[2]+X[1] ) ]
							);

						}

					}
				}
			}
			break;
		case 1:
			{
				alpha*=1;
#pragma omp parallel for num_threads(get_numCPUs())
				for ( std::size_t z=0;z<shape[0];z++ )
				{
					std::size_t Z[3];
					Z[1]=z+shape[0];
					Z[0]=Z[1]-1;
					Z[2]=Z[1]+1;
					Z[0]%=shape[0];
					Z[1]%=shape[0];
					Z[2]%=shape[0];

					Z[0]*=jumpz;
					Z[1]*=jumpz;
					Z[2]*=jumpz;

					for ( std::size_t y=0;y<shape[1];y++ )
					{
						std::size_t Y[3];
						Y[1]=y+shape[1];
						Y[0]=Y[1]-1;
						Y[2]=Y[1]+1;
						Y[0]%=shape[1];
						Y[1]%=shape[1];
						Y[2]%=shape[1];

						Y[0]*=jumpy;
						Y[1]*=jumpy;
						Y[2]*=jumpy;

						for ( std::size_t x=0;x<shape[2];x++ )
						{
							std::size_t X[3];
							X[1]=x+shape[2];
							X[0]=X[1]-1;
							X[2]=X[1]+1;
							X[0]%=shape[2];
							X[1]%=shape[2];
							X[2]%=shape[2];


							std::complex<T> & current=stOut[Z[1]+Y[1]+X[1]];

							if ( clear_field ) current=T ( 0 );

							current+=alpha* (
								stIn[ ( Z[0]+Y[1]+X[1] ) ] +
								stIn[ ( Z[1]+Y[0]+X[1] ) ] +
								stIn[ ( Z[1]+Y[1]+X[0] ) ] -
								T ( 6 ) *stIn[ ( Z[1]+Y[1]+X[1] ) ] +
								stIn[ ( Z[1]+Y[1]+X[2] ) ] +
								stIn[ ( Z[1]+Y[2]+X[1] ) ] +
								stIn[ ( Z[2]+Y[1]+X[1] ) ]
							);
						}

					}
				}
			}
			break;

		default:
			printf ( "unsoported operator\n" );
		}
		return 0;
	}
















#ifdef _STA_LINK_FFTW


	void fft ( const std::complex<double> * IN,
		std::complex<double> * OUT, 
		int shape[],int numComponents,
		bool forward,
		int flag=FFTW_ESTIMATE )
	{
#ifdef _STA_FFT_MULTI_THREAD
		fftw_init_threads();
		fftw_plan_with_nthreads ( get_numCPUs() );
		if ( verbose>0 ) printf ( "FFTW with %d threads \n",get_numCPUs() );
#else
		if ( verbose>0 ) printf ( "FFTW is single threaded\n" );
#endif

		int rank=3;
		int * n=shape;
		int howmany=numComponents;
		fftw_complex * in = ( fftw_complex * ) IN;
		int * inembed=NULL;
		int istride=numComponents;
		int idist=1;
		fftw_complex * out = ( fftw_complex * ) OUT;
		int * onembed=inembed;
		int odist=idist;
		int ostride=istride;
		int sign=FFTW_FORWARD;
		if ( !forward ) sign=FFTW_BACKWARD;
		unsigned flags=flag| FFTW_PRESERVE_INPUT;//FFTW_ESTIMATE;//FFTW_MEASURE |  FFTW_PRESERVE_INPUT;//FFTW_ESTIMATE; //FFTW_MEASURE


#ifdef __linux__    
		char buffer[255];
		gethostname ( buffer,255 );
		std::string s;
		s=std::string ( getenv ( "HOME" ) ) +std::string ( "/.mywisdom_" ) +std::string ( buffer ) +".wisdom";

		FILE *ifp;
		ifp = fopen ( s.c_str(),"r" );
		if ( ifp!=NULL )
		{
			if ( 0==fftw_import_wisdom_from_file ( ifp ) )
				printf ( "Error reading wisdom file: %s!\n",s.c_str() );
			fclose ( ifp );
		}
		else printf ( "Wisdom file does not exist!\n" );
#endif

		fftw_plan plan=fftw_plan_many_dft ( rank,n,howmany,in,inembed,istride, idist,out,onembed, ostride, odist, sign, flags | FFTW_PRESERVE_INPUT );
		if ( plan==NULL )
		{
			printf ( "no plan\n" );
		}

		fftw_execute_dft ( plan,in, out );

#ifdef __linux__        
		ifp = fopen ( s.c_str(),"w" );
		if ( ifp!=NULL )
		{
			fftw_export_wisdom_to_file ( ifp );
			fclose ( ifp );
		}
		else  printf ( "Error creating file!\n" );
#endif

		fftw_destroy_plan ( plan );


		//fftw_cleanup() 
		//fftw_cleanup_threads();

	}



	void fft ( const std::complex<float> * IN,
		std::complex<float> * OUT, 
		int shape[],
		int numComponents, 
		bool forward,
		int flag=FFTW_ESTIMATE )
	{
#ifdef _STA_FFT_MULTI_THREAD
		fftwf_init_threads();
		fftwf_plan_with_nthreads ( get_numCPUs() );
		if ( verbose>0 ) printf ( "FFTW with %d threads \n",get_numCPUs() );
#else
		if ( verbose>0 ) printf ( "FFTW is single threaded\n" );
#endif

		int rank=3;
		int * n=shape;
		int howmany=numComponents;
		fftwf_complex * in = ( fftwf_complex * ) IN;
		int * inembed=NULL;
		int istride=numComponents;
		int idist=1;
		fftwf_complex * out = ( fftwf_complex * ) OUT;
		int * onembed=inembed;
		int odist=idist;
		int ostride=istride;
		int sign=FFTW_FORWARD;
		if ( !forward ) sign=FFTW_BACKWARD;
		unsigned flags=flag| FFTW_PRESERVE_INPUT;//FFTW_ESTIMATE;//FFTW_MEASURE |  FFTW_PRESERVE_INPUT;//FFTW_ESTIMATE; //FFTW_MEASURE


#ifdef __linux__        
		char buffer[255];
		gethostname ( buffer,255 );
		std::string s;
		s=std::string ( getenv ( "HOME" ) ) +std::string ( "/.mywisdom_" ) +std::string ( buffer ) +"_single.wisdom";

		FILE *ifp;
		ifp = fopen ( s.c_str(),"r" );
		if ( ifp!=NULL )
		{
			if ( 0==fftwf_import_wisdom_from_file ( ifp ) )
				printf ( "Error reading wisdom file: %s!\n",s.c_str() );
			fclose ( ifp );
		}
		else printf ( "Wisdom file does not exist!\n" );
#endif

		fftwf_plan plan=fftwf_plan_many_dft ( rank,n,howmany,in,inembed,istride, idist,out,onembed, ostride, odist, sign, flags | FFTW_PRESERVE_INPUT );
		fftwf_execute_dft ( plan,in, out );

#ifdef __linux__        
		ifp = fopen ( s.c_str(),"w" );
		if ( ifp!=NULL )
		{
			fftwf_export_wisdom_to_file ( ifp );
			fclose ( ifp );
		}
		else  printf ( "Error creating file!\n" );
#endif    

		fftwf_destroy_plan ( plan );
	}



#else


	template<typename T>
	void fft ( const std::complex<T> * IN,
		std::complex<T> * OUT, 
		int shape[],
		int numComponents,
		bool forward,
		int flag=0 )
	{
		printf("warning: you forgot to enable FFTW by passing -D_STA_LINK_FFTW to the compiler\n");
	}

#endif 





	/*#####################################################


	FIELD in V^l


	#######################################################*/




	template<typename T>
	T * sta_product_precomputeCGcoefficients_R( int J1,int J2, int J, bool normalized, T fact)
	{
		T norm=(T)1;
		if (normalized)
		{
			//assert((J1+J2+J)%2==0);
			norm=(T)1/(T)hanalysis::clebschGordan(J1,0,J2,0,J,0);
		}


		norm*=fact;
		std::size_t count=0;
		for (int m=-J;m<=0;m++)
		{
			for (int m1=-J1;m1<=J1;m1++)
			{
				int m2=m-m1;
				if (abs(m2)<=J2)
				{
					count++;
				}
			}
		}
		T * cg= new T[count];
		count=0;
		for (int m=-J;m<=0;m++)
		{
			for (int m1=-J1;m1<=J1;m1++)
			{
				int m2=m-m1;
				if (abs(m2)<=J2)
				{
					cg[count++]=norm*(T)hanalysis::clebschGordan(J1,m1,J2,m2,J,m);
				}
			}
		}
		return cg;
	}


	template<typename T,typename S >
	int sta_feature_product_R (
		const std::complex<T> * stIn1,
		const std::complex<T> * stIn2,
		S * stOut ,
		const std::size_t shape[],
		int J,
		T alpha,
		bool normalize,
		int stride_in1 = -1,
		int stride_in2 = -1,
		int stride_out = -1,
		bool clear_field=false)
	{

		T  * Cg= sta_product_precomputeCGcoefficients_R<T> ( J,J, 0,normalize,alpha );

		std::size_t vectorLengthJ1= ( J+1 );
		std::size_t vectorLengthJ2= ( J+1 );
		std::size_t vectorLengthJ= ( 1 );

		if ( stride_in1 == -1 )
			stride_in1 = vectorLengthJ1;
		if ( stride_in2 == -1 )
			stride_in2 = vectorLengthJ2;
		if ( stride_out == -1 )
			stride_out = vectorLengthJ;

		stride_in1*=2; // because input field is complex but pointer are real
		stride_in2*=2;


		std::size_t jumpz=shape[1]*shape[2];

		const T * stIn1R= ( const T * ) stIn1;
		const T * stIn2R= ( const T * ) stIn2;
		S * stOutR= ( S * ) stOut;

#pragma omp parallel for num_threads(get_numCPUs())
		for ( std::size_t z=0;z<shape[0];z++ )
		{
			std::size_t Z=z;
			Z*=jumpz;
			const T * current_J1R=stIn1R+ ( Z*stride_in1+2*J );
			const T * current_J2R=stIn2R+ ( Z*stride_in2+2*J );
			S * current_JR=stOutR+ ( Z*stride_out );

			for ( std::size_t i=0;i<jumpz;i++ )
			{
				std::size_t count=0;
				{
					if ( clear_field )
					{
						current_JR[0]=T ( 0 );
					}

					for ( int m1=-J;m1<=0;m1++ )
					{
						if (m1!=0)
						{
							current_JR[0]+=2*Cg[count]* ( current_J1R[m1*2]*current_J2R[m1*2]
							+current_J1R[m1*2+1]*current_J2R[m1*2+1] );
						}
						else
							current_JR[0]+=Cg[count]* ( current_J1R[m1*2]*current_J2R[m1*2]
						+current_J1R[m1*2+1]*current_J2R[m1*2+1] );		    
					}
				}
				current_J1R+=stride_in1;
				current_J2R+=stride_in2;
				current_JR+=stride_out;
			}
		}
		delete [] Cg;
		return 0;
	}








	template<typename T>
	int sta_derivatives_R(
		const std::complex<T> * stIn,
		std::complex<T> * stOut ,
		const std::size_t shape[],
		int J,
		int Jupdown,  
		bool conjugate=false,
		T alpha=(T)1.0,
		const T  v_size[]=NULL,
		int stride_in = -1,
		int stride_out = -1,
		bool clear_field = false)
	{
		alpha/=T(2);
		if ( abs ( Jupdown ) >1 ) return -1;
		if ( abs ( J+Jupdown ) <0 ) return -1;


		T voxel_size[3];
		voxel_size[0]=voxel_size[1]=voxel_size[2]=T(1);
		if (v_size!=NULL)
		{
			voxel_size[0]/=v_size[0]; // Zdir
			voxel_size[1]/=v_size[1]; // Ydir
			voxel_size[2]/=v_size[2]; // Xdir
		}

		voxel_size[1]*=-1;
		if (conjugate) voxel_size[1]*=T( -1 );

		int J1=(T)(J+Jupdown);

		std::size_t vectorLengthJ=J+1;
		std::size_t vectorLengthJ1=(J1)+1;


		std::size_t jumpz=shape[1]*shape[2];
		std::size_t jumpy=shape[2];

		if (stride_in == -1)
			stride_in = vectorLengthJ;
		if (stride_out == -1)
			stride_out = vectorLengthJ1;




		T * CGTable=new T[3*vectorLengthJ1];
		T shnorm=hanalysis::clebschGordan(1,0,J,0,J1,0);
		if (Jupdown==0) shnorm=1;

		for (int M=-(J1);M<=(0);M++)
		{
			CGTable[M+(J1)]                 =T(1.0/std::sqrt(2.0))*hanalysis::clebschGordan(1,-1,J,M+1,J1,M)/shnorm;;
			CGTable[M+(J1)+vectorLengthJ1]  =voxel_size[0]*hanalysis::clebschGordan(1,0,J,M,J1,M)/shnorm;
			CGTable[M+(J1)+2*vectorLengthJ1]=T(1.0/std::sqrt(2.0))*hanalysis::clebschGordan(1,1,J,M-1,J1,M)/shnorm;
		}
		T * CGTable0=&CGTable[0];
		CGTable0+=(J1);
		T * CGTable1=&CGTable[vectorLengthJ1];
		CGTable1+=(J1);
		T * CGTable2=&CGTable[2*vectorLengthJ1];
		CGTable2+=(J1);


		const T * stIn_r=(const T *)stIn;
		T * stOut_r=(T*)stOut;  
		vectorLengthJ*=2;
		vectorLengthJ1*=2;
		stride_in*=2;
		stride_out*=2;

		int J_times_2=J*2;

#pragma omp parallel for num_threads(get_numCPUs())
		for (std::size_t z=0;z<shape[0];z++)
		{
			std::size_t Z[3];
			Z[1]=z+shape[0];
			Z[0]=Z[1]-1;
			Z[2]=Z[1]+1;
			Z[0]%=shape[0];
			Z[1]%=shape[0];
			Z[2]%=shape[0];

			Z[0]*=jumpz;
			Z[1]*=jumpz;
			Z[2]*=jumpz;

			const T * derivX1;
			const T * derivX0;

			const T * derivY1;
			const T * derivY0;

			const T * derivZ1;
			const T * derivZ0;


			for (std::size_t y=0;y<shape[1];y++)
			{
				std::size_t Y[3];
				Y[1]=y+shape[1];
				Y[0]=Y[1]-1;
				Y[2]=Y[1]+1;
				Y[0]%=shape[1];
				Y[1]%=shape[1];
				Y[2]%=shape[1];

				Y[0]*=jumpy;
				Y[1]*=jumpy;
				Y[2]*=jumpy;

				for (std::size_t x=0;x<shape[2];x++)
				{
					std::size_t X[3];
					X[1]=x+shape[2];
					X[0]=X[1]-1;
					X[2]=X[1]+1;
					X[0]%=shape[2];
					X[1]%=shape[2];
					X[2]%=shape[2];

					derivX1=&stIn_r[(Z[1]+Y[1]+X[0])*stride_in]+J_times_2;
					derivX0=&stIn_r[(Z[1]+Y[1]+X[2])*stride_in]+J_times_2;

					derivY1=&stIn_r[(Z[1]+Y[0]+X[1])*stride_in]+J_times_2;
					derivY0=&stIn_r[(Z[1]+Y[2]+X[1])*stride_in]+J_times_2;

					derivZ1=&stIn_r[(Z[0]+Y[1]+X[1])*stride_in]+J_times_2;
					derivZ0=&stIn_r[(Z[2]+Y[1]+X[1])*stride_in]+J_times_2;

					T * current_r=stOut_r+(Z[1]+Y[1]+X[1])*stride_out;

					for (int M=-(J1);M<=(0);M++)
					{
						T tmp_r=T ( 0 );
						T tmp_i=T ( 0 );

						if (abs(M+1)<=J)  // m1=-1    m2=M+1    M
						{
							int m2=2*(M+1);
							if (M==0)
							{
								m2*=-1;
								tmp_r-=CGTable0[M]*(voxel_size[2]*(derivX0[m2]-derivX1[m2])+voxel_size[1]*(derivY0[m2+1]-derivY1[m2+1]));
								tmp_i-=CGTable0[M]*(voxel_size[2]*(derivX1[m2+1]-derivX0[m2+1])+voxel_size[1]*(derivY0[m2]-derivY1[m2]));
							} else
							{
								tmp_r+=CGTable0[M]*(voxel_size[2]*(derivX0[m2]-derivX1[m2])+voxel_size[1]*(derivY1[m2+1]-derivY0[m2+1]));
								tmp_i+=CGTable0[M]*(voxel_size[2]*(derivX0[m2+1]-derivX1[m2+1])+voxel_size[1]*(derivY0[m2]-derivY1[m2]));
							}
						}
						if (M>=-J)  // m1=0     m2=M        M
						{
							tmp_r+=CGTable1[M]*(derivZ0[M*2]-derivZ1[M*2]);
							tmp_i+=CGTable1[M]*(derivZ0[M*2+1]-derivZ1[M*2+1]);
						}
						if (M-1>=-J)  // m1=1     m2=M-1    M
						{
							int m2=2*(M-1);
							tmp_r+=CGTable2[M]*(voxel_size[2]*(derivX1[m2]-derivX0[m2])+voxel_size[1]*(derivY1[m2+1]-derivY0[m2+1]));
							tmp_i+=CGTable2[M]*(voxel_size[2]*(derivX1[m2+1]-derivX0[m2+1])+voxel_size[1]*(derivY0[m2]-derivY1[m2]));
						}


						if ( clear_field ) 
						{
							(*current_r)=tmp_r*alpha;
							current_r++;
							(*current_r)=tmp_i*alpha;
							current_r++;
						}else
						{
							(*current_r)+=tmp_r*alpha;
							current_r++;
							(*current_r)+=tmp_i*alpha;
							current_r++;		      
						}
					}

				}
			}
		}

		delete [] CGTable;
		return (J1);
	}








	template<typename T,typename S>
	int sta_derivatives_R4th(
		const S * stIn,
		std::complex<T> * stOut ,
		const std::size_t shape[],
		int J,
		int Jupdown,    // either -1 0 or 1
		bool conjugate=false,
		std::complex<T> alpha=(T)1.0,
		const T  v_size[]=NULL,
		int stride_in = -1,
		int stride_out = -1,
		bool clear_field = false)
	{
		alpha/=T(12);
		if ( abs ( Jupdown ) >1 ) return -1;
		if ( abs ( J+Jupdown ) <0 ) return -1;

		std::complex<T> imag=-std::complex<T>(0,1);
		if (conjugate) imag*=T( -1 );

		T voxel_size[3];
		voxel_size[0]=voxel_size[1]=voxel_size[2]=T(1);
		if (v_size!=NULL)
		{
			voxel_size[0]/=v_size[0]; // Zdir
			voxel_size[1]/=v_size[1]; // Ydir
			voxel_size[2]/=v_size[2]; // Xdir
		}

		imag*=voxel_size[1];

		int J1=(T)(J+Jupdown);

		std::size_t vectorLengthJ=J+1;
		std::size_t vectorLengthJ1=(J1)+1;


		if (stride_in == -1)
			stride_in = vectorLengthJ;
		if (stride_out == -1)
			stride_out = vectorLengthJ1;


		std::size_t jumpz=shape[1]*shape[2];
		std::size_t jumpy=shape[2];


		T * CGTable=new T[3*vectorLengthJ1];
		T shnorm=hanalysis::clebschGordan(1,0,J,0,J1,0);
		if (Jupdown==0) shnorm=1;

		for (int M=-(J1);M<=(0);M++)
		{
			CGTable[M+(J1)]                 =T(1.0/std::sqrt(2.0))*hanalysis::clebschGordan(1,-1,J,M+1,J1,M)/shnorm;;
			CGTable[M+(J1)+vectorLengthJ1]  =voxel_size[0]*hanalysis::clebschGordan(1,0,J,M,J1,M)/shnorm;
			CGTable[M+(J1)+2*vectorLengthJ1]=T(1.0/std::sqrt(2.0))*hanalysis::clebschGordan(1,1,J,M-1,J1,M)/shnorm;
		}
		T * CGTable0=&CGTable[0];
		CGTable0+=(J1);
		T * CGTable1=&CGTable[vectorLengthJ1];
		CGTable1+=(J1);
		T * CGTable2=&CGTable[2*vectorLengthJ1];
		CGTable2+=(J1);

#pragma omp parallel for num_threads(get_numCPUs())
		for (std::size_t z=0;z<shape[0];z++)
		{
			std::size_t Z[5];
			Z[2]=z+shape[0];
			Z[0]=Z[2]-2;
			Z[1]=Z[2]-1;
			Z[3]=Z[2]+1;
			Z[4]=Z[2]+2;


			Z[0]%=shape[0];
			Z[1]%=shape[0];
			Z[2]%=shape[0];
			Z[3]%=shape[0];
			Z[4]%=shape[0];


			Z[0]*=jumpz;
			Z[1]*=jumpz;
			Z[2]*=jumpz;
			Z[3]*=jumpz;
			Z[4]*=jumpz;


			const S * derivZ0p=stIn+Z[0]*stride_in+J;
			const S * derivZ1p=stIn+Z[1]*stride_in+J;
			const S * derivZ2p=stIn+Z[2]*stride_in+J;
			const S * derivZ3p=stIn+Z[3]*stride_in+J;
			const S * derivZ4p=stIn+Z[4]*stride_in+J;


			const S * derivX3;
			const S * derivX2;
			const S * derivX1;
			const S * derivX0;

			const S * derivY3;
			const S * derivY2;
			const S * derivY1;
			const S * derivY0;


			const S * derivZ3;
			const S * derivZ2;
			const S * derivZ1;
			const S * derivZ0;


			for (std::size_t y=0;y<shape[1];y++)
			{
				std::size_t Y[5];
				Y[2]=y+shape[1];
				Y[0]=Y[2]-2;
				Y[1]=Y[2]-1;
				Y[3]=Y[2]+1;
				Y[4]=Y[2]+2;



				Y[0]%=shape[1];
				Y[1]%=shape[1];
				Y[2]%=shape[1];
				Y[3]%=shape[1];
				Y[4]%=shape[1];

				Y[0]*=jumpy;
				Y[1]*=jumpy;
				Y[2]*=jumpy;
				Y[3]*=jumpy;
				Y[4]*=jumpy;



				const S * derivZ2Y0p=derivZ2p+Y[0]*stride_in;
				const S * derivZ2Y1p=derivZ2p+Y[1]*stride_in;
				const S * derivZ2Y3p=derivZ2p+Y[3]*stride_in;
				const S * derivZ2Y4p=derivZ2p+Y[4]*stride_in;

				std::size_t tmp=Y[2]*stride_in;
				const S * derivZ0Y2p=derivZ0p+tmp;
				const S * derivZ1Y2p=derivZ1p+tmp;
				const S * derivZ2Y2p=derivZ2p+tmp;
				const S * derivZ3Y2p=derivZ3p+tmp;
				const S * derivZ4Y2p=derivZ4p+tmp;


				for (std::size_t x=0;x<shape[2];x++)
				{

					std::size_t X[5];
					X[2]=x+shape[2];
					X[0]=X[2]-2;
					X[1]=X[2]-1;
					X[3]=X[2]+1;
					X[4]=X[2]+2;

					X[0]%=shape[2];
					X[1]%=shape[2];
					X[2]%=shape[2];
					X[3]%=shape[2];
					X[4]%=shape[2];


					derivX0=derivZ2Y2p+(X[0])*stride_in;
					derivX1=derivZ2Y2p+(X[1])*stride_in;
					derivX2=derivZ2Y2p+X[3]*stride_in;
					derivX3=derivZ2Y2p+(X[4])*stride_in;


					std::size_t tmp=X[2]*stride_in;
					derivY0=derivZ2Y0p+tmp;
					derivY1=derivZ2Y1p+tmp;
					derivY2=derivZ2Y3p+tmp;
					derivY3=derivZ2Y4p+tmp;

					derivZ0=derivZ0Y2p+tmp;
					derivZ1=derivZ1Y2p+tmp;
					derivZ2=derivZ3Y2p+tmp;
					derivZ3=derivZ4Y2p+tmp;

					std::size_t offset=(Z[2]+Y[2]+X[2])*stride_out+J1;

					for (int M=-(J1);M<=(0);M++)
					{
						std::complex<T> & current=stOut[offset+M];
						if ( clear_field ) current=T ( 0 );
						std::complex<T> tmp=T ( 0 );

						if (abs(M+1)<=J)  // m1=-1    m2=M+1    M
						{
							int m2=M+1;
							if (M==0)
							{
								tmp-=CGTable0[M]*(voxel_size[2]*std::conj(derivX0[-m2]+(T)8.0*(derivX2[-m2]-derivX1[-m2])-derivX3[-m2])+
									imag*std::conj(derivY0[-m2]+(T)8.0*(derivY2[-m2]-derivY1[-m2])-derivY3[-m2]));
							} else
								tmp+=CGTable0[M]*(voxel_size[2]*(derivX0[m2]+(T)8.0*(derivX2[m2]-derivX1[m2])-derivX3[m2])+
								imag*(derivY0[m2]+(T)8.0*(derivY2[m2]-derivY1[m2])-derivY3[m2]));
						}
						if (M>=-J)  // m1=0     m2=M        M
						{
							tmp+=CGTable1[M]*(derivZ0[M]+(T)8.0*(derivZ2[M]-derivZ1[M])-derivZ3[M]);
						}
						if (M-1>=-J)  // m1=1     m2=M-1    M
						{
							int m2=M-1;
							tmp+=CGTable2[M]*(-voxel_size[2]*(derivX0[m2]+(T)8.0*(derivX2[m2]-derivX1[m2])-derivX3[m2])+
								imag*(derivY0[m2]+(T)8.0*(derivY2[m2]-derivY1[m2])-derivY3[m2]));
						}
						current+=tmp*alpha;
					}

				}
			}
		}

		delete [] CGTable;
		return (J1);
	}


	template<typename T>
	int sta_product_R (
		const std::complex<T> * stIn1,
		const std::complex<T> * stIn2,
		std::complex<T> * stOut ,
		const std::size_t shape[],
		int J1,
		int J2,
		int J,
		T alpha,
		bool normalize,
		int stride_in1 = -1,
		int stride_in2 = -1,
		int stride_out = -1,
		bool clear_field=false)
	{

		if ( ( std::abs ( J1-J2 ) >J ) || ( J>std::abs ( J1+J2 ) ) )
			return -1;
		if ( ( ( J1+J2+J ) %2!=0 ) && ( normalize ) )
			return -1;


		T  * Cg= sta_product_precomputeCGcoefficients_R<T> ( J1,J2, J,normalize,alpha );

		bool resultInIv= ( ( J1+J2+J ) %2!=0 );

		std::size_t vectorLengthJ1= ( J1+1 );
		std::size_t vectorLengthJ2= ( J2+1 );
		std::size_t vectorLengthJ= ( J+1 );

		if ( stride_in1 == -1 )
			stride_in1 = vectorLengthJ1;
		if ( stride_in2 == -1 )
			stride_in2 = vectorLengthJ2;
		if ( stride_out == -1 )
			stride_out = vectorLengthJ;

		stride_in1*=2; // because input field is complex but pointer are real
		stride_in2*=2;
		stride_out*=2;

		int J2_times_2=J2*2;
		int J1_times_2=J1*2;
		int J_times_2=J*2;


		std::size_t jumpz=shape[1]*shape[2];

		const T * stIn1R= ( const T * ) stIn1;
		const T * stIn2R= ( const T * ) stIn2;
		T * stOutR= ( T * ) stOut;

#pragma omp parallel for num_threads(get_numCPUs())
		for ( std::size_t z=0;z<shape[0];z++ )
		{
			std::size_t Z=z;
			Z*=jumpz;
			const T * current_J1R=stIn1R+ ( Z*stride_in1+J1_times_2 );
			const T * current_J2R=stIn2R+ ( Z*stride_in2+J2_times_2 );
			T * current_JR=stOutR+ ( Z*stride_out+J_times_2 );

			T tmp0R;
			T tmp0I;

			T tmp1R;
			T tmp1I;

			for ( std::size_t i=0;i<jumpz;i++ )
			{
				std::size_t count=0;
				for ( int m=-J;m<=0;m++ )
				{
					if ( clear_field )
					{
						current_JR[m*2]=T ( 0 );
						current_JR[m*2+1]=T ( 0 );
					}
					for ( int m1=-J1;m1<=J1;m1++ )
					{
						int m2=m-m1;
						if ( abs ( m2 ) <=J2 )
						{
							if ( m1>0 )
							{
								if ( m1%2==0 )
								{
									tmp0R=current_J1R[-m1*2];
									tmp0I=-current_J1R[-m1*2+1];
								}
								else
								{
									tmp0R=-current_J1R[-m1*2];
									tmp0I=current_J1R[-m1*2+1];
								}
							}
							else
							{
								tmp0R=current_J1R[m1*2];
								tmp0I=current_J1R[m1*2+1];
							}
							if ( m2>0 )
							{
								if ( m2%2==0 )
								{
									tmp1R=current_J2R[-m2*2];
									tmp1I=-current_J2R[-m2*2+1];
								}
								else
								{
									tmp1R=-current_J2R[-m2*2];
									tmp1I=current_J2R[-m2*2+1];
								}
							}
							else
							{
								tmp1R=current_J2R[m2*2];
								tmp1I=current_J2R[m2*2+1];
							}
							if ( resultInIv )
							{
								current_JR[m*2]-=Cg[count]* ( tmp0R*tmp1I+tmp0I*tmp1R );
								current_JR[m*2+1]+=Cg[count++]* ( tmp0R*tmp1R-tmp0I*tmp1I );
							}
							else
							{		  
								current_JR[m*2]+=Cg[count]* ( tmp0R*tmp1R-tmp0I*tmp1I );
								current_JR[m*2+1]+=Cg[count++]* ( tmp0R*tmp1I+tmp0I*tmp1R );
							}
						}
					}
				}
				current_J1R+=stride_in1;
				current_J2R+=stride_in2;
				current_JR+=stride_out;
			}
		}
		delete [] Cg;
		return 0;
	}


	template<typename T,typename S>
	int sta_product0 (
		const std::complex<T> * stIn1,
		const std::complex<T> * stIn2,
		std::complex<T> * stOut,
		const std::size_t shape[],
		S alpha,
		int  stride_in1 = -1,
		int  stride_in2 = -1,
		int  stride_out = -1,
		bool clear_field = false )
	{
		{
			if (stride_in1 == -1)
				stride_in1 = 1;
			if (stride_in2 == -1)
				stride_in2 = 1;
			if (stride_out == -1)
				stride_out = 1;



			std::size_t jumpz=shape[1]*shape[2];


#pragma omp parallel for num_threads(hanalysis::get_numCPUs())
			for (std::size_t z=0;z<shape[0];z++)
			{
				std::size_t Z=z;
				Z*=jumpz;
				const std::complex<T> * current_J1=&stIn1[Z*stride_in1];
				const std::complex<T> * current_J2=&stIn2[Z*stride_in2];
				std::complex<T> * current_J=&stOut[Z*stride_out];

				for (std::size_t i=0;i<jumpz;i++)
				{
					if (clear_field)
						*current_J=T ( 0 );
					*current_J+=(*current_J1)* (*current_J2)*alpha;
					current_J+=stride_out;
					current_J1+=stride_in1;
					current_J2+=stride_in2;
				}
			}
		}
		return 0;
	}



	/*########################################################



	########################################################*/

	/// tensor field data interpretations according to certain symmetries
	enum STA_FIELD_PROPERTIES {
		STA_FIELD_INVALID_PARAM=0,	
		STA_FIELD_STORAGE_C=2,
		/// we assume the following symmetry: \f$ \mathbf a^\ell_m(\mathbf x)=(-1)^m \overline{\mathbf a^\ell_{-m}(\mathbf x) } \f$.\n
		/// Tensor-data should be arranged as follows: \f$ (\mathbf a^\ell_{-\ell},\mathbf a^\ell_{-(\ell-1)} \cdots,\mathbf a^\ell_{-1},\mathbf a^\ell_{0}) \f$
		STA_FIELD_STORAGE_R=4,
		STA_FIELD_STORAGE_RF=8,
		STA_FIELD_EXPERIMENTAL=16,


		/// tensor field has one single component of rank : \f$ \ell \f$ 
		STA_OFIELD_SINGLE=128,
		STA_OFIELD_FULL=256,
		STA_OFIELD_EVEN=512,
		STA_OFIELD_ODD=1024

	};


	static STA_FIELD_PROPERTIES enumfromstring(std::string s)
	{
		if (s.compare("STA_FIELD_STORAGE_C")==0)
			return STA_FIELD_STORAGE_C;
		//   if (s.compare("STA_FIELD_STORAGE_FULL")==0)
		//     return STA_FIELD_STORAGE_FULL;
		if (s.compare("STA_FIELD_STORAGE_R")==0)
			return STA_FIELD_STORAGE_R;  
		if (s.compare("STA_FIELD_STORAGE_RF")==0)
			return STA_FIELD_STORAGE_RF;  
		if (s.compare("STA_FIELD_EXPERIMENTAL")==0)
			return STA_FIELD_EXPERIMENTAL;  

		if (s.compare("STA_OFIELD_EVEN")==0)
			return STA_OFIELD_EVEN;

		if (s.compare("STA_OFIELD_ODD")==0)
			return STA_OFIELD_ODD;

		if (s.compare("STA_OFIELD_FULL")==0)
			return STA_OFIELD_FULL;

		if (s.compare("STA_OFIELD_SINGLE")==0)
			return STA_OFIELD_SINGLE;
		return STA_FIELD_INVALID_PARAM;
	}



	static std::string enumtostring(STA_FIELD_PROPERTIES p)
	{
		if (p == STA_FIELD_STORAGE_C)
			return "STA_FIELD_STORAGE_C";

		//   if (p == STA_FIELD_STORAGE_FULL)
		//     return "STA_FIELD_STORAGE_FULL";

		if (p == STA_FIELD_STORAGE_R)
			return "STA_FIELD_STORAGE_R";  

		if (p == STA_FIELD_STORAGE_RF)
			return "STA_FIELD_STORAGE_RF";  

		if (p == STA_FIELD_EXPERIMENTAL)
			return "STA_FIELD_EXPERIMENTAL";  

		if (p == STA_OFIELD_EVEN)
			return "STA_OFIELD_EVEN";

		if (p == STA_OFIELD_ODD)
			return "STA_OFIELD_ODD";

		if (p == STA_OFIELD_FULL)
			return "STA_OFIELD_FULL";

		if (p == STA_OFIELD_SINGLE)
			return "STA_OFIELD_SINGLE";

		return "STA_FIELD_INVALID_PARAM";
	}



	static int numComponents2order(
		hanalysis::STA_FIELD_PROPERTIES field_storage,
		hanalysis::STA_FIELD_PROPERTIES field_type,
		int ncomponents)
	{
		if (field_storage==STA_FIELD_STORAGE_C)
		{
			switch (field_type)
			{
			case STA_OFIELD_SINGLE:
				{
					return (ncomponents-1)/2;
				}//break;
			case STA_OFIELD_FULL:
				{
					return std::sqrt((double)ncomponents)-1;
				}//break;
			case STA_OFIELD_ODD:
				{
					return (-3/2+std::sqrt((3.0/2.0)*(3.0/2.0)+2*ncomponents-2));
				}//break;
			case STA_OFIELD_EVEN:
				{
					return (-3/2+std::sqrt((3.0/2.0)*(3.0/2.0)+2*ncomponents-2));
				}//break;
			default:
				return -1;
			}
		} else
		{
			switch (field_type)
			{
			case STA_OFIELD_SINGLE:
				{
					return ncomponents-1;
				}//break;
			case STA_OFIELD_FULL:
				{
					return (-3/2+std::sqrt((3.0/2.0)*(3.0/2.0)+2*ncomponents-2));
				}//break;
			case STA_OFIELD_ODD:
				{
					return std::sqrt((double)(1+4*ncomponents))-2;
				}//break;
			case STA_OFIELD_EVEN:
				{
					return std::sqrt((double)(4*ncomponents))-2;
				}//break;
			default:
				return -1;
			}
			//return -1;
		}
	}




	static int order2numComponents(
		hanalysis::STA_FIELD_PROPERTIES field_storage,
		hanalysis::STA_FIELD_PROPERTIES field_type,
		int L)
	{
		if (field_storage==STA_FIELD_STORAGE_C)
		{
			switch (field_type)
			{
			case STA_OFIELD_SINGLE:
				{
					return 2*L+1;
				}//break;
			case STA_OFIELD_FULL:
				{
					return ((L+1)*(L+1));
				}//break;
			case STA_OFIELD_ODD:
				{
					return ((L+1)*(L+2))/2;
				}//break;
			case STA_OFIELD_EVEN:
				{
					return ((L+1)*(L+2))/2;
				}//break;
			default:
				return -1;
			}
		} else
		{
			switch (field_type)
			{
			case STA_OFIELD_SINGLE:
				{
					return L+1;
				}//break;
			case STA_OFIELD_FULL:
				{
					return ((L+1)*(L+2))/2;
				}//break;
			case STA_OFIELD_ODD:
				{
					return ((L+1)*(L+3))/4;
				}//break;
			case STA_OFIELD_EVEN:
				{
					return ((L+2)*(L+2))/4;
				}//break;
			default:
				return -1;
			}
			//return -1;
		}
	}



	static int getComponentOffset(
		hanalysis::STA_FIELD_PROPERTIES field_storage,
		hanalysis::STA_FIELD_PROPERTIES field_type,
		int L)
	{
		if (field_storage==STA_FIELD_STORAGE_C)
		{
			switch (field_type)
			{
			case STA_OFIELD_SINGLE:
				{
					return 0;
				}//break;
			case STA_OFIELD_FULL:
				{
					return (L)*(L);
				}//break;
			case STA_OFIELD_ODD:
				{
					return ((L-1)*L)/2;
				}//break;
			case STA_OFIELD_EVEN:
				{
					return  ((L-1)*L)/2;
				}//break;
			default:
				return -1;
			}
		} else
		{
			switch (field_type)
			{
			case STA_OFIELD_SINGLE:
				{
					return 0;
				}//break;
			case STA_OFIELD_FULL:
				{
					return (L*(L+1))/2;
				}//break;
			case STA_OFIELD_ODD:
				{
					return ((L+1)*(L+3))/4-L-1;
				}//break;
			case STA_OFIELD_EVEN:
				{
					return ((L+2)*(L+2))/4-L-1;
				}//break;
			default:
				return -1;
			}
			//return -1;
		}
	}



	/*
	enum STA_OFIELD_PROPERTIES {
	STA_OFIELD_SINGLE=0,
	STA_OFIELD_EVEN=1,
	STA_OFIELD_ODD=2,
	STA_OFIELD_FULL=3
	};*/

	///  spherical tensor product:  \f$ \alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \f$ and \f$ \alpha(\mathbf{stIn1} \bullet_{J} \mathbf{stIn2}) \f$, respectively  \n
	/*!
	computes the spherical tensor product \f$ \alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \f$ and \f$ \alpha(\mathbf{stIn1} \bullet_{J} \mathbf{stIn2}) \f$, respectively  \n
	\param stIn1 \f$ \mathbf{stIn1} \in \mathcal T_{J_1}\f$
	\param stIn2 \f$ \mathbf{stIn2} \in \mathcal T_{J_2} \f$
	\param stOut \f$ \alpha(\mathbf{stIn1} \bullet_{J} \mathbf{stIn2}) \in \mathcal T_{J}\f$ if normalized, \f$ \alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \in \mathcal T_{J}\f$  else
	\param shape
	\param J1 \f$ J_1 \in \mathbb N \f$ tensor rank of the first field
	\param J2 \f$ J_2 \in \mathbb N \f$ tensor rank of the second field
	\param J \f$ J \in \mathbb N \f$ tensor rank of the resulting field
	\param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor
	\param normalize normalized tensor products?: true=\f$ \bullet_{J}\f$ , false=\f$ \circ_{J}\f$
	\returns  \f$
	\left\{
	\begin{array}{ll}
	0 &  \mbox{if tensor product exists}\\
	-1 & \mbox{ else }
	\end{array}
	\right.
	\f$
	\warning ensure that stIn1, stIn2, stOut and shape exist
	and have been \b allocated properly!\n
	\warning If \b field_property=STA_FIELD_STORAGE_R and  \f$ (J_1+J2+J)\%2 \neq 0 \f$ the
	function returns \n \f$ (\mathit{i})\alpha(\mathbf{stIn1} \circ_{J} \mathbf{stIn2}) \in \mathcal T_{J}\f$.
	This ensures that \b STA_FIELD_STORAGE_R holds for \f$ \mathbf{stOut} \f$, too. \n
	The same is true for \b field_property=STA_FIELD_STORAGE_RF
	\warning if not \b STA_FIELD_STORAGE_C then \b alpha must be real valued
	*/
	template<typename T>
	int sta_product (
		const std::complex<T> * stIn1,
		const std::complex<T> * stIn2,
		std::complex<T> * stOut,
		const std::size_t shape[],
		int J1,
		int J2,
		int J,
		std::complex<T> alpha,
		bool normalize = false,
		STA_FIELD_PROPERTIES field_property=STA_FIELD_STORAGE_C,
		int  stride_in1 = -1,
		int  stride_in2 = -1,
		int  stride_out = -1,
		bool clear_field = false )
	{
		int result=-1;
		bool alpha_real=(alpha.imag()==0);




		if ((J1==0)&&(J2==0)&&(J==0))
		{
			return  sta_product0(stIn1,
				stIn2,
				stOut,
				shape,
				alpha.real(),
				stride_in1,
				stride_in2,
				stride_out,
				clear_field);
		}    




		if ((field_property==STA_FIELD_STORAGE_R)&&(alpha_real))
		{
			result=sta_product_R (stIn1,
				stIn2,
				stOut,
				shape,
				J1,
				J2,
				J,
				alpha.real(),
				normalize,
				stride_in1,
				stride_in2,
				stride_out,
				clear_field);
			return result;
		}

		printf("please wait for the whole toolbox :-)\n ");

		return result;
	}




	/// computes \f$ \alpha(\mathbf{stIn})  \f$ 
	/*!
	multiplication with a scalar: \f$ \alpha(\mathbf{stIn})  \f$ \n
	\param stIn \f$ \mathbf{stIn1} \in \mathcal T_{J}\f$
	\param stOut \f$ \alpha(\mathbf{stIn}) \in \mathcal T_{J} \f$
	\param shape
	\param ncomponents number of tensor components
	\param alpha \f$ \alpha \in \mathbb C \f$  weighting factor
	\param conjugate returns \f$ \alpha(\overline{\mathbf{stIn}})\f$ if \b true
	\warning ensure that stIn, stOut and shape exist
	and have been \b allocated properly!\n
	*/
	template<typename T,typename S>
	int sta_mult (
		const std::complex<T> * stIn,
		std::complex<T> * stOut,
		const std::size_t shape[],
		int ncomponents,
		S alpha = S ( 1 ),
		bool conjugate=false,
		int  stride_in = -1,
		int  stride_out = -1,
		bool clear_field = false )
	{
		bool doalpha= ( alpha!=S ( 1 ) );

		if (stride_in==-1)
			stride_in=ncomponents;
		if (stride_out==-1)
			stride_out=ncomponents;    


		std::size_t jumpz=shape[1]*shape[2];

		//printf("alpha (%f %f) %d\n",alpha.real(),alpha.imag(),conjugate);

#pragma omp parallel for num_threads(hanalysis::get_numCPUs())
		for (std::size_t a=0;a<shape[0];a++ )
		{
			std::complex<T> *resultp=stOut+a*jumpz*stride_out;
			const std::complex<T> *inp=stIn+a*jumpz*stride_in;

			for ( std::size_t i=0;i<jumpz;i++ )
			{
				for (int b=0;b<ncomponents;b++)
				{
					std::complex<T> tmp=inp[b];
					if ( conjugate )
						tmp=std::conj ( tmp );

					if ( doalpha )
						tmp*=alpha;

					if ( clear_field )
						resultp[b] =tmp;
					else
						resultp[b] +=tmp;
				}
				resultp+=stride_out;
				inp+=stride_in;	    
			}
		}
		return 0;
	}





	/// returns lengths of vectors component by compnent \n
	/*!
	returns lengths of vectors component by compnent \n
	\param stIn \f$ \mathbf{stIn1} \in \mathcal T_{J}\f$
	\param stOut \f$ \alpha(\mathbf{stIn}) \in \mathcal T_{0} \f$
	\param shape
	\param J \f$ J \in \mathbb N \f$ tensor rank of the input field \f$ \mathbf{stIn}  \f$
	*/
	template<typename T>
	int sta_norm (
		const std::complex<T> * stIn,
		std::complex<T> * stOut,
		const std::size_t shape[],
		int J,
		STA_FIELD_PROPERTIES field_property=STA_FIELD_STORAGE_C,
		int  stride_in = -1,
		int  stride_out = -1,
		bool clear_field = false )
	{
		int J1=J+1;
		if (field_property==STA_FIELD_STORAGE_C)
			J1=2*J+1;

		if (stride_in==-1)
			stride_in=J1;
		if (stride_out==-1)
			stride_out=1;


		std::size_t jumpz=shape[1]*shape[2];


#pragma omp parallel for num_threads(hanalysis::get_numCPUs())
		for (std::size_t a=0;a<shape[0];a++ )
		{
			std::complex<T> *resultp=stOut+a*jumpz*stride_out;
			const std::complex<T> *inp=stIn+a*jumpz*stride_in+J;

			for ( std::size_t i=0;i<jumpz;i++ )
			{
				const std::complex<T> * current=inp;

				T tmp=0;

				if (field_property==STA_FIELD_STORAGE_C)
				{
					for (int b=-J;b<=J;b++)
					{
						tmp+=std::norm(current[b]);
					}
				} else
				{
					for (int b=-J;b<0;b++)
					{
						tmp+=T( 2 )*std::norm(current[b]);
					}
					tmp+=std::norm(current[0]);
				}

				if ( clear_field )
					*resultp=T (0);
				*resultp+=std::sqrt(tmp);

				resultp+=stride_out;
				inp+=stride_in;

			}
		}
		return 0;
	}



	/// spherical tensor derivative: \f$ \alpha( {\nabla}  \bullet_{(J+b)}  \mathbf{stIn}) , b \in \{-1,0,1\} \f$
	/*!
	computes the spherical tensor derivative of  \f$ \mathbf{stIn} \in \mathcal T_{J}\f$ \n
	\param stIn \f$ \mathbf{stIn} \in \mathcal T_{J}\f$
	\param stOut \f$ \mathbf{stOut} \in \mathcal T_{(J+Jupdown)}\f$, the spherical tensor derivative of \f$ \mathbf{stIn} \f$
	\param shape
	\param J \f$ J \in \mathbb N \f$ tensor rank of the input field \f$ \mathbf{stIn}  \f$
	\param Jupdown
	\f$
	\left\{
	\begin{array}{ll}
	\mathbf{stOut}=\alpha({\nabla}  \bullet_{(J+1)}  \mathbf{stIn}), &  \mbox{ if } Jupdown=1\\
	\mathbf{stOut}=\alpha({\nabla}  \circ_{J}  \mathbf{stIn}), &  \mbox{ if } Jupdown=0\\
	\mathbf{stOut}=\alpha({\nabla}  \bullet_{(J-1)}  \mathbf{stIn}), &  \mbox{ if } Jupdown=-1
	\end{array}
	\right.
	\f$
	\param conjugate  if \b conjugate=true the  conjugate operator \f$ \overline{{\nabla}} \f$ is used
	\param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor
	\returns  \f$
	\left\{
	\begin{array}{ll}
	J+Jupdown &  \mbox{if derivative exists}\\
	-1 & \mbox{ else }
	\end{array}
	\right.
	\f$
	\warning ensure that stIn, stOut and shape exist
	and have been \b allocated properly!
	\warning if not \b STA_FIELD_STORAGE_C then \b alpha must be real valued
	*/
	template<typename T,typename S>
	int sta_derivatives (
		const S * stIn,
		std::complex<T> * stOut ,
		const std::size_t shape[],
		int J,
		int Jupdown,    // either -1 0 or 1
		bool conjugate=false,
		std::complex<T> alpha= ( T ) 1.0,
		STA_FIELD_PROPERTIES field_property=STA_FIELD_STORAGE_C,
		const T  v_size[]=NULL,
		int stride_in = -1,
		int stride_out = -1,
		bool clear_field = false,
		int accuracy=0)
	{
		int result=-1;
		bool alpha_real=(alpha.imag()==0);

		switch (field_property)
		{
		case STA_FIELD_STORAGE_C:
			{
				printf("please wait for the whole toolbox :-)\n ");
				return -1;

			}break;

		case STA_FIELD_STORAGE_R:
			{
				if (alpha_real)
				{
					if (accuracy==0)
					{
						result=sta_derivatives_R (stIn,
							stOut,
							shape,
							J,
							Jupdown,
							conjugate,
							alpha.real(),
							v_size,
							stride_in,
							stride_out,
							clear_field);
					}
					else
					{
						result=sta_derivatives_R4th (stIn,
							stOut,
							shape,
							J,
							Jupdown,
							conjugate,
							alpha,
							v_size,
							stride_in,
							stride_out,
							clear_field);
					}
				}
			}break;

		default:
			{
				printf("unsupported derivative\n");
			}break;
		}



		return result;
	}



	/// Laplacian: \f$ \alpha\triangle(\mathbf{stIn}) \in \mathcal T_{J}\f$
	/*!
	* computes the Laplacian of \f$ \mathbf{stIn} \in \mathcal T_{J}\f$ component by component \n
	* \param stIn \f$ \mathbf{stIn} \in \mathcal T_{J}\f$
	* \param stOut \f$ \alpha\triangle(\mathbf{stIn}) \in \mathcal T_{J}\f$
	* \param shape
	* \param components \f$ components \in \mathbb N_{>0} \f$ number of tensor components of the input field \f$ \mathbf{stIn}  \f$
	* \param type if \b type=1 the standard 6 neighbours operator is used, if \b type=0 a 18 neighbours operator is used
	*  \param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor
	* \warning ensure that stIn, stOut and shape exist
	* and have been \b allocated properly!
	*
	*/
	template<typename T,typename S>
	int sta_laplace (
		const S * stIn,
		std::complex<T> * stOut ,
		const std::size_t shape[],
		int components=1,
		int type=1,
		std::complex<T> alpha=1,
		STA_FIELD_PROPERTIES field_property=STA_FIELD_STORAGE_C,
		const T  v_size[]=NULL,
		int stride_in = -1,
		int stride_out = -1,
		bool clear_field = false  )
	{
		bool alpha_real=(alpha.imag()==0); 
		if ((!alpha_real) && (field_property!=STA_FIELD_STORAGE_C))
			return -1;


		if ( components<=0 ) return -1;
		if (( components==1 )&&
			(((stride_in==-1)&&(stride_out==-1))||
			((stride_in==1)&&(stride_out==1))))
			sta_laplace_1component ( stIn,stOut,shape,type,alpha,v_size,clear_field);
		else
		{
			printf("please wait for the whole toolbox :-)\n ");
			return -1;

		}
		return 0;
	}




	/// tensor fft component by component 
	/*!
	transforms  a spherical tensor field \f$ \mathbf{stIn} \in \mathcal T_{J}\f$ into Fourier domain (and back) \n
	\param stIn \f$ \mathbf{stIn} \in \mathcal T_{J}\f$
	\param stOut \f$ \mathbf{stOut} \in \mathcal T_{J}\f$
	\param shape
	\param components \f$ components \in \mathbb N_{>0} \f$, number of tensor components of the input field \f$ \mathbf{stIn}  \f$
	\param forward
	\f$
	\left\{
	\begin{array}{ll}
	\mathbf{stOut}=\alpha\mathcal F (\mathbf{stIn}) &  \mbox{ if } forward=true \\
	\mathbf{stOut}=\alpha\mathcal F^{-1} (\mathbf{stIn}) &  \mbox{ if } forward=false
	\end{array}
	\right.
	\f$
	\param conjugate if \b true it computes \f$ \alpha\overline{\mathcal F (\mathbf{stIn})} \f$ and   \f$ \alpha\overline{\mathcal F^{-1} (\mathbf{stIn})} \f$, respectively
	\param alpha \f$ \alpha \in \mathbb C \f$ additional weighting factor
	\warning ensure that stIn, stOut and shape exist
	and have been \b allocated properly! \n
	Consider that  \f$   \frac{\mathcal F^{-1}(\mathcal F (\mathbf{stIn}))}{shape[0] \cdot shape[1] \cdot shape[2]}=\mathbf{stIn}   \f$  !!
	*/
	template<typename T,typename S>
	void sta_fft ( const std::complex<T> * IN,
		std::complex<T> * OUT, 
		const std::size_t shape[],
		int components,
		bool forward,
		bool conjugate=false,
		S alpha = T ( 1 ),
#ifdef _STA_LINK_FFTW	       
		int flag=FFTW_ESTIMATE )
#else
		int flag=0 )
#endif
	{


		int shape_[3];
		shape_[0]=shape[0];
		shape_[1]=shape[1];
		shape_[2]=shape[2];
		fft ( IN,OUT,shape_,components,forward,flag );

		sta_mult (OUT,
			OUT,
			shape,
			components,
			alpha,
			conjugate,
			-1,
			-1,
			true);

#ifndef _STA_LINK_FFTW
		printf("fftw libs have not been linked an no transformation is performed\n");
#endif
	}

}
#endif
