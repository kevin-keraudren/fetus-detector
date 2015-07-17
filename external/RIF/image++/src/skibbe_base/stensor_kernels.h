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

#ifndef STA_STENSOR_KERNELS_H
#define STA_STENSOR_KERNELS_H

#include "stensor.h"
#include <string>
#include <complex>
#include <limits>


namespace hanalysis
{


	enum STA_CONVOLUTION_KERNELS {
		STA_CONV_KERNEL_UNKNOWN=-1,
		STA_CONV_KERNEL_GAUSS_BESSEL=0,
		STA_CONV_KERNEL_GAUSS_LAGUERRE=1,
		STA_CONV_KERNEL_GAUSS=2,
	} ;


	template<typename T>
	class Kernel
	{
	protected:
	public:

		Kernel()
		{

		}
		virtual std::complex<T> get(T rsqr, T theta, T phi,int l,int m)=0;
		virtual std::complex<T> get(T rsqr)=0;
		virtual std::string getName()=0;
		virtual STA_CONVOLUTION_KERNELS getID()=0;
		virtual bool radialComplex()=0;
		virtual std::complex<T> weight()=0;
	};

	template<typename T> class GaussLaguerre;
	template<typename T> class Gauss;
	template<typename T> class GaussBessel;


	template<typename T,typename S>
	int renderKernel(
		//std::complex<T> * & kernel,
		std::complex<T> *  kernel,
		const std::size_t shape[],
		Kernel<S> * kfp,
		int L=0,
		int m=0,
		bool centered=false,
		STA_FIELD_PROPERTIES field_property=STA_FIELD_STORAGE_R,
		const S v_size[]=NULL,
		int stride = -1)
	{
		if (std::abs(m)>L)
			return -1;
		if (L<0)
			return -1;


		bool zero_order=(L==0);

		if (stride==-1)
		{
			if (field_property==STA_FIELD_STORAGE_C)
				stride=2*L+1;
			else
				stride=L+1;
		}

		S voxel_size[3];
		voxel_size[0]=voxel_size[1]=voxel_size[2]=T ( 1 );
		if ( v_size!=NULL )
		{
			voxel_size[0]/=v_size[0]; // Zdir
			voxel_size[1]/=v_size[1]; // Ydir
			voxel_size[2]/=v_size[2]; // Xdir
		}


		Kernel<S> & kf=*kfp;

		std::complex<double> norm=kf.weight();



		int z2=shape[0]/2;
		int y2=shape[1]/2;
		int x2=shape[2]/2;

		std::size_t jumpZ=shape[2]*shape[1];


		int shape_[3];
		shape_[0]=shape[0];
		shape_[1]=shape[1];
		shape_[2]=shape[2];

		//     int shape2[3];
		//     shape2[0]=shape_[0]/2;
		//     shape2[1]=shape_[1]/2;
		//     shape2[2]=shape_[2]/2;


		if (centered)
		{
#pragma omp parallel for num_threads(get_numCPUs())
			for (int z=0;z<shape_[0];z++)
			{
				T vec[3];
				T vecq[3];
				std::complex<T> * K=kernel+z*jumpZ*stride;
				T Z=(T(z-z2))* voxel_size[0];
				vec[0] = Z ;
				vecq[0] = vec[0]*vec[0];

				for (int y=0;y<shape_[1];y++)
				{
					T Y=(T(y-y2))* voxel_size[1];
					vec[1] = Y ;
					vecq[1] = vec[1]*vec[1];
					for (int x=0;x<shape_[2];x++)
					{
						T X=(T(x-x2)) * voxel_size[2];
						vec[2] = X;
						vecq[2] = vec[2]*vec[2];
						T sum = vecq[0]+vecq[1]+vecq[2];
						if (zero_order)
						{
							std::complex<T> & current=(*K);
							current=kf.get(sum);
							current*=norm;
						}
						else
						{
							std::complex<T> & current=K[L+m];
							T length=std::sqrt(sum);
							T theta=0;
							if (length>0)
								theta=std::acos( Z/length);
							T phi=std::atan2( Y, X);
							current=kf.get(sum,-theta,-phi,L,m);
							current*=norm;
						}
						K+=stride;
					}
				}
			}
		} else
		{
#pragma omp parallel for num_threads(get_numCPUs())
			for (int z=0;z<shape_[0];z++)
			{
				T vec[3];
				T vecq[3];
				std::complex<T> * K=kernel+z*jumpZ*stride;

				T Z=(T(z))* voxel_size[0];
				if (z>z2) Z=-(shape[0]-Z);
				vec[0] = Z ;
				vecq[0] = vec[0]*vec[0];

				for (int y=0;y<shape_[1];y++)
				{
					T Y=(T(y))* voxel_size[1];
					if (y>y2) Y=-(shape[1]-Y);
					vec[1] = Y ;
					vecq[1] = vec[1]*vec[1];
					for (int x=0;x<shape_[2];x++)
					{
						T X=(T(x)) * voxel_size[2];
						if (x>x2) X=-(shape[2]-X);
						vec[2] = X;
						vecq[2] = vec[2]*vec[2];
						T sum = vecq[0]+vecq[1]+vecq[2];
						if (zero_order)
						{
							std::complex<T> & current=(*K);
							current=kf.get(sum);
							current*=norm;
						}
						else
						{
							std::complex<T> & current=K[L+m];
							T length=std::sqrt(sum);
							T theta=0;
							if (length>0)
								theta=std::acos( Z/length);
							T phi=std::atan2( Y, X);
							current=kf.get(sum,-theta,-phi,L,m);
							current*=norm;
						}
						K+=stride;
					}
				}
			}
		}
		return 0;
	}



	/*! Gauss Kernel (Solid Harmonics)

	*/
	template<typename T>
	class Gauss: public Kernel<T>
	{
	private:
		std::complex<T> imag;
		T t;
	public:
		std::string getName() {
			return std::string("Gauss");
		};
		STA_CONVOLUTION_KERNELS getID() {
			return STA_CONV_KERNEL_GAUSS;
		}
		bool radialComplex() {
			return false;
		};

		Gauss() : Kernel<T>()
		{
			t=1;
			imag = std::complex<T>(0,1);

		}
		void setSigma(T s) {

			t=s*s;
			if (t==0)
				printf("sigma has been set to 0!!!\n");
		}

		std::complex<T> weight()
		{
			return std::complex<T>(T(1.0)/std::pow(T(t*2*M_PI),T(3.0/2.0)));
		};


		std::complex<T> get(T rsqr, T theta, T phi,int l=0,int m=0)
		{
			T r=std::sqrt(rsqr);
			std::complex<T> tmp=1;
			//tmp*=std::exp(-rsqr/(T(2.0)*t))/std::pow(t,T(l+0.5));
			tmp*=std::exp(-rsqr/(T(2.0)*t));
			if (l>0)
			{
				tmp*=hanalysis::basis_SphericalHarmonicsSemiSchmidt(l,m,(double)theta,(double)phi);
				tmp*=std::pow(r,l);
			}
			return tmp;
		};

		std::complex<T> get(T rsqr)
		{
			std::complex<T> tmp=1;
			tmp*=std::exp(-rsqr/(T(2.0)*t));
			return tmp;
		};

	};


	/*! Gauss-Laguerre Kernel

	*/
	template<typename T>
	class GaussLaguerre: public Kernel<T>
	{
	private:
		int degree;
		T t;
		std::complex<double> imag;

	public:
		std::string getName() {
			return std::string("GaussLaguerre");
		};
		STA_CONVOLUTION_KERNELS getID() {
			return STA_CONV_KERNEL_GAUSS_LAGUERRE;
		}
		bool radialComplex() {
			return false;
		};

		GaussLaguerre() : Kernel<T>()
		{
			imag = std::complex<T>(0,1);
			degree=0;
			t=1;
		}
		void setDegree(int d) {
			degree=d;
		}
		void setSigma(T s) {

			t=s*s;
			if (t==0)
				printf("sigma has been set to 0!!!\n");
		}


		std::complex<T> weight()
		{
			return std::complex<T>(1);
			//return std::complex<double>(std::pow(double(t*2*M_PI),3.0/2.0)/(std::pow((double)2,(double)degree)*gsl_sf_fact(degree)));
			//return std::complex<double>(std::pow(t,l-degree)*gsl_sf_doublefact(l)*std::pow(double(t*2*M_PI),3.0/2.0)/(std::pow((double)2,(double)degree)*gsl_sf_fact(degree)));
		};

		std::complex<T> get(T rsqr, T theta, T phi,int l=0,int m=0)
		{
			T r=std::sqrt(rsqr);
			std::complex<T> tmp=1;
			//tmp*=std::exp(-rsqr/(2.0*t))/std::pow(t,l+0.5);
			tmp*=std::exp(-rsqr/(T(2.0)*t));
			tmp*=gsl_sf_laguerre_n ((double)degree,l+0.5,rsqr/(2.0*t));
			if (l>0)
			{
				tmp*=hanalysis::basis_SphericalHarmonicsSemiSchmidt(l,m,(double)theta,(double)phi);
				tmp*=std::pow(r,l);
			}
			return tmp;
		};

		std::complex<T> get(T rsqr)
		{
			//T r=std::sqrt(rsqr);
			std::complex<T> tmp=1;
			tmp*=std::exp(-rsqr/(T(2.0)*t));
			tmp*=gsl_sf_laguerre_n ((double)degree,0.5,rsqr/(2.0*t));
			return tmp;
		};
	};



	/*! Gauss-Bessel Kernel (Gabor)

	*/
	template<typename T>
	class GaussBessel: public Kernel<T>
	{
	private:
		T freq;
		T s;
		T t;
		T sqrtt;
	public:
		std::string getName() {
			return std::string("GaussBessel");
		};
		STA_CONVOLUTION_KERNELS getID() {
			return STA_CONV_KERNEL_GAUSS_BESSEL;
		}
		bool radialComplex() {
			return false;
		};
		GaussBessel() : Kernel<T>()
		{
			freq=1.0;
			t=1;
			sqrtt=std::sqrt(t);
			s=1;
			// 		scale=1.0;
		}
		void setFreq(T f) {
			freq=f;
		}
		void setSigma(T s) {
			t=s*s;
			sqrtt=s;
			if (t==0)
				printf("sigma has been set to 0!!!\n");
		}
		void setGauss(T s) {
			this->s=s;
			if (t==0)
				printf("sigma has been set to 0!!!\n");
		}

		std::complex<T> weight()
		{
			return 1;
			//std::complex<T>(std::pow(T(t*2*M_PI),T(3.0/2.0)));
		};

		std::complex<T> get(T rsqr, T theta, T phi,int l=0,int m=0)
		{
			T r=std::sqrt(rsqr);
			std::complex<T> tmp=1;
			tmp*=std::exp(-rsqr/(T(2.0)*s*t));

			if (l>0)
			{
				tmp*=hanalysis::basis_SphericalHarmonicsSemiSchmidt(l,m,(double)theta,(double)phi);
				// 			tmp*=std::pow(-1.0,l);
				double tmp2=0;
				for (int i=0;i<=l;i++)
				{
					double fact=gsl_sf_fact(l)/(gsl_sf_fact(i)*gsl_sf_fact(l-i));
					tmp2+=fact*std::pow(1.0/(s*t),l-i)*std::pow(freq/(sqrtt),i)*std::pow(r,l-i)*gsl_sf_bessel_jl (i,freq*r/sqrtt);
				}
				tmp*=tmp2;
			} else
			{
				tmp*=gsl_sf_bessel_j0 (freq*r/sqrtt);
			}
			return tmp;
		};

		std::complex<T> get(T rsqr)
		{
			T r=std::sqrt(rsqr);
			std::complex<T> tmp=1;
			//j_0 = sinx/x
			tmp*=gsl_sf_bessel_j0 (freq*r/sqrtt);
			tmp*=std::exp(-rsqr/(2.0*s*t));
			return tmp;
		};

	};





}

#endif