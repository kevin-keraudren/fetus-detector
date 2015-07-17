/*=========================================================================

Library   : Image Registration Toolkit (IRTK)
Module    : Cpp version of Rotation Invariant Feature filtering
Fast Rotation Invariant 3D Feature Computation utilizing Efficient Local Neighborhood Operators
Henrik Skibbe, M. Reisert, Thorsten Schmidt, Thomas Brox, Olaf Ronneberger, Hans Burkhardt
IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(8): 1563 - 1575 , Aug 2012
Needs: FFTW!
Implements currently only SGD features!!
THIS IS ONLY A SIMPIFIED CPP VERSION OF MY CUDA IMPLEMENTATION!
The output of this filter is 4D for a 3D input! x-dim is feature space!
Copyright : Imperial College, Department of Computing
Visual Information Processing (VIP), 2011 onwards
Date      : 05/07/2013
Version   : 0.001 beta
Changes   : $Author: bkainz $

=========================================================================*/

#include "skibbe_base/stafield.h"
#include "irtkRotationInvariantFeaturesFilter.h"
#include <math.h>
#include <cstddef>
#include <complex>
#include <cmath>
#include <sstream>
#include <cstddef>
#include <vector>
#ifndef _WIN32
#include <unistd.h>
#endif
#define M_PI 3.14159265358979323846
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string>
#include <limits>


template<typename T>
int getLap() {
	return 1;
};
template<>
int getLap<float>() {
	return 0;
};

template <class VoxelType> irtkRotationInvariantFeaturesFilter<VoxelType>::irtkRotationInvariantFeaturesFilter(filterParams params)
{
	_params = params;
}

template <class VoxelType> irtkRotationInvariantFeaturesFilter<VoxelType>::irtkRotationInvariantFeaturesFilter(const char* filename)
{
	this->readParamsFromFile(filename);
}

template <class VoxelType> irtkRotationInvariantFeaturesFilter<VoxelType>::~irtkRotationInvariantFeaturesFilter(void)
{

}


template <class VoxelType> bool irtkRotationInvariantFeaturesFilter<VoxelType>::RequiresBuffering(void)
{
	return false;
}

template <class VoxelType> const char *irtkRotationInvariantFeaturesFilter<VoxelType>::NameOfClass()
{
	return "irtkRotationInvariantFeaturesFilter";
}

//TODO parallel TBB implementation not supported yet
//template <class VoxelType> double 
//	irtkRotationInvariantFeaturesFilter<VoxelType>::Run(int x, int y, int z, int t)
//{

//each voxel transform 
//	return 0.0;
//}

template <class VoxelType> void irtkRotationInvariantFeaturesFilter<VoxelType>::Initialize()
{
	if(this->_input->GetXSize() != this->_input->GetYSize() || 
		this->_input->GetXSize() != this->_input->GetZSize() || 
		this->_input->GetYSize() != this->_input->GetZSize() )
		std::cerr << "WARNING you should resample your image to isotropic voxel size for a proper use of this filter!" << std::endl;

        this->_featureoutput = new irtkGenericImage<double>(this->getFeatureDim(),
                                                            this->_input->GetX(),
                                                            this->_input->GetY(),
                                                            this->_input->GetZ());

}

template <class VoxelType> void irtkRotationInvariantFeaturesFilter<VoxelType>::Run()
{
	//TODO recursive derivatives per element
	//this->irtkImageToImage<VoxelType>::Run();
	this->Initialize();

        transform2featureSpace(this->_input, _featureoutput->GetPointerToVoxels());

	for(int t = 0; t < _featureoutput->GetNumberOfVoxels(); t++)
	{
		_featureoutput->GetPointerToVoxels()[t] = sqrt(_featureoutput->GetPointerToVoxels()[t]);
	}

}

template <class VoxelType> void irtkRotationInvariantFeaturesFilter<VoxelType>::Finalize()
{
	//TODO Finalize -- only 4D double output is supported and required by this filter ImageToImage does not support this
	this->irtkImageToImage<VoxelType>::Finalize();

}

template <class VoxelType> void irtkRotationInvariantFeaturesFilter<VoxelType>::readParamsFromFile(std::string filename)
{
	_params.kparams.clear();

	string line;
    ifstream myfile(filename.c_str());
	if (myfile.is_open())
	{
		int c = 0;
		int i = 0;
		int j = 0;
		std::vector<double> pvec;
		while ( myfile.good() )
		{
			getline (myfile,line);
			if(c==0)
			{
				_params.BW = atof(line.c_str());
				cout << _params.BW << endl;
			}
			if(c==1)
			{
				_params.kname = line;
				cout << _params.kname << endl;
			}
			if(c==2)
			{
				_params.Lap = atof(line.c_str());
				cout << _params.Lap << endl;
			}
			if(c>2)
			{
				pvec.push_back(atof(line.c_str()));
				j++;
				if(j>2)
				{
					_params.kparams.push_back(pvec);
					pvec.clear();
					j=0;
					i++;
				}
			}
			c++;
		}

		myfile.close();

		for(i = 0; i <  _params.kparams.size(); i++)
		{
			cout << _params.kparams[i][0] << endl;
			cout << _params.kparams[i][1] << endl;
			cout << _params.kparams[i][2] << endl;
		}

	}
	else cout << "Unable to open file"; 
}

template <class VoxelType> void irtkRotationInvariantFeaturesFilter<VoxelType>::writeParamsToFile(std::string filename)
{
	std::ofstream fst;
    fst.open(filename.c_str());
	if (fst.is_open())
	{
		fst << _params.BW <<  std::endl << _params.kname <<  std::endl << _params.Lap << std::endl;
		for(int i = 0; i < _params.kparams.size(); i++)
		{
			for(int j = 0; j < _params.kparams[i].size(); j++)
			{
				fst << _params.kparams[i][j] << std::endl;
			}
		}
		fst.close();
	}
	else cout << "Unable to open file";
}

template <class VoxelType> int irtkRotationInvariantFeaturesFilter<VoxelType>::getFeatureDim()
{
	if(_params.kparams.size() == 0)
	{
		std::cout << "ERROR: you must read or set some filter parameter. not filtering!" << std::endl;
		return 0;
	}

	return _params.kparams.size()*_params.BW+_params.kparams.size();
}

template <class VoxelType> void 
	irtkRotationInvariantFeaturesFilter<VoxelType>::transform2featureSpace(irtkGenericImage<VoxelType>* img, double* outdata)
{

    std::size_t* shape = new std::size_t[3];
	shape[0] = img->GetZ();
	shape[1] = img->GetY();
	shape[2] = img->GetX();

	std::complex<double> * data = new std::complex<double>[img->GetNumberOfVoxels()];

        for(int i = 0; i < img->GetNumberOfVoxels(); i++)
	{
		data[i] = std::complex<double>(img->GetPointerToVoxels()[i], 0.0);
	}

	//USE of original (slightly adapted) skibbe implementation
	int num_kernels = _params.kparams.size();
	int featureDim=(_params.Lap+1)*(_params.BW+1)*num_kernels;

	hanalysis::stafield<double> ifield = hanalysis::stafield<double>(shape, 0, 
		hanalysis::STA_FIELD_STORAGE_R, hanalysis::STA_OFIELD_SINGLE, data);

	hanalysis::stafield<double> ifield_ft=ifield.fft(true);

	int current_feat=0;
	int ndims[4];
	ndims[0] = featureDim;
	ndims[1]=ifield.getShape()[2];
	ndims[2]=ifield.getShape()[1];
	ndims[3]=ifield.getShape()[0];

	hanalysis::stafield<double>  convoluion_kernel_ft;
	hanalysis::stafield<double>  convoluion_kernel;

	std::size_t size_bufferA=hanalysis::order2numComponents(ifield.getStorage(),ifield.getType(),(_params.BW+1))*ifield.getNumVoxel();
	std::size_t size_bufferB=hanalysis::order2numComponents(ifield.getStorage(),ifield.getType(),(_params.BW))*ifield.getNumVoxel();
	std::complex<double> * bufferA=NULL;
	std::complex<double> * bufferB=NULL;
	bufferA=new std::complex<double>[size_bufferA];
	if (size_bufferB>0)
		bufferB=new std::complex<double>[size_bufferB];
	bool largeBufferA=true;


	convoluion_kernel=hanalysis::stafield<double>(
		ifield.getShape(),
		0,
		ifield.getStorage(),
		hanalysis::STA_OFIELD_SINGLE);
	convoluion_kernel_ft=hanalysis::stafield<double>(
		ifield.getShape(),
		0,
		ifield.getStorage(),
		hanalysis::STA_OFIELD_SINGLE);
	convoluion_kernel_ft.switchFourierFlag();

	for (int k=0;k<num_kernels;k++)
	{
		double t=1;
		double normfact=1;
		if (_params.kname=="gauss")
		{
			t=_params.kparams[k][0]*_params.kparams[k][0]; // t=sigma^2
			normfact=std::pow(2*t*M_PI,3.0/2.0);
		}
		if (_params.kname=="gaussBessel")
		{
			t=_params.kparams[k][0]*_params.kparams[k][0]; // t=sigma^2
			normfact=std::pow(2*t*M_PI,3.0/2.0);
		}

		printf("(I*k) = ");

		hanalysis::stafield<double>::makeKernel(
			_params.kname,
			_params.kparams[k],
			convoluion_kernel);

		hanalysis::stafield<double>::FFT(
			convoluion_kernel,
			convoluion_kernel_ft,
			true,
			false,
			(double)(1.0)/(double)(ifield.getNumVoxel()));

		hanalysis::stafield<double> viewA=hanalysis::stafield<double>(
			ifield.getShape(),
			0,
			ifield.getStorage(),
			hanalysis::STA_OFIELD_SINGLE,
			bufferA);
		viewA.switchFourierFlag();

		hanalysis::stafield<double>::Prod(
			convoluion_kernel_ft,
			ifield_ft,
			viewA,
			0,
			true,1,true);

		hanalysis::stafield<double>::FFT(
			viewA,
			convoluion_kernel,
			false);


		hanalysis::stafield<double> * lap_imageA=&convoluion_kernel;
		convoluion_kernel_ft.switchFourierFlag();
		hanalysis::stafield<double> * lap_imageB=&convoluion_kernel_ft;

		for (int lap=0;lap<=_params.Lap;lap++)
		{
			if (lap>0)
			{
				if (_params.kname=="gauss")
					normfact*=(double)std::max(2*lap-1,1)/(double)(2*lap);
				printf("        ");
				hanalysis::stafield<double>::Lap(*lap_imageA,*lap_imageB,t/(lap*2),true,getLap<double>());
				std::swap(lap_imageA,lap_imageB);
			}

			if (_params.BW>0)
				printf("a(%d,%d)->",lap,lap);
			else
				printf("a(%d,%d)\n",lap,lap);            

			hanalysis::sta_feature_product_R<double,double>(
				lap_imageA->getDataConst(),
				lap_imageA->getDataConst(),
				outdata+(current_feat++),
				lap_imageA->getShape(),
				lap_imageA->getRank(),
				1/std::sqrt(normfact),
				false,
				lap_imageA->getStride(),
				lap_imageA->getStride(),
				featureDim,
				true);

			double normfact_deriv=normfact;


			if (((_params.BW%2==0)&&(largeBufferA))||(((_params.BW%2==1)&&(!largeBufferA))))
			{
				std::swap(bufferA,bufferB);
				largeBufferA=!largeBufferA;
			}

			for (int l=0;l<_params.BW;l++)
			{
				if (l<_params.BW-1)
					printf("a(%d,%d)->",lap,lap+l+1);
				else
					printf("a(%d,%d)\n",lap,lap+l+1);

				if (_params.kname=="gauss")
					normfact_deriv*=std::max(2*l-1,1)*t;
				if (_params.kname=="gaussBessel")
					normfact_deriv*=t;

				hanalysis::stafield<double> viewA;
				if (l==0)
					viewA=lap_imageA->get(0);
				else
					viewA=hanalysis::stafield<double>(
					ifield.getShape(),
					l,
					ifield.getStorage(),
					hanalysis::STA_OFIELD_SINGLE,
					bufferA).get(l);

				hanalysis::stafield<double> viewB=hanalysis::stafield<double>(
					ifield.getShape(),
					l+1,
					ifield.getStorage(),
					hanalysis::STA_OFIELD_SINGLE,
					bufferB);

				hanalysis::stafield<double>::Deriv(viewA,viewB,1,true,t,true);
				std::swap(bufferA,bufferB);
				largeBufferA=!largeBufferA;

				hanalysis::sta_feature_product_R<double,double>(
					viewB.getDataConst(),
					viewB.getDataConst(),
					outdata+(current_feat++),
					viewB.getShape(),
					viewB.getRank(),
					1/std::sqrt(normfact_deriv),
					false,
					viewB.getStride(),
					viewB.getStride(),
					featureDim,
					true);
			}
			if (lap<_params.Lap)
			{
				printf("        ");
				printf("|\n");
				printf("        ");
				printf("V\n");
			} else
				printf("\n");

		}
	}

	if (bufferA!=NULL) delete [] bufferA;
	if (bufferB!=NULL) delete [] bufferB;

}

template class irtkRotationInvariantFeaturesFilter<char>;
template class irtkRotationInvariantFeaturesFilter<unsigned char>;
template class irtkRotationInvariantFeaturesFilter<short>;
template class irtkRotationInvariantFeaturesFilter<unsigned short>;
template class irtkRotationInvariantFeaturesFilter<int>;
template class irtkRotationInvariantFeaturesFilter<float>;
template class irtkRotationInvariantFeaturesFilter<double>;
