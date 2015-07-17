/*=========================================================================

  Library   : Image Registration Toolkit (IRTK)
  Module    : Cpp version of Rotation Invariant Feature filtering
			  Fast Rotation Invariant 3D Feature Computation utilizing Efficient Local Neighborhood Operators
			  Henrik Skibbe, M. Reisert, Thorsten Schmidt, Thomas Brox, Olaf Ronneberger, Hans Burkhardt
			  IEEE Transactions on Pattern Analysis and Machine Intelligence, 34(8): 1563 - 1575 , Aug 2012
			  Needs: FFTW!
			  Implements currently only SGD features!!
  Copyright : Imperial College, Department of Computing
              Visual Information Processing (VIP), 2011 onwards
  Date      : 05/07/2013
  Version   : 0.001 beta
  Changes   : $Author: bkainz $

=========================================================================*/

#ifndef _irtkRotationInvariantFeaturesFilter_H
#define _irtkRotationInvariantFeaturesFilter_H


#include <iostream>
#include <stdlib.h>

#include <irtkImage.h>
#include <irtkImageToImage.h>
#include "fftw3.h"

#include <vector>
using namespace std;


/*
Transforming the image to STA rotation invariant feature space
similar to:
Skibbe, H.; Reisert, M.; Schmidt, T.; Brox, T.; Ronneberger, O.; 
Burkhardt, H., "Fast Rotation Invariant 3D Feature Computation 
Utilizing Efficient Local Neighborhood Operators," Pattern Analysis and 
Machine Intelligence, IEEE Transactions on , vol.34, no.8, pp.1563,1575, 
Aug. 2012
*/

template <class VoxelType> 
class irtkRotationInvariantFeaturesFilter : public irtkImageToImage<VoxelType>
{

protected:
	irtkGenericImage<double> *_featureoutput;

	  /** Run This method is protected and should only
  *  be called from within public member function Run().
  */
	//TODO parallel TBB implementation not supported yet
   //virtual double Run(int, int, int, int);

public:
	struct filterParams {   
		filterParams() : BW(5), Lap(0), kname("gaussBessel"){};
		int BW;
		int Lap;
		std::string kname;
		std::vector< std::vector<double> > kparams;
	};

	//constructor using params vector
	irtkRotationInvariantFeaturesFilter(filterParams params);
	//constructor loading parameter from file
	irtkRotationInvariantFeaturesFilter(const char* filename);
	~irtkRotationInvariantFeaturesFilter();

	/// Initialize the filter
	virtual void Initialize();

	/// Finalize the filter
	virtual void Finalize();

	/// Run
	virtual void Run();

	//void setInput(irtkCUGenericImage<VoxelType>* _input);
	//void setParams(filterParams params);

	//TODO a float output should be also possible
	//irtkGenericImage<double>* getFeatureImage();
	void readParamsFromFile(std::string filename);
	void writeParamsToFile(std::string filename);
	filterParams getParams(){return _params;};
	int getFeatureDim();

	//SetMacro(input, irtkGenericImage<irtkGreyPixel>*);
	SetMacro(featureoutput, irtkGenericImage<double>*);
	GetMacro(featureoutput, irtkGenericImage<double>*);

	/// Returns the name of the class
	virtual const char *NameOfClass();

	/// Returns whether the filter requires buffering
	virtual bool RequiresBuffering();

	//in this case we need to return an output image of different dimensions
	//irtkGenericImage<double>* GetFeatureOutput(){return _featureoutput;};


private:
	filterParams _params;

	void transform2featureSpace(irtkGenericImage<VoxelType>* img, double* outdata);
};


#endif
