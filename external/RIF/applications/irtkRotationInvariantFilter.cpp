#include <iostream>
#include <stdlib.h>

#include <irtkImage.h>
#include <limits>

#include <irtkResampling.h>

#include <irtkRotationInvariantFeaturesFilter.h>


int main(int argc, char **argv)
{

	if(argc < 3) 
		std::cout << "usage: rotationInvariantFilter <inputImage.nii> <outputImage.nii> [param_file.txt <optional>]" << std::endl;

	irtkGenericImage<irtkGreyPixel>* image = new irtkGenericImage<irtkGreyPixel>(argv[1]);
	irtkRotationInvariantFeaturesFilter<irtkGreyPixel>* testRIFFilter;

	if(argc < 4)
	{
		std::cout << "using standard foetal brain paramters." << std::endl;

		irtkRotationInvariantFeaturesFilter<irtkGreyPixel>::filterParams params; 
		std::vector<double> p1(3);
		p1[0] = 4.0;
		p1[1] = 0.0;
		p1[2] = 0.5;
		params.kparams.push_back(p1);
		std::vector<double> p2(3);
		p2[0] = 4.0;
		p2[1] = M_PI;
		p2[2] = 0.5;
		params.kparams.push_back(p2);
		std::vector<double> p3(3);
		p3[0] = 4.0;
		p3[1] = 2*M_PI;
		p3[2] = 0.5;
		params.kparams.push_back(p3);
		params.BW = 20;
		params.Lap = 0;
		params.kname = std::string("gaussBessel");
		testRIFFilter = new irtkRotationInvariantFeaturesFilter<irtkGreyPixel>(params);
	}
	else
	{
		testRIFFilter = new irtkRotationInvariantFeaturesFilter<irtkGreyPixel>(argv[3]);
	}

	testRIFFilter->SetInput(image);
	testRIFFilter->SetOutput(image); //This is just to go in line with the ImageToImage class
	testRIFFilter->Run();

	irtkGenericImage<double>* output = testRIFFilter->Getfeatureoutput();
	std::cout << "output size " << output->GetX() << " " << output->GetY() << " " << output->GetZ() << " " << output->GetT() << std::endl;

	output->Write(argv[2]);
	//testRIFFilter->writeParamsToFile("irtkRotationInvariantFilter_foetal_brain_params.txt");

	return EXIT_SUCCESS;
}
