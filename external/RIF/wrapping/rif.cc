#include <rif.h>

void extractRIF( short* img_in,
                 double* pixelSize,
                 double* xAxis,
                 double* yAxis,
                 double* zAxis,
                 double* origin,
                 int* dim,
                 double* img_out,
                 int* dim_out ) {

    irtkGenericImage<short> irtk_image;
    py2irtk<short>( irtk_image,
                    img_in,
                    pixelSize,
                    xAxis,
                    yAxis,
                    zAxis,
                    origin,
                    dim );
           
    irtkRotationInvariantFeaturesFilter<short>::filterParams params;
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
    irtkRotationInvariantFeaturesFilter<short> testRIFFilter(params);
    
    testRIFFilter.SetInput(&irtk_image);
    testRIFFilter.Run();

    irtkGenericImage<double>* output = testRIFFilter.Getfeatureoutput();
    dim_out[0] = output->GetX();
    dim_out[1] = output->GetY();
    dim_out[2] = output->GetZ();
    dim_out[3] = output->GetT();
    
    irtk2py<double>( *output,
                     img_out,
                     pixelSize,
                     xAxis,
                     yAxis,
                     zAxis,
                     origin,
                     dim_out ); 
}
