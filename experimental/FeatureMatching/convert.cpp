#include "convert.h"

cv::Mat rgb_to_grayscale(cv::Mat &img)
{
    int from_to[] = {0, 0};
    cv::Mat img_gray = cv::Mat(img.rows, img.cols, CV_8UC1);
    cv::mixChannels(&img, 1, &img_gray, 1, from_to, 1);
    return img_gray;
}


std::vector<cv::Mat> rgb_to_luv(const cv::Mat &input_color_image)
{
    std::vector<cv::Mat> luvImage(3);
    for (int idxC = 0; idxC < 3; ++idxC) {
        luvImage[idxC].create(input_color_image.rows, input_color_image.cols, CV_32F);
    }

    //init
    const float y0 =(float) ((6.0/29)*(6.0/29)*(6.0/29));
    const float a = (float) ((29.0/3)*(29.0/3)*(29.0/3));
    const double XYZ[3][3] = {  {  0.430574,  0.341550,  0.178325 },
                                {  0.222015,  0.706655,  0.071330 },
                                {  0.020183,  0.129553,  0.939180 }   };

    const double Un_prime = 0.197833;
    const double Vn_prime = 0.468331;
    const double maxi     = 1.0/270;
    const double minu     = -88*maxi;
    const double minv     = -134*maxi;
    const double Lt       = 0.008856;
    static float lTable[1064];
    for(int i=0; i<1025; i++)
    {
        float y = (float) (i/1024.0);
        float l = y>y0 ? 116*(float)pow((double)y,1.0/3.0)-16 : y*a;
        lTable[i] = l*maxi;
    }

    cv::Mat in(input_color_image);

    cv::Mat out1(luvImage[0]);
    cv::Mat out2(luvImage[1]);
    cv::Mat out3(luvImage[2]);

    for (int i = 0; i < in.rows; i++)
    {
        uchar* pixelin = in.ptr<uchar>(i);  // point to first color in row
        float* pixelout1 = out1.ptr<float>(i);  // point to first color in row
        float* pixelout2 = out2.ptr<float>(i);  // point to first color in row
        float* pixelout3 = out3.ptr<float>(i);  // point to first color in row
        for (int j = 0; j < in.cols; j++)//row
        {
            //cv::Vec3b rgb = in.at<cv::Vec3b>(j,i);
            float b = *pixelin++ / 255.0f;
            float g = *pixelin++ / 255.0f;
            float r = *pixelin++ / 255.0f;

            //RGB to LUV conversion

            //delcare variables
            float  x, y, z, u_prime, v_prime, constant, L, u, v;

            //convert RGB to XYZ...
            x       = XYZ[0][0]*r + XYZ[0][1]*g + XYZ[0][2]*b;
            y       = XYZ[1][0]*r + XYZ[1][1]*g + XYZ[1][2]*b;
            z       = XYZ[2][0]*r + XYZ[2][1]*g + XYZ[2][2]*b;

            //convert XYZ to LUV...

            //compute ltable(y*1024)
            L = lTable[(int)(y*1024)];

            //compute u_prime and v_prime
            constant    = 1/(x + 15 * y + 3 * z + 1e-35);   //=z

            u_prime = (4 * x) * constant;   //4*x*z
            v_prime = (9 * y) * constant;

            //compute u* and v*
            u = (float) (13 * L * (u_prime - Un_prime)) - minu;
            v = (float) (13 * L * (v_prime - Vn_prime)) - minv;

            *pixelout1++ = L*270*2.55;
            *pixelout2++ = ((u*270-88)+ 134.0)* 255.0 / 354.0;
            *pixelout3++ = ((v*270-134)+ 140.0)* 255.0 / 256.0;
        }
    }

    return luvImage;
}

