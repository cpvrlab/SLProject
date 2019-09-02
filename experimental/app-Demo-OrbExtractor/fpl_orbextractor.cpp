#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>

#include "wai_orb.cpp"

void cv_extractAndDrawKeyPoints(OrbExtractionState*    state,
                                cv::Mat                image,
                                std::vector<KeyPoint>& keyPoints,
                                u8**                   descriptors)
{
    cv::Mat grayscaleImg = cv::Mat(image.rows,
                                   image.cols,
                                   CV_8UC1);

    int from_to[] = {0, 0};
    cv::mixChannels(&image, 1, &grayscaleImg, 1, from_to, 1);

    FrameBuffer grayscaleBuffer;
    grayscaleBuffer.memory        = grayscaleImg.data;
    grayscaleBuffer.width         = grayscaleImg.cols;
    grayscaleBuffer.height        = grayscaleImg.rows;
    grayscaleBuffer.bytesPerPixel = 1;
    grayscaleBuffer.pitch         = (i32)grayscaleImg.step;

    keyPoints = detectFastCorners(state,
                                  &grayscaleBuffer,
                                  20,
                                  ORB_EDGE_THRESHOLD);

    const cv::Point*       pattern0 = (cv::Point*)bit_pattern_31_;
    std::vector<cv::Point> pattern;
    i32                    pointCount = ORB_PATTERN_VALUE_COUNT / 2;
    std::copy(pattern0, pattern0 + pointCount, std::back_inserter(pattern));

    *descriptors = (u8*)malloc(ORB_DESCRIPTOR_COUNT * keyPoints.size() * sizeof(u8));

    u8* keyPointDescriptor = *descriptors;
    for (KeyPoint keyPoint : keyPoints)
    {
        computeOrbDescriptor(&grayscaleBuffer,
                             &keyPoint,
                             bit_pattern_31_,
                             keyPointDescriptor);

        keyPointDescriptor += ORB_DESCRIPTOR_COUNT;
    }

    for (KeyPoint keyPoint : keyPoints)
    {
        cv::rectangle(image,
                      cv::Rect((int)keyPoint.x - 3, (int)keyPoint.y - 3, 7, 7),
                      cv::Scalar(0, 0, 255));
    }
}

int main(int argc, char** argv)
{
    cv::Mat image1, image2;
    image1 = cv::imread("/home/jdellsperger/projects/WAI/data/images/textures/Lena.tiff", CV_LOAD_IMAGE_COLOR);
    image2 = cv::imread("/home/jdellsperger/projects/WAI/data/images/textures/Lena_s.tiff", CV_LOAD_IMAGE_COLOR);

    if (!image1.data || !image2.data)
    {
        printf("Could not load image.\n");
        return -1;
    }

    OrbExtractionState state = {};

    // This is for orientation
    // pre-compute the end of a row in a circular patch
    int          v, v0;
    int          vmax = cvFloor(ORB_HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int          vmin = cvCeil(ORB_HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2  = ORB_HALF_PATCH_SIZE * ORB_HALF_PATCH_SIZE;
    for (v = 0; v <= vmax; v++)
        state.umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    for (v = ORB_HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (state.umax[v0] == state.umax[v0 + 1])
        {
            v0++;
        }

        state.umax[v] = v0;
        v0++;
    }

    std::vector<KeyPoint> keyPointsImage1;
    std::vector<KeyPoint> keyPointsImage2;
    u8*                   descriptorsImage1 = nullptr;
    u8*                   descriptorsImage2 = nullptr;
    cv_extractAndDrawKeyPoints(&state, image1, keyPointsImage1, &descriptorsImage1);
    cv_extractAndDrawKeyPoints(&state, image2, keyPointsImage2, &descriptorsImage2);

    cv::Mat concatenatedImage;
    cv::hconcat(image1, image2, concatenatedImage);

    for (int a = 0; a < keyPointsImage1.size(); a++)
    {
        int minDist      = INT_MAX;
        int minDistIndex = -1;
        for (int b = 0; b < keyPointsImage2.size(); b++)
        {
            int dist = descriptorDistance(&descriptorsImage1[a], &descriptorsImage2[b]);

            if (dist < minDist)
            {
                minDist      = dist;
                minDistIndex = b;
            }
        }

        cv::line(concatenatedImage,
                 cv::Point((int)keyPointsImage1[a].x, (int)keyPointsImage1[a].y),
                 cv::Point((int)keyPointsImage2[minDistIndex].x + image2.cols, (int)keyPointsImage2[minDistIndex].y),
                 cv::Scalar(255, 0, 0));
    }

    cv::namedWindow("orbextractor", CV_WINDOW_AUTOSIZE);

    cv::imshow("orbextractor", concatenatedImage);

    cv::waitKey(0);

    return 0;
}
