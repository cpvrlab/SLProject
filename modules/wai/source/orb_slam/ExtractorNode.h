#ifndef EXTRACTOR_NODE_H 
#define EXTRACTOR_NODE_H

#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include <WAIHelper.h>
#include <orb_slam/KPextractor.h>


namespace ORB_SLAM2
{
class ExtractorNode
{
    public:
    ExtractorNode() : bNoMore(false) {}

    void DivideNode(ExtractorNode& n1, ExtractorNode& n2, ExtractorNode& n3, ExtractorNode& n4);

    std::vector<cv::KeyPoint>          vKeys;
    cv::Point2i                        UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool                               bNoMore;
};

}

#endif
