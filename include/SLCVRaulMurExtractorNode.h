#ifndef SLCVRAULMUREXTRACTORNODE_H
#define SLCVRAULMUREXTRACTORNODE_H

#include <vector>
#include <list>
#include <opencv/cv.h>
#include <SLCVRaulMurExtractorNode.h>

class SLCVRaulMurExtractorNode
{
public:
    SLCVRaulMurExtractorNode():bNoMore(false){}

    void DivideNode(SLCVRaulMurExtractorNode &n1, SLCVRaulMurExtractorNode &n2, SLCVRaulMurExtractorNode &n3, SLCVRaulMurExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<SLCVRaulMurExtractorNode>::iterator lit;
    bool bNoMore;
};
#endif // SLCVRAULMUREXTRACTORNODE_H
