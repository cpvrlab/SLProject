//#############################################################################
//  File:      SLCVRaulMurExtractorNode.h
//  Author:    Pascal Zingg, Timon Tschanz
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVRAULMUREXTRACTORNODE_H
#define SLCVRAULMUREXTRACTORNODE_H

#include <SLCV.h>
#include <SLCVRaulMurExtractorNode.h>

//-----------------------------------------------------------------------------
//!Datastructure used to subdivide the Image with keypoints into segments.

class SLCVRaulMurExtractorNode
{
public:
    SLCVRaulMurExtractorNode():bNoMore(false){}

    void DivideNode(SLCVRaulMurExtractorNode &n1, SLCVRaulMurExtractorNode &n2, SLCVRaulMurExtractorNode &n3, SLCVRaulMurExtractorNode &n4);

    SLCVVKeyPoint   vKeys;
    SLCVPoint2i     UL, UR, BL, BR;
    std::list<SLCVRaulMurExtractorNode>::iterator lit;
    bool bNoMore;
};
//-----------------------------------------------------------------------------
#endif // SLCVRAULMUREXTRACTORNODE_H
