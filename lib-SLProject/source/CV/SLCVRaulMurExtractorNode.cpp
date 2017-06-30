//#############################################################################
//  File:      SLCVRaulMurExtractorNode.h
//  Author:    Pascal Zingg, Timon Tschanz
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <stdafx.h>
#include <SLCV.h>
#include <SLCVRaulMurExtractorNode.h>

const int PATCH_SIZE = 31;
const int HALF_PATCH_SIZE = 15;
const int EDGE_THRESHOLD = 19;

//-----------------------------------------------------------------------------
//!???
void SLCVRaulMurExtractorNode::DivideNode(SLCVRaulMurExtractorNode &n1, 
                                          SLCVRaulMurExtractorNode &n2, 
                                          SLCVRaulMurExtractorNode &n3, 
                                          SLCVRaulMurExtractorNode &n4)
{
    const int halfX = (int)(ceil(static_cast<float>(UR.x-UL.x)/2));
    const int halfY = (int)(ceil(static_cast<float>(BR.y-UL.y)/2));

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = SLCVPoint2i(UL.x+halfX,UL.y);
    n1.BL = SLCVPoint2i(UL.x,UL.y+halfY);
    n1.BR = SLCVPoint2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = SLCVPoint2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = SLCVPoint2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for(size_t i=0;i<vKeys.size();i++)
    {
        const SLCVKeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;
}
//-----------------------------------------------------------------------------
