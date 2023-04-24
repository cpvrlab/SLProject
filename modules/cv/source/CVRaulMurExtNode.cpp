//#############################################################################
//  File:      CVRaulMurExtNode.h
//  Purpose:   Declares the Raul Mur ORB feature detector and descriptor
//  Source:    This File is based on the ORB Implementation of ORB_SLAM
//             https://github.com/raulmur/ORB_SLAM2
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Marcus Hudritsch, Michael Goettlicher, Pascal Zingg, Timon Tschanz
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#include <CVTypedefs.h>
#include <CVRaulMurExtNode.h>
#include <algorithm> // std::max

//-----------------------------------------------------------------------------
//! Divides the current ExtractorNode into four ExtractorNodes.
//! The Keypoints are also divided between the four ExtractorNodes by space.
void CVRaulMurExtNode::DivideNode(CVRaulMurExtNode& n1,
                                  CVRaulMurExtNode& n2,
                                  CVRaulMurExtNode& n3,
                                  CVRaulMurExtNode& n4)
{
    const int halfX = (int)(ceil(static_cast<float>(UR.x - UL.x) / 2));
    const int halfY = (int)(ceil(static_cast<float>(BR.y - UL.y) / 2));

    // Define boundaries of childs
    n1.UL = UL;
    n1.UR = CVPoint2i(UL.x + halfX, UL.y);
    n1.BL = CVPoint2i(UL.x, UL.y + halfY);
    n1.BR = CVPoint2i(UL.x + halfX, UL.y + halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = CVPoint2i(UR.x, UL.y + halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = CVPoint2i(n1.BR.x, BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    // Associate points to childs
    for (size_t i = 0; i < vKeys.size(); i++)
    {
        const CVKeyPoint& kp = vKeys[i];
        if (kp.pt.x < n1.UR.x)
        {
            if (kp.pt.y < n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if (kp.pt.y < n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    if (n1.vKeys.size() == 1)
        n1.bNoMore = true;
    if (n2.vKeys.size() == 1)
        n2.bNoMore = true;
    if (n3.vKeys.size() == 1)
        n3.bNoMore = true;
    if (n4.vKeys.size() == 1)
        n4.bNoMore = true;
}
//-----------------------------------------------------------------------------
