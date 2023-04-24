//#############################################################################
//  File:      CVRaulMurExtNode.h
//  Purpose:   Declares the Raul Mur ORB feature detector and descriptor
//  Source:    This File is based on the ORB Implementation of ORB_SLAM
//             https://github.com/raulmur/ORB_SLAM2
//  Date:      Spring 2017
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/SLProject-Coding-Style
//  Authors:   Pascal Zingg, Timon Tschanz, Michael Goettlicher, Marcus Hudritsch
//             This softwareis provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef CVRAULMUREXTNODE_H
#define CVRAULMUREXTNODE_H

#include <CVTypedefs.h>

using std::list;

//-----------------------------------------------------------------------------
//! Data structure used to subdivide the Image with key points into segments.
class CVRaulMurExtNode
{
public:
    CVRaulMurExtNode() : bNoMore(false) {}

    void DivideNode(CVRaulMurExtNode& n1,
                    CVRaulMurExtNode& n2,
                    CVRaulMurExtNode& n3,
                    CVRaulMurExtNode& n4);

    CVVKeyPoint                      vKeys;
    CVPoint2i                        UL, UR, BL, BR;
    list<CVRaulMurExtNode>::iterator lit;
    bool                             bNoMore;
};
//-----------------------------------------------------------------------------
#endif // CVRAULMUREXTRACTORNODE_H
