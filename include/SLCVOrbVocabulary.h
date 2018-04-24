//#############################################################################
//  File:      SLCVOrbVocabulary.h
//  Author:    Michael Goettlicher
//  Date:      April 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCV_ORBVOCABULARY_H
#define SLCV_ORBVOCABULARY_H

/*!Singleton class used to load, store and delete ORB_SLAM2::ORBVocabulary instance.
*/
class SLCVOrbVocabulary
{
public:
    ~SLCVOrbVocabulary();

    //!get vocabulary
    static ORB_SLAM2::ORBVocabulary* get();
    //!delete vocabulary (free storage)
    static void free();

private:
    static SLCVOrbVocabulary& instance()
    {
        static SLCVOrbVocabulary s_instance;
        return s_instance;
    }

    void loadFromFile();
    void doFree();
    ORB_SLAM2::ORBVocabulary* doGet();

    ORB_SLAM2::ORBVocabulary* _vocabulary = NULL;
};

#endif // !SLCV_ORBVOCABULARY_H
