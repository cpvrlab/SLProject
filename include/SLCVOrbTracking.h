//#############################################################################
//  File:      SLCVOrbTracking.h
//  Author:    Jan Dellsperger
//  Date:      Apr 2018
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Michael Goettlicher
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

#ifndef SLCVORBTRACKING_H
#define SLCVORBTRACKING_H

#include <condition_variable>
#include <SLCVStateEstimator.h>
#include <SLCVMapTracking.h>
#include <SLCVCalibration.h>

class SLCVOrbTracking : public SLCVMapTracking
{
public:
  string getPrintableState() {
    switch (mState)
      {
      case SYSTEM_NOT_READY:
	return "SYSTEM_NOT_READY";
      case NO_IMAGES_YET:
	return "NO_IMAGES_YET";
      case NOT_INITIALIZED:
	return "NOT_INITIALIZED";
      case OK:
	if (!mbVO) {
	  if (!mVelocity.empty())
	    return "OK_MM"; //motion model tracking
	  else
	    return "OK_RF"; //reference frame tracking
	}
	else {
	  return "OK_VO";
	}
	return "OK";
      case LOST:
	return "LOST";

	return "";
      }
    }
  
  SLCVOrbTracking(SLCVStateEstimator* stateEstimator,
		  SLCVKeyFrameDB* keyFrameDB,
		  SLCVMap* map,
		  SLCVMapNode* mapNode,
		  ORBVocabulary* vocabulary,
		  bool serial = false);
  SLCVOrbTracking(SLCVStateEstimator* stateEstimator,
		  SLCVMapNode* mapNode,
		  ORBVocabulary* vocabulary,
		  bool serial = false);
  ~SLCVOrbTracking();

  void calib(SLCVCalibration* calib);
  bool serial();

  void trackOrbs();
  
protected:
  //Motion Model
  cv::Mat mVelocity;

  // In case of performing only localization, this flag is true when there are no matches to
  // points in the map. Still tracking will continue if there are enough matches with temporal points.
  // In that case we are doing visual odometry. The system will try to do relocalization to recover
  // "zero-drift" localization to the map.
  bool mbVO = false;
  
  bool Relocalization();
  bool TrackWithMotionModel();
  bool TrackLocalMap();
  void SearchLocalPoints();
  bool TrackReferenceKeyFrame();
  void UpdateLastFrame();

  void UpdateLocalMap();
  void UpdateLocalPoints();
  void UpdateLocalKeyFrames();
  
private:
  SLCVStateEstimator* _stateEstimator;
  bool _running = true;
  bool _serial = false;
  SLCVCalibration* _calib = nullptr;
  //Last Frame, KeyFrame and Relocalisation Info
  unsigned int mnLastRelocFrameId = 0;

  // Lists used to recover the full camera trajectory at the end of the execution.
  // Basically we store the reference keyframe for each frame and its relative transformation
  list<cv::Mat> mlRelativeFramePoses;
  list<SLCVKeyFrame*> mlpReferences;
  list<double> mlFrameTimes;
  list<bool> mlbLost;

  //New KeyFrame rules (according to fps)
  // Max/Min Frames to insert keyframes and to check relocalisation
  int mMinFrames = 0;
  int mMaxFrames = 30; //= fps

  // ORB vocabulary used for place recognition and feature matching.
  ORBVocabulary* mpVocabulary;
  //extractor instance
  ORB_SLAM2::ORBextractor* _extractor = NULL;
  
  std::mutex _runLock;
  std::mutex _calibLock;
  std::condition_variable _calibReady;
  std::thread _trackingThread;
  
  void trackOrbsContinuously();
  bool running();
  void running(bool running);
};

#endif
