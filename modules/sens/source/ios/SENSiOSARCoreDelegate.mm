#import "SENSiOSARCoreDelegate.h"

@interface SENSiOSARCoreDelegate () {

@private
    ARSession*       _arSession;
    ARConfiguration* _arConfig;
}

@end

@implementation SENSiOSARCoreDelegate

- (id)init
{
    //Initialize the parent class(es) up the hierarchy and create self:
    self = [super init];

    //Initialize members (not necessary with ARC)
    _arSession = nil;
    _arConfig  = nil;
    if (self)
    {
        [self initARKit];
    }

    return self;
}

- (void)initARKit
{
    if (ARWorldTrackingConfiguration.isSupported)
    {
        // Create an ARSession
        _arSession          = [ARSession new];
        _arSession.delegate = self;

        _arConfig = [ARWorldTrackingConfiguration new];

        for (int i = 0; i < ARWorldTrackingConfiguration.supportedVideoFormats.count; ++i)
        {
            CGSize s = ARWorldTrackingConfiguration.supportedVideoFormats[i].imageResolution;
            NSLog(NSStringFromCGSize(s));
        }

        _arConfig.videoFormat = ARWorldTrackingConfiguration.supportedVideoFormats.lastObject;
    }
}

- (BOOL)isAvailable
{
    return ARWorldTrackingConfiguration.isSupported;
}

- (BOOL)run
{
    if (ARWorldTrackingConfiguration.isSupported && _arSession)
    {
        [_arSession runWithConfiguration:_arConfig];
        return YES;
    }
    else
        return NO;
}

- (void)pause
{
    if (_arSession)
    {
        [_arSession pause];
    }
}

- (BOOL)reset
{
    if (ARWorldTrackingConfiguration.isSupported && _arSession)
    {
        [_arSession runWithConfiguration:_arConfig options:ARSessionRunOptionResetTracking | ARSessionRunOptionRemoveExistingAnchors];
        return YES;
    }
    else
        return NO;
}

- (void)latestFrame:(cv::Mat*)pose withImg:(cv::Mat*)imgBGR AndIntrinsic:(cv::Mat*)intrinsic AndImgWidth:(int*)w AndImgHeight:(int*)h IsTracking:(BOOL*)isTracking
{
    //Reference the current ARFrame (I think the referenced "currentFrame" may change during this function call)
    ARFrame*  frame  = _arSession.currentFrame;
    ARCamera* camera = frame.camera;

    //copy camera pose
    *pose = cv::Mat_<float>(4, 4);
    for (int i = 0; i < 4; ++i)
    {
        simd_float4 col       = camera.transform.columns[i];
        pose->at<float>(0, i) = (float)col[0];
        pose->at<float>(1, i) = (float)col[1];
        pose->at<float>(2, i) = (float)col[2];
        pose->at<float>(3, i) = (float)col[3];
    }

    //copy intrinsic
    *intrinsic = cv::Mat_<double>(3, 3);
    for (int i = 0; i < 3; ++i)
    {
        simd_float3 col             = camera.intrinsics.columns[i];
        intrinsic->at<double>(0, i) = (double)col[0];
        intrinsic->at<double>(1, i) = (double)col[1];
        intrinsic->at<double>(2, i) = (double)col[2];
    }

    if (frame.camera.trackingState == ARTrackingStateNormal)
        *isTracking = YES;
    else
        *isTracking = NO;

    //copy the image as in camera
    CVImageBufferRef pixelBuffer = frame.capturedImage;

    CVReturn ret         = CVPixelBufferLockBaseAddress(pixelBuffer, 0);
    OSType   pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer);
    //This is NV12, so the order is U/V (NV12: YYYYUV NV21: YYYYVU)
    if (ret == kCVReturnSuccess && pixelFormat == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange)
    {
        size_t imgWidth  = CVPixelBufferGetWidth(pixelBuffer);
        size_t imgHeight = CVPixelBufferGetHeight(pixelBuffer);
        *w               = (int)imgWidth;
        *h               = (int)imgHeight;

        uint8_t* yPlane = (uint8_t*)CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0);

        cv::Mat yuvImg((int)imgHeight + ((int)imgHeight / 2), (int)imgWidth, CV_8UC1, yPlane);
        cv::cvtColor(yuvImg, *imgBGR, cv::COLOR_YUV2BGR_NV12, 3);
    }
    CVPixelBufferUnlockBaseAddress(pixelBuffer, 0);
}

#pragma mark - ARSessionDelegate

- (void)session:(ARSession*)session didFailWithError:(NSError*)error
{
    // Present an error message to the user
}

- (void)session:(ARSession*)session cameraDidChangeTrackingState:(ARCamera*)camera
{
    /*
    switch (camera.trackingState)
    {
        case ARTrackingStateNormal:
            NSLog(@"Tracking is Normal.\n");
            break;
        case ARTrackingStateLimited:
            NSLog(@"Tracking is limited: ");
            switch (camera.trackingStateReason)
            {
                case ARTrackingStateReasonNone:
                    NSLog(@"Tracking is not limited.\n");
                    break;
                case ARTrackingStateReasonInitializing:
                    NSLog(@"Tracking is limited due to initialization in progress.\n");
                    break;
                case ARTrackingStateReasonExcessiveMotion:
                    NSLog(@"Tracking is limited due to a excessive motion of the camera.\n");
                    break;
                case ARTrackingStateReasonInsufficientFeatures:
                    NSLog(@"Tracking is limited due to a lack of features visible to the camera.\n");
                    break;
                case ARTrackingStateReasonRelocalizing:
                    NSLog(@"Tracking is limited due to a relocalization in progress.\n");
                    break;
                default:
                    break;
            }
            break;
        case ARTrackingStateNotAvailable:
            NSLog(@"Tracking is not available.\n");
            break;
        default:
            break;
    }
     */
}

- (void)sessionWasInterrupted:(ARSession*)session
{
    // Inform the user that the session has been interrupted, for example, by presenting an overlay
}

- (void)sessionInterruptionEnded:(ARSession*)session
{
    // Reset tracking and/or remove existing anchors if consistent tracking is required
}

@end