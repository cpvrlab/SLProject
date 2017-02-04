package ch.fhnw.comgr.camera;

import android.graphics.ImageFormat;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CaptureRequest;
import android.media.ImageReader;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import java.util.ArrayList;
import java.util.List;

import ch.fhnw.comgr.CameraService;

@SuppressWarnings("MissingPermission")
public class PreviewCallback {
    private static final String TAG = PreviewCallback.class.getSimpleName();

    private Size mTargetPreviewSize;
    private CaptureRequest.Builder mPreviewRequestBuilder;
    private CameraDevice mCameraDevice;
    private CameraCaptureSession mActiveCaptureSession;
    private Surface surface;
    private CameraService.ImageAvailableListener imageAvailableListener;

    public PreviewCallback(Size[] outputSizes,
                           CameraService.ImageAvailableListener imageAvailableListener,
                           CameraDevice mCameraDevice) throws CameraAccessException
    {
        this.imageAvailableListener = imageAvailableListener;
        this.mCameraDevice = mCameraDevice;

        // TODO: Smaller sizes
        if (outputSizes.length > 0) {
            mTargetPreviewSize = outputSizes[0];
        }

        setupSurface();
    }

    private void setupSurface()
    {
        ImageReader mImageReader = ImageReader.newInstance(mTargetPreviewSize.getWidth(),
                                                           mTargetPreviewSize.getHeight(),
                                                           ImageFormat.RAW_SENSOR, /*maxImages*/ 2);
        mImageReader.setOnImageAvailableListener(imageAvailableListener, null);
        surface = mImageReader.getSurface();
    }


    //Request for a basic preview
    protected CaptureRequest.Builder createPreviewRequestBuilder() throws CameraAccessException
    {
        return getCameraDevice().createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
    }

    //Return all target surfaces for camera frames
    protected List<Surface> getCaptureTargets()
    {
        List<Surface> baseTargets = new ArrayList<>();
        baseTargets.add(surface);
        return baseTargets;
    }

    /*
     * The same builder is used for all repeated requests, and some
     * state is shared between them. The object is lazily created
     * the first time it is needed.
     */
    protected final CaptureRequest.Builder getPreviewRequestBuilder() throws CameraAccessException
    {
        if (mPreviewRequestBuilder == null)
            mPreviewRequestBuilder = createPreviewRequestBuilder();

        return mPreviewRequestBuilder;
    }

    protected final CameraDevice getCameraDevice() {
        return mCameraDevice;
    }

    private void setActiveCaptureSession(CameraCaptureSession session) {
        mActiveCaptureSession = session;
    }

    protected final CameraCaptureSession getActiveCaptureSession() {
        return mActiveCaptureSession;
    }

    public void cancelActiveCaptureSession() {
        if (mActiveCaptureSession != null) {
            mActiveCaptureSession.close();
            mActiveCaptureSession = null;
        }
    }

    //Restart preview with existing camera settings
    public void restartPreview(int effect) throws CameraAccessException {
        final CaptureRequest.Builder builder = getPreviewRequestBuilder();
        builder.set(CaptureRequest.CONTROL_EFFECT_MODE, effect);
        getActiveCaptureSession().setRepeatingRequest(builder.build(), null, null);
    }

    /*
     * Begin streaming preview data.
     */
    public void startPreviewSession() throws CameraAccessException
    {
        //Cancel any active sessions
        cancelActiveCaptureSession();

        //Preview request contains state we need to reset
        // when we start a new preview session.
        mPreviewRequestBuilder = null;

        // Configure the size of default buffer match the camera preview.
        //surface.setDefaultBufferSize(mTargetPreviewSize.getWidth(), mTargetPreviewSize.getHeight());

        // We set up a CaptureRequest.Builder with the output Surface.
        CaptureRequest.Builder builder = getPreviewRequestBuilder();
        builder.addTarget(surface);

        // Here, we create a CameraCaptureSession for camera preview.
        getCameraDevice().createCaptureSession(getCaptureTargets(), new PreviewSessionCallback(builder), null);
    }

    //Callback to react to creation of the preview session
    private class PreviewSessionCallback extends CameraCaptureSession.StateCallback {
        private final CaptureRequest.Builder mBuilder;

        public PreviewSessionCallback(CaptureRequest.Builder builder) {
            mBuilder = builder;
        }

        @Override
        public void onConfigured(CameraCaptureSession captureSession) {
            // The camera is already closed
            if (null == getCameraDevice()) {
                return;
            }

            // When the session is ready, we start displaying the preview.
            setActiveCaptureSession(captureSession);
            try {
                // Finally, we start displaying the camera preview.
                CaptureRequest previewRequest = mBuilder.build();
                getActiveCaptureSession().setRepeatingRequest(previewRequest, null, null);
            } catch (CameraAccessException e) {
                e.printStackTrace();
            }
        }

        @Override
        public void onConfigureFailed(CameraCaptureSession captureSession) {
            Log.w(TAG, "Failed to Create Camera Preview");
        }
    }
}
