//#############################################################################
//  File:      GLES3View.java
//  Author:    Marcus Hudritsch, Zingg Pascal
//  Date:      Spring 2017
//  Purpose:   Android Java toplevel windows interface into the SLProject demo
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Zingg Pascal
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

package ch.cpvr.wai;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class GLES3View extends GLSurfaceView
{
    private static String TAG = "WAIApp";
    private static final boolean DEBUG = false;
    private static final int VT_NONE = 0;
    private static final int VT_MAIN = 1;
    private static final int VT_SCND = 2;
    private static final int VT_FILE = 3;

    public GLES3View(Context context)
    {
        super(context);

        setEGLConfigChooser(8, 8, 8, 0, 16, 0);
        //More detailed: Configer context with ConfigChooser class
        //setEGLConfigChooser(new ConfigChooser(8, 8, 8, 0, 16, 0));

        setEGLContextClientVersion(3);
        //More detailed: Creatext context with ContextFactory class
        //setEGLContextFactory(new ContextFactory());

        // Set the renderer responsible for frame rendering
        setRenderer(new Renderer());

        // From Android r15
        setPreserveEGLContextOnPause(true);

        // Render only when needed. Without this it would render continuously with lots of power consumption
        setRenderMode(RENDERMODE_WHEN_DIRTY);
    }

    /**
     * The renderer implements the major callback for the OpenGL ES rendering:
     * - onSurfaceCreated calls SLProjects onInit
     * - onSurfaceChanged calls SLProjects onResize
     * - onDrawFrame      calls SLProjects onUpdateAndPaint
     * Be aware that the renderer runs in a separate thread. Calling something in the
     * activity cross thread invocations.
     */
    private static class Renderer implements GLSurfaceView.Renderer {
        protected Handler mainLoop;
        int _w, _h;
        boolean _initialized = false;

        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            Log.i(TAG, "Renderer.onSurfaceCreated");
            _w = GLES3Lib.view.getWidth();
            _h = GLES3Lib.view.getHeight();

            // Get main event handler of UI thread
            mainLoop = new Handler(Looper.getMainLooper());
        }

        public void onSurfaceChanged(GL10 gl, int width, int height) {
            Log.i(TAG, "Renderer.onSurfaceChanged");
            GLES3Lib.onResize(width, height);
            GLES3Lib.view.requestRender();
        }

        public void onDrawFrame(GL10 gl) {
            if (!GLES3Lib.activity.isPermissionReadStorageGranted() || !GLES3Lib.activity.isPermissionWriteStorageGranted()) return;

            if (!_initialized)
            {
                GLES3Lib.onInit(_w, _h,
                        GLES3Lib.dpi,
                        GLES3Lib.App.getApplicationContext().getFilesDir().getAbsolutePath());
                _initialized = true;
            }

            int videoType = GLES3Lib.getVideoType();
            int sizeIndex = GLES3Lib.getVideoSizeIndex();
            boolean usesRotation = GLES3Lib.usesRotation();
            boolean usesLocation = GLES3Lib.usesLocation();

            if (videoType==VT_MAIN || videoType==VT_SCND)
                 mainLoop.post(new Runnable() {@Override public void run() {GLES3Lib.activity.cameraStart(videoType, sizeIndex);}});
            else mainLoop.post(new Runnable() {@Override public void run() {GLES3Lib.activity.cameraStop();}});

            if (usesRotation)
                 mainLoop.post(new Runnable() {@Override public void run() {GLES3Lib.activity.rotationSensorStart();}});
            else mainLoop.post(new Runnable() {@Override public void run() {GLES3Lib.activity.rotationSensorStop();}});

            if (usesLocation)
                 mainLoop.post(new Runnable() {@Override public void run() {GLES3Lib.activity.locationSensorStart();}});
            else mainLoop.post(new Runnable() {@Override public void run() {GLES3Lib.activity.locationSensorStop();}});

            if (videoType==VT_FILE)
                GLES3Lib.grabVideoFileFrame();

            //////////////////////////////////////////////////////
            Boolean doRepaint = GLES3Lib.onUpdate();
            //Boolean sceneUpdated    = GLES3Lib.onUpdateScene();
            //Boolean viewUpdated     = GLES3Lib.onPaintAllViews();
            //////////////////////////////////////////////////////

            //Boolean doRepaint = trackingUpdated || sceneUpdated || viewUpdated;

            // Only request new rendering for non-live video
            // For live video the camera service will call requestRenderer
            if (doRepaint && (videoType==VT_NONE || videoType==VT_FILE))
                GLES3Lib.view.requestRender();

            if (videoType!=VT_NONE)
                GLES3Lib.lastVideoImageIsConsumed.set(true);
        }
    }
}
