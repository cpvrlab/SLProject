//#############################################################################
//  File:      GLES3lib.java
//  Date:      Spring 2017
//  Purpose:   Android Java native interface into the SLProject C++ library
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Authors:   Marcus Hudritsch, Zingg Pascal
//  License:   This software is provided under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

// Please do not change the name space. The SLProject app is identified in the app-store with it.
package ch.bfh.cpvrlab;

import android.app.Application;
import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLContext;

// Java class that encapsulates the native C-functions into SLProject
public class GLES3Lib {

    static {
        System.loadLibrary("native-lib");
    }

    public static Application App = null;
    public static String FilesPath = null;
    public static GLES3View view;
    public static GLES3Activity activity;
    public static int dpi;
    public static boolean RTIsRunning = false;

    // flag to indicate if the last video images was displayed at all
    public static AtomicBoolean lastVideoImageIsConsumed = new AtomicBoolean(false);

    public static final int VIDEO_TYPE_NONE = 0;    // No video at all is used
    public static final int VIDEO_TYPE_MAIN = 1;    // Maps to Androids back facing camera
    public static final int VIDEO_TYPE_SCND = 2;    // Maps to Androids front facing camera
    public static final int VIDEO_TYPE_FILE = 3;    // Maps to Androids front facing camera

    public static native void    onInit             (int width, int height, int dotsPerInch, String FilePath);
    public static native boolean onUpdateVideo      ();
    public static native boolean onUpdateParallelJob();
    public static native boolean onPaintAllViews    ();
    public static native void    onResize           (int width, int height);
    public static native void    onMouseDown        (int button, int x, int y);
    public static native void    onMouseUp          (int button, int x, int y);
    public static native void    onMouseMove        (int x, int y);
    public static native void    onTouch2Down       (int x1, int y1, int x2, int y2);
    public static native void    onTouch2Move       (int x1, int y1, int x2, int y2);
    public static native void    onTouch2Up         (int x1, int y1, int x2, int y2);
    public static native void    onDoubleClick      (int button, int x, int y);
    public static native void    onRotationQUAT     (float quatX, float quatY, float quatZ, float quatW);
    public static native void    onClose            ();
    public static native boolean usesRotation       ();
    public static native boolean usesLocation       ();
    public static native void    onLocationLatLonAlt      (double latitudeDEG, double longitudeDEG, double altitudeM, float accuracyM);
    public static native int     getVideoType       ();
    public static native int     getVideoSizeIndex  ();
    public static native void    grabVideoFileFrame ();
    public static native void    copyVideoImage     (int imgWidth, int imgHeight, byte[] imgBuffer);
    public static native void    copyVideoYUVPlanes (int srcW, int srcH,
                                                     byte[] y, int ySize, int yPixStride, int yLineStride,
                                                     byte[] u, int uSize, int uPixStride, int uLineStride,
                                                     byte[] v, int vSize, int vPixStride, int vLineStride);
    public static native void    onSetupExternalDir (String externalDirPath);
    public static native void    setCameraSize      (int sizeIndex, int sizeIndexMax, int width, int height);
    public static native void    setDeviceParameter (String parameter, String value);
    public static native void    initMediaPipeAssetManager(Context androidContext, String cacheDirPath);

    /**
     * The Raytracing Callback function is used to repaint the ray tracing image during the
     * ray tracing process. Only the GUI bound OpenGL context can call the swap the buffer
     * for the OpenGL display. This is an example for a native C++ callback into managed
     * Java. See also the Java_renderRaytracingCallback in SLInterface that calls this
     * function.
     */
    public static boolean RaytracingCallback() {
        // calls the OpenGL rendering to display the RT image on a simple rectangle
        boolean stopSignal = GLES3Lib.onPaintAllViews();

        // Do the OpenGL back to front buffer swap
        EGL10 mEgl = (EGL10) EGLContext.getEGL();
        mEgl.eglSwapBuffers(mEgl.eglGetCurrentDisplay(), mEgl.eglGetCurrentSurface(EGL10.EGL_READ));
        return RTIsRunning;
    }

    /**
     * Extracts the data folder from the assets in our private storage on the device's
     * internal storage. We extract all files and creates sub-folders recursively.
     * Folders that already exist are deleted.
     * This has to be done because most files in the apk/assets
     * folder are compressed and can not be read with standard C-file IO.
     */
    public static void extractAPK() throws IOException {
        FilesPath = App.getApplicationContext().getFilesDir().getAbsolutePath();
        Log.i("SLProject", "Destination: " + FilesPath);

        extractRecursive(App.getAssets(), FilesPath, "data");
    }

    private static void extractRecursive(AssetManager assetManager, String path, String assetFolder) throws IOException{
        File directory = new File(path, assetFolder);
        createEmptyDirectory(directory);
        String[] assets = assetManager.list(assetFolder);

        for (String asset : assets) {
            // FIXME: How do we determine if an asset is a directory?
            if (asset.contains(".") && !asset.equals("2.0")) {
                // assumes file
                InputStream in = assetManager.open(assetFolder + File.separator + asset);
                File destinationFile = new File(directory, asset);
                copy(in, destinationFile);
            } else {
                // assume directory
                extractRecursive(assetManager, path, assetFolder + File.separator + asset);
            }
        }
    }

    private static void createEmptyDirectory(File dir){
        deleteRecursive(dir);
        if(!dir.exists()){
            dir.mkdirs();
        }
    }

    private static void copy(InputStream in, File destination) throws IOException {
        OutputStream out = new FileOutputStream(destination);
        // Transfer bytes from in to out
        byte[] buffer = new byte[1024];
        int len;
        while ((len = in.read(buffer)) > 0) {
            out.write(buffer, 0, len);
        }
        in.close();
        out.close();
    }

    public static void deleteRecursive(File fileOrDirectory) {
        if (fileOrDirectory.isDirectory())
            for (File child : fileOrDirectory.listFiles())
                deleteRecursive(child);

        fileOrDirectory.delete();
    }
}
