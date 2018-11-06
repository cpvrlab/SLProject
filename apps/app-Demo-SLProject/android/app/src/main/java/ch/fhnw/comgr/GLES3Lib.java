//#############################################################################
//  File:      GLES3lib.java
//  Author:    Marcus Hudritsch, Zingg Pascal
//  Date:      Spring 2017
//  Purpose:   Android Java native interface into the SLProject C++ library
//  Codestyle: https://github.com/cpvrlab/SLProject/wiki/Coding-Style-Guidelines
//  Copyright: Marcus Hudritsch, Zingg Pascal
//             This software is provide under the GNU General Public License
//             Please visit: http://opensource.org/licenses/GPL-3.0
//#############################################################################

// Please do not change the name space. The SLProject app is identified in the app-store with it.
package ch.fhnw.comgr;

import android.app.ActivityManager;
import android.app.Application;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.concurrent.atomic.AtomicBoolean;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLContext;

import static android.content.Context.ACTIVITY_SERVICE;

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
    public static native boolean onUpdateAndPaint   ();
    public static native void    onResize           (int width, int height);
    public static native void    onMouseDown        (int button, int x, int y);
    public static native void    onMouseUp          (int button, int x, int y);
    public static native void    onMouseMove        (int x, int y);
    public static native void    onTouch2Down       (int x1, int y1, int x2, int y2);
    public static native void    onTouch2Up         (int x1, int y1, int x2, int y2);
    public static native void    onTouch2Move       (int x1, int y1, int x2, int y2);
    public static native void    onDoubleClick      (int button, int x, int y);
    public static native void    onRotationQUAT     (float quatX, float quatY, float quatZ, float quatW);
    public static native void    onClose            ();
    public static native boolean shouldClose        ();
    public static native void    shouldClose        (boolean doClose);
    public static native boolean usesRotation       ();
    public static native boolean usesLocation       ();
    public static native void    onLocationLLA      (double latitudeDEG, double longitudeDEG, double altitudeM, float accuracyM);
    public static native int     getVideoType       ();
    public static native int     getVideoSizeIndex  ();
    public static native void    grabVideoFileFrame ();
    public static native void    copyVideoImage     (int imgWidth, int imgHeight, byte[] imgBuffer);
    public static native void    copyVideoYUVPlanes (int srcW, int srcH,
                                                     byte[] y, int ySize, int yPixStride, int yLineStride,
                                                     byte[] u, int uSize, int uPixStride, int uLineStride,
                                                     byte[] v, int vSize, int vPixStride, int vLineStride);
    public static native void    onSetupExternalDirectories(String externalDirPath);


    /**
     * The RaytracingCallback function is used to repaint the ray tracing image during the
     * ray tracing process. Only the GUI bound OpenGL context can call the swap the buffer
     * for the OpenGL display. This is an example for a native C++ callback into managed
     * Java. See also the Java_renderRaytracingCallback in SLInterface that calls this
     * function.
     */
    public static boolean RaytracingCallback() {
        // calls the OpenGL rendering to display the RT image on a simple rectangle
        boolean stopSignal = GLES3Lib.onUpdateAndPaint();

        // Do the OpenGL back to front buffer swap
        EGL10 mEgl = (EGL10) EGLContext.getEGL();
        mEgl.eglSwapBuffers(mEgl.eglGetCurrentDisplay(), mEgl.eglGetCurrentSurface(EGL10.EGL_READ));
        return RTIsRunning;
    }

    /**
     * Extracts the relevant folders from the assets in our private storage on the device
     * internal storage. We extract all files from the folders textures, models, shaders, etc.
     * into the corresponding folders. This has to be done because most files in the apk/assets
     * folder are compressed and can not be read with standard C-file IO.
     */
    public static void extractAPK() throws IOException {
        FilesPath = App.getApplicationContext().getFilesDir().getAbsolutePath();
        Log.i("SLProject", "Destination: " + FilesPath);
        extractAPKFolder(FilesPath, "textures");
        extractAPKFolder(FilesPath, "videos");
        extractAPKFolder(FilesPath, "fonts");
        extractAPKFolder(FilesPath, "models");
        extractAPKFolder(FilesPath, "shaders");
        extractAPKFolder(FilesPath, "calibrations");
        extractAPKFolder(FilesPath, "config");
    }

    /**
     * Extract a folder inside the APK File. If we have a subfolder we just skip it.
     * If a file already exists we skip it => we don't update it!
     *
     * @param FilesPath where we want to store the path. Usually this is some writable path on the device. No / at the end
     * @param AssetPath path inside the asset folder. No leading or closing /
     * @throws IOException
     */
    public static void extractAPKFolder(String FilesPath, String AssetPath) throws IOException {
        String[] files = App.getAssets().list(AssetPath);

        for (String file : files)
        {
            if (!file.contains("."))
                continue;

            File f = new File(FilesPath + "/" + AssetPath + "/" + file);
            if (f.exists())
                continue;

            if (createDir(FilesPath + "/" + AssetPath))
                Log.i("SLProject", "Folder created: " + FilesPath + "/" + AssetPath + "/ -------------------------------------------\r\n");

            copyFile(App.getAssets().open(AssetPath + "/" + file), new FileOutputStream(FilesPath + "/" + AssetPath + "/" + file));
            Log.i("SLProject", "File: " + FilesPath + "/" + AssetPath + "/" + file + "\r\n");
        }
    }

    /**
     * Create the full directory tree up to our path
     *
     * @param Path to create
     * @throws IOException
     */
    public static boolean createDir(String Path) throws IOException {
        File directory = new File(Path);
        if (directory.exists())
            return false;
        if (!directory.mkdirs())
            throw new IOException("Directory couldn't be created: " + Path);
        return true;
    }

    /**
     * Copy from an Input to an Output stream. We use a buffer of 1024 and close both streams at the end
     *
     * @param is
     * @param os
     * @throws IOException
     */
    public static void copyFile(InputStream is, OutputStream os) throws IOException {
        byte[] buffer = new byte[1024];

        int length;
        while ((length = is.read(buffer)) > 0)
        {
            os.write(buffer, 0, length);
        }
        os.flush();
        os.close();
        is.close();
    }

}
