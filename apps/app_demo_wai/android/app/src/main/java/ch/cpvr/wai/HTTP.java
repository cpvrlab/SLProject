package ch.cpvr.wai;

import android.util.Base64;
import android.util.Log;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.Authenticator;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.PasswordAuthentication;
import java.net.URL;
import java.net.URLConnection;
import java.util.Arrays;

public class HTTP {

    public static void DownloadFiles(String url, String to, String user, String password){

        try {
            URLConnection c = new URL(url).openConnection();
            c.setRequestProperty("Authorization", "basic " + Arrays.toString(Base64.encode((user+ ":" + password).getBytes(), Base64.NO_WRAP)));
            c.setUseCaches(false);
            c.connect();

            URL u = c.getURL();
            //URL u = new URL(url);
            InputStream is = u.openStream();

            DataInputStream dis = new DataInputStream(is);
            byte[] buffer = new byte[1024];
            int length;
            FileOutputStream fos = new FileOutputStream(new File(to));

            while ((length = dis.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }

        } catch (MalformedURLException mue) {
            Log.e("SYNC getUpdate", "malformed url error", mue);
        } catch (IOException ioe) {
            Log.e("SYNC getUpdate", "io error", ioe);
        } catch (SecurityException se) {
            Log.e("SYNC getUpdate", "security error", se);
        }
    }

    public static void DownloadFiles(String url, String to){

        try {
            URL u = new URL(url);
            InputStream is = u.openStream();

            DataInputStream dis = new DataInputStream(is);
            byte[] buffer = new byte[1024];
            int length;
            FileOutputStream fos = new FileOutputStream(new File(to));

            while ((length = dis.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }

        } catch (MalformedURLException mue) {
            Log.e("SYNC getUpdate", "malformed url error", mue);
        } catch (IOException ioe) {
            Log.e("SYNC getUpdate", "io error", ioe);
        } catch (SecurityException se) {
            Log.e("SYNC getUpdate", "security error", se);
        }
    }

    /*
    public static byte[] encryptAES(SecretKey key, byte[] clear) {
    try {

        SecretKeySpec skeySpec = new SecretKeySpec(key.getEncoded(), "AES");
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.ENCRYPT_MODE, skeySpec);
        byte[] encrypted = cipher.doFinal(clear);
        return encrypted;
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
    }

    public static byte[] decryptAES(SecretKey key, byte[] encrypted) {
    try {

        SecretKeySpec skeySpec = new SecretKeySpec(key.getEncoded(), "AES");
        Cipher cipher = Cipher.getInstance("AES");
        cipher.init(Cipher.DECRYPT_MODE, skeySpec);
        byte[] decrypted = cipher.doFinal(encrypted);
        return decrypted;
    } catch (Exception e) {
        e.printStackTrace();
    }
    return null;
    }
*/
}
