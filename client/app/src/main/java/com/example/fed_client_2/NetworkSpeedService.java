package com.example.fed_client_2;

import android.app.Service;
import android.content.Intent;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.IBinder;
import android.util.Log;
import java.io.BufferedInputStream;
import java.io.ByteArrayOutputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import android.os.ResultReceiver;

public class NetworkSpeedService extends Service {
    //private static final String TAG = "NetworkSpeedService";
    private static final String TAG = "network";
    public static final String EXTRA_RECEIVER = "extra_receiver";
    public static final int RESULT_CODE_SPEED = 1;

    private static final int FILE_SIZE_BYTES = 10 * 1024 * 1024;  // 10 MB

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        ResultReceiver receiver = intent.getParcelableExtra(EXTRA_RECEIVER);
        String hostUrl = intent.getStringExtra("hostUrl");
        if (hostUrl != null && !hostUrl.isEmpty() && receiver != null) {
            Log.d(TAG, "hostUrl: " + hostUrl);
            new SpeedTestTask(hostUrl, receiver).execute();
        } else {
            Log.e(TAG, "Host IP is missing! || No ResultReceiver provided");
        }
        //new SpeedTestTask().execute();
        return START_NOT_STICKY;
    }

//    @Override
//    protected void onPostExecute(SpeedResult result) {
//        // Send result back to UI
//        Intent intent = new Intent("com.yourapp.NETWORK_SPEED_RESULT");
//        intent.putExtra("download_speed", result.downloadSpeedMbps);
//        intent.putExtra("upload_speed", result.uploadSpeedMbps);
//        sendBroadcast(intent);
//    }

    @Override
    public IBinder onBind(Intent intent) {
        return null;
    }

    private class SpeedTestTask extends AsyncTask<Void, Void, Bundle> {
        private String SERVER_URL;
        private final ResultReceiver receiver;

        SpeedTestTask(String hostUrl, ResultReceiver receiver) {
            this.SERVER_URL = hostUrl;
            this.receiver = receiver;
        }

        @Override
        protected Bundle doInBackground(Void... params) {
            Bundle results = new Bundle();
            double downloadSpeed = measureDownloadSpeed();
            double uploadSpeed = measureUploadSpeed();
            results.putDouble("download_speed_mbps", downloadSpeed);
            results.putDouble("upload_speed_mbps", uploadSpeed);
            return results;
        }

        @Override
        protected void onPostExecute(Bundle results) {
            if (receiver != null) {
                receiver.send(RESULT_CODE_SPEED, results);
            }
            stopSelf();
        }

        private double measureDownloadSpeed() {
            double speedMbps = -1.0d;
            try {
                URL url = new URL(SERVER_URL + "/download");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("GET");

                long startTime = System.currentTimeMillis();
                BufferedInputStream in = new BufferedInputStream(conn.getInputStream());
                ByteArrayOutputStream out = new ByteArrayOutputStream();
                byte[] buffer = new byte[8192];
                int bytesRead;
                int totalBytes = 0;

                while ((bytesRead = in.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                    totalBytes += bytesRead;
                }
                long endTime = System.currentTimeMillis();

                in.close();
                out.close();
                conn.disconnect();

                double timeDiffSeconds = (endTime - startTime) / 1000.0;
                speedMbps = (totalBytes * 8.0 / timeDiffSeconds) / 1_000_000;
                //Log.d(TAG, String.format("Download Speed: %.2f Mbps, Size: %d bytes, Time: %.2f s",
                        //speedMbps, totalBytes, timeDiffSeconds));
            } catch (Exception e) {
                Log.e(TAG, "Download error: " + e.getMessage());
            }
            return speedMbps;
        }

        private double measureUploadSpeed() {
            double speedMbps = -1.0d;
            try {
                URL url = new URL(SERVER_URL + "/upload");
                HttpURLConnection conn = (HttpURLConnection) url.openConnection();
                conn.setRequestMethod("POST");
                conn.setDoOutput(true);
                conn.setRequestProperty("Content-Type", "application/octet-stream");

                byte[] data = new byte[FILE_SIZE_BYTES];  // 10 MB of dummy data
                java.util.Arrays.fill(data, (byte) 0);  // Fill with zeros

                long startTime = System.currentTimeMillis();
                OutputStream out = conn.getOutputStream();
                out.write(data);
                out.flush();
                out.close();

                // Read response to ensure upload completed
                conn.getResponseCode();
                long endTime = System.currentTimeMillis();

                conn.disconnect();

                double timeDiffSeconds = (endTime - startTime) / 1000.0;
                speedMbps = (FILE_SIZE_BYTES * 8.0 / timeDiffSeconds) / 1_000_000;
                //Log.d(TAG, String.format("Upload Speed: %.2f Mbps, Size: %d bytes, Time: %.2f s",
                        //speedMbps, FILE_SIZE_BYTES, timeDiffSeconds));
            } catch (Exception e) {
                Log.e(TAG, "Upload error: " + e.getMessage());
            }
            return speedMbps;
        }
    }

//    private static class SpeedResult {
//        double downloadSpeedMbps;
//        double uploadSpeedMbps;
//
//        SpeedResult(double download, double upload) {
//            this.downloadSpeedMbps = download;
//            this.uploadSpeedMbps = upload;
//        }
//    }
}