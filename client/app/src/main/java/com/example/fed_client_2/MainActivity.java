package com.example.fed_client_2;

import android.app.Activity;
import android.content.Context;
import android.icu.text.SimpleDateFormat;
import android.os.BatteryManager;
import android.os.Bundle;
import android.os.AsyncTask;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Handler;
import android.os.Looper;
import android.text.TextUtils;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import android.os.Debug;
import android.os.Build;
// grpc
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import com.google.protobuf.ByteString; // to send weights as List of byte buffers
// flower_server
import io.grpc.flower_server.Parameters;
import io.grpc.flower_server.Scalar;
import io.grpc.stub.StreamObserver; // for client-to-server stream and bi-directional streaming
import io.grpc.flower_server.ClientMessage;
import io.grpc.flower_server.ServerMessage;
import io.grpc.flower_server.FlowerServiceGrpc;
import io.grpc.flower_server.FlowerServiceGrpc.FlowerServiceStub;
// battery gauge
import claug.batterygauge.logging.BatteryGauge;
import claug.batterygauge.logging.BatteryGaugeConfig;
// Network Statistics
import android.net.wifi.WifiInfo;
import android.net.wifi.WifiManager;
import fr.bmartel.speedtest.SpeedTestSocket;
import fr.bmartel.speedtest.SpeedTestReport;
import fr.bmartel.speedtest.model.SpeedTestError;
import fr.bmartel.speedtest.inter.ISpeedTestListener;
import fr.bmartel.speedtest.utils.SpeedTestUtils;
import android.content.Intent;
import android.os.ResultReceiver;

public class MainActivity extends AppCompatActivity {
    // Front End
    private String host;
    private EditText ip;
    private EditText port;
    private Button connectButton;
    private Button trainButton;
    private TextView resultText;
    private ManagedChannel channel;

    private DataLoader trainLoader;
    private DataLoader testLoader;
    private String modelName = "transferred_model.txt";
    private double n_epochs = 1.0;
    private int batchSize = 32;
    private int inputLayer;
    private int hiddenLayer;
    private int outputLayer;

    private String energyBefore = null;
    private String energyAfter = null;

    private final List<Integer> rssiList = new ArrayList<>();

    private WifiManager wifiManager;

//    private ScheduledExecutorService executorService;
//    private AtomicBoolean isTraining = new AtomicBoolean(false);
//    private Map<String, Float> performanceMetrics = new HashMap<>(); // To store averages

    private int totalPacketsReceived;

    private int totalPacketsSend;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        resultText = (TextView) findViewById(R.id.grpc_response_text);
        resultText.setMovementMethod(new ScrollingMovementMethod());
        resultText.setMaxLines(1000);
        ip = (EditText) findViewById(R.id.serverIP);
        port = (EditText) findViewById(R.id.serverPort);
        connectButton = (Button) findViewById(R.id.connect);
        trainButton = (Button) findViewById(R.id.trainFederated);
        wifiManager = (WifiManager) getApplicationContext().getSystemService(Context.WIFI_SERVICE);
    }
    public static void hideKeyboard(Activity activity) {
        InputMethodManager imm = (InputMethodManager) activity.getSystemService(Activity.INPUT_METHOD_SERVICE);
        View view = activity.getCurrentFocus();
        if (view == null) {
            view = new View(activity);
        }
        imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
    }


    public void setResultText(String text) {
        SimpleDateFormat dateFormat = new SimpleDateFormat("HH:mm:ss", Locale.GERMANY);
        String time = dateFormat.format(new Date());
        resultText.append("\n" + time + "   " + text);
    }

    public void cleatText() {
        resultText.setText("");
    }

    private void print(String output){
        Log.d("test",output);
    }

    private void timer(int seconds){
        try {
            TimeUnit.SECONDS.sleep(seconds); // dummy synchronization
        }
        catch (Exception e){
            Log.e("debug","Timer error");
        }
    }

    public void connect(View view) {
        host = ip.getText().toString();
        String portStr = port.getText().toString();

        if (TextUtils.isEmpty(host) || TextUtils.isEmpty(portStr)) {
            host = "192.168.1.1";
            portStr = "8080";
            Toast.makeText(this, "Connected to server with IP: " + host + " port: " + portStr, Toast.LENGTH_LONG).show();
        }
        int port = TextUtils.isEmpty(portStr) ? 0 : Integer.parseInt(portStr);
        channel = ManagedChannelBuilder.forAddress(host, port).maxInboundMessageSize(100 * 1024 * 1024).usePlaintext().build();

        // Start speed test and get results
        SpeedResultReceiver receiver = new SpeedResultReceiver(new Handler());
        Intent intent = new Intent(this, NetworkSpeedService.class);
        intent.putExtra(NetworkSpeedService.EXTRA_RECEIVER, receiver);
        String port2 = "8088";
        String hostUrl = String.format("http://%s:%s", host, port2);
        intent.putExtra("hostUrl", hostUrl);
        startService(intent);
        //new SpeedTestTask(true).execute();
        // Measure upload speed
        //new SpeedTestTask(false).execute();


        hideKeyboard(this);
        trainButton.setEnabled(true);
        connectButton.setEnabled(false);
        setResultText("Channel object created. Ready to train!");
    }

    public void runGrpc(View view){
        MainActivity activity = this;
        ExecutorService executor = Executors.newSingleThreadExecutor();
        Handler handler = new Handler(Looper.getMainLooper());

        executor.execute(new Runnable() {
            private String result;
            @Override
            public void run() {
                try {
                    (new FlowerServiceRunnable()).run(FlowerServiceGrpc.newStub(channel), activity); // this handles the FL
                    result =  "Connection to the FL server successful \n";
                } catch (Exception e) {
                    StringWriter sw = new StringWriter();
                    PrintWriter pw = new PrintWriter(sw);
                    e.printStackTrace(pw);
                    pw.flush();
                    result = "Failed to connect to the FL server \n" + sw;
                }
                handler.post(() -> {
                    setResultText(result);
                    trainButton.setEnabled(false);
                });
            }
        });
    }

    private static class FlowerServiceRunnable{
        protected Throwable failed;
        private StreamObserver<ClientMessage> requestObserver;

        public void run(FlowerServiceStub asyncStub, MainActivity activity) {
            join(asyncStub, activity);
        }

        private void join(FlowerServiceStub asyncStub, MainActivity activity) throws RuntimeException {
            final CountDownLatch finishLatch = new CountDownLatch(1);
            // timers to measure download speed
            requestObserver = asyncStub.join(
                    new StreamObserver<ServerMessage>() {
                        @Override
                        public void onNext(ServerMessage msg) {
                            handleMessage(msg, activity);
                            Log.d("network", "onNext triggered");
                        }

                        @Override
                        public void onError(Throwable t) {
                            t.printStackTrace();
                            failed = t;
                            finishLatch.countDown();
                            Log.e("network","Connection error my friend: " + t.getMessage());
                        }

                        @Override
                        public void onCompleted() {
                            Log.e("network", "on complete triggered");
                            finishLatch.countDown();
                        }
                    });
        }

        private void handleMessage(ServerMessage message, MainActivity activity) {
            try {
                ClientMessage toSend = null;
                //activity.cleatText(); // clear text view

                // Get Network Statistics
                // a. RSSI
                WifiManager wifiManager = (WifiManager) activity.getApplicationContext().getSystemService(Context.WIFI_SERVICE);
                if (wifiManager != null) {
                    WifiInfo wifiInfo = wifiManager.getConnectionInfo();
                    int rssi = wifiInfo.getRssi();

                    synchronized (activity.rssiList) {
                        activity.rssiList.add(rssi);
                    }
                }

                // b. ping (client-server)
                long ping = activity.pingServer(activity.host);
                Log.d("network", "ping: " + ping);

                // c. BW
                double dl = MainActivity.SpeedResultReceiver.getDownloadSpeed();
                double ul = MainActivity.SpeedResultReceiver.getUploadSpeed();
                Log.d("network", String.format("Received - Download: %.2f Mbps, Upload: %.2f Mbps",
                        dl, ul));

                //long nativeHeapSize = Debug.getNativeHeapSize();
                //long nativeHeapFreeSize = Debug.getNativeHeapFreeSize();
                //long nativeHeapAllocatedSize = Debug.getNativeHeapAllocatedSize();
                //Log.d("Memory", "Native Heap Size: " + nativeHeapSize / 1024 / 1024 + " MB");
                //Log.d("Memory", "Free Heap Size: " + nativeHeapFreeSize / 1024 / 1024 + " MB");
                //Log.d("Memory", "Allocated Heap Size: " + nativeHeapAllocatedSize / 1024 / 1024 + " MB");
//                Runtime runtime = Runtime.getRuntime();
//                long maxMemory = runtime.maxMemory() / 1024 / 1024;
//                Log.d("HeapSize", "Max heap size: " + maxMemory + " MB");

                if (message.hasEvaluateIns()){
                    activity.print("--- Loading Data ---");
                    //activity.setResultText("--- Loading data ---");

                    // 1. catch data
                    List<ByteString> data = message.getEvaluateIns().getParameters().getTensorsList(); // data : X_train,y_train,X_test,y_test

                    byte[] X_train = data.get(0).toByteArray();
                    byte[] y_train = data.get(1).toByteArray();
                    byte[] X_test = data.get(2).toByteArray();
                    byte[] y_test = data.get(3).toByteArray();

                    // 2. get configuration file: model configuration
                    Scalar inputLayer_config = message.getEvaluateIns().getConfigMap().getOrDefault("n_inputs",Scalar.newBuilder().build());
                    Scalar hiddenLayer_config = message.getEvaluateIns().getConfigMap().getOrDefault("n_hidden",Scalar.newBuilder().build());
                    Scalar outputLayer_config = message.getEvaluateIns().getConfigMap().getOrDefault("n_outputs",Scalar.newBuilder().build());
                    activity.inputLayer = (int) inputLayer_config.getSint64();
                    activity.hiddenLayer = (int) hiddenLayer_config.getSint64();
                    activity.outputLayer = (int) outputLayer_config.getSint64();

                    // 3. construct dataLoaders: e.x (X_train,y_train,n_samples) <- trainLoader.next()
                    activity.trainLoader = new DataLoader(X_train,y_train,activity.batchSize,activity.inputLayer);
                    activity.testLoader = new DataLoader(X_test,y_test,activity.batchSize, activity.inputLayer);

                    // print info
                    activity.print("input: " + activity.inputLayer + " hidden: " + activity.hiddenLayer + " output: " + activity.outputLayer);
                    //activity.setResultText("model shape: input: " + activity.inputLayer + " hidden: " + activity.hiddenLayer + " output: " + activity.outputLayer);
                    activity.print("n_train_samples: " + activity.trainLoader.getNumSamples());
                    activity.print("n_test_samples: " + activity.testLoader.getNumSamples());
                }
                else if (message.hasFitIns()) { // <- get model(i). fit model(i + 1). send model(i + 1)
                    // Get configuration file from server
                    Scalar n_epochs_config = message.getFitIns().getConfigMap().getOrDefault("n_epochs",Scalar.newBuilder().setDouble(activity.n_epochs).build());
                    float n_epochs = (float) n_epochs_config.getDouble();
                    Scalar operation_mode_config = message.getFitIns().getConfigMap().getOrDefault("operation_mode",Scalar.newBuilder().setString("train").build());
                    String operation_mode = (String) operation_mode_config.getString();

                    String operation_mode_msg = operation_mode.equals("train") ? "--- Start training routine ---"
                            : "--- Start Inference routine -";
                    if (operation_mode.equals("train")) {
                        //activity.setResultText("----------------------------");
                    }
                    activity.print(operation_mode_msg);

                    List<ByteString> modelFileWrapper = message.getFitIns().getParameters().getTensorsList();
                    ByteString modelFile = modelFileWrapper.get(0);

                    // load model to TFLite model wrapper
                    TransferLearningModelWrapper modelWrapper =  new TransferLearningModelWrapper(formatModelFile(modelFile,activity), activity.inputLayer, activity.hiddenLayer, activity.outputLayer, activity);

                    // we will send the energy measurements at inference
                    // in order to keep the train packets only for weights
                    // we do not to "litter" the BW measurements with unecessary data
                    if (operation_mode.equals("train")) {
                        // get total packets received from server (training only)
                        activity.totalPacketsReceived = message != null ? message.getSerializedSize() : -1;
                        // compute battery statistics before training (log at BatteryGauge)
                        BatteryGaugeConfig config = new BatteryGaugeConfig(activity.getApplicationContext()).filterNa(true);
                        BatteryGauge.init(config);
                        activity.energyBefore = BatteryGauge.getStatistics();
                        Log.d("energy", "energy_before: " + activity.energyBefore);
                        // call training routine
                        activity.print("model size: " + modelFile.size()  + " bytes" + ", n_epochs: " + n_epochs);
                        try {
                            modelWrapper.fit(activity.trainLoader, n_epochs);
                        } catch (Exception e) {
                                Log.e("test","Training error my Friend: " + e);
                        }

                        // get battery statistics after training
                        activity.energyAfter = BatteryGauge.getStatistics();
                        Log.d("energy", "energy_before: " + activity.energyBefore);
                        BatteryGauge.destroy();

                        // train set statistics
                        try {
                            Metrics metrics = modelWrapper.evaluate(activity.trainLoader);
                            Float accuracy = metrics.getAccuracy();
                            activity.print("train set accuracy: " + accuracy);
                            //activity.setResultText("train set accuracy: " + accuracy);
                        } catch (Exception e) {
                            Log.e("test","Evaluation error my Friend: " + e);
                        }

                        // train set statistics
                        // get trained weights
                        List<ByteString> weights = modelWrapper.getWeights();

                        // construct client message
                        toSend = fitResAsProto(weights,activity.trainLoader.getNumSamples());
                        activity.totalPacketsSend = toSend != null ? toSend.getSerializedSize() : -1;
                    } else if (operation_mode.equals("inference")) {
                        // call inference routine
                        activity.print("model size: " + modelFile.size()  + " bytes");
                        Float accuracy = null;
                        List<Float> yTrue = null;
                        List<Float> yPred = null;
                        try {
                            Metrics metrics = modelWrapper.evaluate(activity.testLoader);
                            accuracy = metrics.getAccuracy();
                            yTrue = metrics.getYtrue();
                            yPred = metrics.getYpred();
                            activity.print("test set accuracy: " + accuracy);
                            //activity.setResultText("test set accuracy: " + accuracy);
                        }
                        catch (Exception e) {
                            Log.e("test","Evaluation error my Friend: " + e);
                        }
                        int numSamples = activity.testLoader.getNumSamples();

                        // get average rssi
                        int sumRssi = 0;
                        List<Integer> rssiList = activity.getAndClearRssiList();
                        for (Integer rssi : rssiList) {
                            sumRssi += rssi;
                            Log.d("network", "rssi: " + rssi);
                        }
                        int avgRssi = rssiList != null && !rssiList.isEmpty() ? sumRssi / rssiList.size() : -1;

                        toSend = evaluateResAsProto(accuracy, yTrue, yPred, numSamples, activity.energyBefore, activity.energyAfter,
                                avgRssi, ping, dl, ul, activity.totalPacketsReceived, activity.totalPacketsSend);
                    }
                }
                else {
                    Log.d("test", "mode not supported");
                }
                requestObserver.onNext(toSend);
            }
            catch (Exception e){
                Log.e("test","runGrpc error: " + e.getMessage());
            }
        }
    }

    private static MappedByteBuffer formatModelFile(ByteString fileChunk,MainActivity activity){
        // save fileChunk to runtime file path
        File file = new File(activity.getFilesDir(),activity.modelName);
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(file);
            fileChunk.writeTo(fileOutputStream); // single time transfer
            fileOutputStream.close();
            //activity.print("File saved successfully at path: " + activity.getFilesDir().getAbsolutePath());
        } catch (IOException e) {
            e.printStackTrace();
            Log.e("test","Error saving file: " + e);
        }

        // get model from runtime file path
        File fileToLoad = new File(activity.getFilesDir(), activity.modelName);
        MappedByteBuffer model = null;
        try {
            FileInputStream fileInputStream = new FileInputStream(fileToLoad);
            FileChannel fileChannel = fileInputStream.getChannel();

            // Map the file into a MappedByteBuffer
            long fileLength = fileChannel.size();
            model = fileChannel.map(FileChannel.MapMode.READ_ONLY, 0, fileLength);

            // Close the file input stream when done
            fileInputStream.close();
        } catch (IOException e) {
            // Handle file read error
            e.printStackTrace();
            Log.e("test","Error loading file: " + e);
        }

        return model;
    }

    private static ClientMessage fitResAsProto(List<ByteString> weights, int trainingSize) {
        //for (ByteString w : weights) {
        //    Log.e("test","To send: " + w.size());
        //}
        Parameters p = Parameters.newBuilder().addAllTensors(weights).setTensorType("float32").build();
        //Scalar lossScalar = Scalar.newBuilder().setString(String.valueOf(statsBefore)).build();
        ClientMessage.FitRes res = ClientMessage.FitRes.newBuilder()
                .setParameters(p).setNumExamples(trainingSize).build();
        return ClientMessage.newBuilder().setFitRes(res).build();
    }

    private static ClientMessage evaluateResAsProto(float accuracy, List<Float> yTrue, List<Float> yPred,
                                                    int numSamples, String energyBefore, String energyAfter,
                                                    int rssi, long ping, double downloadSpeed, double uploadSpeed,
                                                    int totalPacketsReceived, int totalPacketsSend) {
        // we would pass y_true, y_pred as Parameters yArray = concat(yTrue, yPred)
        // we should also convert them to bytes
        List<ByteString> yArray = new ArrayList<>();
        yArray.add(floatList2ByteString(yTrue));
        yArray.add(floatList2ByteString(yPred));

        Parameters p = Parameters.newBuilder().addAllTensors(yArray).setTensorType("float32").build();
        Scalar deviceNameScalar = Scalar.newBuilder().setString(String.valueOf(getDeviceName())).build();
        Scalar energyAfterScalar = Scalar.newBuilder().setString(String.valueOf(energyAfter)).build();
        Scalar energyBeforeScalar = Scalar.newBuilder().setString(String.valueOf(energyBefore)).build();
        Scalar accScalar = Scalar.newBuilder().setString(String.valueOf(accuracy)).build();
        Scalar rssiScalar = Scalar.newBuilder().setString(String.valueOf(rssi)).build();
        Scalar pingScalar = Scalar.newBuilder().setString(String.valueOf(ping)).build();
        Scalar dlSpeedScalar = Scalar.newBuilder().setString(String.valueOf(downloadSpeed)).build();
        Scalar ulSpeedScalar = Scalar.newBuilder().setString(String.valueOf(uploadSpeed)).build();
        Scalar totalReceivedScalar = Scalar.newBuilder().setString(String.valueOf(totalPacketsReceived)).build();
        Scalar totalSendScalar = Scalar.newBuilder().setString(String.valueOf(totalPacketsSend)).build();

        ClientMessage.FitRes res = ClientMessage.FitRes.newBuilder().putMetrics("device_name", deviceNameScalar)
                .putMetrics("energy_before", energyBeforeScalar).putMetrics("energy_after", energyAfterScalar)
                .putMetrics("avg_rssi", rssiScalar).putMetrics("latency", pingScalar)
                .putMetrics("download_speed", dlSpeedScalar).putMetrics("upload_speed", ulSpeedScalar)
                .putMetrics("rx_data", totalReceivedScalar).putMetrics("tx_data", totalSendScalar)
                .putMetrics("accuracy", accScalar).setParameters(p).setNumExamples(numSamples).build();
        //Log.d("test","evaluate res: " + res.toString());
        return ClientMessage.newBuilder().setFitRes(res).build();
    }

    private static ByteString floatList2ByteString(List<Float> y) {
        ByteBuffer byteBuffer = ByteBuffer.allocate(y.size() * Float.BYTES);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        //Log.d("test","########### 1D #############");
        for (Float value : y) {
            //Log.d("test","predition/true value: " + value);
            byteBuffer.putFloat(value);
        }
        byteBuffer.flip(); // Prepare the buffer for reading
        return ByteString.copyFrom(byteBuffer);
    }

    private static String getDeviceName() {
        String manufacturer = Build.MANUFACTURER;
        String model = Build.MODEL;

        if (model.toLowerCase().startsWith(manufacturer.toLowerCase())) {
            return model;
        } else {
            return manufacturer + " " + model;
        }
    }

    private List<Integer> getAndClearRssiList() {
        synchronized (rssiList) {
            List<Integer> copy = new ArrayList<>(rssiList);
            rssiList.clear();  // Clear after retrieving
            return copy;
        }
    }

    private long pingServer(String ipAddress) {
        try {
            String command = "/system/bin/ping -c 4 " + ipAddress; // full path to ping binary
            Process process = Runtime.getRuntime().exec(command);

            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            long totalTime = 0;
            int count = 0;

            while ((line = reader.readLine()) != null) {
                if (line.contains("time=")) {
                    // Find time= value using regex or string matching
                    int timeIndex = line.indexOf("time=");
                    int msIndex = line.indexOf(" ms", timeIndex);
                    if (timeIndex != -1 && msIndex != -1) {
                        String timeSubstring = line.substring(timeIndex + 5, msIndex); // extract just the number
                        try {
                            double timeMs = Double.parseDouble(timeSubstring);
                            totalTime += (long) timeMs;
                            count++;
                        } catch (NumberFormatException e) {
                            Log.d("ping", "Failed to parse time: " + timeSubstring);
                        }
                    }
                }
            }

            process.waitFor();

            if (count > 0) {
                return totalTime / count;
            } else {
                Log.d("ping", "No valid ping results");
                return -1;
            }

        } catch (Exception e) {
            Log.d("ping", "Ping error: " + e.getMessage());
            e.printStackTrace();
            return -1;
        }
    }

    public static class SpeedResultReceiver extends ResultReceiver {

        private static double downloadSpeed = -1.0d;
        private static double uploadSpeed = -1.0d;
        public SpeedResultReceiver(Handler handler) {
            super(handler);
        }

        public static double getDownloadSpeed () {
            return downloadSpeed;
        }

        public static double getUploadSpeed () {
            return uploadSpeed;
        }

        @Override
        protected void onReceiveResult(int resultCode, Bundle resultData) {
            if (resultCode == NetworkSpeedService.RESULT_CODE_SPEED) {
                downloadSpeed = resultData.getDouble("download_speed_mbps", -1.0);
                uploadSpeed = resultData.getDouble("upload_speed_mbps", -1.0);
                //Log.d("network", String.format("Received - Download: %.2f Mbps, Upload: %.2f Mbps", downloadSpeed, uploadSpeed));
            }
        }
    }

    private float getMemoryUsage() {
        Runtime runtime = Runtime.getRuntime();
        long usedMemory = runtime.totalMemory() - runtime.freeMemory();
        long maxMemory = runtime.maxMemory();
        return (usedMemory / (float) maxMemory) * 100.0f; // Percentage of max memory used
    }
}