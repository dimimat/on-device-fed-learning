package com.example.fed_client_2;

import android.content.Context;
import android.content.Context;
import android.os.ConditionVariable;
import android.util.Log;
import java.text.DecimalFormat;
import com.google.protobuf.ByteString;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
//TFLite
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.label.TensorLabel;

public class TransferLearningModelWrapper {
    private Interpreter interpreter;
    private int N_FEATURES;
    private int N_CLASSES;
    private int N_HIDDEN;
    private MainActivity activity;

    // interpreter input,output tags
    private static final String INFERENCE_INPUT_KEY = "feature";
    private static final String INFERENCE_OUTPUT_KEY = "output";
    private static final String INFERENCE_KEY = "infer";
    private static final String TRAINING_INPUT_FEATURE_KEY = "feature";
    private static final String TRAINING_INPUT_LABELS_KEY = "label";
    private static final String TRAINING_OUTPUT_KEY = "loss";
    private static final String TRAINING_KEY = "train";


    private final ConditionVariable isTraining = new ConditionVariable();
    private final ConditionVariable isTesting = new ConditionVariable();

    // stats
    public int totalCorrect = 0; // catch inference routine statistics
    private List<Float>  yTrue = null;
    private List<Float>  yPred = null;
    public TransferLearningModelWrapper(MappedByteBuffer modelFile, int n_features, int n_hidden, int n_classes, MainActivity activity){
        this.interpreter = new Interpreter(modelFile);
        this.N_FEATURES = n_features;
        this.N_HIDDEN = n_hidden;
        this.N_CLASSES = n_classes;
        this.activity = activity;
    }

    public Interpreter getInterpreter(){ return this.interpreter; }

    private void trainRoutine(DataLoader trainLoader,float n_epochs){
        int numSamples = trainLoader.getNumSamples();
        int batchSize = trainLoader.getBatchSize();
        int n_iters = (int) (n_epochs * (float) Math.ceil((double) numSamples / (double) batchSize));
        DecimalFormat df = new DecimalFormat("0.#####");
        new Thread(() -> {
            for (int loop_counter = 0; loop_counter <= (n_iters - 1); loop_counter++){
                // 1. Get a batch of (X_train,y_train,n_samples)
                BatchOfData batchData = trainLoader.next();
                float[][] X_train_batch = batchData.getX();
                float[] y_train_batch = batchData.getY();
                //int nSamples = batchData.getNumSamples();
                // print shape of arrays

                //if (y_train_batch.length != 32 || X_train_batch.length != 32 || X_train_batch[0].length != 16) {
                //    Log.d("test","###################");
                //    Log.d("test","1D array: " + y_train_batch.length);
                //    Log.d("test","2D array n_rows: " + X_train_batch.length + " n_columns: " + X_train_batch[0].length);
                //}

                // 2. start training
                float loss = train(X_train_batch,y_train_batch);

                // print data
                if (loop_counter % 100 == 0){
                    //this.activity.setResultText("loss: " + df.format(loss)); // print to app stdout
                    Log.d("test","loss: " + loss);
                }
            }
            isTraining.open();
        }).start();
    }

    private float train(float[][] features, float[] labels){
        // format data to TFLite way
        Map<String, Object> trainInputs = new HashMap<>();
        trainInputs.put(TRAINING_INPUT_FEATURE_KEY, features);
        trainInputs.put(TRAINING_INPUT_LABELS_KEY, labels);
        Map<String, Object> trainOutputs = new HashMap<>();
        FloatBuffer loss = ByteBuffer.allocateDirect(4).asFloatBuffer();
        trainOutputs.put(TRAINING_OUTPUT_KEY,loss);

        // start training
        this.interpreter.runSignature(trainInputs,trainOutputs,TRAINING_KEY);
        loss.rewind();

        return loss.get();
    }

    private void inferenceRoutine(DataLoader testLoader){
        new Thread(() -> {
            int numSamples = testLoader.getNumSamples();
            int batchSize = testLoader.getBatchSize();
            int n_iters = (int) Math.ceil((double) numSamples / (double) batchSize);
            for (int loop_counter = 0; loop_counter <= (n_iters - 1); loop_counter++){
                // 1. get a batch of (X_train,y_train,n_samples)
                BatchOfData batchData = testLoader.next();
                float[][] X_test_batch = batchData.getX();
                float[] y_test_batch = batchData.getY();
                int nSamples = batchData.getNumSamples();

                // 2. get inference probs
                float[][] y_pred_probs = inference(X_test_batch, nSamples);

                // debugging
                //for (float[] value : y_pred_probs){
                //    Log.d("test","==============================");
                //    Log.d("test","prob: " + Arrays.toString(value));
                //}

                // 3. analyze data
                float[] y_pred = getPredictions(y_pred_probs,nSamples);
                int n_correct = correctPredictions(y_test_batch,y_pred);
                this.totalCorrect += n_correct;

                // collect data
                for (float value : y_test_batch) {
                    yTrue.add(value);
                }

                for (float value : y_pred) {
                    yPred.add(value);
                }
            }
            isTesting.open();
        }).start();
    }

    private float[][] inference(float[][] feature, int nSamples){
        // format data to TFLite way
        Map<String, Object> testInputs = new HashMap<>();
        testInputs.put(INFERENCE_INPUT_KEY, feature);
        Map<String, Object> testOutputs = new HashMap<>();
        float[][] outputBuffer = new float[nSamples][this.N_CLASSES];
        testOutputs.put(INFERENCE_OUTPUT_KEY, outputBuffer);

        // start inference
        this.interpreter.runSignature(testInputs,testOutputs,INFERENCE_KEY);

        return outputBuffer;
    }

    private float[] toFloat(List<Float> input){
        float[] output = new float[input.size()];
        for (int i = 0; i < input.size() ; i++){
            output[i] = input.get(i);
        }
        return output;
    }
    private float[][] convert2d(List<float[]> feature , int n_samples){
        // convert feature to float[][]
        float[][] feature_vec = new float[n_samples][this.N_FEATURES];
        int counter = 0;
        for (float[] sample : feature) {
            feature_vec[counter] = sample;
            counter++;
        }
        return feature_vec;
    }

    private float[] getPredictions(float[][] prediction_probs, int nSamples){
        // prediction_probs := n_samples x n_classes

        float[] y_pred = new float[nSamples];
        int counter = 0;
        for (float[] probs : prediction_probs){
            y_pred[counter] = (float) argMax(probs);
            //Log.d("test","prediction: " + (float) argMax(probs));
            counter++;
        }
        return y_pred;
    }

    private int argMax(float[] vec){
        int arg_max = 0;
        float max = vec[arg_max];
        for (int i = 1; i <vec.length; i++) {
            if (vec[i] > max){
                max = vec[i];
                arg_max = i;
            }
        }
        return arg_max;
    }

    private int correctPredictions(float[] y_true, float[] y_pred){
        int n_correct = 0;
        for (int i = 0; i < y_true.length; i++){
            if (y_true[i] == y_pred[i]){
                n_correct++;
            }
        }
        return n_correct;
    }

    public List<ByteString> getWeights() {
        Map<String, Object> inputs = new HashMap<>(); // set dummy input
        inputs.put("dummy_load", new float[]{0});

        // set output
        float[][] w1 = new float[this.N_FEATURES][this.N_HIDDEN];
        float[] b1 = new float[this.N_HIDDEN];
        float[][] w2 = new float[this.N_HIDDEN][this.N_HIDDEN];
        float[] b2 = new float[this.N_HIDDEN];
        float[][] w3 = new float[this.N_HIDDEN][this.N_CLASSES];
        float[] b3 = new float[this.N_CLASSES];

        String W1 = "layer1/kernel";
        String B1 = "layer1/bias";
        String W2 = "layer2/kernel";
        String B2 = "layer2/bias";
        String W3 = "layer3/kernel";
        String B3 = "layer3/bias";

        Map<String, Object> outputs = new HashMap<>();
        outputs.put(W1, w1);
        outputs.put(B1, b1);
        outputs.put(W2, w2);
        outputs.put(B2, b2);
        outputs.put(W3, w3);
        outputs.put(B3, b3);

        this.interpreter.runSignature(inputs, outputs, "get_weights");

        // convert floats to ByteBuffer
        List<ByteString> weights = new ArrayList<>();
        weights.add(ByteString.copyFrom(floatArray2DToByteBuffer(w1)));
        weights.add(ByteString.copyFrom(floatArrayToByteBuffer(b1)));
        weights.add(ByteString.copyFrom(floatArray2DToByteBuffer(w2)));
        weights.add(ByteString.copyFrom(floatArrayToByteBuffer(b2)));
        weights.add(ByteString.copyFrom(floatArray2DToByteBuffer(w3)));
        weights.add(ByteString.copyFrom(floatArrayToByteBuffer(b3)));

        return weights;
    }

    private ByteBuffer floatArrayToByteBuffer(float[] vec){
        ByteBuffer byteBuffer = ByteBuffer.allocate(vec.length * Float.BYTES);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        //Log.d("test","########### 1D #############");
        for (float value : vec) {
            //Log.d("test","float value: " + value);
            byteBuffer.putFloat(value);
        }
        byteBuffer.flip(); // Prepare the buffer for reading
        return byteBuffer;
    }

    private ByteBuffer floatArray2DToByteBuffer(float[][] vec2D){
        // compute number of values
        int n_values = 0;
        for (float[] vec : vec2D){
            n_values += vec.length;
        }

        ByteBuffer byteBuffer = ByteBuffer.allocate(n_values * Float.BYTES);
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        //Log.d("test","########### 2D #############");
        for (float[] vec : vec2D) {
            for (float value : vec){
                //Log.d("test","value: " + value);
                byteBuffer.putFloat(value);
            }
        }
        byteBuffer.flip(); // Prepare the buffer for reading
        return byteBuffer;
    }

    // wrapper function for trainRoutine : add syncrhonization mechanism
    public void fit(DataLoader trainLoader,float n_epochs){
        isTraining.close();
        trainRoutine(trainLoader,n_epochs);
        isTraining.block();
    }

    // wrapper function for inferenceRoutine : return test statistics , add synchronization mechanism just to be sure
    //public float evaluate(DataLoader testLoader) {
    //    this.totalCorrect = 0; // erase previous value
    //    int numSamples = testLoader.getNumSamples();

    //    isTesting.close();
    //    inferenceRoutine(testLoader);
    //    isTesting.block();

    //    float accuracy = (float) this.totalCorrect / (float) numSamples;
    //    Log.d("test","test -> total_correct: " + this.totalCorrect + " total_samples: " + numSamples);
    //    return accuracy;
    //}
    public Metrics evaluate(DataLoader testLoader) {
        // erase previous value
        this.totalCorrect = 0;
        this.yTrue = new ArrayList<>();
        this.yPred = new ArrayList<>();


        isTesting.close();
        inferenceRoutine(testLoader);
        isTesting.block();

        int numSamples = testLoader.getNumSamples();
        float accuracy = (float) this.totalCorrect / (float) numSamples;
        Log.d("test","test -> total_correct: " + this.totalCorrect + " total_samples: " + numSamples);

        return new Metrics(accuracy, this.yTrue, this.yPred);
    }
}