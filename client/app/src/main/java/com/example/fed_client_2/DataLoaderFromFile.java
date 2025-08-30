package com.example.fed_client_2;

import android.content.Context;
import android.util.Log;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import android.content.Context;
import android.util.Log;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

// ----- old implementation ------ //

public class DataLoaderFromFile {
    public final int batch_size;
    private final String mode;
    public int n_samples = 0;
    public int device_id;
    public static Map<String, Float> stoi;
    public static Map<Float, String> itos;
    public int currentPosition = 0;
    private Context context;

    // return variables from .next() method
    private List<float[]> batchOfFeatures;
    private List<Float> batchOfLabels;
    private int current_samples;

    public DataLoaderFromFile(Context context, Boolean train, int batchSize, int device_id){
        // init variables
        this.batch_size = batchSize;
        this.mode = (train) ? "train" : "test";
        this.context = context;
        this.device_id = device_id;

        // init label mapping
        stoi();
        itos();

        // calculate n_samples
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/y_" + (this.mode) + "/partition_" + (this.device_id) + ".txt")));
            while (reader.readLine() != null) {this.n_samples++;}
            reader.close();
        }
        catch (IOException ex) {
            ex.printStackTrace();
            Log.e("data","DataLoading error");
        }
    }

    public void next(){ // at first return objects via return functions
        // return (X_train,y_train,n_samples) one batch at a time : for example X_train, y_train , n_samples = next(iter(train_loader))

        List<Float> batchOfLabels = new ArrayList<>();
        List<float[]> batchOfFeatures = new ArrayList<>();
        try {
            BufferedReader label_reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/y_" + (this.mode) + "/partition_" + (this.device_id) + ".txt")));
            BufferedReader feature_reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/X_" + (this.mode) + "/partition_" + (this.device_id) + ".txt")));

            // ----------- get batch of labels : List<Float> ---------------
            String line;
            int lineCounter = 0;
            int current_samples = 0;
            while ((line = label_reader.readLine()) != null) {
                if (lineCounter >= this.currentPosition && lineCounter <= (this.currentPosition + this.batch_size - 1)){
                    batchOfLabels.add(stoi.get(line));
                    current_samples++;
                }

                if (lineCounter > (currentPosition + batch_size)){
                    break;
                }
                lineCounter++;
                //current_samples++;
            }
            label_reader.close();

            // ----------- get batch of features : List<float[]> ---------------
            lineCounter = 0;
            while ((line = feature_reader.readLine()) != null) {
                // get batch of features
                if (lineCounter >= this.currentPosition && lineCounter <= (this.currentPosition + this.batch_size - 1)){
                    batchOfFeatures.add(getFeature(line));
                }
                if (lineCounter > (currentPosition + batch_size)){
                    break;
                }
                lineCounter++;
            }
            feature_reader.close();

            // if end of samples rewind.
            if (current_samples < this.batch_size) {
                currentPosition = 0;
            }
            else {
                currentPosition += batch_size;
            }

            // save data to return variables
            this.current_samples = current_samples;
            this.batchOfFeatures = batchOfFeatures;
            this.batchOfLabels = batchOfLabels;

        }
        catch (IOException ex){
            Log.e("test","data error my friend");
            ex.printStackTrace();
        }
    }

    private float[] getFeature(String line){
        String[] lineSplit = line.split("\\s+");
        float[] feature_vec = new float[lineSplit.length];
        int counter = 0;
        for (String s : lineSplit){
            feature_vec[counter] = Float.parseFloat(s);
            counter++;
        }
        return feature_vec;
    }

    // return functions : try to clear lists after sending them
    public int getCurrentSamples(){ return this.current_samples; }

    public List<Float> getBatchOfLabels(){ return this.batchOfLabels;}

    public List<float[]> getBatchOfFeatures(){ return this.batchOfFeatures;}

    // dictionary to map to/from integers(our model uses floats)
    private void stoi(){
        stoi = new HashMap<>();
        stoi.put("Running",0f);
        stoi.put("Sitting",1f);
        stoi.put("Standing",2f);
        stoi.put("Walking",3f);
        stoi.put("downstaires",4f);
        stoi.put("upstaires",5f);
    }
    private void itos(){
        itos = new HashMap<>();
        itos.put(0f,"Running");
        itos.put(1f,"Sitting");
        itos.put(2f,"Standing");
        itos.put(3f,"Walking");
        itos.put(4f,"downstaires");
        itos.put(5f,"upstaires");
    }
}