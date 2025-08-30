package com.example.fed_client_2;

import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class DataLoader {
    private byte[] X;
    private byte[] y;
    private int batchSize;
    private int numFeatures;
    private int numSamples;
    private int samplePointer;  // (logical) pointer to current sample: use the same for X,y

    public DataLoader(byte[] X, byte[] y, int batchSize, int numFeatures) {
        this.X = X;
        this.y = y;
        this.batchSize = batchSize;
        this.numFeatures = numFeatures;
        this.numSamples = y.length / 4; // since we are using 32bit data
        this.samplePointer = 0;
    }
    public BatchOfData next(){
        // check if we have enough samples for batchSize: currentSamples < batchSize if 4 * nSamples * nFeatures % batchSize != 0
        int currentSamples = this.samplePointer + this.batchSize < this.numSamples ? this.batchSize : this.numSamples - this.samplePointer;

        float[][] X_batch = new float[currentSamples][this.numFeatures];
        float[] y_batch = new float[currentSamples];

        // placeholder variable
        byte[] byteValue;
        for (int i = 0; i < currentSamples; i++) {
            // decode y
            byteValue = new byte[] {this.y[4 * this.samplePointer], this.y[4 * this.samplePointer + 1],
                    this.y[4 * this.samplePointer + 2], this.y[4 * this.samplePointer + 3]};

            float temp = (float) ByteBuffer.wrap(byteValue).order(ByteOrder.LITTLE_ENDIAN).getInt();
            //Log.d("test","value: " + temp);
            y_batch[i] = (float) ByteBuffer.wrap(byteValue).order(ByteOrder.LITTLE_ENDIAN).getInt();
            for (int j = 0; j < this.numFeatures; j++) {
                // decode X
                byteValue = new byte[] {this.X[4 * this.samplePointer * this.numFeatures + 4*j],
                        this.X[4 * this.samplePointer * this.numFeatures + 4*j + 1],
                        this.X[4 * this.samplePointer * this.numFeatures + 4*j + 2],
                        this.X[4 * this.samplePointer * this.numFeatures + 4*j + 3]};

                X_batch[i][j] = ByteBuffer.wrap(byteValue).order(ByteOrder.LITTLE_ENDIAN).getFloat();
            }
            this.samplePointer += 1;
        }

        // rewind if samplesPointer after traversing every sample
        this.samplePointer = (currentSamples < this.batchSize || this.samplePointer == this.numSamples) ? 0 : this.samplePointer;

        // construct return tuple: (X,y,current_samples) : convert primitive data types to corresponding classes
        BatchOfData data = new BatchOfData(X_batch, y_batch, currentSamples);
        return data;
    }
    public int getNumSamples() {return this.numSamples;}
    public int getBatchSize() {return this.batchSize;}
}

class BatchOfData {
    private float[][] X;
    private float[] y;
    private int numSamples;

    public BatchOfData(float[][] X, float[] y, int numSamples){
        this.X = X;
        this.y = y;
        this.numSamples = numSamples;
    }

    public float[][] getX(){
        return this.X;
    }

    public float[] getY(){
        return this.y;
    }

    public int getNumSamples(){
        return this.numSamples;
    }
}