package com.example.fed_client_2;

import java.util.List;

public class Metrics {
        private Float accuracy;
        private List<Float> yPred;
        private List<Float> yTrue;

        public Metrics(Float accuracy, List<Float> yTrue, List<Float> yPred) {
            this.accuracy = accuracy;
            this.yTrue = yTrue;
            this.yPred = yPred;
        }

        public Float getAccuracy() {
            return this.accuracy;
        }

        public List<Float> getYtrue() {
            return this.yTrue;
        }

        public List<Float> getYpred() {
            return this.yPred;
        }
}
