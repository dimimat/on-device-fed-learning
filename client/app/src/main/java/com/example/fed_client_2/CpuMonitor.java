package com.example.fed_client_2;

import java.io.BufferedReader;
import java.io.FileReader;
import android.util.Log;

public class CpuMonitor {
    private long lastIdle = 0;
    private long lastTotal = 0;

    public float getCpuUtilization() {
        try {
            BufferedReader reader = new BufferedReader(new FileReader("/proc/stat"));
            String line = reader.readLine();
            reader.close();

            if (line != null && line.startsWith("cpu ")) {
                String[] tokens = line.split("\\s+");
                long idle = Long.parseLong(tokens[4]);
                long total = 0;
                for (int i = 1; i < tokens.length; i++) {
                    total += Long.parseLong(tokens[i]);
                }

                if (lastTotal == 0) { // First reading
                    lastIdle = idle;
                    lastTotal = total;
                    return 0.0f;
                }

                long idleDelta = idle - lastIdle;
                long totalDelta = total - lastTotal;
                float utilization = (1.0f - (idleDelta / (float) totalDelta)) * 100.0f;

                lastIdle = idle;
                lastTotal = total;
                return utilization;
            }
        } catch (Exception e) {
            Log.e("CpuMonitor", "Error reading CPU stats: " + e);
        }
        return -1.0f;
    }
}
