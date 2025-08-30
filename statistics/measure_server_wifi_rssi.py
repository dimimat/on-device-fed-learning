import subprocess

def get_macos_wifi_signal():
    try:
        output = subprocess.check_output([
            "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
            "-I"
        ], text=True)
        for line in output.split("\n"):
            if "agrCtlRSSI" in line:
                rssi = int(line.split(":")[1].strip())
                return rssi
    except Exception as e:
        print("Error:", e)
        return None

if __name__ == "__main__":
    signal_strength = get_macos_wifi_signal()
    print(f"Server Wi-Fi Signal Strength: {signal_strength} dBm")

