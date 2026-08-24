import serial
import time


# ============================================================
# MINDLINK CONTINUOUS EEG TEST
# ============================================================

PORT = "COM3"
BAUDRATE = 57600

print("=" * 60)
print("NEUROSTATE AI - CONTINUOUS MINDLINK EEG")
print("=" * 60)

print()
print("Keep MindLink OFF initially.")
print("Wear the MindLink correctly.")
print("Start this program.")
print("Then turn MindLink ON.")
print()
print("Waiting for COM3...")
print()


ser = None


# ============================================================
# WAIT FOR MINDLINK
# ============================================================

while ser is None:

    try:

        ser = serial.Serial(
            port=PORT,
            baudrate=BAUDRATE,
            timeout=0.1
        )

        print()
        print("=" * 60)
        print("🟢 MINDLINK CONNECTED")
        print("=" * 60)
        print()
        print("Receiving EEG continuously...")
        print("Press CTRL+C to stop.")
        print()

    except Exception:

        time.sleep(0.1)


# ============================================================
# CONTINUOUS EEG READING
# ============================================================

total_bytes = 0
packet_count = 0

last_display = time.time()


try:

    while True:

        waiting = ser.in_waiting

        if waiting > 0:

            data = ser.read(waiting)

            total_bytes += len(data)

            # Count AA AA packet headers
            for i in range(len(data) - 1):

                if (
                    data[i] == 0xAA
                    and data[i + 1] == 0xAA
                ):

                    packet_count += 1


        # Display status once every second
        if time.time() - last_display >= 1:

            print(
                f"🟢 LIVE | "
                f"Bytes: {total_bytes:,} | "
                f"Packets: {packet_count:,}"
            )

            last_display = time.time()


        time.sleep(0.005)


except KeyboardInterrupt:

    print()
    print()
    print("=" * 60)
    print("STOPPING MINDLINK")
    print("=" * 60)

    print()
    print(
        f"Total bytes received : {total_bytes:,}"
    )

    print(
        f"Total packets        : {packet_count:,}"
    )


finally:

    if ser is not None and ser.is_open:

        ser.close()

    print()
    print("COM3 closed.")
    print("Program stopped.")