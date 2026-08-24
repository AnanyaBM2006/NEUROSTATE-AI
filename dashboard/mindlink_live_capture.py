import serial
import time


# ============================================================
# MINDLINK LIVE CONNECTION
# ============================================================

PORT = "COM3"
BAUDRATE = 57600


print("=" * 60)
print("NEUROSTATE AI - LIVE MINDLINK CAPTURE")
print("=" * 60)

print()
print("PORT :", PORT)
print("BAUD :", BAUDRATE)

print()
print("IMPORTANT:")
print("1. Keep MindLink OFF initially.")
print("2. Wear the MindLink correctly.")
print("3. Keep this program running.")
print("4. Then turn MindLink ON / connect it.")
print()
print("Waiting for COM3...")
print()


ser = None


# ============================================================
# WAIT FOR COM3
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
        print("COM3 CONNECTED!")
        print("=" * 60)
        print()
        print("STARTING EEG CAPTURE...")
        print()

    except Exception:

        # COM3 is not available yet.
        # Keep checking.

        time.sleep(0.1)


# ============================================================
# READ DATA IMMEDIATELY
# ============================================================

start_time = time.time()

total_bytes = 0
sync_count = 0

buffer = bytearray()


try:

    while time.time() - start_time < 20:

        waiting = ser.in_waiting

        if waiting > 0:

            data = ser.read(waiting)

            total_bytes += len(data)

            buffer.extend(data)

            # ------------------------------------------------
            # Count AA AA packet headers
            # ------------------------------------------------

            for i in range(len(data) - 1):

                if (
                    data[i] == 0xAA
                    and data[i + 1] == 0xAA
                ):

                    sync_count += 1


            # ------------------------------------------------
            # Show live information
            # ------------------------------------------------

            if total_bytes % 1000 < len(data):

                print(
                    f"Bytes received: {total_bytes}"
                )

                print(
                    f"AA AA packets: {sync_count}"
                )

        time.sleep(0.005)


except KeyboardInterrupt:

    print()
    print("Capture stopped by user.")


finally:

    if ser is not None and ser.is_open:

        ser.close()


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 60)
print("MINDLINK CAPTURE RESULTS")
print("=" * 60)

print()
print(
    "Total bytes received:",
    total_bytes
)

print(
    "AA AA packet headers:",
    sync_count
)

print()


if total_bytes > 0:

    print("SUCCESS!")
    print()
    print(
        "MindLink EEG data was received through COM3."
    )

    print()
    print("First 100 bytes:")

    print(
        buffer[:100].hex(" ")
    )

else:

    print("NO DATA RECEIVED.")


print()
print("=" * 60)
print("CAPTURE COMPLETE")
print("=" * 60)