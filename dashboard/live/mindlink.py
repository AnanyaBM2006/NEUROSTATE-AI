import serial
import time
import csv
import os
import threading
from datetime import datetime


DEFAULT_PORT = "COM3"
BAUDRATE = 57600

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "live_mindlink"
)


class MindLinkReader:

    def __init__(
        self,
        port=DEFAULT_PORT,
        baudrate=BAUDRATE,
        save_csv=
    ):

        self.port = port
        self.baudrate = baudrate
        self.save_csv = save_csv

        self.ser = None
        self.running = False
        self.connected = False

        self.thread = None
        self.lock = threading.Lock()

        # Actual decoded EEG samples
        self.raw_eeg_buffer = []

        self.latest_raw_eeg = None
        self.latest_attention = None
        self.latest_meditation = None
        self.latest_signal_quality = None

        self.sample_count = 0
        self.packet_count = 0
        self.valid_packet_count = 0

        self.csv_file = None
        self.csv_writer = None

    # ========================================================
    # CSV
    # ========================================================

    def _open_csv(self):

        if not self.save_csv:
            return

        os.makedirs(
            OUTPUT_DIR,
            exist_ok=True
        )

        filename = (
            "mindlink_live_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".csv"
        )

        filepath = os.path.join(
            OUTPUT_DIR,
            filename
        )

        self.csv_file = open(
            filepath,
            "w",
            newline="",
            encoding="utf-8"
        )

        self.csv_writer = csv.writer(
            self.csv_file
        )

        self.csv_writer.writerow([
            "timestamp",
            "raw_eeg",
            "attention",
            "meditation",
            "signal_quality"
        ])

        self.csv_file.flush()

        print()
        print("Recording live EEG to:")
        print(filepath)
        print()

    def _close_csv(self):

        if self.csv_file is not None:

            try:
                self.csv_file.close()

            except Exception:
                pass

        self.csv_file = None
        self.csv_writer = None

    # ========================================================
    # CONNECT
    # ========================================================

    def connect(self):

        if self.connected:
            return True

        try:

            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1
            )

            self.ser.reset_input_buffer()

            self.connected = True

            print()
            print("🟢 MindLink connected on", self.port)

            return True

        except Exception:

            self.ser = None
            self.connected = False

            return False

    # ========================================================
    # DISCONNECT
    # ========================================================

    def disconnect(self):

        self.connected = False

        if self.ser is not None:

            try:

                if self.ser.is_open:
                    self.ser.close()

            except Exception:
                pass

        self.ser = None

    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self.running:
            return

        self.running = True

        self._open_csv()

        self.thread = threading.Thread(
            target=self._reader_loop,
            daemon=True
        )

        self.thread.start()

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        self.disconnect()

        self._close_csv()

    # ========================================================
    # READ PACKET
    # ========================================================

    def _read_packet(self):

        if not self.connected:
            return None

        # Find AA AA
        while self.running:

            first = self.ser.read(1)

            if not first:
                return None

            if first[0] != 0xAA:
                continue

            second = self.ser.read(1)

            if not second:
                return None

            if second[0] == 0xAA:
                break

        # Payload length
        length_byte = self.ser.read(1)

        if not length_byte:
            return None

        payload_length = length_byte[0]

        if payload_length > 169:
            return None

        # Payload
        payload = self.ser.read(
            payload_length
        )

        if len(payload) != payload_length:
            return None

        # Checksum
        checksum = self.ser.read(1)

        if not checksum:
            return None

        received_checksum = checksum[0]

        calculated_checksum = (
            (~sum(payload)) & 0xFF
        )

        if received_checksum != calculated_checksum:
            return None

        return payload

    # ========================================================
    # PARSE PAYLOAD
    # ========================================================

    def _parse_payload(self, payload):

        i = 0

        raw_eeg = None
        attention = None
        meditation = None
        signal_quality = None

        while i < len(payload):

            code = payload[i]

            # -----------------------------------------------
            # SIGNAL QUALITY
            # -----------------------------------------------

            if code == 0x02:

                if i + 1 >= len(payload):
                    break

                signal_quality = payload[i + 1]

                i += 2

            # -----------------------------------------------
            # ATTENTION
            # -----------------------------------------------

            elif code == 0x04:

                if i + 1 >= len(payload):
                    break

                attention = payload[i + 1]

                i += 2

            # -----------------------------------------------
            # MEDITATION
            # -----------------------------------------------

            elif code == 0x05:

                if i + 1 >= len(payload):
                    break

                meditation = payload[i + 1]

                i += 2

            # -----------------------------------------------
            # RAW EEG
            # -----------------------------------------------

            elif code == 0x80:

                if i + 3 >= len(payload):
                    break

                value_length = payload[i + 1]

                if value_length != 2:

                    i += 2 + value_length
                    continue

                high = payload[i + 2]
                low = payload[i + 3]

                raw_eeg = (
                    (high << 8) | low
                )

                # signed 16-bit
                if raw_eeg >= 32768:
                    raw_eeg -= 65536

                i += 4

            # -----------------------------------------------
            # EXTENDED VALUE
            # -----------------------------------------------

            elif code >= 0x80:

                if i + 1 >= len(payload):
                    break

                value_length = payload[i + 1]

                i += 2 + value_length

            # -----------------------------------------------
            # UNKNOWN
            # -----------------------------------------------

            else:

                i += 2

        return {
            "raw_eeg": raw_eeg,
            "attention": attention,
            "meditation": meditation,
            "signal_quality": signal_quality
        }

    # ========================================================
    # UPDATE
    # ========================================================

    def _update(self, data):

        raw_eeg = data["raw_eeg"]

        attention = data["attention"]
        meditation = data["meditation"]
        signal_quality = data["signal_quality"]

        with self.lock:

            if raw_eeg is not None:

                self.latest_raw_eeg = raw_eeg

                self.raw_eeg_buffer.append(
                    raw_eeg
                )

                self.sample_count += 1

            if attention is not None:
                self.latest_attention = attention

            if meditation is not None:
                self.latest_meditation = meditation

            if signal_quality is not None:
                self.latest_signal_quality = signal_quality

        # ----------------------------------------------------
        # Save ONLY when an actual raw EEG sample exists
        # ----------------------------------------------------

        if (
            raw_eeg is not None
            and self.csv_writer is not None
        ):

            timestamp = (
                datetime.now()
                .isoformat(timespec="milliseconds")
            )

            self.csv_writer.writerow([
                timestamp,
                raw_eeg,
                self.latest_attention,
                self.latest_meditation,
                self.latest_signal_quality
            ])

            self.csv_file.flush()

    # ========================================================
    # READER LOOP
    # ========================================================

    def _reader_loop(self):

        last_status = time.time()

        while self.running:

            if not self.connected:

                self.connect()

                if not self.connected:

                    time.sleep(0.5)
                    continue

            try:

                payload = self._read_packet()

                if payload is None:
                    continue

                self.packet_count += 1

                parsed = self._parse_payload(
                    payload
                )

                if parsed is None:
                    continue

                if parsed["raw_eeg"] is not None:

                    self.valid_packet_count += 1

                    self._update(parsed)

                else:

                    # Still update attention/
                    # meditation if present
                    self._update(parsed)

            except Exception:

                self.disconnect()

                time.sleep(0.5)

            # ------------------------------------------------
            # Status once every second
            # ------------------------------------------------

            if time.time() - last_status >= 1:

                with self.lock:

                    print(
                        f"🟢 LIVE | "
                        f"Raw EEG: {self.latest_raw_eeg} | "
                        f"Attention: {self.latest_attention} | "
                        f"Meditation: {self.latest_meditation} | "
                        f"Samples: {self.sample_count:,} | "
                        f"Packets: {self.packet_count:,}"
                    )

                last_status = time.time()

    # ========================================================
    # GET CURRENT DATA
    # ========================================================

    def get_latest(self):

        with self.lock:

            return {
                "connected": self.connected,
                "raw_eeg": self.latest_raw_eeg,
                "attention": self.latest_attention,
                "meditation": self.latest_meditation,
                "signal_quality": self.latest_signal_quality,
                "sample_count": self.sample_count,
                "packet_count": self.packet_count
            }

    # ========================================================
    # GET EEG BUFFER
    # ========================================================

    def get_raw_eeg(self):

        with self.lock:

            return list(
                self.raw_eeg_buffer
            )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("NEUROSTATE AI")
    print("MINDLINK LIVE EEG RECORDER")
    print("=" * 60)

    print()
    print("Keep MindLink OFF.")
    print("Wear it correctly.")
    print("Start this program.")
    print("Then turn MindLink ON.")
    print()

    reader = MindLinkReader()

    reader.start()

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        print()
        print("Stopping...")

    finally:

        reader.stop()

        print()
        print("Recording stopped.")
        print(
            f"Actual EEG samples recorded: "
            f"{reader.sample_count:,}"
        )