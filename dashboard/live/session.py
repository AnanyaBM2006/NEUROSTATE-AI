# ============================================================
# NEUROSTATE AI
# LIVE EEG SESSION CONTROLLER
# ============================================================

import time


# ============================================================
# AVAILABLE SESSION DURATIONS
# ============================================================

SESSION_DURATIONS = {
    "1 Minute": 1 * 60,
    "5 Minutes": 5 * 60,
    "10 Minutes": 10 * 60,
    "20 Minutes": 20 * 60,
    "30 Minutes": 30 * 60,
    "45 Minutes": 45 * 60,
    "1 Hour": 60 * 60,
    "1.5 Hours": 90 * 60,
    "2 Hours": 120 * 60,
}


# ============================================================
# SESSION CONTROLLER
# ============================================================

class EEGSession:

    def __init__(self, duration_seconds):

        self.duration_seconds = int(
            duration_seconds
        )

        self.start_time = None

        self.end_time = None

        self.running = False

        self.completed = False


    # ========================================================
    # START SESSION
    # ========================================================

    def start(self):

        self.start_time = time.time()

        self.end_time = (
            self.start_time
            +
            self.duration_seconds
        )

        self.running = True

        self.completed = False


    # ========================================================
    # REMAINING TIME
    # ========================================================

    def remaining_seconds(self):

        if not self.running:

            return 0

        remaining = (
            self.end_time
            -
            time.time()
        )

        return max(
            0,
            remaining
        )


    # ========================================================
    # ELAPSED TIME
    # ========================================================

    def elapsed_seconds(self):

        if self.start_time is None:

            return 0

        return max(
            0,
            time.time()
            -
            self.start_time
        )


    # ========================================================
    # CHECK WHETHER SESSION IS COMPLETE
    # ========================================================

    def is_finished(self):

        if not self.running:

            return False

        if time.time() >= self.end_time:

            self.running = False

            self.completed = True

            return True

        return False


    # ========================================================
    # STOP MANUALLY
    # ========================================================

    def stop(self):

        self.running = False

        self.completed = True


    # ========================================================
    # FORMATTED TIME
    # ========================================================

    def formatted_remaining(self):

        remaining = int(
            self.remaining_seconds()
        )

        hours = remaining // 3600

        minutes = (
            remaining % 3600
        ) // 60

        seconds = (
            remaining % 60
        )

        if hours > 0:

            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )


    # ========================================================
    # FORMATTED ELAPSED
    # ========================================================

    def formatted_elapsed(self):

        elapsed = int(
            self.elapsed_seconds()
        )

        hours = elapsed // 3600

        minutes = (
            elapsed % 3600
        ) // 60

        seconds = (
            elapsed % 60
        )

        if hours > 0:

            return (
                f"{hours:02d}:"
                f"{minutes:02d}:"
                f"{seconds:02d}"
            )

        return (
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )