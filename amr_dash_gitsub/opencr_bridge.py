import time
import serial
from PyQt5.QtCore import QObject, pyqtSignal, QThread

class OpenCRBridge(QObject):
    telemetry_sent_signal = pyqtSignal(str, bool) # (msg, is_connected)
    
    def __init__(self, port='/dev/ttyACM0', baudrate=115200):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.is_connected = False
        self.last_tx_msg = "Awaiting first command..."
        self.init_serial()

    def init_serial(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.is_connected = True
            print(f"[OpenCRBridge] Successfully connected to OpenCR on {self.port} @ {self.baudrate} baud", flush=True)
        except Exception as e:
            self.is_connected = False
            print(f"[OpenCRBridge] Serial connection warning for {self.port}: {e}", flush=True)

    def send_safety_command(self, safety_status):
        """
        Formats and transmits safety directional telemetry to OpenCR over UART.
        Packet example:
        CMD:ACT=FORWARD|STAT:F=1(1.45),B=1(2.10),L=0(0.28),R=1(0.92)\n
        """
        rec_cmd = safety_status.get('recommended_cmd', 'HALT')
        
        cmd_char = 's'
        if rec_cmd == 'FORWARD':
            cmd_char = 'f'
        elif rec_cmd == 'BACKWARD':
            cmd_char = 'b'
        elif rec_cmd == 'LEFT':
            cmd_char = 'l'
        elif rec_cmd == 'RIGHT':
            cmd_char = 'r'
            
        msg = f"{cmd_char}\n"
        self.last_tx_msg = msg.strip()

        if self.is_connected and self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.write(msg.encode('ascii'))
                self.serial_conn.flush()
            except Exception as e:
                print(f"[OpenCRBridge] Tx error: {e}", flush=True)
                self.is_connected = False
        else:
            # Try to reconnect if dropped
            if not self.is_connected:
                self.init_serial()

        self.telemetry_sent_signal.emit(self.last_tx_msg, self.is_connected)

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
            except Exception:
                pass
        self.is_connected = False
