import socket

class Connector():
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        print(f"Unity ({self.ip}:{self.port}) aranıyor...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        print("Bağlandı!")
        self._buf = ""  # <-- EKLENDİ

    def sendCs(self, data):
        msg = ",".join(map(str, data)) + "\n"
        self.sock.sendall(msg.encode("utf-8"))

    def readCs(self):
        # newline gelene kadar oku (bufferlı)
        while "\n" not in self._buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Unity bağlantısı kapandı (recv=0).")
            self._buf += chunk.decode("utf-8", errors="replace")

        line, self._buf = self._buf.split("\n", 1)
        line = line.strip()

        # BOŞ SATIR GELİRSE: yut ve tekrar oku
        if line == "":
            return self.readCs()

        return line
