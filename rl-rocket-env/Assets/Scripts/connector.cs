using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;


    internal class connector : MonoBehaviour
    {
        public env envScript;

        public int port = 5000;
        public string ip = "127.0.0.1";

        private TcpListener server;
        private TcpClient client;
        private Thread serverThread;

        // Thread-safe paylaşım
        private readonly object _lock = new object();

        private string _incomingLine = null;      // Update() bunu işleyecek
        private string _outgoingLine = "";        // thread bunu gönderecek

        private bool _messageReady = false;       // Update() için bayrak
        private bool _unityResponded = false;     // thread için bayrak

        private bool _stopRequested = false;

        // TCP framing buffer
        private string _recvBuffer = "";

        void Start()
        {
            Physics.simulationMode = SimulationMode.Script;

            serverThread = new Thread(StartServerLoop);
            serverThread.IsBackground = true;
            serverThread.Start();
        }

        void StartServerLoop()
        {
            try
            {
                server = new TcpListener(IPAddress.Parse(ip), port);
                server.Start();
                Debug.Log("Server is open, waiting python...");

                client = server.AcceptTcpClient();
                Debug.Log("Python connected");

                NetworkStream stream = client.GetStream();
                byte[] buffer = new byte[4096];

                while (!_stopRequested && client.Connected)
                {
                    int bytesRead = stream.Read(buffer, 0, buffer.Length);
                    if (bytesRead <= 0) break;

                    string chunk = Encoding.UTF8.GetString(buffer, 0, bytesRead);

                    // Buffer'a ekle
                    _recvBuffer += chunk;

                    // \n ile gelen tüm satırları işle
                    while (true)
                    {
                        int nl = _recvBuffer.IndexOf('\n');
                        if (nl < 0) break;

                        // 1 satır komut al
                        string line = _recvBuffer.Substring(0, nl).Trim();
                        _recvBuffer = _recvBuffer.Substring(nl + 1);

                        if (string.IsNullOrEmpty(line))
                            continue;

                        // Unity tarafına işi ver
                        lock (_lock)
                        {
                            _incomingLine = line;
                            _messageReady = true;
                            _unityResponded = false;
                        }

                        // Unity Update() cevap üretip bayrak çevirene kadar bekle
                        while (!_stopRequested && client.Connected)
                        {
                            bool responded;
                            lock (_lock) { responded = _unityResponded; }
                            if (responded) break;
                            Thread.Sleep(1);
                        }

                        // Cevabı gönder (satır sonu ile!)
                        string toSend;
                        lock (_lock) { toSend = _outgoingLine; }

                        if (!toSend.EndsWith("\n"))
                            toSend += "\n";

                        byte[] sendMsg = Encoding.UTF8.GetBytes(toSend);
                        stream.Write(sendMsg, 0, sendMsg.Length);
                        stream.Flush();
                    }
                }
            }
            catch (Exception ex)
            {
                Debug.LogError("ServerLoop exception: " + ex.Message);
            }
            finally
            {
                try { client?.Close(); } catch { }
                try { server?.Stop(); } catch { }
            }
        }

        void Update()
        {
            bool hasMsg;
            string msg;

            lock (_lock)
            {
                hasMsg = _messageReady;
                msg = _incomingLine;
                if (hasMsg) _messageReady = false;
            }

            if (!hasMsg || string.IsNullOrEmpty(msg))
                return;

            // 1) aksiyonu uygula
            envScript.doAction(msg);

            // 2) 1 physics step ilerlet
            Physics.Simulate(Time.fixedDeltaTime);

            // 3) state üret
            string state = envScript.getStates();

            // 4) thread'e cevabı ver
            lock (_lock)
            {
                _outgoingLine = state;  // newline eklemeyi thread yapıyor
                _unityResponded = true;
            }
        }

        void OnApplicationQuit()
        {
            Physics.simulationMode = SimulationMode.FixedUpdate;

            _stopRequested = true;

            try { client?.Close(); } catch { }
            try { server?.Stop(); } catch { }

            // Thread.Abort yok — güvenli kapanış
            if (serverThread != null && serverThread.IsAlive)
            {
                serverThread.Join(200); // kısa bekleme
            }
        }
    }
