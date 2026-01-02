using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;
using System.Globalization;

public class env : MonoBehaviour
{
    [Header("Physics & Components")]
    public Rigidbody rb;
    public Transform enginePoint; // Kuvvetin uygulanaca�� nokta
    public Transform targetPoint; // �ni� yap�lacak hedef (Pist)

    [Header("Effects")]
    public ParticleSystem engineParticles;
    public ParticleSystem.EmissionModule emissionModule;
    public ParticleSystem.MainModule mainModule;

    [Header("Thrust & Control Settings")]
    public float mainThrustPower = 150f;
    public float rcsPower = 10f; // Eski koddaki torque g�c�

    private Vector3 feetOffset; // Ayaklar�n taban noktas� mesafesi

    void Start()
    {
            // Kısıtlamaları KALDIRIYORUZ. 
            // Roket tamamen serbest olsun, dengesini yapay zeka sağlasın.
        rb.constraints = RigidbodyConstraints.None; 

            // Otomatik Ayak Mesafesi Hesaplama (Önceki konuşmamızdan)
        Collider col = GetComponent<Collider>();
        if (col != null)
        {
            float halfHeight = col.bounds.extents.y / transform.localScale.y;
            feetOffset = new Vector3(0, -halfHeight, 0);
        }
    }

    void FixedUpdate()
    {
            
    }

    public void ResetEnv(float x, float y, float z, float pitch, float yaw)
    {
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;

        transform.position = new Vector3(x, y, z);
            
            // Başlangıçta düzgün doğsun ama sonra serbest kalsın
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);
            
            // Kısıtlama yok
        rb.constraints = RigidbodyConstraints.None;

        if (engineParticles != null)
        {
            engineParticles.Stop();
            engineParticles.Clear();
        }

        Debug.Log("........ORTAM SIFIRLANDI..........");
    }

    // --- RL �LET���M METODLARI ---

    // Python'dan gelen komutlar� i�ler
    public void doAction(string dataString)
    {
        dataString = dataString.Replace("[", "").Replace("]", "").Replace(" ", "").Trim();

        if (string.IsNullOrEmpty(dataString)) return;

        string[] parts = dataString.Split(',');

        if (parts.Length < 1) return;

        if (!float.TryParse(parts[0], NumberStyles.Any, CultureInfo.InvariantCulture, out float modeRaw))
        {
            return; // Mod okunamadıysa çık
        }

        int mode = (int)modeRaw;

        switch (mode)
        {
            case 1: // RESET
                if (parts.Length >= 6)
                {
                    float x = ParseFloat(parts[1]);
                    float y = ParseFloat(parts[2]);
                    float z = ParseFloat(parts[3]);
                    float pitch = ParseFloat(parts[4]);
                    float yaw = ParseFloat(parts[5]);

                    ResetEnv(x, y, z, pitch, yaw);
                }
                break;

            case 0:
                if (parts.Length >= 5)
                {
                    float pitch = ParseFloat(parts[1]);
                    float yaw = ParseFloat(parts[2]);
                    float thrust = ParseFloat(parts[3]);
                    float roll = ParseFloat(parts[4]);

                    ApplyPhysics(pitch, yaw, thrust, roll);
                }
                break;
        }
    }

    // Roketin durumunu Python'a string olarak g�nderir
    private float ParseFloat(string value)
    {
        if (float.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out float result))
        {
            return result;
        }
        return 0.0f;
    }

    private void ApplyPhysics(float pitch, float yaw, float thrust, float roll)
    {
        float motorGucu = Mathf.Clamp01(thrust);
        rb.AddRelativeForce(Vector3.up * motorGucu * mainThrustPower);

        // EKSENLERİ DOĞRU OTURTMA:
        // transform.right   (X) -> PITCH (Öne arkaya yatma)
        // transform.forward (Z) -> YAW (Sağa sola yatma)
        // transform.up      (Y) -> ROLL (Kendi ekseninde dönme / Spin)
        
        Vector3 pitchTork = transform.right * pitch * rcsPower;
        Vector3 yawTork   = transform.forward * yaw * rcsPower;
        Vector3 rollTork  = transform.up * roll * (rcsPower * 0.1f); // Artık roll çalışıyor

        rb.AddTorque(pitchTork + yawTork + rollTork);

        // Efekt Kodları Aynen Kalabilir
        if (engineParticles != null)
        {
            var emission = engineParticles.emission;
            if (motorGucu > 0.01f)
            {
                if (!engineParticles.isPlaying) engineParticles.Play();
                emission.rateOverTime = motorGucu * 500f;
            }
            else
            {
                emission.rateOverTime = 0f;
            }
        }
    }

    // connector.cs ilet
    public string getStates()
    {
        if (targetPoint == null || rb == null) return "";
        Vector3 globalFeetPos = transform.TransformPoint(feetOffset);

        float dx = targetPoint.position.x - globalFeetPos.x;
        float dz = targetPoint.position.z - globalFeetPos.z;
        float dy = globalFeetPos.y - targetPoint.position.y;
        Vector3 velocity = rb.linearVelocity;
        Vector3 angularVel = rb.angularVelocity;

        Quaternion rotation = transform.rotation;

        // Python beklenen sıra: dx, dy, dz, vx, vy, vz, wx, wy, wz, qx, qy, qz, qw
        // InvariantCulture kullanarak ondalık ayırıcıyı nokta (.) yapıyoruz
        string states =
            dx.ToString(CultureInfo.InvariantCulture) + "," +
            dy.ToString(CultureInfo.InvariantCulture) + "," +
            dz.ToString(CultureInfo.InvariantCulture) + "," +
            velocity.x.ToString(CultureInfo.InvariantCulture) + "," +
            velocity.y.ToString(CultureInfo.InvariantCulture) + "," +
            velocity.z.ToString(CultureInfo.InvariantCulture) + "," +
            angularVel.x.ToString(CultureInfo.InvariantCulture) + "," +
            angularVel.y.ToString(CultureInfo.InvariantCulture) + "," +
            angularVel.z.ToString(CultureInfo.InvariantCulture) + "," +
            rotation.x.ToString(CultureInfo.InvariantCulture) + "," +
            rotation.y.ToString(CultureInfo.InvariantCulture) + "," +
            rotation.z.ToString(CultureInfo.InvariantCulture) + "," +
            rotation.w.ToString(CultureInfo.InvariantCulture);

        return  states;
    }
}