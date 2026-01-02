using UnityEngine;
using System.Globalization; // Ondalýk sayý (nokta/virgül) ayrýmý için kritik

public class env : MonoBehaviour
{
    [Header("Physics & Components")]
    public Rigidbody rb;
    public Transform enginePoint; // Kuvvetin uygulanacaðý nokta
    public Transform targetPoint; // Ýniþ yapýlacak hedef (Pist)

    [Header("Effects")]
    public ParticleSystem engineParticles;
    private ParticleSystem.EmissionModule emissionModule;
    private ParticleSystem.MainModule mainModule;

    [Header("Thrust & Control Settings")]
    public float mainThrustPower = 150f;
    public float rcsPower = 10f; // Eski koddaki torque gücü

    private Vector3 feetOffset; // Ayaklarýn taban noktasý mesafesi

    void Start()
    {
        if (rb == null) rb = GetComponent<Rigidbody>();

        // Roket serbest kalsýn, dengeyi yapay zeka saðlasýn
        rb.constraints = RigidbodyConstraints.None;
        rb.interpolation = RigidbodyInterpolation.Interpolate;

        // Partikül hazýrlýðý
        if (engineParticles != null)
        {
            emissionModule = engineParticles.emission;
            mainModule = engineParticles.main;
        }

        // Otomatik Ayak Mesafesi Hesaplama (Geliþtirilmiþ)
        // Roketin en alt noktasýný (Foot_Pad) bulmak için
        CalculateFeetOffset();
    }

    void CalculateFeetOffset()
    {
        // En basit haliyle roketin altýna doðru bir mesafe tanýmlýyoruz
        // Senin tasarýmýna göre (Foot_Pad Y: -3.35 idi), yaklaþýk -3.4f diyebiliriz
        feetOffset = new Vector3(0, -3.4f, 0);
    }

    // --- RL ÝLETÝÞÝM METODLARI ---

    // Python'dan gelen komutlarý iþler
    public void doAction(string dataString)
    {
        dataString = dataString.Replace("[", "").Replace("]", "").Replace(" ", "").Trim();
        if (string.IsNullOrEmpty(dataString)) return;

        string[] parts = dataString.Split(',');
        if (parts.Length < 1) return;

        if (!float.TryParse(parts[0], NumberStyles.Any, CultureInfo.InvariantCulture, out float modeRaw)) return;
        int mode = (int)modeRaw;

        switch (mode)
        {
            case 1: // RESET MODU
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

            case 0: // UÇUÞ MODU
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

    // Roketin durumunu Python'a string olarak gönderir
    public string getStates()
    {
        if (targetPoint == null || rb == null) return "";

        // Ayaklarýn dünya koordinatýndaki yeri
        Vector3 globalFeetPos = transform.TransformPoint(feetOffset);

        // Hedefe olan uzaklýk (Mesafe hatasý / Error)
        float dx = targetPoint.position.x - globalFeetPos.x;
        float dy = globalFeetPos.y - targetPoint.position.y; // Yükseklik
        float dz = targetPoint.position.z - globalFeetPos.z;

        // Hýzlar (Unity 6: linearVelocity ve angularVelocity)
        Vector3 vel = rb.linearVelocity;
        Vector3 angVel = rb.angularVelocity;
        Quaternion rot = transform.rotation;

        // Verileri Python'un beklediði sýrayla birleþtir
        return string.Format(CultureInfo.InvariantCulture,
            "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}",
            dx, dy, dz, vel.x, vel.y, vel.z, angVel.x, angVel.y, angVel.z, rot.x, rot.y, rot.z, rot.w);
    }

    // --- FÝZÝK VE KONTROL ---

    private void ApplyPhysics(float pitch, float yaw, float thrust, float roll)
    {
        float motorGucu = Mathf.Clamp01(thrust);

        // Ana Ýtki (Thrust) - EnginePoint üzerinden
        rb.AddForceAtPosition(transform.up * motorGucu * mainThrustPower, enginePoint.position, ForceMode.Force);

        // Tork (Yönlendirme) - Eski kodundaki eksen mantýðýyla
        Vector3 pitchTork = transform.right * pitch * rcsPower;
        Vector3 yawTork = transform.forward * yaw * rcsPower;
        Vector3 rollTork = transform.up * roll * (rcsPower * 0.5f);

        rb.AddTorque(pitchTork + yawTork + rollTork, ForceMode.Force);

        // Görsel Efekt Güncelleme
        UpdateEffects(motorGucu);
    }

    public void ResetEnv(float x, float y, float z, float pitch, float yaw)
    {
        rb.linearVelocity = Vector3.zero;
        rb.angularVelocity = Vector3.zero;
        transform.position = new Vector3(x, y, z);
        transform.rotation = Quaternion.Euler(pitch, yaw, 0f);

        if (engineParticles != null) { engineParticles.Stop(); engineParticles.Clear(); }
        Debug.Log("Environment Reseted");
    }

    private void UpdateEffects(float thrust)
    {
        if (engineParticles == null) return;
        if (thrust > 0.01f)
        {
            emissionModule.rateOverTime = thrust * 200f;
            mainModule.startSpeed = 10f + (thrust * 20f);
            if (!engineParticles.isPlaying) engineParticles.Play();
        }
        else
        {
            emissionModule.rateOverTime = 0f;
            if (engineParticles.isPlaying) engineParticles.Stop();
        }
    }

    private float ParseFloat(string value)
    {
        return float.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out float res) ? res : 0f;
    }
}