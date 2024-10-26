import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Faktor skala yang lebih besar
scale_factor = 100000  # Perbesar skala data untuk melihat perbedaan

# Baca data CSV dengan pemisah titik koma
data = pd.read_csv('insert file path here', sep=';')

# Konversi kolom yang mengandung percepatan menjadi numerik
data['ax (m/s^2)'] = pd.to_numeric(data['ax (m/s^2)'], errors='coerce')
data['ay (m/s^2)'] = pd.to_numeric(data['ay (m/s^2)'], errors='coerce')
data['az (m/s^2)'] = pd.to_numeric(data['az (m/s^2)'], errors='coerce')
data['aT (m/s^2)'] = pd.to_numeric(data['aT (m/s^2)'], errors='coerce')
data['time'] = pd.to_numeric(data['time'], errors='coerce')

# Drop NaN values
time = data['time'].dropna().values  # Kolom waktu dalam detik (ambil sebagai array)
accel_x = data['ax (m/s^2)'].dropna().values  # Percepatan di sumbu X
accel_y = data['ay (m/s^2)'].dropna().values  # Percepatan di sumbu Y
accel_z = data['az (m/s^2)'].dropna().values  # Percepatan di sumbu Z
accel_total = data['aT (m/s^2)'].dropna().values  # Percepatan total

# Terapkan faktor skala pada percepatan agar lebih terlihat pada grafik
accel_x *= scale_factor
accel_y *= scale_factor
accel_z *= scale_factor
accel_total *= scale_factor

# Fungsi untuk menerapkan filter Butterworth dengan pengecekan panjang data
def butterworth_filter(data, cutoff=0.1, fs=50, order=5):
    if len(data) > 18:  # Cek apakah data cukup panjang
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)
    else:
        # Jika data kurang dari 18 sampel, tidak melakukan filtering
        return data

# Terapkan filter pada data percepatan jika data cukup panjang
filtered_accel_x = butterworth_filter(accel_x)
filtered_accel_y = butterworth_filter(accel_y)
filtered_accel_z = butterworth_filter(accel_z)

# Hitung total percepatan setelah filtering
filtered_accel_total = np.sqrt(filtered_accel_x**2 + filtered_accel_y**2 + filtered_accel_z**2)

# Fungsi manual untuk integrasi kumulatif menggunakan metode trapezoidal
def cumtrapz_manual(y, x):
    if len(x) > 1:
        return np.concatenate([[0], np.cumsum(np.diff(x) * (y[1:] + y[:-1]) / 2)])
    else:
        return np.array([0])

# Integrasi untuk mendapatkan kecepatan
velocity_x = cumtrapz_manual(filtered_accel_x, time)
velocity_y = cumtrapz_manual(filtered_accel_y, time)
velocity_z = cumtrapz_manual(filtered_accel_z, time)
velocity_total = np.sqrt(velocity_x**2 + velocity_y**2 + velocity_z**2)

# Sesuaikan panjang time dengan data terpendek
min_length = min(len(time), len(velocity_x), len(velocity_y), len(velocity_z))

time_shortened = time[:min_length]
velocity_x = velocity_x[:min_length]
velocity_y = velocity_y[:min_length]
velocity_z = velocity_z[:min_length]
velocity_total = velocity_total[:min_length]
filtered_accel_x = filtered_accel_x[:min_length]
filtered_accel_y = filtered_accel_y[:min_length]
filtered_accel_z = filtered_accel_z[:min_length]
filtered_accel_total = filtered_accel_total[:min_length]

# Terapkan faktor skala pada kecepatan dan percepatan
velocity_x *= scale_factor
velocity_y *= scale_factor
velocity_z *= scale_factor
velocity_total *= scale_factor

# Integrasi untuk mendapatkan perpindahan
displacement_x = cumtrapz_manual(velocity_x, time_shortened)
displacement_y = cumtrapz_manual(velocity_y, time_shortened)
displacement_z = cumtrapz_manual(velocity_z, time_shortened)
displacement_total = np.sqrt(displacement_x**2 + displacement_y**2 + displacement_z**2)

# Pangkas perpindahan agar sesuai dengan time_shortened
displacement_x = displacement_x[:min_length]
displacement_y = displacement_y[:min_length]
displacement_z = displacement_z[:min_length]
displacement_total = displacement_total[:min_length]

# Terapkan faktor skala pada perpindahan
displacement_x *= scale_factor
displacement_y *= scale_factor
displacement_z *= scale_factor
displacement_total *= scale_factor

# Hitung gaya total (dengan gaya gravitasi)
mass = 1.0  # Massa benda dalam kg, sesuaikan sesuai kebutuhan
force_total = mass * filtered_accel_total

# Terapkan faktor skala pada gaya total
force_total *= scale_factor

# Hitung usaha total dengan interval rata-rata waktu
if len(time_shortened) > 1:
    delta_t = np.diff(time_shortened).mean()  # Hitung selisih waktu rata-rata
    work_done = np.sum(force_total * displacement_total) * delta_t
else:
    work_done = 0  # Jika data waktu kurang dari 2 poin, usaha diatur ke 0

# Hitung energi kinetik
kinetic_energy = 0.5 * mass * velocity_total**2

# Terapkan faktor skala pada energi kinetik
kinetic_energy *= scale_factor

# Hitung energi potensial (asumsikan z sebagai ketinggian)
g = 9.81  # Gravitasi
potential_energy = mass * g * displacement_z

# Terapkan faktor skala pada energi potensial
potential_energy *= scale_factor

# Plot data
plt.figure(figsize=(14, 10))

# Plot percepatan
plt.subplot(3, 2, 1)
plt.plot(time_shortened, filtered_accel_x, label='Accel X')
plt.plot(time_shortened, filtered_accel_y, label='Accel Y')
plt.plot(time_shortened, filtered_accel_z, label='Accel Z')
plt.plot(time_shortened, filtered_accel_total, label='Accel Total')
plt.title('Filtered Acceleration (Scaled)')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.legend()

# Plot kecepatan
plt.subplot(3, 2, 2)
plt.plot(time_shortened, velocity_x, label='Velocity X')
plt.plot(time_shortened, velocity_y, label='Velocity Y')
plt.plot(time_shortened, velocity_z, label='Velocity Z')
plt.plot(time_shortened, velocity_total, label='Velocity Total')
plt.title('Velocity (Scaled)')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()

# Plot perpindahan
plt.subplot(3, 2, 3)
plt.plot(time_shortened, displacement_x, label='Displacement X')
plt.plot(time_shortened, displacement_y, label='Displacement Y')
plt.plot(time_shortened, displacement_z, label='Displacement Z')
plt.plot(time_shortened, displacement_total, label='Displacement Total')
plt.title('Displacement (Scaled)')
plt.xlabel('Time (s)')
plt.ylabel('Displacement (m)')
plt.legend()

# Plot gaya total
plt.subplot(3, 2, 4)
plt.plot(time_shortened, force_total, label='Force Total')
plt.title('Total Force (Scaled)')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()

# Plot energi kinetik
plt.subplot(3, 2, 5)
plt.plot(time_shortened, kinetic_energy, label='Kinetic Energy')
plt.title('Kinetic Energy (Scaled)')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()

# Plot energi potensial
plt.subplot(3, 2, 6)
plt.plot(time_shortened, potential_energy, label='Potential Energy')
plt.title('Potential Energy (Scaled)')
plt.xlabel('Time (s)')
plt.ylabel('Energy (J)')
plt.legend()

plt.tight_layout()
plt.show()
