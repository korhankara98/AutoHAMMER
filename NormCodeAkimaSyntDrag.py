import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import find_peaks

# Kısayol tuşları:
# X - Nokta seçimi yapar
# P - Normalizasyon işlemini başlatır ve veriyi kaydeder
# U - Son seçilen noktayı geri alır
# B - İki nokta arasındaki maksimumları seçer
# M - Mevcut noktayı siler
# Mouse Sol Tık - Noktaları mouse ile sürükler

fits_file = '/Users/korhankara/Desktop/MScStellae/delOri/3800_7800_feros/delOri_4200-4500_feros.fits' # Dosya yolunu tekrar yaz

def load_fits_file(fits_file):
    with fits.open(fits_file) as hdul:
        print(f"Dosya adı: {fits_file}")
        hdul.info()
        
        # Veri HDU'sunu belirle
        if len(hdul) > 1 and hdul[1].data is not None:
            data = hdul[1].data
        else:
            data = hdul[0].data

        # Veri kolonları var mı kontrol et
        if data.dtype.names is not None:
            print("Kolon isimleri:", data.dtype.names)

            if 'WAVELENGTH' in data.dtype.names:
                wavelength = data['WAVELENGTH']
            elif 'lambda' in data.dtype.names:
                wavelength = data['lambda']
            else:
                wavelength = data.field(0)
                print("İlk kolon dalga boyu olarak kullanılıyor.")

            if 'FLUX' in data.dtype.names:
                flux = data['FLUX']
            elif 'NORMALIZED_FLUX' in data.dtype.names:
                flux = data['NORMALIZED_FLUX']
            elif 'flux' in data.dtype.names:
                flux = data['flux']
            else:
                flux = data.field(1)
                print("İkinci kolon akı olarak kullanılıyor.")
        else:
            # Eğer veri tek boyutlu bir dizi ise
            wavelength = None
            flux = data
            print("Veri tek boyutlu bir dizide bulundu.")

        return wavelength, flux

def normalize_spectrum(wavelength, flux, selected_wavelengths, selected_fluxes):
    """
    Tayfı normalize etmek için seçilen noktalara Akima1DInterpolator fit uygular ve akıyı bu eğriye böler.
    """
    sorted_indices = np.argsort(selected_wavelengths)
    sorted_wavelengths = np.array(selected_wavelengths)[sorted_indices]
    sorted_fluxes = np.array(selected_fluxes)[sorted_indices]

    if len(sorted_wavelengths) < 5:
        print("Akima1DInterpolator için en az 5 nokta gereklidir. Normalizasyon işlemi iptal edildi.")
        return flux, np.ones_like(flux)

    try:
        interpolator = Akima1DInterpolator(sorted_wavelengths, sorted_fluxes)
        continuum_fit = interpolator(wavelength)
        normalized_flux = flux / continuum_fit
        return normalized_flux, continuum_fit
    except Exception as e:
        print(f"Akima1DInterpolator hatası: {e}")
        return flux, np.ones_like(flux)

wavelength, flux = load_fits_file(fits_file)

# Eğer dalga boyu verisi yoksa, orijinal dosyadan yükle
if wavelength is None:
    original_fits_file = '/Users/korhankara/Desktop/MScStellae/delOri/3800_7800_feros/delOri_4200-4500_feros.fits' # Dosya yolunu tekrar yaz
    wavelength, _ = load_fits_file(original_fits_file)
    if wavelength is None:
        print("Dalga boyu verisi bulunamadı. İşlem sonlandırılıyor.")
        import sys
        sys.exit(1)

selected_wavelengths = []
selected_fluxes = []
selected_points = []

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

line1, = ax1.plot(wavelength, flux, 'g-', label='Akı vs Dalga Boyu')
normalization_line, = ax1.plot([], [], 'k--', label='Normalizasyon Eğrisi')
normalized_line, = ax2.plot([], [], 'g-', label='Normalize Edilmiş Akı', linewidth=2)

data_file = '/Users/korhankara/Desktop/Deneme/delOri_4200-4500.10'

try:
    # Verileri yükle
    data = np.loadtxt(data_file)
    wavelength_file = data[:, 0]
    flux_file = data[:, 1]

    # Verileri ikinci grafik üzerine çizdir
    ax2.plot(wavelength_file, flux_file, 'k-', label='Synthetic Data')

    # Grafiği güncelle
    ax2.legend()
    ax2.relim()
    ax2.autoscale_view()
except Exception as e:
    print(f"Dosya okunurken hata oluştu: {e}")

def add_point(x, y):
    selected_wavelengths.append(x)
    selected_fluxes.append(y)
    point, = ax1.plot(x, y, 'ro', picker=5)  # Enable picking with picker=5
    selected_points.append(point)
    update_normalization_line()
    plt.draw()

def remove_closest_point(x, y):
    if selected_wavelengths and selected_fluxes:
        distances = np.sqrt((np.array(selected_wavelengths) - x)**2 + (np.array(selected_fluxes) - y)**2)
        closest_idx = np.argmin(distances)
        if distances[closest_idx] < 0.5:  # Yakınlık eşiği
            selected_wavelengths.pop(closest_idx)
            selected_fluxes.pop(closest_idx)
            point = selected_points.pop(closest_idx)
            point.remove()
            update_normalization_line()
            plt.draw()

def find_peaks_custom(x_segment, y_segment):
    peaks, _ = find_peaks(y_segment)
    return peaks

def update_normalization_line():
    if len(selected_wavelengths) >= 5:
        sorted_indices = np.argsort(selected_wavelengths)
        sorted_wavelengths = np.array(selected_wavelengths)[sorted_indices]
        sorted_fluxes = np.array(selected_fluxes)[sorted_indices]

        try:
            interpolator = Akima1DInterpolator(sorted_wavelengths, sorted_fluxes)
            continuum_fit = interpolator(wavelength)
            normalization_line.set_data(wavelength, continuum_fit)

            # Normalize edilmiş akıyı hesapla ve güncelle
            normalized_flux = flux / continuum_fit
            normalized_line.set_data(wavelength, normalized_flux)

            ax2.relim()
            ax2.autoscale_view()
        except Exception as e:
            print(f"Akima1DInterpolator hatası: {e}")
            normalization_line.set_data([], [])
            normalized_line.set_data([], [])
    else:
        normalization_line.set_data([], [])
        normalized_line.set_data([], [])
    plt.draw()

def on_key(event):
    global selected_wavelengths, selected_fluxes, selected_points
    if event.key == 'x':  # Nokta seçimi yapar
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            add_point(x, y)
    elif event.key == 'p':  # Normalizasyon işlemini başlatır ve veriyi kaydeder
        if len(selected_wavelengths) >= 5:
            normalized_flux, continuum_fit = normalize_spectrum(
                wavelength, flux, selected_wavelengths, selected_fluxes
            )

            normalization_line.set_data(wavelength, continuum_fit)
            normalized_line.set_data(wavelength, normalized_flux)
            ax1.legend()
            ax2.legend()
            plt.draw()

            # Normalize edilmiş veriyi FITS dosyasına kaydetme
            from astropy.io import fits

            col1 = fits.Column(name='WAVELENGTH', array=wavelength, format='D')
            col2 = fits.Column(name='NORMALIZED_FLUX', array=normalized_flux, format='D')
            cols = fits.ColDefs([col1, col2])

            hdu = fits.BinTableHDU.from_columns(cols)
            new_fits_file = '/Users/korhankara/Desktop/MScStellae/delOri/3800_7800_feros/delOri_4200-4500_feros_Norm.fits' # _Norm.fits olarak bitsin örn. feros.fits --> feros_Norm.fits olarak yaz
            hdu.writeto(new_fits_file, overwrite=True)
            print(f'{new_fits_file} dosyası başarıyla IRAF uyumlu olarak oluşturuldu.')

            # Normalize edilmiş veriyi TXT dosyasına kaydetme
            txt_file = '/Users/korhankara/Desktop/MScStellae/delOri/3800_7800_feros/delOri_4200-4500_feros_Norm.txt' # _Norm.txt olarak bitecek
            np.savetxt(txt_file, np.column_stack((wavelength, normalized_flux)), fmt='%.6f')
            print(f'{txt_file} dosyasına normalize edilmiş veri başarıyla kaydedildi.')

            selected_wavelengths = []
            selected_fluxes = []
            for point in selected_points:
                point.remove()
            selected_points.clear()
            update_normalization_line()
        else:
            print("Normalizasyon için en az 5 nokta seçmelisiniz.")
    elif event.key == 'u':  # Son seçilen noktayı geri alır
        if selected_wavelengths and selected_fluxes:
            selected_wavelengths.pop()
            selected_fluxes.pop()
            point = selected_points.pop()
            point.remove()
            update_normalization_line()
            plt.draw()
    elif event.key == 'b':  # İki nokta arasındaki maksimumları seçer
        if len(selected_wavelengths) >= 2:
            x1, x2 = selected_wavelengths[-2], selected_wavelengths[-1]
            if x1 > x2:
                x1, x2 = x2, x1
            mask = (wavelength >= x1) & (wavelength <= x2)
            if np.any(mask):
                x_segment = wavelength[mask]
                y_segment = flux[mask]
                peak_indices = find_peaks_custom(x_segment, y_segment)
                peak_x = x_segment[peak_indices]
                peak_y = y_segment[peak_indices]
                for px, py in zip(peak_x, peak_y):
                    add_point(px, py)
    elif event.key == 'm':  # Mevcut noktayı siler
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            remove_closest_point(x, y)

# Variables to track dragging state
dragging_point = [None]  # Use a list to make it mutable in nested functions

def on_pick(event):
    # Identify which point was picked
    artist = event.artist
    if artist in selected_points:
        dragging_point[0] = selected_points.index(artist)

def on_motion(event):
    if dragging_point[0] is None:
        return
    if event.inaxes != ax1:
        return
    x_new, y_new = event.xdata, event.ydata
    if x_new is None or y_new is None:
        return
    # Update the data
    selected_wavelengths[dragging_point[0]] = x_new
    selected_fluxes[dragging_point[0]] = y_new
    # Update the plot point with lists
    selected_points[dragging_point[0]].set_data([x_new], [y_new])
    # Update normalization
    update_normalization_line()
    plt.draw()

def on_release(event):
    dragging_point[0] = None

# Connect the event handlers
fig.canvas.mpl_connect('key_press_event', on_key)
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)
fig.canvas.mpl_connect('button_release_event', on_release)

# İlk grafik ayarları
ax1.set_xlabel('Dalga Boyu')
ax1.set_ylabel('Akı')
ax1.set_title('Dalga Boyu vs Akı ve Normalizasyon Eğrisi')
ax1.legend()
# ax1.set_facecolor('black')
ax1.grid(True)

# İkinci grafik ayarları
ax2.set_xlabel('Dalga Boyu')
ax2.set_ylabel('Normalize Edilmiş Akı')
ax2.set_title('Dalga Boyu vs Normalize Edilmiş Akı')
ax2.legend()
# ax2.set_facecolor('black')
ax2.grid(True)

plt.tight_layout()
plt.show()

# Normalize edilmiş veriyi yükleme ve çizdirme
new_fits_file = '/Users/korhankara/Desktop/MScStellae/delOri/3800_7800_feros/delOri_4200-4500_feros_Norm.fits' # _Norm olarak biten kısmı 193. satırdakini yaz
wavelength_norm, normalized_flux_data = load_fits_file(new_fits_file)

plt.figure(figsize=(10, 6))
plt.plot(wavelength_norm, normalized_flux_data, 'g-', label='Normalize Edilmiş Akı')
plt.xlabel('Dalga Boyu')
plt.ylabel('Normalize Edilmiş Akı')
plt.title('Dalga Boyu - Normalize Edilmiş Akı')
plt.legend()
plt.grid(True)
plt.show()