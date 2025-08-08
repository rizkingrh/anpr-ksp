@echo off
REM Mendapatkan IP address dari PC (IPv4)
FOR /F "tokens=2 delims=:" %%A IN ('ipconfig ^| findstr /C:"IPv4 Address"') DO (
    SET IP_PC=%%A
)

REM Menghapus spasi awal dari IP
SET IP_PC=%IP_PC:~1%

REM Menjalankan Laravel server
start "" cmd /k "cd /d C:\laragon\www\anpr-dashboard && php artisan serve --host=%IP_PC%"

REM Menunggu sebentar agar server sempat start
timeout /t 10 > nul

REM Membuka browser ke dashboard
start http://%IP_PC%:8000
