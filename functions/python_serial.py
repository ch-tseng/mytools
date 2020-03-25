import serial

COM_PORT = 'COM4'
BAUD_RATES = 9600    # 設定傳輸速率
ser = serial.Serial(COM_PORT, BAUD_RATES)

try:
    while True:

        while ser.in_waiting:
            data_raw = ser.readline()
            data = data_raw.decode('utf8', 'ignore')
            print('接收到的原始資料：', data_raw)
            print('接收到的資料：', data)

except KeyboardInterrupt:
    ser.close()
    print('再見！')