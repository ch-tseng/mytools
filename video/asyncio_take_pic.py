import asyncio
import aiofiles
import cv2
import numpy as np

cam_id = r'C:\Users\ch.tseng\Videos\tcar01.mp4'
cam_id = cam_id.replace('\\', '/')
video_size = (1920,1080)
camera = cv2.VideoCapture(cam_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

print("USB Camera's resolution is: %d x %d" % (width, height))

async def take_photo():
    # 模擬使用 OpenCV 拍照
    print("Taking photo...")
    # 這裡使用 OpenCV 的模擬函式生成一張圖片
    (grabbed, frame) = camera.read()
    #await asyncio.sleep(2)  # 模擬耗時的拍照操作
    print("Photo taken")
    return frame

async def write_to_file(photo_data):
    # 將圖片數據寫入文件
    async with aiofiles.open('photo.jpg', 'wb') as file:
        print("Writing to file...")
        await file.write(cv2.imencode('.jpg', photo_data)[1])
        print("Write to file complete")

async def main():
    (grabbed, frame) = camera.read()
    while grabbed is True:
        cv2.imshow("Frame", frame)
        k = cv2.waitKey(1)
        if k == ord('p'):
            # 啟動 take_photo 協程，並獲取拍照後的圖片數據
            if grabbed is True:
                photo_data = await take_photo()

                # 使用 async with 開始非同步寫入文件操作
                await write_to_file(photo_data)

        (grabbed, frame) = camera.read()

# 創建事件循環
loop = asyncio.get_event_loop()

# 使用 run_until_complete 啟動 main 協程
loop.run_until_complete(main())

# 關閉事件循環
loop.close()
