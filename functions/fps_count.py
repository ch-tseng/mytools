#fps count
start = time.time()
last_time = time.time()
last_frames = 0

def fps_count(num_frames):
    end = time.time()
    seconds = end - start
    fps  = num_frames / seconds;
    print("FPS: {0}".format(fps))
    return fps

def fps_count2(total_frames):
    global last_time, last_frames

    timenow = time.time()
    if(timenow - last_time)>60:
        fps  = (total_frames - last_frames) / (timenow - last_time)
        print("FPS: {0}".format(fps))

        last_time  = timenow
        last_frames = total_frames

def fps_count3(total_frames):
    global last_time, last_frames

    timenow = time.time()
    if(timenow - last_time)>6:
        fps  = (total_frames - last_frames) / (timenow - last_time)
        print("FPS: {0}".format(fps))

        last_time  = timenow
        last_frames = total_frames
