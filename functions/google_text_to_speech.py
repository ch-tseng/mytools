#List all langs: gtts-cli --all
from gtts import gTTS
import tempfile
from pygame import mixer
 
def say(text, filename=None):
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        tts = gTTS(text, lang='zh-tw',slow=False)
        if filename is None:
            filename = "{}.mp3".format(temp.name)
        tts.save(filename)
        mixer.init()
        mixer.music.load(filename)
        mixer.music.play()
        while mixer.music.get_busy() == True:
            continue
        mixer.quit()
say("教務觸報告，請以下同學，下課時間溪帶紙與筆到2樓穿堂集合")


#say("有了這一行，你就能把這句話錄音成MP3了", "introduction.mp3")
