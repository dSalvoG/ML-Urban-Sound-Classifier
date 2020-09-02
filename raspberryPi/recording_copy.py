import pyaudio
import wave
import time
import datetime

# Sample rate: 48 kHz. Resolution: 16 bits. Channel: 1
chunk = 960
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000

# num_max: number of wav files to record.
RECORD_SECONDS = 4
# 50 min of recordings with 4 seconds
num_max = 750
times = 0

while times < num_max:
    start_time = time.time()
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    
    print("* recording")
    frames = []
    print(" times = %i" % times)

    for i in range(0, int(RATE / chunk * RECORD_SECONDS)):
        data = stream.read(chunk)
        frames.append(data)

    print("finished recording")
    todaydate = datetime.date.today()
    today = todaydate.strftime("%d_%m_%Y")
    file_name_with_extension = "a-a-audio-0-" + today + str(times) + ".wav"

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open("audio/input/" + file_name_with_extension, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print("Saved wav file: %s" % file_name_with_extension )
    time.sleep(25) # Time in seconds. Se tiene que calibrar
    times += 1
    # Tiempo que ha tardado en ejecutarse.
    print("--- %s seconds ---" % (time.time() - start_time))