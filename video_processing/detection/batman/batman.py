import cv2
import numpy as np
import peakutils
import matplotlib as mpl
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import subprocess
from scipy.io import wavfile
import matplotlib.pyplot as plt


######################## Setup #######################

DIR = "C:\\Users\\XXXXX\\Desktop\\test_\\"  # if changing the root directory - also change manually in row 121
NAME_TAG = "day4short"  # name of the source video file
VIDEO_TYPE = ".mp4"  # change according to file type
VIDEO_PATH = DIR + NAME_TAG + VIDEO_TYPE
TEXT_PATH = DIR + NAME_TAG + "_vid.txt"
AUDIO_PATH = DIR + NAME_TAG + ".wav"
PLOT_PATH = DIR + NAME_TAG + ".png"
FPS = 25
md = 5  # mod divider for down-sampling


def downSample(arr, n):
    end = n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


def mulArr(arr, mul):
    return map(lambda a: a * mul, arr)


################## Video Processing ###################

print ""
print ""
print "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
print "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
print ":::::::::::::::::::::::::::::::::::::::::::::-'    `-:::::::::::::::::::"
print "::::::::::::::::::::::::::::::::::::::::::-'          `:::::::::::::::::"
print ":::::::::::::::::::::::::::::::::::::::-  '   /(_M_)\\  `::::::::::::::::"
print ":::::::::::::::::::::::::::::::::::-'        |       |  ::::::::::::::::"
print "::::::::::::::::::::::::::::::::-         .   \\/~V~\\/  ,::::::::::::::::"
print "::::::::::::::::::::::::::::-'             .          ,:::::::::::::::::"
print ":::::::::::::::::::::::::-'                 `-.    .-:::::::::::::::::::"
print ":::::::::::::::::::::-'                  _,,-:::::::::::::::::::::::::::"
print "::::::::::::::::::-'                _,--::::::::::::::::::::::::::::::::"
print "::::::::::::::-'               _.--::::::::::::::::::::::#####::::::::::"
print ":::::::::::-'             _.--:::::::::::::::::::::::::::#####:::::#####"
print "::::::::'    ##     ###.-::::::###:::::::::::::::::::::::#####:::::#####"
print "::::-'       ###_.::######:::::###::::::::::::::#####:##########:::#####"
print ":'         .:###::########:::::###::::::::::::::#####:##########:::#####"
print "     ...--:::###::########:::::###:::::######:::#####:##########:::#####"
print " _.--:::##:::###:#########:::::###:::::######:::#####:##################"
print "'#########:::###:#########::#########::######:::#####:##################"
print ":#########:::#############::#########::######:::########################"
print "##########:::########################::#################################"
print "##########:::########################::#################################"
print "##########:::########################::#################################"
print "########################################################################"
print "################################################# Calling BATMAN #######"
print "############ (Ben's Algorithm for Tracking of Movement After Noise) ####"
print "########################################################################"
print ""
print ""


f = open(TEXT_PATH, 'w')  # text file for keeping the movement pixels
cap = cv2.VideoCapture(VIDEO_PATH)  # capture the input video
fgbg = cv2.BackgroundSubtractorMOG2()  # movement detection algorithm || py3: cv2.createBackgroundSubtractorMOG2()
white = 0  # white pixels counter
l = 0  # frames counter

while(1):

    ret, frame = cap.read()
    if not ret:
        break

    if l % md == 0:
        fgmask = fgbg.apply(frame)
        fgmask = cv2.resize(fgmask, (0, 0), fx=0.25, fy=0.25)
        # startY, endY, startX, endX = 50, 250, 150, 350
        # fgmask = cv2.rectangle(fgmask, (startX, startY), (endX, endY), (255, 255, 255), 3)  # (x,y), (x+w,y+h)
        cv2.imshow('frame', fgmask)
        cv2.moveWindow('frame', 100, 20)
        # fgmask = fgmask[startY:endY, startX:endX]  # ROI
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        h, w = fgmask.shape
        if l == md:
            print h
            print w

        for i in range(h):  # sum up the movement pixels
            for k in range(w):
                pixel = fgmask[i][k]
                if pixel == 255:
                    white += 1

        f.write("{}\n".format(white))
        white = 0

    l += 1

f.close()
cv2.destroyWindow('frame')


with open(TEXT_PATH) as g:
    lines = g.readlines()
    W = [line.split("\n")[0] for line in lines]

g.close()


################## Audio Processing ###################

# create audio file from video, notice that the path is in a different format
command = "ffmpeg -i C:/Users/XXXXX/Desktop/test_/" + NAME_TAG + VIDEO_TYPE + " -ab 160k -ac 2 -ar 44100 -vn C:/Users/XXXXX/Desktop/test_/" + NAME_TAG + ".wav"
subprocess.call(command, shell=True)


sampFreq, snd = wavfile.read(AUDIO_PATH)
ch1 = snd[:, 0]
cb = np.array(ch1)
ds = downSample(cb, sampFreq/(2*FPS/md))  # (44100Hz || 25fps || md = 5) => 4410
np.savetxt(DIR + NAME_TAG + "_aud.txt", ds, fmt='%f')


################## Peaks and Plots ###################

audTimeArray = np.arange(0, ds.shape[0], 1)

audPeaks = peakutils.indexes(ds, thres=0.3, min_dist=300.0)  # thres=0.2~0.3, min_dist=200.0~500.0
np.savetxt(DIR + NAME_TAG + "_audPeaks.txt", audPeaks/2, fmt='%d')
x = audPeaks
y = [ds[j] for j in audPeaks]
y = mulArr(y, 50)
ds = mulArr(ds, 50)

mpl.rcParams['agg.path.chunksize'] = 10000
plt.ion()
plt.scatter(x/2, y, color='b')
plt.plot(audTimeArray/2, ds, color='r')


vidTimeArray = np.arange(0, len(W), 1)
whiteIntList = map(int, W) # in py2: whiteIntList = map(int, W)  # in py3: whiteIntList = list(map(int, W))
whitesArray = np.array(whiteIntList)
vidPeaks = peakutils.indexes(whitesArray, thres=0.05, min_dist=30.0)
np.savetxt(DIR + NAME_TAG + "_vidPeaks.txt", vidPeaks, fmt='%d')
vid_x = vidPeaks
vid_y = [whitesArray[j] for j in vidPeaks]


mpl.rcParams['agg.path.chunksize'] = 10000
plt.scatter(vid_x, vid_y, color='g')
plt.plot(vidTimeArray, whitesArray, color='k')
plt.ylim(0)
# plt.ylim(0, 3000)  # if there is an inorganic peak, limit the y-axis for better view
plt.xlabel('frame (5fps)')
plt.title('Noise: black | Peaks: blue')
plt.show()
plt.savefig(PLOT_PATH, bbox_inches='tight', dpi=200)
plt.close()


################## Extracting Parts of Interest ###################

c = 1
for i in vidPeaks:
    r = range(max(i-30, 0), i)
    for j in audPeaks:
        if j/2 in r:
            t1 = max(i-40, 0)/5
            t2 = min(i+15, len(vidTimeArray)-1)/5
            ffmpeg_extract_subclip(VIDEO_PATH, t1, t2,
                                   targetname=DIR + NAME_TAG + "_crop_" + str(c) + "_" + str(t1) + "sec.mp4")
            c += 1


################## Extracting Parts of Interest FROM AUDIO ONLY ###################

# c = 1
#
# for j in audPeaks:
#     t1 = max(j-40, 0)/5
#     t2 = min(j+15, len(audTimeArray)-1)/5
#     ffmpeg_extract_subclip(VIDEO_PATH, t1, t2,
#                            targetname=DIR + NAME_TAG + "_aud_crop_" + str(c) + "_" + str(t1) + "sec.mp4")
#     c += 1


######################### Batman Ends #############################

print ""
print ""
print "Found BATMAN, look in " + DIR
print ""
print "      _==/           i     i           \\==_"
print "     /XX/            |\\___/|            \\XX\\"
print "   /XXXX\\            |XXXXX|            /XXXX\\"
print "  |XXXXXX\\_         _XXXXXXX_         _/XXXXXX|"
print " XXXXXXXXXXXxxxxxxxXXXXXXXXXXXxxxxxxxXXXXXXXXXXX"
print "|XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|"
print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
print "|XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX|"
print " XXXXXX/^^^^^\\XXXXXXXXXXXXXXXXXXXXX/^^^^^\\XXXXXX"
print "  |XXX|       \\XXX/^^\\XXXXX/^^\\XXX/       |XXX|"
print "    \\XX\\       \\X/    \\XXX/    \\X/       /XX/"
print "       \"\\       \"      \\X/      \"       /\""
print "                        \""
print "                                         Ben Arnon, 2017"
