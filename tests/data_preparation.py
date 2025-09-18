import sys, time, os, tqdm, glob, subprocess, warnings, cv2, pickle, numpy

from types import SimpleNamespace
from scipy import signal
from shutil import rmtree
from scipy.io import wavfile
from scipy.interpolate import interp1d

from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # for model import

from model.faceDetector.s3fd import S3FD

warnings.filterwarnings("ignore")

# =========================
# Global configuration
# Configure these variables instead of using CLI args.
# All other preprocessing settings (ffmpeg/scenedetect/S3FD, thresholds, etc.)
# are kept identical to Columbia_test.py
# =========================

# Input selection
VIDEO_NAME = "0003"            # Demo video name (without extension)
VIDEO_FOLDER = "demo"  # Path containing the input video and where outputs will be created

# Preprocessing parameters (identical defaults to Columbia_test.py)
N_DATALOADER_THREAD = 10       # Number of workers/threads used in ffmpeg calls
FACEDET_SCALE = 0.25           # Scale factor for face detection (frames scaled to 0.25 of orig.)
MIN_TRACK = 10                 # Minimum frames for a valid track/shot
NUM_FAILED_DET = 10            # Missed detections allowed before stopping a track
MIN_FACE_SIZE = 1              # Minimum face size in pixels
CROP_SCALE = 0.40              # Scale bounding box when cropping face clips

# Time range (0 means full video)
START = 0
DURATION = 0

# Output directory names (more human-readable than original)
MEDIA_DIRNAME = "media"              # Was: pyavi
FRAMES_DIRNAME = "frames"            # Was: pyframes
INTERMEDIATE_DIRNAME = "intermediate"# Was: pywork
FACECLIPS_DIRNAME = "face_clips"     # Was: pycrop


def _build_args() -> SimpleNamespace:
	"""Create a simple args-like object from global config for internal use."""
	save_path = os.path.join(VIDEO_FOLDER, VIDEO_NAME)
	media_path = os.path.join(save_path, MEDIA_DIRNAME)
	frames_path = os.path.join(save_path, FRAMES_DIRNAME)
	intermediate_path = os.path.join(save_path, INTERMEDIATE_DIRNAME)
	faceclips_path = os.path.join(save_path, FACECLIPS_DIRNAME)

	# Find input video (match any extension like .mp4, .avi)
	candidates = glob.glob(os.path.join(VIDEO_FOLDER, VIDEO_NAME + ".*"))
	if not candidates:
		raise FileNotFoundError(
			f"No input video found for '{VIDEO_NAME}' in '{VIDEO_FOLDER}'. Expected something like {VIDEO_NAME}.mp4"
		)
	video_path = candidates[0]

	return SimpleNamespace(
		# IO paths
		videoFolder=VIDEO_FOLDER,
		savePath=save_path,
		videoPath=video_path,
		mediaPath=media_path,
		framesPath=frames_path,
		intermediatePath=intermediate_path,
		faceClipsPath=faceclips_path,

		# Parameters
		nDataLoaderThread=N_DATALOADER_THREAD,
		facedetScale=FACEDET_SCALE,
		minTrack=MIN_TRACK,
		numFailedDet=NUM_FAILED_DET,
		minFaceSize=MIN_FACE_SIZE,
		cropScale=CROP_SCALE,
		start=START,
		duration=DURATION,
	)

def scene_detect(args):
	# CPU: Scene detection, output is the list of each shot's time duration
	videoManager = VideoManager([args.videoFilePath])
	statsManager = StatsManager()
	sceneManager = SceneManager(statsManager)
	sceneManager.add_detector(ContentDetector())
	baseTimecode = videoManager.get_base_timecode()
	videoManager.set_downscale_factor()
	videoManager.start()
	sceneManager.detect_scenes(frame_source = videoManager)
	sceneList = sceneManager.get_scene_list(baseTimecode)
	savePath = os.path.join(args.intermediatePath, 'scene.pckl')
	if sceneList == []:
		sceneList = [(videoManager.get_base_timecode(),videoManager.get_current_timecode())]
	with open(savePath, 'wb') as fil:
		pickle.dump(sceneList, fil)
		sys.stderr.write('%s - scenes detected %d\n'%(args.videoFilePath, len(sceneList)))
	return sceneList

def inference_video(args):
	# GPU: Face detection, output is the list contains the face location and score in this frame
	DET = S3FD(device='cuda')
	flist = glob.glob(os.path.join(args.framesPath, '*.jpg'))
	flist.sort()
	dets = []
	for fidx, fname in enumerate(flist):
		image = cv2.imread(fname)
		imageNumpy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		bboxes = DET.detect_faces(imageNumpy, conf_th=0.9, scales=[args.facedetScale])
		dets.append([])
		for bbox in bboxes:
					dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]}) # dets has the frames info, bbox info, conf info
		sys.stderr.write('%s-%05d; %d dets\r' % (args.videoFilePath, fidx, len(dets[-1])))
	savePath = os.path.join(args.intermediatePath,'faces.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(dets, fil)
	return dets

def bb_intersection_over_union(boxA, boxB, evalCol = False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou

def track_shot(args, sceneFaces):
	# CPU: Face tracking
	iouThres  = 0.5     # Minimum IOU between consecutive face detections
	tracks    = []
	while True:
		track     = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= args.numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > args.minTrack:
			frameNum    = numpy.array([ f['frame'] for f in track ])
			bboxes      = numpy.array([numpy.array(f['bbox']) for f in track])
			frameI      = numpy.arange(frameNum[0],frameNum[-1]+1)
			bboxesI    = []
			for ij in range(0,4):
				interpfn  = interp1d(frameNum, bboxes[:,ij])
				bboxesI.append(interpfn(frameI))
			bboxesI  = numpy.stack(bboxesI, axis=1)
			if max(numpy.mean(bboxesI[:,2]-bboxesI[:,0]), numpy.mean(bboxesI[:,3]-bboxesI[:,1])) > args.minFaceSize:
				tracks.append({'frame':frameI,'bbox':bboxesI})
	return tracks

def crop_video(args, track, cropFile):
	# CPU: crop the face clips
	flist = glob.glob(os.path.join(args.framesPath, '*.jpg')) # Read the frames
	flist.sort()
	vOut = cv2.VideoWriter(cropFile + 't.avi', cv2.VideoWriter_fourcc(*'XVID'), 25, (224,224))# Write video
	dets = {'x':[], 'y':[], 's':[]}
	for det in track['bbox']: # Read the tracks
		dets['s'].append(max((det[3]-det[1]), (det[2]-det[0]))/2) 
		dets['y'].append((det[1]+det[3])/2) # crop center x 
		dets['x'].append((det[0]+det[2])/2) # crop center y
	dets['s'] = signal.medfilt(dets['s'], kernel_size=13)  # Smooth detections 
	dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
	dets['y'] = signal.medfilt(dets['y'], kernel_size=13)
	for fidx, frame in enumerate(track['frame']):
		cs  = args.cropScale
		bs  = dets['s'][fidx]   # Detection box size
		bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount 
		image = cv2.imread(flist[frame])
		frame = numpy.pad(image, ((bsi,bsi), (bsi,bsi), (0, 0)), 'constant', constant_values=(110, 110))
		my  = dets['y'][fidx] + bsi  # BBox center Y
		mx  = dets['x'][fidx] + bsi  # BBox center X
		face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
		vOut.write(cv2.resize(face, (224, 224)))
	audioTmp    = cropFile + '.wav'
	audioStart  = (track['frame'][0]) / 25
	audioEnd    = (track['frame'][-1]+1) / 25
	vOut.release()
	command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 -threads %d -ss %.3f -to %.3f %s -loglevel panic" % \
		      (args.audioFilePath, args.nDataLoaderThread, audioStart, audioEnd, audioTmp)) 
	output = subprocess.call(command, shell=True, stdout=None) # Crop audio file
	_, audio = wavfile.read(audioTmp)
	command = ("ffmpeg -y -i %st.avi -i %s -threads %d -c:v copy -c:a copy %s.avi -loglevel panic" % \
			  (cropFile, audioTmp, args.nDataLoaderThread, cropFile)) # Combine audio and video file
	output = subprocess.call(command, shell=True, stdout=None)
	os.remove(cropFile + 't.avi')
	return {'track':track, 'proc_track':dets}

def extract_MFCC(file, outPath):
	# Deprecated in this data preparation script (ASD step removed).
	# Keeping as a stub in case you want MFCCs later.
	pass

"""
ASD model inference and visualization steps were intentionally removed.
This script now focuses purely on data preparation:
- Extract media (audio/video)
- Extract frames
- Scene detection
- Face detection
- Face tracking
- Crop per-face clips with synchronized audio
"""

pass

pass

def main():
	# This preprocessing is modified from the original Columbia_test script and
	# now focuses on preparing data only (no ASD/visualization).
	#
	# Output structure:
	#.
	# ├── media
	# │   ├── audio.wav (Audio extracted from input video)
	# │   └── video.avi (Copy/re-encoded input video)
	# ├── face_clips (Detected face clips: per-track audio+video)
	# │   ├── 00000.avi
	# │   ├── 00000.wav
	# │   ├── 00001.avi
	# │   ├── 00001.wav
	# │   └── ...
	# ├── frames (All frames from the video)
	# │   ├── 000001.jpg
	# │   ├── 000002.jpg
	# │   └── ...
	# └── intermediate
	#     ├── faces.pckl (face detection result)
	#     ├── scene.pckl (scene detection result)
	#     └── tracks.pckl (face tracking result)

	args = _build_args()

	# Clean output dir
	if os.path.exists(args.savePath):
		rmtree(args.savePath)

	# Create directories
	os.makedirs(args.mediaPath, exist_ok=True)
	os.makedirs(args.framesPath, exist_ok=True)
	os.makedirs(args.intermediatePath, exist_ok=True)
	os.makedirs(args.faceClipsPath, exist_ok=True)

	# Extract video
	args.videoFilePath = os.path.join(args.mediaPath, 'video.avi')
	if args.duration == 0:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -async 1 -r 25 %s -loglevel panic" %
				   (args.videoPath, args.nDataLoaderThread, args.videoFilePath))
	else:
		command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -ss %.3f -to %.3f -async 1 -r 25 %s -loglevel panic" %
				   (args.videoPath, args.nDataLoaderThread, args.start, args.start + args.duration, args.videoFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Extracted video to {args.videoFilePath}\n")

	# Extract audio
	args.audioFilePath = os.path.join(args.mediaPath, 'audio.wav')
	command = ("ffmpeg -y -i %s -qscale:a 0 -ac 1 -vn -threads %d -ar 16000 %s -loglevel panic" %
			   (args.videoFilePath, args.nDataLoaderThread, args.audioFilePath))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Extracted audio to {args.audioFilePath}\n")

	# Extract the video frames
	command = ("ffmpeg -y -i %s -qscale:v 2 -threads %d -f image2 %s -loglevel panic" %
			   (args.videoFilePath, args.nDataLoaderThread, os.path.join(args.framesPath, '%06d.jpg')))
	subprocess.call(command, shell=True, stdout=None)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Extracted frames to {args.framesPath}\n")

	# Scene detection for the video frames
	scene = scene_detect(args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Scene detection saved in {args.intermediatePath}\n")

	# Face detection for the video frames
	faces = inference_video(args)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Face detections saved in {args.intermediatePath}\n")

	# Face tracking
	allTracks, vidTracks = [], []
	for shot in scene:
		if shot[1].frame_num - shot[0].frame_num >= args.minTrack:
			# 'frame' indexes this track's timesteps; 'bbox' contains face locations
			allTracks.extend(track_shot(args, faces[shot[0].frame_num:shot[1].frame_num]))
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Face tracking found {len(allTracks)} tracks\n")

	# Face clips cropping (audio + video per track)
	for ii, track in tqdm.tqdm(enumerate(allTracks), total=len(allTracks)):
		vidTracks.append(crop_video(args, track, os.path.join(args.faceClipsPath, '%05d' % ii)))
	savePath = os.path.join(args.intermediatePath, 'tracks.pckl')
	with open(savePath, 'wb') as fil:
		pickle.dump(vidTracks, fil)
	sys.stderr.write(time.strftime("%Y-%m-%d %H:%M:%S") + f" Cropped face clips saved in {args.faceClipsPath}\n")

if __name__ == '__main__':
	main()
