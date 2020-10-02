import cv2
import numpy as np

if __name__ == '__main__':

	cam = 0
	cap=cv2.VideoCapture(cam,cv2.CAP_DSHOW)#电脑自带摄像头为0,usb外接摄像头为1
	
	K = np.array([[246.0,0.0,320.0],[0.0,251.7,236.4],[0.0,0.0,1.0]])
	D = np.array([[-0.05727062404195109],[0.031232780647544825],[-0.07873088848068864],[0.057029373751815604]])

	width,height= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	#优化内参教和畸变系数
	P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K,D,(width,height),None)

	#此处计算花费时间较大，需从循环中抽取出来
	mapx2,mapy2=cv2.fisheye.initUndistortRectifyMap(K,D,None,P,(width,height),cv2.CV_32F)
	print('mapx2=',mapx2)
	print('mapy2=',mapy2)

	while (True):
		ret,frame = cap.read()
		cv2.imshow('raw',frame)

		#畸变矫正
		frame_rectified = cv2.remap(frame,mapx2,mapy2,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

		cv2.namedWindow("RectifiedImage",cv2.WINDOW_FREERATIO)
		cv2.resizeWindow("RectifiedImage",960,720)#调整显示窗口大小
		cv2.imshow('RectifiedImage',frame_rectified)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break 
	cap.release()
	cv2.destroyAllWindows()