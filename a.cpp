void CaptureVideo::CaptureProc() {
	AVFrame* pFrameBuf = av_frame_alloc();
	AVFrame* pFrameBuf_cali = av_frame_alloc();
	memcpy(pFrameBuf->linesize, pFrame->linesize, AV_NUM_DATA_POINTERS * sizeof(int));
	memcpy(pFrameBuf_cali->linesize, pFrame->linesize, AV_NUM_DATA_POINTERS * sizeof(int));
	std::vector<std::future<void>> capFuture(cam_nums_);
	float totalClock = 0.0;
	while (init_succ)
	{
		if(CamerasIsGrabbing()){

			string errormessage;
			bool iscali= isCalibrate;
			pFrameBuf->data[0] = pFrameBuffer->back(0);
			pFrameBuf->data[1] = pFrameBuffer->back(1);
			//线程池同步提交任务
			bool detectSucess  = true;
			for (int cam_idx = 0; cam_idx < cam_nums_; ++cam_idx) {
				capFuture[cam_idx] = globalExecutor.commit(CameraCapture, GrabRes_.data(), pSubFrames, pFrameBuf, cam_idx, iscali, &detectSucess, &errormessage);
			}
			for (int cam_idx = 0; cam_idx < cam_nums_; ++cam_idx) {
				capFuture[cam_idx].get();
			}

			if (iscali) {
				for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
					H_matrix[cam_idx].release();
				}
				message.clear();
				if(detectSucess){
					std::vector<std::future<void>> calcFuture(cam_nums_);
					for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
						cout << cam_idx << endl;
						calcFuture[cam_idx] = globalExecutor.commit(CalcHmat, cam_idx);

					}
					for (int cam_idx = 0; cam_idx < NUM_CAMERAS; ++cam_idx) {
						calcFuture[cam_idx].get();
					}
				}
				isCalibrate = false;
				iscali = isCalibrate;
				if (errormessage.empty()) {
					message = "标定完成，可以执行应用程序！";
					FileStorage H_Info;
					H_Info.open("D://H//H_mat.xml", cv::FileStorage::WRITE);

					for (int i = 0; i < NUM_CAMERAS; i++)
					{
						H_Info << "H_mat" + to_string(i) << H_matrix[i];
					}
				}
				else {
					String_t SerialNumber = cameras_[atoi(errormessage.c_str())].GetDeviceInfo().GetSerialNumber();
					message = std::string("相机SN[") + SerialNumber.c_str() + "] 标定失败，确认相机该是否覆盖全部标定版，或者灯光是否足够亮！";
				}
			}
			pFrameBuffer->push_back();
			pFrameBuffer->push_back();
			
			video_mat_.zeros(FRAME_H, FRAME_W, CV_8UC3);
			cv::Mat bgr_mat(FRAME_H, FRAME_W, CV_8UC3);
			cv::Mat yuv_mat(FRAME_H * 3 / 2, FRAME_W, CV_8UC1);
			CHECK(cudaMemcpy2D(yuv_mat.data, yuv_mat.step, pFrameBuf->data[0], pFrameBuf->linesize[0], FRAME_W, FRAME_H, cudaMemcpyDefault));
			CHECK(cudaMemcpy2D(yuv_mat.data + yuv_mat.step * FRAME_H, yuv_mat.step, pFrameBuf->data[1], pFrameBuf->linesize[1], FRAME_W, FRAME_H / 2, cudaMemcpyDefault));
			cv::cvtColor(yuv_mat, video_mat_, cv::COLOR_YUV2BGR_NV12);


			pFrameBuffer->front(0);
			pFrameBuffer->front(1);
			pFrameBuffer->pop();
			pFrameBuffer->pop();

		}
	}
