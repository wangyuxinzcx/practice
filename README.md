//单个相机的图像采集
void CaptureVideo::CameraCapture(CGrabResultPtr* GrabRes, AVFrame** tarFrames,AVFrame* finalFrame, int cam_idx,bool isCali, bool *detectSucess,string *errorMessage)
{
	cameras_[cam_idx].RetrieveResult(5000, GrabRes[cam_idx], TimeoutHandling_ThrowException);
	if (GrabRes[cam_idx]->GrabSucceeded())
	{
		cv::cuda::GpuMat& bayerGR_mat = FrameProc::bayerGR_mat[cam_idx];
		cv::cuda::GpuMat& bgr_mat = FrameProc::bgr_mat[cam_idx];
		cv::cuda::GpuMat& bgr_mat_res = FrameProc::bgr_mat_res[cam_idx];
		int offset = (CAPTURE_H - VIEW_H)/2 * bgr_mat_res.step + (CAPTURE_W - VIEW_W) / 2 * 3;

		AVFrame* tarFrame = tarFrames[cam_idx];
		CHECK(cudaMemcpy2D(bayerGR_mat.data, bayerGR_mat.step, GrabRes[cam_idx]->GetBuffer(),
			GrabRes[cam_idx]->GetWidth(), CAPTURE_W, CAPTURE_H, cudaMemcpyDefault));
		cv::cuda::cvtColor(bayerGR_mat, bgr_mat, cv::COLOR_BayerRGGB2BGR);
		if (isCali)
		{
			cv::Mat bgrmat;
			bgr_mat.download(src_gray_mat_[cam_idx]);
			*detectSucess = capture_and_detect(src_gray_mat_[cam_idx], cam_idx,errorMessage);
		}
		cv::Mat h_frame;
		bgr_mat.download(h_frame); // 首先将GpuMat下载到CPU
		// cv::imwrite（to_string(cam_idx) + "");
		// 在CPU上的Mat对象上绘制字符串
		String_t SerialNumber = cameras_[cam_idx].GetDeviceInfo().GetSerialNumber();
		string SN =std::string("SN:") + SerialNumber.c_str();
		int font_face = cv::FONT_HERSHEY_DUPLEX;
		double font_scale = 3.0;
		cv::Point text_position((CAPTURE_W- VIEW_W)/2 + 80, CAPTURE_H - VIEW_H - 200  );
		cv::Scalar text_color(255, 255, 255); // 白色文本
		cv::putText(h_frame, SN, text_position, font_face, font_scale, text_color);
		bgr_mat.upload(h_frame);

		CHECK(BGRToNV12(tarFrame->data[0], tarFrame->linesize[0], tarFrame->data[1], tarFrame->linesize[1],
			bgr_mat.data + offset, bgr_mat.step, VIEW_W, VIEW_H));

		FrameProc::frame_stitch(finalFrame, tarFrame, cam_idx);
	}
	else
	{
		cout << "Error: " << std::hex << GrabRes[cam_idx]->GetErrorCode() << std::dec << " " << GrabRes[cam_idx]->GetErrorDescription() << endl;
	}
}
