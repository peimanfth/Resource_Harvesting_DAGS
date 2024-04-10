from invoker_v2 import VA_ImgRec_input_extractor, VA_input_extractor


#test case for VideoAnalytics feature extraction
def test_VA_input_extractor():
    DAG_Input = {
        "video": "car",
        "num_frames": 3
    }
    func_Input = {'request_ids': ['58fdb1e2-8565-4b3b-a083-244da2c861e9-recog1', '58fdb1e2-8565-4b3b-a083-244da2c861e9-recog2'], 'num_frames': 2, 'db_name': 'video-bench', 'images': [['2.jpg'], ['0.jpg', '1.jpg']]}
    idx = 1
    assert VA_input_extractor(DAG_Input) == 3
    assert VA_ImgRec_input_extractor(func_Input,idx) == 2

if __name__ == "__main__":
    test_VA_input_extractor()
    print("Video Analytics test cases passed")