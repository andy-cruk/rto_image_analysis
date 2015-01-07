__author__ = 'paters01'

class ImageAggregation:

    def __init__(self):
        # raw data for class
        _id = 0
        _activated_at = 0
        _classification_count = 0
        _created_at = 0
        _group_id = 0
        _group_name = 0
        _group_zooniverse_id = 0
        _group_id = 0
        _location_standard = 0
        _location_thumbnail = 0
        _meatdata_collection = 0
        _meatdata_id_no = 0
        _meatdata_index = 0
        _meatdata_orig_directory = 0
        _meatdata_orig_file_name = 0
        _meatdata_score = 0
        _meatdata_stain_type = 0
        _metadata_answer_counts_a_1_1 = 0 # cancer_yes
        _metadata_answer_counts_a_1_2 = 0 # cancer_no
        _metadata_answer_counts_a_2_0 = 0 # stained_na
        _metadata_answer_counts_a_2_1 = 0 # stained_none
        _metadata_answer_counts_a_2_2 = 0 # stained_1_25
        _metadata_answer_counts_a_2_3 = 0 # stained_25_50
        _metadata_answer_counts_a_2_4 = 0 # stained_50_75
        _metadata_answer_counts_a_2_5 = 0 # stained_75_95
        _metadata_answer_counts_a_2_6 = 0 # stained_95_100
        _metadata_answer_counts_a_3_0 = 0 # bright_na
        _metadata_answer_counts_a_3_1 = 0 # bright_weak
        _metadata_answer_counts_a_3_2 = 0 # bright_medium
        _metadata_answer_counts_a_3_3 = 0 # bright_strong
        _metadata_collection = 0
        _metadata_id_no = 0
        _metadata_index = 0
        _metadata_orig_directory = 0
        _metadata_orig_file_name = 0
        _metadata_score = 0
        _metadata_stain_type = 0
        _project_id = 0
        _random = 0
        _state = 0
        _updated_at = 0
        _workflow_ids = 0
        _zooniverse_id = 0

    def aggregate_from_classifications(self):
        # need to write this if we need to process raw classification data
        for classification in self._classifications:
            pass

    def calculate_medians(self):
        self.calculate_median1()
        self.calculate_median2()

    def calculate_median1(self):
        pass

    def calculate_median2(self):
        pass

class Sample:

    def __init__(self):
        # collection of aggregations for this sample
        _image_aggregations = None
        pass

    def calculate_medians(self):
        pass


