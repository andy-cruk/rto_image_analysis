__author__ = 'paters01'

class ImageAggregation:

    def __init__(self):
        # raw data for class
        #
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
        _cancer_yes = 0 # metadata_answer_counts_a_1_1
        _cancer_no = 0 # metadata_answer_counts_a_1_2
        _stained_na = 0 # metadata_answer_counts_a_2_0
        _stained_none = 0 # metadata_answer_counts_a_2_1
        _stained_1_25 = 0 # metadata_answer_counts_a_2_2
        _stained_25_50 = 0 # metadata_answer_counts_a_2_3
        _stained_50_75 = 0 # metadata_answer_counts_a_2_4
        _stained_75_95 = 0 # metadata_answer_counts_a_2_5
        _stained_95_100 = 0 # metadata_answer_counts_a_2_6
        _bright_na = 0 # metadata_answer_counts_a_3_0
        _bright_weak = 0 # metadata_answer_counts_a_3_1
        _bright_medium = 0 # metadata_answer_counts_a_3_2
        _bright_strong = 0 # metadata_answer_counts_a_3_3
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

class Core:

    def __init__(self):
        # collection of aggregations for this sample
        _image_aggregations = None
        pass

    def calculate_medians(self):
        pass

def create_image_aggregations():
    # create empty dictionary
    # connect to mongo raw image data
    # for each row in raw image data
        # get subject id
        # if subject id in dictionary
            # get value (ImageAggregation object)
        # else (new image)
            # create new ImageAggregation object
            # do any initialisation required
            # add this to dictionanry with subject id as key
        # process row to increment counts in ImageAggregation object



