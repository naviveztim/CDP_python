import unittest
from Utils.utils import from_ucr_txt, subsequent_distance
from ShapeletDataMining.shapelet_classifier import ShapeletClassifier


class TestPSO(unittest.TestCase):

    def test_read_from_ucr_txt(self):
        filepath = "D:\\Research\\Datasets\\UCR Time series\\UCR_TS_Archive_2015\\BeetleFly\\BeetleFly_TRAIN"
        test_pdf = from_ucr_txt(filepath)
        print(test_pdf.shape)
        print(test_pdf.head().to_string())

    def test_shapelet_classifier(self):
        # Define the train dataset of time series
        train_filepath = "D:\\Research\\Datasets\\UCR Time series\\UCR_TS_Archive_2015\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN_2_samples.txt"
        classifiers_folder = "./Classifiers"
        shapelet_classifier = ShapeletClassifier(class_index_a=1
                                                , class_index_b=2
                                                , dataset_filepath=train_filepath
                                                , classifiers_folder=classifiers_folder)

        shapelet = shapelet_classifier.find_shapelet()

        print(shapelet)

    '''
    def test_pso(self):

        # Define the train dataset of time series
        train_filepath = "D:\\Research\\Datasets\\UCR Time series\\UCR_TS_Archive_2015\\ItalyPowerDemand\\ItalyPowerDemand_TRAIN_2_samples.txt"
        train_time_series_dataframe = from_ucr_txt(train_filepath)

        min_train_value = min(train_time_series_dataframe['values'].explode())
        max_train_value = max(train_time_series_dataframe['values'].explode())

        # Start PSO algorithm to find shapelet that separate two classes
        shapelet_pso = ShapeletsPso(min_length=3
                                    , max_length=5
                                    , step=1
                                    , min_position=min_train_value
                                    , max_position=max_train_value
                                    , min_velocity=min_train_value
                                    , max_velocity=max_train_value
                                    , train_dataframe=train_time_series_dataframe)

        shapelet_pso.start_pso()

        # Test so found shapelet
        test_filepath = "D:\\Research\\Datasets\\UCR Time series\\UCR_TS_Archive_2015\\ItalyPowerDemand\\ItalyPowerDemand_TEST_2_samples.txt"
        test_time_series_dataframe = from_ucr_txt(test_filepath)

        test_time_series_dataframe = ShapeletsPso.norm(test_time_series_dataframe)

        # Check  split results
        class_a = []
        class_b = []
        for _, time_series in test_time_series_dataframe.iterrows():
            distance = subsequent_distance(time_series["values"], shapelet_pso.best_particle.best_position)

            print(f'distance: {distance}')
            if distance > shapelet_pso.best_particle.optimal_split_distance:
                class_a.append(time_series["class_index"])
            else:
                class_b.append(time_series["class_index"])

        print(class_a)
        print(class_b)

    def test_shapelet(self):
        sh1 = Shapelet(values=[1, 2, 3, 4]
                       , optimal_split_distance=0.5
                       , best_information_gain=0.9
                       , left_class_index=1
                       , right_class_index=0)
        sh2 = Shapelet(values=[1, 2, 3, 4]
                       , optimal_split_distance=0.5
                       , best_information_gain=0.9
                       , left_class_index=0
                       , right_class_index=1)

        assert(sh1.compare(sh2) == -1)
    '''


if __name__ == '__main__':
    unittest.main()


