import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class Predictpipeline:
    def __init__(self):
        pass
    
    def predict(self,feature):
        try:
            model_path = "artifacts\model.pkl"
            preprocessor_path = "artifacts\preprocessor.pkl"
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scale = preprocessor.transform(feature)
            preds = model.predict(data_scale)
            return preds
        except Exception as e:
              raise CustomException(e,sys)

class Customdata:
    def  __init__(self, 
                  gender : str,
                  race_ethnicity : str,
                  parental_level_of_education ,
                  lunch :int,
                  test_preparation_coures:int,
                  reading_score: int,
                  writing_score: int):

                  self.gender  = gender
                  self.race_ethnicity = race_ethnicity 
                  self.parental_level_of_education =parental_level_of_education
                  self.lunch = lunch
                  self.test_preparation_coures = test_preparation_coures 
                  self.reading_score = reading_score
                  self.writing_score =  writing_score
    def  get_data_as_data_frame(self):
        try:
              Custom_data_input_dict = {
                    "gender" : [self.gender],
                    "race_ethnicity" : [self.race_ethnicity],
                    "parental_level_of_education" : [self.parental_level_of_education],
                    "lunch" : [self.lunch],
                    "test_preparation_coures" : [ self.test_preparation_coures],
                    "reading_score" : [self.reading_score],
                    "writing_score" : [self.writing_score]
                    }
              return pd.DataFrame(Custom_data_input_dict)
        except Exception as e:
              raise CustomException(e,sys)
        
        
        
       