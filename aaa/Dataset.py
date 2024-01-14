import json



class DataContainer(object):
    """
    Container for data
    TODO: subclass for each task
    """
    
    def __init__(self):
        return None

    def __next__(self):
        return None
    



class QAContainer(DataContainer):
    """
    QAContainer for Q&A Data. Dataset should be stored in jsonl file, with each dictionary in the 
    format of {'context':context, 'question':question,'answer':answer}
    TODO: assertion
    """
    
    def __init__(self, filepath):
        super().__init__()
        with open(filepath, "r") as q_file:
            self.__questions = (json.loads(q) for q in list(q_file))
        

    def __next__(self):
        q = next(self.__questions)
        return q


        
