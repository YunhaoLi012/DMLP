import json



class FileConverter(object):
    """
    Container for data
    TODO: subclass for each task
    """
    
    def __init__(self) -> None:
        pass

    def __next__(self):
        pass
    



class QAFileConvertor(FileConverter):
    """
    A tool for combining Q&A dataset into one generator.
    Question_file: A list of dictionaries with context, question, potential answers. 
    Example: {"context": "Tracy didn't go home that evening and resisted Riley's attacks.", 
                "question": "What does Tracy need to do before this?", 
                "answerA": "make a new plan", 
                "answerB": "Go home and see Riley", 
                "answerC": "Find somewhere to go"}
    Answer_file: A .lst file with correct answers

    Output: A generator of dictionaries where each dictionary has the question and answer.
    Example: {"context": "Tracy didn't go home that evening and resisted Riley's attacks.", 
                "question": "What does Tracy need to do before this?", 
                "answerA": "make a new plan", 
                "answerB": "Go home and see Riley", 
                "answerC": "Find somewhere to go",
                "answer": 3}

    Source of data: https://leaderboard.allenai.org/socialiqa/submissions/get-started
    """
    
    def __init__(self, question_path, answer_path):
        super().__init__()
        with open(question_path, "r") as q_file:
            self.__questions = (json.loads(q) for q in list(q_file))
        
        with open(answer_path, "r")  as a_file:
            answers = list(a_file)
            self.__answers = (int(a[0]) for a in answers)

    def __next__(self):
        q = next(self.__questions)
        a = next(self.__answers)
        q['answer'] = a
        return q
    
    def convert(self, target_file_path):
        """
        Create the converted dataset and save it into file_path
        """
        with open(target_file_path,"w") as file:
            ii = 1
            while ii:
                ii=next(self,0)
                if ii !=0:
                    out = {}
                    out['context'] = ii['context']
                    out['question'] = ii['question']
                    if ii['answer'] == 1:
                        out['answer'] = ii['answerA']
                    elif ii['answer'] ==2:
                        out['answer'] = ii['answerB']
                    else:
                        out['answer'] = ii['answerC']
                    json.dump(out, file)
                    file.write('\n')


class Tokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, container):
        """
        dataset should be a DataContainer object
        TODO: Alternatively, can we just ask the user to pass in a file path instead of a container?
        """
        text=1
        small_instance_list = []
        while text!=0:
            text = next(container,0)
            if text!=0:
                context = text['context']
                question = text['question']
                answer = text['answer']
                tokenized_context = self.tokenizer(context,padding="max_length", truncation=True,max_length=16)
                tokenized_question = self.tokenizer(question, padding = 'max_length',truncation = True, max_length = 16)
                tokenized_answer = self.tokenizer(answer, padding='max_length', truncation=True, max_length = 16)
                ### Check the tokenized output first, this is not a single array.
                small_instance(context,question,answer,tokenized_context,tokenized_question,tokenized_answer)
                small_instance_list.append(small_instance)

        return small_instance_list

# For vae.py
import torch
def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        return m + torch.log(sum_exp)