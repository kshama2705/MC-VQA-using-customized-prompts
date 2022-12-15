import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import json
from pathlib import Path
import glob

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

def get_ans_choices(dct):
    lst = [x["answer"] for x in dct]
    return list(set(lst))

class VQAv1Dataset(Dataset):
    """
     Args:
        root (string): Root directory of the VQAv2 Dataset.
        split (string, optional): The dataset split, either "train" (default) or "val"
        image_transforms: torchvision transforms to apply to the images
        question_transforms: 
    """

    IMAGE_PATH = {"train": ("train2014", "MultipleChoice_mscoco_train2014_questions.json", "mscoco_train2014_annotations.json"), 
                  "val": ("val2014", "MultipleChoice_mscoco_val2014_questions.json", "mscoco_val2014_annotations.json"),  
                  "test": ("test2015", "MultipleChoice_mscoco_test2014_questions.json")}

    def __init__(self,root,split="val",image_transforms=None,question_transforms=None,answer_selection=most_common_from_dict,tokenize=None,get_ans_choices=get_ans_choices):
        self.split=split
        self.image_transforms=image_transforms
        self.question_transforms=question_transforms
        self.answer_selection=answer_selection
        self.tokenize=tokenize
        self.get_ans_choices=get_ans_choices


        self.root=root

        #IMAGES
        #images=sorted(glob.glob(str(root/f"{self.IMAGE_PATH[self.split][0]}/*.jpg")))

        # Questions
        path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[split][1]))
        with open(path, 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data["questions"])
        df["image_path"] = df["image_id"].apply(lambda x: f"{self.IMAGE_PATH[split][0]}/COCO_{self.IMAGE_PATH[split][0]}_{x:012d}.jpg")
        
        #ANNOTATIONS
        annotations_path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[split][2]))

        with open(annotations_path, 'r') as f:
            data=json.load(f)
        df_annotations = pd.DataFrame(data["annotations"])

        df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='left')
        df["image_id"] = df["image_id_x"]
        if not all(df["image_id_y"] == df["image_id_x"]):
            print("There is something wrong with image_id")
        del df["image_id_x"]
        del df["image_id_y"]
        self.df = df
        self.n_samples = self.df.shape[0]

        print("Loading VQAv2 Dataset done")

    
    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["image_id"]
        image_path = self.df.iloc[index]["image_path"]
        question = self.df.iloc[index]["question"]
        question_id = self.df.iloc[index]["question_id"]
        split = self.split   
        main_answer = self.df.iloc[index]["multiple_choice_answer"] # Already extracted main answer

        answers = self.df.iloc[index]["answers"] # list of dicts: [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}, ...]
        #selected_answers = self.answer_selection(self.df.iloc[index]["answers"]) # Apply answer_selection() function to list of dict
        ans_choices=self.df.iloc[index]['multiple_choices']
        #ans_choices=[ans[0] for ans in ans_choices]
        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.root, image_path))        
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.load()
        
        if self.image_transforms:
            img = self.image_transforms(img)

        # Load, transform and tokenize question
        if self.question_transforms: 
            question = self.question_transforms(question)
        if self.tokenize:
            question = self.tokenize(question)

        # Return
        
        #return {"img": img, "image_id": image_id, "question_id": question_id, "question": question, 
        #        "main_answer": main_answer, "ans_choices":ans_choices, "answers": answers, "selected_answers": selected_answers}

        return {"img": img, "image_id": image_id, "question_id": question_id, "question": question, 
                "main_answer": main_answer, "ans_choices":ans_choices, "answers": answers}
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples







