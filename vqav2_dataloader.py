import torch,torchvision
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

class VQAv2Dataset(Dataset):
    """
     Args:
        root (string): Root directory of the VQAv2 Dataset.
        split (string, optional): The dataset split, either "train" (default) or "val"
        image_transforms: torchvision transforms to apply to the images
        question_transforms: 
    """

    IMAGE_PATH = {"train": ("train2014", "v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json"), 
                  "val": ("val2014", "v2_OpenEnded_mscoco_val2014_questions.json", "v2_mscoco_val2014_annotations.json"),  
                  "test": ("test2015", "v2_OpenEnded_mscoco_test2014_questions.json")}

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
        #annotations_path = os.path.expanduser(os.path.join(root, self.IMAGE_PATH[split][2]))

        #with open(annotations_path, 'r') as f:
        #    data=json.load(f)
        #df_annotations = pd.DataFrame(data["annotations"])

        #Questions- Annotations
        qna_path = "CPL-master/CPL/prompt/other_T5filtered_answers.json"

        with open(qna_path) as f:
            data = [json.loads(line) for line in f]

        #with open("yes_no_10k_qns.json") as f:
        #    yes_no_data = json.load(f)

        

        df_annotations = pd.DataFrame(data)
        df_annotations['qid']=df_annotations['qid'].astype(int)
        
        #df_yesno_annotations = pd.DataFrame(yes_no_data)
        #df_yesno_annotations['qid']=df_yesno_annotations['qid'].astype(int)


        df = pd.merge(df, df_annotations, left_on='question_id', right_on='qid', how='left')
        #df=pd.merge(df, df_yesno_annotations, left_on='question_id', right_on='qid', how='left')

        df=df.dropna()  #Removes the rows with missing values
        #print(df)
        #df = df[df['image'].notna()]
     

        """
        df["image_id"] = df["image_id_x"]
        if not all(df["image_id_y"] == df["image_id_x"]):
            print("There is something wrong with image_id")
        del df["image_id_x"]
        del df["image_id_y"]
        """
        self.df = df
        self.n_samples = self.df.shape[0]

        print("Loading VQAv2 Dataset done")

    
    def __getitem__(self, index):


        # image input
        img_path=self.df.iloc[index]["image_path"]
        image_id = self.df.iloc[index]["image_id"]
        image_path = self.df.iloc[index]["image_path"]
        question = self.df.iloc[index]["question_x"]
        question_id = self.df.iloc[index]["question_id"]
        split = self.split   
        #main_answer = self.df.iloc[index]["multiple_choice_answer"] # Already extracted main answer
        main_answer = self.df.iloc[index]["answer"] # Already extracted main answer

        ans_choices=self.df.iloc[index]['labels']
        prompts=self.df.iloc[index]['prompts']
        #ans_choices=[ans[0] for ans in ans_choices]
        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.root, image_path))        
        with open(image_path, "rb") as f:
            img = Image.open(f)
            img.load()
        
        if img.mode=="L":
            img=img.convert('RGB')
        
        if self.image_transforms:
            img = self.image_transforms(img)
            
        if self.question_transforms:
            question=self.question_transforms(question)

        # Load, transform and tokenize question
        if self.question_transforms: 
            question = self.question_transforms(question)
        if self.tokenize:
            question = self.tokenize(question)

        # Return
        
        #return {"img": img, "image_id": image_id, "question_id": question_id, "question": question, 
        #        "main_answer": main_answer, "ans_choices":ans_choices, "answers": answers, "selected_answers": selected_answers}

        return {"img": img, "question": question, "img_path":img_path,
                "main_answer": main_answer, "ans_choices":ans_choices, "prompts":prompts}
    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


def collater(samples):
        image_list, question_list, answer_list, correct_ans,prompt_list = [], [], [], [], []
        

        
        for sample in samples:
            image_list.append(sample["img"])
            question_list.append(sample["question"])
            correct_ans.append(sample['main_answer'])

            answers = sample["ans_choices"]

            answer_list.append(answers)
            
            prompt_list.append(sample["prompts"])

        return {
            "image": torch.stack(image_list, dim=0),
            "question": question_list,
            "correct_ans": correct_ans,
            "ans_choices": answer_list,
            "prompt":prompt_list
        }
    
