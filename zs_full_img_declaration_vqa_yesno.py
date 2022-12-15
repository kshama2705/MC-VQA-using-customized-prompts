import torch
from torch.utils.data import DataLoader
import CLIP.clip as clip
from tqdm import tqdm 
from vqav2_dataloader_yesno import VQAv2Dataset,collater




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


clip_model, clip_preprocess = clip.load("RN50")


vqav2=VQAv2Dataset(root="dataset",image_transforms=clip_preprocess)

test_loader= DataLoader(vqav2, batch_size=1, shuffle=True)

print("Total Test Batches", len(test_loader)) 

classification = 0
classification_top5 = 0

test_loader_loop = tqdm(test_loader)

with torch.no_grad():
    for batch in test_loader_loop:
        
        image_tensor=batch['img'].to(device)
        question=batch['question']
        options=["yes","no"]
        correct_answer= batch['main_answer'][0] 
        
        prompts = batch["prompts"] 


        correct_answer_idx=0

        image_input = image_tensor.to(device)
        correct_answer_idx = torch.tensor(correct_answer_idx).to(device)

        text_inputs = torch.cat(
            [clip.tokenize(prompt) for prompt in prompts]
        ).to(device)
        
        

        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        pred_labels = torch.argmax(similarity, dim=1)
       
        classification += torch.sum(correct_answer_idx == pred_labels)

        test_loader_loop.set_postfix(
            classification=classification.item(),
        )
print("\n")
print("Accuracies")
print("Top-1 Accuracy %.2f" % (100 * classification.item() / len(test_loader)))