import torch
from torch.utils.data import DataLoader
import CLIP.clip as clip
from lavis.models import load_model_and_preprocess
from tqdm import tqdm 
from vqav2_dataloader import VQAv2Dataset,collater
from patchify_img import image_to_patches




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


clip_model, clip_preprocess = clip.load("RN50")
itm_model, itm_vis_processors, itm_txt_processors = load_model_and_preprocess(name="pnp_vqa", model_type="base", is_eval=True, device=device)

vqav2=VQAv2Dataset(root="dataset",image_transforms=itm_vis_processors["eval"],question_transforms=itm_txt_processors["eval"])

test_loader= DataLoader(vqav2, batch_size=1, shuffle=True,collate_fn=collater)

print("Total Test Batches", len(test_loader)) 

classification = 0
classification_top5 = 0

test_loader_loop = tqdm(test_loader)

with torch.no_grad():
    for batch in test_loader_loop:
        
        image_tensor=batch['image'].to(device)
        question=batch['question']
        
        samples = {"image": image_tensor, "text_input": question}
        
        samples = itm_model.forward_itm(samples=samples)
        gradcam = samples['gradcams']
        
        
        
        scores,scores_idx=torch.sort(gradcam.squeeze(0), descending=True)
        
        
        patches=image_to_patches(image_tensor).squeeze(0)
        
        relevant_patches=patches.clone()
        relevant_patches=relevant_patches[scores_idx]
        
        
        k=20
        sampled_patches=relevant_patches.clone()
        sampled_patches=sampled_patches[0:k]
        
        clip_patches=sampled_patches.clone()
        
        patches_clip=[clip_preprocess(clip_patches[i,:,:,:]) for i in range(clip_patches.shape[0])]


        patches_clip=torch.stack(patches_clip)
        image_input=patches_clip.to(device)
    
        options=batch['ans_choices'][0] 

        # Collapse List of tuples into a single list
        #options=list(sum(options, ())) 

        correct_answer= batch['correct_ans'][0] 

        if correct_answer not in options: 
            options.append(correct_answer) 

        correct_answer_idx=options.index(correct_answer)

        prompt = batch["prompt"][0][0] 

        prompt_split=prompt.split("<extra_id_0>") 

        prompt_prefix,prompt_suffix=prompt_split[0],prompt_split[-1]

        correct_answer_idx = torch.tensor(correct_answer_idx).to(device)

        text_inputs = torch.cat(
            [clip.tokenize(prompt_prefix+option+prompt_suffix ) for option in options]
        ).to(device)
        
        

        image_features = clip_model.encode_image(image_input)
        text_features = clip_model.encode_text(text_inputs)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        
        # Average scores across patches 
        similarity=torch.mean(similarity,dim=0,keepdims=True) 
        pred_labels = torch.argmax(similarity, dim=1)
        
        # # If majority vote rule 
        # pred_labels=torch.argmax(similarity,dim=1)  
        # majority_label=torch.mode(pred_labels) 
        
        
        
        if len(options)<5: 
            pred_labels_top5=similarity.topk(len(options),1,largest=True,sorted=True)[1]
        else:
            pred_labels_top5 = similarity.topk(5, 1, largest=True, sorted=True)[1]
        classification += torch.sum(correct_answer_idx == pred_labels)
        classification_top5 += torch.sum(
            torch.any(correct_answer_idx.reshape(-1, 1) == pred_labels_top5, dim=1)
        )

        test_loader_loop.set_postfix(
            classification=classification.item(),
            classification_top5=classification_top5.item(),
        )
print("\n")
print("Accuracies")
print("Top-1 Accuracy %.2f" % (100 * classification.item() / len(test_loader)))
print("Top-5 Accuracy %.2f" % (100 * classification_top5.item() / len(test_loader)))
